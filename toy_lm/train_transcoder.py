"""Fit a transcoder to the MLP sublayer of a frozen transformer.

The transformer is only ever run forward: each batch it is fed a window of the
corpus, a hook keeps whatever went into and came out of `model.mlp`, and the
transcoder is trained to map the one to the other, position by position.

    python train_transcoder.py
    python train_transcoder.py --ckpt checkpoints/toy_lm_0.pt --steps 5000
"""

import argparse
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm

from dataloader import CorpusExhausted, StreamingCorpus
from model import Transformer
from transcoder import Transcoder
from utils import MetricsLog, load_tokenizer, lr_at, pick_device, set_seed

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
CONFIG_PATH = ROOT / "transcoder_config.yaml"
ARCH_KEYS = ("vocab_size", "embedding_dim", "num_heads", "seq_len")


def load_config(path):
    """Flatten the sectioned transcoder_config.yaml into one dict of argparse defaults."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    flat = {}
    for section in config.values():
        flat.update(section)
    return flat


class MLPTap:
    """Keeps the input and output of `mlp` from the most recent forward pass."""

    def __init__(self, mlp):
        self.acts = self.target = None
        mlp.register_forward_hook(self._capture)

    def _capture(self, module, args, output):
        self.acts, self.target = args[0], output


def loss_fn(transcoder, acts, target, l1_coeff):
    """Reconstruction plus an L1 pull towards sparsity, both summed over features
    and averaged over positions. Also returns the two numbers worth watching:
    fraction of the MLP output's variance left unexplained, and mean live
    features per position.
    """
    recon, hidden = transcoder(acts)
    mse = (recon - target).pow(2).sum(-1).mean()
    loss = mse + l1_coeff * hidden.abs().sum(-1).mean()
    fvu = mse / (target - target.mean((0, 1))).pow(2).sum(-1).mean()
    l0 = (hidden > 0).sum(-1).float().mean()
    return loss, fvu, l0


@torch.no_grad()
def evaluate(transcoder, model, tap, data, l1_coeff, batches=20):
    transcoder.eval()
    totals = torch.zeros(3)
    for _ in range(batches):
        inputs, _ = data.batch("val")
        model(tokens=inputs)
        totals += torch.tensor([v.item() for v in
                                loss_fn(transcoder, tap.acts, tap.target, l1_coeff)])
    transcoder.train()
    return (totals / batches).tolist()


def main():
    # transcoder_config.yaml supplies the defaults; anything passed on the CLI
    # overrides it. The transformer's own architecture comes from its checkpoint.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG_PATH)
    cfg = load_config(pre.parse_known_args()[0].config)

    sources = [name for name, key in (("english", "english_data"), ("code", "code_data"))
               if cfg[key]]
    if not sources:
        raise SystemExit("config: english_data and code_data are both false, "
                         "enable at least one")

    if not 0.0 <= cfg["min_lr_ratio"] <= 1.0:
        raise SystemExit("config: min_lr_ratio must be in 0..1, "
                         f"got {cfg['min_lr_ratio']}")

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--ckpt", type=Path, default=ROOT / cfg["ckpt"])
    p.add_argument("--steps", type=int, default=cfg["steps"])
    p.add_argument("--num-features", type=int, default=cfg["num_features"])
    p.add_argument("--epsilon", type=float, default=cfg["epsilon"])
    p.add_argument("--l1-coeff", type=float, default=cfg["l1_coeff"])
    p.add_argument("--shards", type=int, default=10, help="shards to stream, 1..10")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--log-name", type=str, default="transcoder")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "transcoder.pt")
    args = p.parse_args()

    if not args.ckpt.exists():
        raise SystemExit(f"no checkpoint at {args.ckpt} -- train a transformer first")

    set_seed(args.seed)
    device = pick_device()

    # weights_only would reject the Path objects the checkpoint keeps in "args"
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("config") or {}
    saved.setdefault("vocab_size", ckpt["model"]["lm_head.weight"].shape[0])
    arch = {k: saved[k] for k in ARCH_KEYS}

    tokenizer = load_tokenizer(args.tokenizer, arch["vocab_size"])
    vocab_size = tokenizer.get_vocab_size()
    assert vocab_size == arch["vocab_size"], (vocab_size, arch["vocab_size"])

    data = StreamingCorpus(tokenizer, arch["seq_len"], cfg["batch_size"], device,
                           english=cfg["english_data"], code=cfg["code_data"],
                           num_shards=args.shards, val_batches=cfg["eval_batches"])

    model = Transformer(vocab_size, arch["embedding_dim"], arch["num_heads"],
                        arch["seq_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval().requires_grad_(False)
    tap = MLPTap(model.mlp)

    transcoder = Transcoder(args.epsilon, arch["embedding_dim"], arch["embedding_dim"],
                            args.num_features).to(device)
    optimizer = torch.optim.AdamW(transcoder.parameters(), lr=cfg["lr"],
                                  betas=tuple(cfg["betas"]), weight_decay=0.0)

    params = sum(p.numel() for p in transcoder.parameters())
    print(f"device {device} | vocab {vocab_size} | "
          f"stream {'+'.join(sources)} x{args.shards} | "
          f"{args.num_features} features | {params/1e6:.1f}M params")
    print(f"loaded {args.ckpt}")

    metrics = MetricsLog(args.log_dir, args.log_name)
    print(f"logging to {metrics.path}")

    transcoder.train()
    start = time.time()
    step_tokens = cfg["batch_size"] * arch["seq_len"]
    bar = tqdm(total=args.steps * step_tokens, unit="tok", unit_scale=True,
               dynamic_ncols=True, desc="transcoder", smoothing=0.05)
    for step in range(1, args.steps + 1):
        try:
            inputs, _ = data.batch("train")
        except CorpusExhausted as exc:
            bar.write(f"stream exhausted at step {step} ({exc})")
            break

        with torch.no_grad():
            model(tokens=inputs)
        loss, fvu, l0 = loss_fn(transcoder, tap.acts, tap.target, args.l1_coeff)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # returns the total norm *before* clipping, which is what we log
        max_grad_norm = cfg["max_grad_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            transcoder.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"))

        lr = lr_at(step, cfg["lr"], cfg["warmup_steps"], args.steps, cfg["min_lr_ratio"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        train_loss = loss.item()
        bar.update(step_tokens)
        bar.set_postfix(loss=f"{train_loss:.3f}", fvu=f"{fvu.item():.3f}",
                        l0=f"{l0.item():.1f}", refresh=False)

        val_loss = None
        if step % cfg["eval_every"] == 0 or step == args.steps:
            val_loss, val_fvu, val_l0 = evaluate(transcoder, model, tap, data,
                                                 args.l1_coeff, cfg["eval_batches"])
            # bar.write keeps these lines from being overwritten by the bar
            bar.write(f"step {step:>6} | train {train_loss:.3f} | val {val_loss:.3f} "
                      f"| fvu {val_fvu:.3f} | l0 {val_l0:.1f} "
                      f"| {time.time() - start:.1f}s")

        metrics.log(step, train_loss, grad_norm.item(), lr,
                    time.time() - start, val_loss=val_loss)

    bar.close()
    metrics.close()

    data.close()
    print(f"streamed {data.tokens_seen/1e6:.1f}M tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"transcoder": transcoder.state_dict(), "args": vars(args),
                "config": cfg, "arch": arch}, args.out)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
