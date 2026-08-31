"""Forward-only counterpart to main.py: load a checkpoint, run N steps, plot the loss.

    python eval.py
    python eval.py --ckpt weights/toy_lm.pt --steps 5000
    python eval.py --english --no-code
"""

import argparse
import random
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm

import plot
from dataloader import CorpusExhausted, StreamingCorpus
from model import Transformer
from utils import MetricsLog, pick_device

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
CONFIG_PATH = ROOT / "config.yaml"
ARCH_KEYS = ("embedding_dim", "num_heads", "seq_len")


def default_checkpoint():
    """weights/ holds checkpoints pulled from the Hub, checkpoints/ ones trained here."""
    pulled = ROOT / "weights" / "toy_lm.pt"
    return pulled if pulled.exists() else ROOT / "checkpoints" / "toy_lm.pt"


def load_config(path):
    """Flatten the sectioned config.yaml into one dict of argparse defaults."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    flat = {}
    for section in config.values():
        flat.update(section)
    return flat


def loss_fn(model, inputs, targets):
    logits = model(tokens=inputs)
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))


def main():
    # config.yaml supplies the defaults, the checkpoint's own config overrides it,
    # and anything passed on the CLI wins over both.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG_PATH)
    pre.add_argument("--ckpt", type=Path, default=default_checkpoint())
    known = pre.parse_known_args()[0]

    if not known.ckpt.exists():
        raise SystemExit(f"no checkpoint at {known.ckpt} -- train one first")

    cfg = load_config(known.config)
    # weights_only would reject the Path objects the checkpoint keeps in "args"
    ckpt = torch.load(known.ckpt, map_location="cpu", weights_only=False)
    saved = ckpt.get("config") or {}
    stale = [k for k in ARCH_KEYS if k in saved and saved[k] != cfg.get(k)]
    cfg.update(saved)
    if stale:
        print(f"note: using checkpoint's {', '.join(stale)} over config.yaml")

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--steps", type=int, default=1000, help="forward passes to run")
    p.add_argument("--batch-size", type=int, default=cfg["batch_size"])
    p.add_argument("--eval-every", type=int, default=cfg["eval_every"],
                   help="how often to write a running-mean point")
    p.add_argument("--english", action=argparse.BooleanOptionalAction,
                   default=cfg["english_data"])
    p.add_argument("--code", action=argparse.BooleanOptionalAction,
                   default=cfg["code_data"])
    p.add_argument("--shards", type=int, default=1, help="shards to stream, 1..10")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--log-name", type=str, default="eval")
    p.add_argument("--prompt", type=str, default="To be or not to be, ")
    p.add_argument("--plot", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--ema-span", type=int, default=50)
    args = p.parse_args()

    sources = [name for name, on in (("english", args.english), ("code", args.code)) if on]
    if not sources:
        raise SystemExit("--no-english and --no-code leave nothing to evaluate, "
                         "enable at least one")

    torch.manual_seed(args.seed)
    random.seed(args.seed)  # the loader's shuffle pool draws from `random`
    device = pick_device()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()

    # val_batches=0: every batch here is already unseen, so a held-out split
    # would only take batches away from the run.
    data = StreamingCorpus(tokenizer, cfg["seq_len"], args.batch_size, device,
                           english=args.english, code=args.code,
                           num_shards=args.shards, val_batches=0)
    model = Transformer(vocab_size, cfg["embedding_dim"], cfg["num_heads"], cfg["seq_len"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | vocab {vocab_size} | "
          f"stream {'+'.join(sources)} x{args.shards} | {params/1e6:.1f}M params")
    print(f"loaded {args.ckpt}")

    metrics = MetricsLog(args.log_dir, args.log_name)
    print(f"logging to {metrics.path}")

    total = 0.0
    steps_done = 0
    start = time.time()
    step_tokens = args.batch_size * cfg["seq_len"]
    bar = tqdm(total=args.steps * step_tokens, unit="tok", unit_scale=True,
               dynamic_ncols=True, desc="eval", smoothing=0.05)
    with torch.no_grad():
        for step in range(1, args.steps + 1):
            try:
                inputs, targets = data.batch("train")
            except CorpusExhausted as exc:
                bar.write(f"stream exhausted at step {step} ({exc})")
                break

            loss = loss_fn(model, inputs, targets).item()
            total += loss
            steps_done = step
            running = total / step

            bar.update(step_tokens)
            bar.set_postfix(loss=f"{loss:.3f}", mean=f"{running:.3f}", refresh=False)

            mean_point = None
            if step % args.eval_every == 0 or step == args.steps:
                mean_point = running
                bar.write(f"step {step:>6} | loss {loss:.3f} | mean {running:.3f} "
                          f"| {time.time() - start:.1f}s")

            metrics.log(step, loss, elapsed_s=time.time() - start, val_loss=mean_point)

    bar.close()
    metrics.close()

    data.close()
    if not steps_done:
        raise SystemExit("no batches were served -- nothing to plot")

    mean = total / steps_done
    print(f"streamed {data.tokens_seen/1e6:.1f}M tokens over {steps_done} steps, "
          f"freed {data.bytes_freed/1e9:.1f}GB of shards")
    print(f"mean loss {mean:.4f} | perplexity {torch.tensor(mean).exp().item():.2f}")

    if args.plot:
        plot.render(metrics.path, ema_span_hint=args.ema_span,
                    train_name="loss", val_name="running mean")

    prompt = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    sample = model.generate(prompt, max_new_tokens=32)
    print("sample:", tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
