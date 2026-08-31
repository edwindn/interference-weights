import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer
from tqdm import tqdm

from dataloader import CorpusExhausted, StreamingCorpus
from model import Transformer
from utils import MetricsLog, pick_device, set_seed

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
CONFIG_PATH = ROOT / "config.yaml"


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


@torch.no_grad()
def evaluate(model, data, batches=20):
    model.eval()
    losses = [loss_fn(model, *data.batch("val")).item() for _ in range(batches)]
    model.train()
    return sum(losses) / len(losses)


def main():
    # config.yaml supplies the defaults; anything passed on the CLI overrides it.
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=CONFIG_PATH)
    cfg = load_config(pre.parse_known_args()[0].config)

    # Which halves of the corpus to train on. Both is a full mix; neither is a
    # config mistake, so say so before anything is downloaded.
    sources = [name for name, key in (("english", "english_data"), ("code", "code_data"))
               if cfg[key]]
    if not sources:
        raise SystemExit("config: english_data and code_data are both false, "
                         "enable at least one")

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--shards", type=int, default=10, help="shards to stream, 1..10")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    p.add_argument("--seed", type=int, default=cfg["seed"])
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--log-name", type=str, default="train")
    p.add_argument("--prompt", type=str, default="To be or not to be, ")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "toy_lm.pt")
    args = p.parse_args()

    set_seed(args.seed)
    device = pick_device()

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()

    data = StreamingCorpus(tokenizer, cfg["seq_len"], cfg["batch_size"], device,
                           english=cfg["english_data"], code=cfg["code_data"],
                           num_shards=args.shards, val_batches=cfg["eval_batches"])
    model = Transformer(vocab_size, cfg["embedding_dim"], cfg["num_heads"], cfg["seq_len"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  weight_decay=0.0)

    params = sum(p.numel() for p in model.parameters())
    print(f"device {device} | vocab {vocab_size} | "
          f"stream {'+'.join(sources)} x{args.shards} | {params/1e6:.1f}M params")

    metrics = MetricsLog(args.log_dir, args.log_name)
    print(f"logging to {metrics.path}")

    model.train()
    start = time.time()
    # The run is budgeted in tokens, so the bar tracks tokens rather than steps.
    step_tokens = cfg["batch_size"] * cfg["seq_len"]
    bar = tqdm(total=cfg["steps"] * step_tokens, unit="tok", unit_scale=True,
               dynamic_ncols=True, desc="train", smoothing=0.05)
    for step in range(1, cfg["steps"] + 1):
        try:
            inputs, targets = data.batch("train")
        except CorpusExhausted as exc:
            bar.write(f"stream exhausted at step {step} ({exc})")
            break
        loss = loss_fn(model, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # returns the total norm *before* clipping, which is what we log
        max_grad_norm = cfg["max_grad_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"))
        optimizer.step()

        train_loss = loss.item()
        bar.update(step_tokens)
        bar.set_postfix(loss=f"{train_loss:.3f}", gnorm=f"{grad_norm.item():.2f}",
                        refresh=False)

        val_loss = None
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            val_loss = evaluate(model, data, cfg["eval_batches"])
            # bar.write keeps these lines from being overwritten by the bar
            bar.write(f"step {step:>6} | train {train_loss:.3f} | val {val_loss:.3f} "
                      f"| gnorm {grad_norm.item():.2f} | {time.time() - start:.1f}s")

        metrics.log(step, train_loss, grad_norm.item(), cfg["lr"],
                    time.time() - start, val_loss=val_loss)

    bar.close()
    metrics.close()

    data.close()
    print(f"streamed {data.tokens_seen/1e6:.1f}M tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args), "config": cfg}, args.out)
    print(f"saved {args.out}")

    prompt = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    sample = model.generate(prompt, max_new_tokens=32)
    print("sample:", tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
