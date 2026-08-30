import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from tokenizers import Tokenizer

from dataloader import CorpusExhausted, StreamingCorpus
from model import Transformer
from utils import MetricsLog

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
DATA_PATH = ROOT.parent / "data" / "input.txt"
CONFIG_PATH = ROOT / "config.yaml"


def load_config(path):
    """Flatten the sectioned config.yaml into one dict of argparse defaults."""
    with open(path) as f:
        config = yaml.safe_load(f) or {}
    flat = {}
    for section in config.values():
        flat.update(section)
    return flat


class TextData:
    """Tokenises a text file once, then serves random (input, target) windows."""

    def __init__(self, path, tokenizer, seq_len, batch_size, device, val_frac=0.05):
        text = Path(path).read_text(encoding="utf-8")
        ids = tokenizer.encode(text).ids
        tokens = torch.tensor(ids, dtype=torch.long)

        split = int(len(tokens) * (1 - val_frac))
        self.splits = {"train": tokens[:split], "val": tokens[split:]}
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.device = device

        for name, part in self.splits.items():
            if len(part) < seq_len + 1:
                raise ValueError(f"{name} split has {len(part)} tokens, need > {seq_len}")

    def __len__(self):
        return len(self.splits["train"]) + len(self.splits["val"])

    def batch(self, split="train"):
        tokens = self.splits[split]
        idx = torch.randint(len(tokens) - self.seq_len - 1, (self.batch_size,))
        inputs = torch.stack([tokens[i:i + self.seq_len] for i in idx])
        targets = torch.stack([tokens[i + 1:i + self.seq_len + 1] for i in idx])
        return inputs.to(self.device), targets.to(self.device)


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

    p = argparse.ArgumentParser(parents=[pre])
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--stream", action="store_true",
                   help="stream PleIAs/common_corpus instead of reading --data")
    p.add_argument("--stream-mode", choices=("english", "code"), default="english")
    p.add_argument("--shards", type=int, default=10, help="shards to stream, 1..10")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-dir", type=Path, default=ROOT / "logs")
    p.add_argument("--log-name", type=str, default="train")
    p.add_argument("--prompt", type=str, default="To be or not to be, ")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "toy_lm.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()

    if args.stream:
        data = StreamingCorpus(tokenizer, cfg["seq_len"], cfg["batch_size"], device,
                               mode=args.stream_mode, num_shards=args.shards,
                               val_batches=cfg["eval_batches"])
    else:
        data = TextData(args.data, tokenizer, cfg["seq_len"], cfg["batch_size"], device)
    model = Transformer(vocab_size, cfg["embedding_dim"], cfg["num_heads"], cfg["seq_len"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], betas=tuple(cfg["betas"]),
                                  weight_decay=0.0)

    params = sum(p.numel() for p in model.parameters())
    source = f"stream {args.stream_mode} x{args.shards}" if args.stream else f"{len(data)} tokens"
    print(f"device {device} | vocab {vocab_size} | {source} | {params/1e6:.1f}M params")

    metrics = MetricsLog(args.log_dir, args.log_name)
    print(f"logging to {metrics.path}")

    model.train()
    start = time.time()
    for step in range(1, cfg["steps"] + 1):
        try:
            inputs, targets = data.batch("train")
        except CorpusExhausted as exc:
            print(f"stream exhausted at step {step} ({exc})")
            break
        loss = loss_fn(model, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        # returns the total norm *before* clipping, which is what we log
        max_grad_norm = cfg["max_grad_norm"]
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_grad_norm if max_grad_norm > 0 else float("inf"))
        optimizer.step()

        val_loss = None
        if step % cfg["eval_every"] == 0 or step == cfg["steps"]:
            val_loss = evaluate(model, data, cfg["eval_batches"])
            print(f"step {step:>5} | train {loss.item():.3f} | val {val_loss:.3f} "
                  f"| gnorm {grad_norm.item():.2f} | {time.time() - start:.1f}s")

        metrics.log(step, loss.item(), grad_norm.item(), cfg["lr"],
                    time.time() - start, val_loss=val_loss)

    metrics.close()

    if args.stream:
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
