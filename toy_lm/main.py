import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from tokenizers import Tokenizer

from dataloader import CorpusExhausted, StreamingCorpus
from model import Transformer

ROOT = Path(__file__).resolve().parent
TOKENIZER_PATH = ROOT.parent / "tokenizer" / "Pleias-1.2b-Preview" / "tokenizer.json"
DATA_PATH = ROOT.parent / "data" / "input.txt"


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
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=Path, default=DATA_PATH)
    p.add_argument("--stream", action="store_true",
                   help="stream PleIAs/common_corpus instead of reading --data")
    p.add_argument("--stream-mode", choices=("english", "code"), default="english")
    p.add_argument("--shards", type=int, default=10, help="shards to stream, 1..10")
    p.add_argument("--tokenizer", type=Path, default=TOKENIZER_PATH)
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--embedding-dim", type=int, default=256)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--prompt", type=str, default="To be or not to be, ")
    p.add_argument("--out", type=Path, default=ROOT / "checkpoints" / "toy_lm.pt")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    torch.manual_seed(0)
    device = torch.device(args.device)

    tokenizer = Tokenizer.from_file(str(args.tokenizer))
    vocab_size = tokenizer.get_vocab_size()

    if args.stream:
        data = StreamingCorpus(tokenizer, args.seq_len, args.batch_size, device,
                               mode=args.stream_mode, num_shards=args.shards)
    else:
        data = TextData(args.data, tokenizer, args.seq_len, args.batch_size, device)
    model = Transformer(vocab_size, args.embedding_dim, args.num_heads, args.seq_len).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.95), weight_decay=0.1)

    params = sum(p.numel() for p in model.parameters())
    source = f"stream {args.stream_mode} x{args.shards}" if args.stream else f"{len(data)} tokens"
    print(f"device {device} | vocab {vocab_size} | {source} | {params/1e6:.1f}M params")

    model.train()
    start = time.time()
    for step in range(1, args.steps + 1):
        try:
            inputs, targets = data.batch("train")
        except CorpusExhausted as exc:
            print(f"stream exhausted at step {step} ({exc})")
            break
        loss = loss_fn(model, inputs, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % args.eval_every == 0 or step == args.steps:
            print(f"step {step:>5} | train {loss.item():.3f} | val {evaluate(model, data):.3f} "
                  f"| {time.time() - start:.1f}s")

    if args.stream:
        data.close()
        print(f"streamed {data.tokens_seen/1e6:.1f}M tokens")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "args": vars(args)}, args.out)
    print(f"saved {args.out}")

    prompt = torch.tensor([tokenizer.encode(args.prompt).ids], device=device)
    sample = model.generate(prompt, max_new_tokens=32)
    print("sample:", tokenizer.decode(sample[0].tolist()))


if __name__ == "__main__":
    main()
