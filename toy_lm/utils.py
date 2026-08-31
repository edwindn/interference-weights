import csv
import json
import math
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer


def set_seed(seed):
    """Seed every RNG a run draws from: `random` (the loader's shuffle pool),
    numpy, and torch on cpu plus whichever accelerator is in use.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # covers cuda and mps too
    torch.cuda.manual_seed_all(seed)  # no-op without cuda


def pick_device():
    """cuda, else Apple Metal; never a silent cpu fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    raise SystemExit("no gpu available: neither cuda nor Apple Metal (mps)")


def load_tokenizer(path, vocab_size=None):
    """The tokenizer with every id >= `vocab_size` dropped. Lossless above 246."""
    path = Path(path)
    full = Tokenizer.from_file(str(path))
    if vocab_size is None or vocab_size == full.get_vocab_size():
        return full
    if not 0 < vocab_size <= full.get_vocab_size():
        raise SystemExit(f"config: vocab_size must be in 1..{full.get_vocab_size()}, "
                         f"got {vocab_size}")

    spec = json.loads(path.read_text())
    kept = {s: i for s, i in spec["model"]["vocab"].items() if i < vocab_size}
    lost = [i for s, i in spec["model"]["vocab"].items() if len(s) == 1 and i >= vocab_size]
    if lost:
        raise SystemExit(f"config: vocab_size {vocab_size} drops {len(lost)} "
                         f"single-character token(s); use at least {max(lost) + 1}")
    spec["model"]["vocab"] = kept
    spec["model"]["merges"] = [m for m in spec["model"]["merges"]
                              if m[0] in kept and m[1] in kept and m[0] + m[1] in kept]
    spec["added_tokens"] = [a for a in spec.get("added_tokens", []) if a["id"] < vocab_size]
    return Tokenizer.from_str(json.dumps(spec))


def lr_at(step, max_lr, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Linear warmup to max_lr at `warmup_steps`, then cosine to
    min_lr_ratio*max_lr at `total_steps`. Steps are 1-based.
    """
    if step <= warmup_steps:
        return max_lr * step / max(warmup_steps, 1)
    progress = min((step - warmup_steps) / max(total_steps - warmup_steps, 1), 1.0)
    min_lr = max_lr * min_lr_ratio
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


def sinusoidal_encoding(seq_len, dim, max_timescale=2**16):
    PE = np.empty((dim, seq_len))
    pos = np.arange(seq_len).reshape(1, -1)
    k = (np.arange(dim) // 2).reshape(-1, 1)
    inv = max_timescale ** (2 * k / dim)
    PE[::2,:] = np.sin(pos / inv[::2])
    PE[1::2,:] = np.cos(pos / inv[1::2])
    return torch.tensor(PE, dtype=torch.float32)

class MetricsLog:
    FIELDS = ["step", "train_loss", "val_loss", "grad_norm", "lr", "elapsed_s"]

    def __init__(self, log_dir, name="train"):
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{name}_{stamp}.log"
        self._file = self.path.open("w", newline="")
        self._writer = csv.DictWriter(self._file, fieldnames=self.FIELDS)
        self._writer.writeheader()

        latest = log_dir / "latest.log"
        latest.unlink(missing_ok=True)
        try:
            latest.symlink_to(self.path.name)
        except OSError:
            pass

    def log(self, step, train_loss, grad_norm=None, lr=None, elapsed_s=0.0, val_loss=None):
        """val_loss, grad_norm and lr are blank on rows that have no such value."""
        self._writer.writerow({
            "step": step,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": "" if val_loss is None else f"{val_loss:.6f}",
            "grad_norm": "" if grad_norm is None else f"{grad_norm:.6f}",
            "lr": "" if lr is None else f"{lr:.8g}",
            "elapsed_s": f"{elapsed_s:.3f}",
        })
        self._file.flush()

    def close(self):
        self._file.close()
