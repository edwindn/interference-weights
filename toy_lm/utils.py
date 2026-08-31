import csv
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

def pick_device():
    """cuda, else Apple Metal; never a silent cpu fallback."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return torch.device("mps")
    raise SystemExit("no gpu available: neither cuda nor Apple Metal (mps)")


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
