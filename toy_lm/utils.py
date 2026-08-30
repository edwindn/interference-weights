import numpy as np
import torch

def sinusoidal_encoding(seq_len, dim, max_timescale=2**16):
    PE = np.empty((dim, seq_len))
    pos = np.arange(seq_len).reshape(1, -1)
    k = (np.arange(dim) // 2).reshape(-1, 1)
    inv = max_timescale ** (2 * k / dim)
    PE[::2,:] = np.sin(pos / inv[::2])
    PE[1::2,:] = np.cos(pos / inv[1::2])
    return torch.tensor(PE, dtype=torch.float32)