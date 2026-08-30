import torch


def loss_func(y_pred, y, num_features, importance_base):
    importance = importance_base ** torch.arange(
        num_features, dtype=y.dtype, device=y.device
    )
    return (((y_pred - y) ** 2) * importance).sum(dim=-1).mean()
