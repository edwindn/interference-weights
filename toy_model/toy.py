import torch.nn as nn
import torch
import math


class Toy(nn.Module):
    def __init__(self,
                in_dim: int,
                hidden_dim: int,
                init_method: str,
                activation: str | None,
                activation_after: bool,
                seed: int):

        super().__init__()

        torch.manual_seed(seed)

        self.activation_after = activation_after

        self.weight = nn.Parameter(torch.empty(in_dim, hidden_dim))
        self.bias = nn.Parameter(torch.empty(in_dim,))

        if init_method == "normal":
            nn.init.normal_(self.weight, mean=0.0, std=1 / math.sqrt(in_dim))
            nn.init.zeros_(self.bias)
        elif init_method == "xavier":
            nn.init.xavier_normal_(self.weight)
            nn.init.zeros_(self.bias)
        elif init_method == "kaiming":
            nn.init.kaiming_normal_(
                self.weight,
                mode="fan_in",
                nonlinearity="relu",
            )
            nn.init.zeros_(self.bias)
        elif init_method == "uniform":
            bound = 1 / math.sqrt(in_dim)
            nn.init.uniform_(self.weight, -bound, bound)
            nn.init.uniform_(self.bias, -bound, bound)
        else:
            raise ValueError(f"unexpected init method {init_method}")

        if activation == 'relu':
            self.act = nn.ReLU()
        elif activation == 'silu':
            self.act = nn.SiLU()
        elif activation == 'gelu':
            self.act = nn.GELU()
        elif activation == None:
            self.act = nn.Identity()
        else:
            raise ValueError(f"unexpected activation {activation}") 

    def forward(self, x):
        h = x @ self.weight
        if self.activation_after:
            xp = self.act(h @ self.weight.T + self.bias)
        else:
            h = self.act(h)
            xp = h @ self.weight.T + self.bias
        return xp, h