import torch
import torch.nn as nn


class Transcoder(nn.Module):
    def __init__(self, epsilon, in_dim, out_dim, num_features=4096):
        super().__init__()

        self.epsilon = epsilon
        self.W_in = nn.Parameter(torch.randn(in_dim, num_features) / in_dim**0.5)
        self.b = nn.Parameter(torch.zeros(num_features))
        self.W_out = nn.Parameter(torch.randn(num_features, out_dim) / num_features**0.5)

    def forward(self, acts):
        hidden = acts @ self.W_in + self.b
        hidden = hidden * (hidden > self.epsilon)
        return hidden @ self.W_out, hidden
    