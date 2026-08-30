import torch.nn as nn
import torch
from torch.nn.functional import scaled_dot_product_attention

from functools import partial
LinearNoBias = partial(nn.Linear, bias=False)

from utils import sinusoidal_encoding

class MLP(nn.Module):
    def __init__(self, embedding_dim, activation=None):
        super().__init__()

        if activation is None:
            activation = nn.ReLU

        self.mlp = nn.Sequential(
            LinearNoBias(embedding_dim, 4 * embedding_dim),
            activation(),
            LinearNoBias(4 * embedding_dim, embedding_dim),
        )

    def forward(self, x):
        return self.mlp(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads):
        super().__init__()

        self.proj = LinearNoBias(embedding_dim, 3 * embedding_dim)
        self.out_proj = LinearNoBias(embedding_dim, embedding_dim)
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

    def forward(self, x):
        b, t, c = x.size()
        qkv = self.proj(x)
        q, k, v = qkv.split(self.embedding_dim, dim=2)
        q = q.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3) # batch, head dim, seq length (T), size
        k = k.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(b, t, self.num_heads, c // self.num_heads).permute(0, 2, 1, 3)

        out = scaled_dot_product_attention(q, k, v, is_causal=True)
        out = out.permute(0, 2, 1, 3).reshape(b, t, c)
        return self.out_proj(out)


class Transformer(nn.Module):
    pos_embed: torch.Tensor
    def __init__(self, vocab_size, embedding_dim, num_heads, max_seq_len):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # buffer so it follows the module across .to(device) / state_dict
        self.register_buffer("pos_embed", sinusoidal_encoding(max_seq_len, embedding_dim).T.contiguous())
        self.attention = MultiHeadAttention(embedding_dim, num_heads)
        self.mlp = MLP(embedding_dim)
        self.lm_head = LinearNoBias(embedding_dim, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, tokens=None, latents=None):
        if tokens is not None and latents is not None:
            raise ValueError("Pass only one of tokens or latents")
        if tokens is not None:
            latents = self.embedding(tokens)
        if latents is None:
            raise ValueError("Must provide one of tokens or latents")

        b, t, _ = latents.size()
        assert t <= self.max_seq_len

        latents = latents + self.pos_embed[:t, :].to(latents.dtype)

        x = latents
        x = x + self.attention(x)
        x = x + self.mlp(x)
        return self.lm_head(x)

    @torch.no_grad()
    def generate(self, tokens, max_new_tokens, temperature=1.0, top_k=50):
        for _ in range(max_new_tokens):
            logits = self(tokens=tokens[:, -self.max_seq_len:])[:, -1, :]
            logits = logits.float() / max(temperature, 1e-5)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1)), dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = torch.softmax(logits, dim=-1)
            tokens = torch.cat([tokens, torch.multinomial(probs, 1)], dim=1)
        return tokens