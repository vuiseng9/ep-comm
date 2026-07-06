import torch
import torch.nn as nn


class RoPE(nn.Module):
    def __init__(self, Dh, S=4096, base=10000):
        super().__init__()
        assert Dh % 2 == 0
        self.Dh = Dh # head dim
        self.S = S # max sequence length

        inv_freq = 1.0 / (base ** (torch.arange(0, Dh, 2).float() / Dh))
        pos = torch.arange(S).float()

        freqs = torch.outer(pos, inv_freq)  # [S, Dh/2]

        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def extra_repr(self):
        return f"Dh={self.Dh}, S={self.S}"
    
    def forward(self, x):
        # x: [B, S, H, Dh]
        B, S, H, Dh = x.shape

        cos = self.cos[:S].to(dtype=x.dtype, device=x.device)  # [S, Dh/2]
        sin = self.sin[:S].to(dtype=x.dtype, device=x.device)

        cos = cos[None, :, None, :]  # [1, S, 1, Dh/2]
        sin = sin[None, :, None, :]

        x_even = x[..., 0::2]
        x_odd  = x[..., 1::2]

        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos

        return out