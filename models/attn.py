import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.cuda.nvtx import range as nvtx_range


class Attn(nn.Module):
    """
    attention module supporting MHA (G=1), GQA (1<G<H), and MQA (G=H).
    """
    def __init__(self, D, H, G=1, rope=None):
        super().__init__()
        assert D % H == 0, "D must be divisible by H"
        assert H % G == 0, "H must be divisible by G"

        self.D = D
        self.H = H
        self.Dh = D // H
        self.G = G
        self.Hkv = H // G           # 1 KV head per G query heads
        self.Dkv = self.Hkv * self.Dh

        self.norm = nn.RMSNorm(D)
        self.qkv_proj = nn.Linear(D, D + 2 * self.Dkv)
        self.rope = rope
        self.o_proj = nn.Linear(D, D)

    def __repr__(self):
        if self.G == 1:
            extra = f"(MHA: H={self.H}, Dh={self.Dh})"
        elif self.G == self.H:
            extra = f"(MQA: H={self.H}, Dh={self.Dh}, Hkv={self.Hkv}, G={self.G})"
        else:
            extra = f"(GQA: H={self.H}, Dh={self.Dh}, Hkv={self.Hkv}, G={self.G})"
        head, tail = super().__repr__().split("\n", 1)
        return head + "\n  " + extra + "\n" + tail

    @nvtx_range("fw.Attn")
    def forward(self, x, mask=None):
        B, L, D = x.shape
        residual = x

        qkv = self.qkv_proj(self.norm(x))
        q, k, v = qkv.split([self.D, self.Dkv, self.Dkv], dim=-1)

        # B,L,D -> B,L,H,Dh
        q = q.view(B, L, self.H, self.Dh)
        k = k.view(B, L, self.Hkv, self.Dh)
        v = v.view(B, L, self.Hkv, self.Dh)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)

        # B,L,H,Dh -> B,H,L,Dh
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Expand KV heads to match Q heads (no-op when G=1 / MHA)
        if self.G > 1:
            k = k.repeat_interleave(self.G, dim=1)
            v = v.repeat_interleave(self.G, dim=1)

        # FlashAttention > memory-efficient > math (in priority order; math excluded as slowest)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=0.1 if self.training else 0.0,
            )

        # B,H,L,Dh -> B,L,H,Dh -> B,L,D
        out = self.o_proj(attn.transpose(1, 2).reshape(B, L, D))

        return out + residual
