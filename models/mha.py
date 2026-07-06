import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import sdpa_kernel, SDPBackend
from torch.cuda.nvtx import range as nvtx_range
    
class MHA(nn.Module):
    def __init__(self, D, H, rope=None):
        super().__init__()
        assert D%H==0, "D must be divisible by H"
        self.D = D
        self.H = H
        self.Dh = D//H

        # D -> 3D (Q,K,V)
        self.norm = nn.RMSNorm(D)
        self.qkv_proj = nn.Linear(D, 3*D)
        self.rope = rope
        self.o_proj = nn.Linear(D, D)

    @nvtx_range("fw.Attn_MHA")
    def forward(self, x, mask=None):
        B, L, D = x.shape
        residual = x

        qkv = self.qkv_proj(self.norm(x))
        q,k,v = qkv.chunk(3, dim=-1)

        # B,L,D -> B,L,H,Dh -> B,H,L,Dh
        q = q.view(B, L, self.H, self.Dh)
        k = k.view(B, L, self.H, self.Dh)
        v = v.view(B, L, self.H, self.Dh)

        if self.rope is not None:
            q = self.rope(q)
            k = self.rope(k)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # FlashAttention > memory-efficient > math (in priority order; math excluded as slowest)
        with sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION]):
            attn = F.scaled_dot_product_attention(
                q, k, v,
                is_causal=True,
                dropout_p=0.1 if self.training else 0.0,
            )

        # attn[B,H,L,Dh], concat-ing H into hidden dim of D 
        # essentially permuting it to [B,L,H,Dh] then reshape [B,L,D]
        # # B,H,L,Dh -> B,L,H,Dh -> B,L,D  
        out = self.o_proj( attn.transpose(1, 2).reshape(B, L, D) )

        return out + residual