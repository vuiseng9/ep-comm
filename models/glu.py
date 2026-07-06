import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from contextlib import nullcontext

class GluMLP(nn.Module):
    def __init__(self, D, Dff):
        super().__init__()
        self.up = nn.Linear(D, Dff)
        self.gate = nn.Linear(D, Dff)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=0.1)
        self.down = nn.Linear(Dff, D)

    def forward(self, x):
        out = self.down( self.drop( self.up(x) * self.act(self.gate(x)) ) )
        return out
    

class GroupedMMFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, W, offs, fwd_ctx_fn, bwd_ctx_fn):           # x:(N,IC)  W:(G,OC,IC)  offs:(G,) int32
        with fwd_ctx_fn():
            y = F.grouped_mm(x, W.transpose(-2, -1), offs=offs)   # (OC,IC)->(IC,OC) per group; y:(N,OC)
        
        ctx.save_for_backward(x, W, offs)
        ctx.fwd_ctx_fn = fwd_ctx_fn
        ctx.bwd_ctx_fn = bwd_ctx_fn
        return y

    @staticmethod
    def backward(ctx, dy):                   # dy:(N,OC)
        x, W, offs = ctx.saved_tensors
        dW = F.grouped_mm(dy.transpose(0, 1), x, offs=offs)   # (OC,N)x(N,IC)    -> (G,OC,IC)

        with ctx.bwd_ctx_fn():
            dx = F.grouped_mm(dy, W, offs=offs)                   # (N,OC)x(G,OC,IC) -> (N,IC)
        return dx, dW, None, None, None


class GroupedLinear(nn.Module):
    def __init__(self, group_size, ic, oc,
                 fwd_ctx_fn=nullcontext, bwd_ctx_fn=nullcontext):
        super().__init__()
        self.gs = group_size
        self.ic = ic
        self.oc = oc
        self.fwd_ctx_fn = fwd_ctx_fn
        self.bwd_ctx_fn = bwd_ctx_fn
        self.weights = nn.Parameter(torch.empty((group_size, oc, ic)))
        self.reset_parameters()
    
    def extra_repr(self):
        return f"n_group={self.gs}, oc={self.oc}, ic={self.ic}, biasless | " \
               f"fw: {self.fwd_ctx_fn.__name__} | bw: {self.bwd_ctx_fn.__name__}"

    def reset_parameters(self): 
        nn.init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        
    def forward(self, x, offs):
        return GroupedMMFunc.apply(x, self.weights, offs, self.fwd_ctx_fn, self.bwd_ctx_fn)


class GroupedGluMLP(nn.Module):
    def __init__(self, group_size, D, Dff, symm_mem_pool_fn=nullcontext):
        super().__init__()
        self.up_gate =  GroupedLinear(group_size, D, Dff*2)
        self.act = nn.SiLU()
        self.drop = nn.Dropout(p=0.1)
        self.down = GroupedLinear(group_size, Dff, D)
        self.set_symm_mem_pool_fn(symm_mem_pool_fn)

    def set_symm_mem_pool_fn(self, symm_mem_pool_fn):
        self.up_gate.bwd_ctx_fn = symm_mem_pool_fn
        self.down.fwd_ctx_fn = symm_mem_pool_fn

    def forward(self, x, offs):
        up, gate = self.up_gate(x, offs).chunk(2, dim=-1)
        return self.down( self.drop( up * self.act(gate) ), offs )


if __name__ == '__main__':
    torch.manual_seed(42)

    # grouped_mm is a CUDA-only CUTLASS kernel; tensors must be on GPU + bf16
    device = torch.device('cuda')
    dtype  = torch.bfloat16

    E = 4        # number of groups / experts
    D = 16       # model dim
    Dff = 32     # feed-forward dim

    # variable token counts per group
    tokens_per_group = [3, 5, 2, 4]
    total = sum(tokens_per_group)

    # offs: 1-D int32 exclusive-end offsets required by F.grouped_mm
    offs = torch.cumsum(torch.tensor(tokens_per_group), dim=0).to(torch.int32).to(device)
    print(f"offs: {offs.tolist()}  (total tokens: {total})")

    # NOTE: y.sum().backward() creates a broadcast grad with strides [0,0]
    # which grouped_mm backward rejects.  Explicitly supplying ones_like(y)
    # as the incoming gradient avoids the issue.
    def backward(out):
        out.backward(torch.ones_like(out))

    # ── GroupedLinear ────────────────────────────────────────────────────────
    print("\n--- GroupedLinear ---")
    gl = GroupedLinear(group_size=E, ic=D, oc=Dff).to(device=device, dtype=dtype)
    x_gl = torch.randn(total, D, device=device, dtype=dtype, requires_grad=True)

    out_gl = gl(x_gl, offs)
    print(f"  forward : {x_gl.shape} -> {out_gl.shape}")

    backward(out_gl)
    print(f"  x.grad  : {x_gl.grad.shape}")
    print(f"  W.grad  : {gl.weights.grad.shape}")
    assert x_gl.grad is not None and x_gl.grad.shape == x_gl.shape
    assert gl.weights.grad is not None and gl.weights.grad.shape == gl.weights.shape
    print("  PASSED")

    # ── GroupedGluMLP ────────────────────────────────────────────────────────
    print("\n--- GroupedGluMLP ---")
    glu = GroupedGluMLP(E=E, D=D, Dff=Dff).to(device=device, dtype=dtype)
    x_glu = torch.randn(total, D, device=device, dtype=dtype, requires_grad=True)

    out_glu = glu(x_glu, offs)
    print(f"  forward : {x_glu.shape} -> {out_glu.shape}")

    backward(out_glu)
    print(f"  x.grad            : {x_glu.grad.shape}")
    print(f"  up_gate W.grad    : {glu.up_gate.weights.grad.shape}")
    print(f"  down    W.grad    : {glu.down.weights.grad.shape}")
    assert x_glu.grad is not None and x_glu.grad.shape == x_glu.shape
    assert glu.up_gate.weights.grad is not None
    assert glu.down.weights.grad is not None
    print("  PASSED")
