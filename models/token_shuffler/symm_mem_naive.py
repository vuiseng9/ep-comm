import torch
import torch.distributed as dist
import torch.distributed._symmetric_memory as symm_mem
from nvtx import annotate as nvtx_annotate
from .base import MoETokenShuffler

_symm_mem_backend_initialized = False

# ────────────────────────────────────────────────
# EP Token Shuffler via Symmetric Memory 
# ────────────────────────────────────────────────
class NaiveSymmMemTokenShuffler(MoETokenShuffler):
    """
    Expert-Parallel Token Shuffler using 
    torch.distributed._symmetric_memory,
    dispatch via torch.ops.symm_mem.all_to_all_vdev_2d,
    combine via torch.ops.symm_mem.all_to_all_vdev_2d.

    Naive Implementation:
    - Allocate symm mem on the fly and 
    - copy data to it for dispatch and combine operations.
    """
    def __init__(self, module_of_experts, E, K, ep_group):
        super().__init__(module_of_experts, E, K, ep_group)

        global _symm_mem_backend_initialized
        if not _symm_mem_backend_initialized:
            symm_mem.set_backend("NVSHMEM") # globally once
            _symm_mem_backend_initialized = True
        
        _torch_ver = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
        if _torch_ver <= (2, 11):
            symm_mem.enable_symm_mem_for_group(ep_group.group_name)
    
    def __call__(self, tokens, k_eids, k_weights):
        self.max_in = k_eids.numel()
        self.max_out = max(int(self.max_in*1.15), 256)

        with nvtx_annotate("dispatch.fw", color="darkorange"):
            tokens_by_eid_order, ntok_per_eid, weights_by_eid_order = self.permute_for_dispatch(tokens, k_eids, k_weights)
            routed = self.dispatch(tokens_by_eid_order, ntok_per_eid)
            offset = self.out_so[0].view(self.EPR, self.EP).sum(dim=1).cumsum(dim=0).to(torch.int32)
        
        with nvtx_annotate("experts.fw", color="lightskyblue"):
            expert_computed_tokens = self.expert_compute(routed, offset)

        with nvtx_annotate("combine.fw", color="deeppink"):
            combined_tokens, _ = self.combine(expert_computed_tokens)
            combined_tokens = self.apply_router_probs(combined_tokens, weights_by_eid_order.unsqueeze(-1))
            moe_outputs = self.restore_token_order(combined_tokens)

        return moe_outputs

    def permute_for_dispatch(self, tokens: torch.Tensor, k_eids: torch.Tensor, k_weights: torch.Tensor):
        flat_k_eids = k_eids.reshape(-1) # T*K, index is token id
        k_expanded_tok_ids_by_eid_order = flat_k_eids.argsort() # T*K
        self.inv_perm = k_expanded_tok_ids_by_eid_order.argsort() 

        tok_ids_by_eid_order = k_expanded_tok_ids_by_eid_order // self.K # T*K
        slot_by_eid_order    = k_expanded_tok_ids_by_eid_order  % self.K # T*K
        
        # permute tokens, calc offsets
        tokens_by_eid_order = tokens[tok_ids_by_eid_order]
        
        ntok_per_eid = torch.bincount(flat_k_eids, minlength=self.E)
        # eid_offs = ntok_per_eid.cumsum(dim=0).to(torch.int32)

        # permute weights
        weights_by_eid_order = k_weights[tok_ids_by_eid_order, slot_by_eid_order]        
        return tokens_by_eid_order, ntok_per_eid, weights_by_eid_order
    
    def dispatch(self, tokens, expert_splits):
        # tokens: [m, H] rank-major dense; in_splits: [E] int64 rank-major
        expert_major_tokens, self.out_so = NaiveSymmMemDispatchFunc.apply(
            tokens, expert_splits, self.ep_group.group_name,
            self.max_in, self.max_out
        )
        return expert_major_tokens

    def permute_for_expert_compute(self, dispatched_tokens):
        raise NotImplementedError("Not required, torch symm_mem.all_to_all already handles the permutation")
    
    def expert_compute(self, tokens_by_expert, offset):
        return self.experts(tokens_by_expert, offset)

    def unpermute_for_combine(self, expert_computed_tokens):
        raise NotImplementedError("Not required, torch symm_mem.all_to_all already handles the permutation")
        
    def combine(self, expert_computed_tokens):
        return NaiveSymmMemCombineFunc.apply(
                expert_computed_tokens, self.out_so, self.ep_group.group_name, 
                self.max_out, self.max_in
            )
    
    def apply_router_probs(self, tokens_symmem, probs):
        return MulProbsFunc.apply(tokens_symmem, probs)

    def restore_token_order(self, weighted_expert_computed_tokens):
        moe_outputs = weighted_expert_computed_tokens[self.inv_perm]
        return moe_outputs


# ────────────────────────────────────────────────
# Dispatch Func
# ────────────────────────────────────────────────
class NaiveSymmMemDispatchFunc(torch.autograd.Function):
    @staticmethod
    @nvtx_annotate("fw.a2a_dispatch", color="darkorange")
    def forward(ctx, inp, in_splits, group_name, max_in_numel, max_out_numel):
        device, H = inp.device, inp.shape[1]
        E = in_splits.shape[0]

        inp_symm = symm_mem.empty(max_in_numel, H, dtype=inp.dtype, device=device)
        inp_symm[: inp.shape[0]].copy_(inp)
        in_splits_symm = symm_mem.empty(E, dtype=torch.int64, device=device).copy_(in_splits)
        out_symm = symm_mem.empty(max_out_numel, H, dtype=inp.dtype, device=device)
        out_so = symm_mem.empty((2, E), dtype=torch.int64, device=device)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            inp_symm, out_symm, in_splits_symm, out_so, group_name, major_align=1
        )

        ctx.group_name = group_name
        ctx.max_in_numel = max_in_numel
        ctx.max_out_numel = max_out_numel
        ctx.in_shape = tuple(inp.shape)
        ctx.save_for_backward(out_so)

        return out_symm, out_so

    @staticmethod
    @nvtx_annotate("bw.a2a_dispatch", color="darkorange")
    def backward(ctx, grad_out, _grad_so):
        (out_so,) = ctx.saved_tensors
        device, H = grad_out.device, grad_out.shape[1]
        E = out_so.shape[1]
        

        g_symm = symm_mem.empty(ctx.max_out_numel, H, dtype=grad_out.dtype, device=device)
        g_symm[: grad_out.shape[0]].copy_(grad_out)
        in_so = symm_mem.empty((2, E), dtype=torch.int64, device=device).copy_(out_so)
        grad_inp_symm = symm_mem.empty(ctx.max_in_numel, H, dtype=grad_out.dtype, device=device)
        out_so_bwd = symm_mem.empty((2, E), dtype=torch.int64, device=device)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            g_symm, grad_inp_symm, in_so, out_so_bwd, ctx.group_name
        )

        m = ctx.in_shape[0]
        grad_inp = grad_inp_symm[:m]
        return grad_inp, None, None, None, None


# ────────────────────────────────────────────────
# Combine Func
# ────────────────────────────────────────────────
class NaiveSymmMemCombineFunc(torch.autograd.Function):
    @staticmethod
    @nvtx_annotate("fw.a2a_combine", color="deeppink")
    def forward(ctx, inp, in_splits_offsets, group_name,
                max_in_numel, max_out_numel):
        device, H = inp.device, inp.shape[1]
        E = in_splits_offsets.shape[1]

        inp_symm = symm_mem.empty(max_in_numel, H, dtype=inp.dtype, device=device)
        inp_symm[: inp.shape[0]].copy_(inp)
        in_so = symm_mem.empty((2, E), dtype=torch.int64, device=device)
        in_so.copy_(in_splits_offsets)
        out_symm = symm_mem.empty(max_out_numel, H, dtype=inp.dtype, device=device)
        out_so = symm_mem.empty((2, E), dtype=torch.int64, device=device)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d_offset(
            inp_symm, out_symm, in_so, out_so, group_name
        )


        ctx.group_name = group_name
        ctx.max_in_numel = max_in_numel
        ctx.max_out_numel = max_out_numel
        ctx.in_shape = tuple(inp.shape)
        ctx.save_for_backward(out_so)

        return out_symm, out_so

    @staticmethod
    @nvtx_annotate("bw.a2a_combine", color="deeppink")
    def backward(ctx, grad_out, _grad_out_so):
        (in_so,) = ctx.saved_tensors
        rank_major_splits = in_so[0]  
        device, H = grad_out.device, grad_out.shape[1]
        E = rank_major_splits.shape[0]

        g_symm = symm_mem.empty(ctx.max_out_numel, H, dtype=grad_out.dtype, device=device)
        g_symm[: grad_out.shape[0]].copy_(grad_out)
        in_splits = symm_mem.empty(E, dtype=torch.int64, device=device).copy_(rank_major_splits)
        grad_inp_symm = symm_mem.empty(ctx.max_in_numel, H, dtype=grad_out.dtype, device=device)
        bwd_so = symm_mem.empty((2, E), dtype=torch.int64, device=device)

        torch.cuda.synchronize(device)
        dist.barrier()

        torch.ops.symm_mem.all_to_all_vdev_2d(
            g_symm, grad_inp_symm, in_splits, bwd_so, ctx.group_name,
            major_align=1,
        )

        m = ctx.in_shape[0]
        grad_inp = grad_inp_symm[:m]
        return grad_inp, None, None, None, None


# ────────────────────────────────────────────────
# Mul Func with grad returned in symm_mem
# ────────────────────────────────────────────────
class MulProbsFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tokens_symmem, probs):
        weighted = tokens_symmem * probs
        ctx.save_for_backward(tokens_symmem, probs)
        return weighted
    
    @staticmethod
    def backward(ctx, grad_out):
        tokens_symmem, probs = ctx.saved_tensors
        grad_tokens_symmem = grad_out * probs
        grad_probs = grad_out * tokens_symmem
        return grad_tokens_symmem, grad_probs