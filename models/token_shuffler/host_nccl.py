import torch
import torch.distributed as dist
from torch.cuda.nvtx import range as nvtx_range
from .base import MoETokenShuffler

# ────────────────────────────────────────────────
# EP Token Shuffler using host-initiated NCCL all-to-all 
# ────────────────────────────────────────────────
class HostNcclTokenShuffler(MoETokenShuffler):
    """
    Expert-Parallel Token Shuffler using 
    host-side NCCL all-to-all 
    (torch.distributed.all_to_all_single)
    """
    def __call__(self, tokens, k_eids, k_weights):
        with nvtx_range(f"dispatch.fw"):
            tokens_rank_major, ntok_per_eid, weights_rank_major =self.permute_for_dispatch(tokens, k_eids, k_weights)
            dispatched_tokens = self.dispatch(tokens_rank_major, ntok_per_eid)
            expert_major_tokens = self.permute_for_expert_compute(dispatched_tokens)
            eid_offs = self.recv_expert_splits_2d.sum(0).cumsum(0).to(torch.int32)

        with nvtx_range(f"experts.fw"):    
            expert_computed_tokens = self.expert_compute(expert_major_tokens, eid_offs)
        
        with nvtx_range(f"combine.fw"):
            expert_computed_tokens =self.unpermute_for_combine(expert_computed_tokens)
            combined_tokens = self.combine(expert_computed_tokens)
            combined_tokens = self.apply_router_probs(combined_tokens, weights_rank_major)
            moe_outputs =  self.restore_token_order(combined_tokens)
            
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
        # --- sync expert splits a priori (metadata all-to-all, not differentiable) 
        dest_expert_splits_2d = expert_splits.view(self.EP, self.EPR).contiguous()
        self.recv_expert_splits_2d = torch.empty_like(dest_expert_splits_2d)
        dist.all_to_all_single(self.recv_expert_splits_2d, dest_expert_splits_2d, group=self.ep_group)

        self.dest_rank_splits = dest_expert_splits_2d.sum(1).tolist()
        self.recv_rank_splits = self.recv_expert_splits_2d.sum(1).tolist() 
        
        return AlltoAllFunc.apply(tokens, self.recv_rank_splits, self.dest_rank_splits, self.ep_group)
    
    def permute_for_expert_compute(self, dispatched_tokens):
        # --- build/initialize expert id of dispatched tokens
        dispatched_eids = torch.repeat_interleave(
                torch.arange(self.EPR, device=dispatched_tokens.device).repeat(self.EP),
                self.recv_expert_splits_2d.reshape(-1)
            )
        # --- make (un)permute indices to expert-major for expert compute
        sort_idx = torch.argsort(dispatched_eids, stable=True)    # [total_recv]
        self.inv_idx = torch.empty_like(sort_idx)
        self.inv_idx[sort_idx] = torch.arange(dispatched_tokens.shape[0], device=dispatched_tokens.device)

        return dispatched_tokens[sort_idx]

    def expert_compute(self, tokens_by_expert, offset):
        return self.experts(tokens_by_expert, offset)
         
    def unpermute_for_combine(self, expert_computed_tokens):
        # return expert_computed_tokens[self.inv_idx]
        return expert_computed_tokens.index_select(0, self.inv_idx)

    def combine(self, expert_computed_tokens):
        return AlltoAllFunc.apply(
            expert_computed_tokens, self.dest_rank_splits, self.recv_rank_splits, self.ep_group)

    def apply_router_probs(self, expert_computed_tokens, weights_by_eid_order):
        return expert_computed_tokens * weights_by_eid_order.unsqueeze(-1)
    
    def restore_token_order(self, weighted_expert_computed_tokens):
        moe_outputs = weighted_expert_computed_tokens[self.inv_perm]
        return moe_outputs
    
# ────────────────────────────────────────────────
# Differentiable All-to-All 
# ────────────────────────────────────────────────
class AlltoAllFunc(torch.autograd.Function):
    @staticmethod
    @nvtx_range("fw.AlltoAllFunc")
    def forward(ctx, inp, out_splits, in_splits, group):
        out = inp.new_empty(sum(out_splits), inp.shape[1])
        dist.all_to_all_single(out, inp.contiguous(), out_splits, in_splits, group=group)
        ctx.out_splits, ctx.in_splits, ctx.group = out_splits, in_splits, group
        ctx.in_rows = inp.shape[0]
        return out

    @staticmethod
    @nvtx_range("bw.AlltoAllFunc")
    def backward(ctx, grad_out):
        grad_in = grad_out.new_empty(ctx.in_rows, grad_out.shape[1])
        # transpose: swap the split roles
        dist.all_to_all_single(
            grad_in, grad_out.contiguous(), ctx.in_splits, ctx.out_splits, group=ctx.group
        )
        return grad_in, None, None, None