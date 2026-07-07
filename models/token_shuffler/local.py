import torch
from nvtx import annotate as nvtx_annotate
from .base import MoETokenShuffler

# ────────────────────────────────────────────────
# Token Shuffler for rank-local moe execution
# ────────────────────────────────────────────────
class LocalTokenShuffler(MoETokenShuffler):
    """
    Rank-local implementation for data-parallel MoE,
    no token exchange between ranks. 
    Or usable for single gpu MoE execution.
    """
    def __call__(self, tokens, k_eids, k_weights):

        with nvtx_annotate("dispatch.fw", color="darkorange"):
            tokens_by_eid_order, eid_offs, weights_by_eid_order =self.permute_for_expert_compute(tokens, k_eids, k_weights)
        
        with nvtx_annotate("experts.fw", color="lightskyblue"):  
            expert_computed_tokens = self.expert_compute(tokens_by_eid_order, eid_offs)
        
        with nvtx_annotate("combine.fw", color="deeppink"):
            weighted_expert_computed_tokens = self.apply_router_probs(expert_computed_tokens, weights_by_eid_order)
            moe_outputs = self.restore_token_order(weighted_expert_computed_tokens)

        return moe_outputs
    
    def permute_for_dispatch(self):
        raise NotImplementedError("No dispatch required for LocalTokenShuffler")

    def dispatch(self):
        raise NotImplementedError("No dispatch required for LocalTokenShuffler")

    def permute_for_expert_compute(self, tokens, k_eids, k_weights):
        flat_k_eids = k_eids.reshape(-1) # T*K, index is token id
        k_expanded_tok_ids_by_eid_order = flat_k_eids.argsort() # T*K
        self.inv_perm = k_expanded_tok_ids_by_eid_order.argsort() 

        tok_ids_by_eid_order = k_expanded_tok_ids_by_eid_order // self.K # T*K
        slot_by_eid_order    = k_expanded_tok_ids_by_eid_order  % self.K # T*K
        
        # permute tokens, calc offsets
        tokens_by_eid_order = tokens[tok_ids_by_eid_order]
        ntok_per_eid = torch.bincount(flat_k_eids, minlength=self.E)
        eid_offs = ntok_per_eid.cumsum(dim=0).to(torch.int32)

        # permute weights
        weights_by_eid_order = k_weights[tok_ids_by_eid_order, slot_by_eid_order]        
        return tokens_by_eid_order, eid_offs, weights_by_eid_order

    def expert_compute(self, tokens_by_expert, offset):
        return self.experts(tokens_by_expert, offset)
         
    def unpermute_for_combine(self):
        raise NotImplementedError("No combine required for LocalTokenShuffler")

    def combine(self):
        raise NotImplementedError("No combine required for LocalTokenShuffler")

    def apply_router_probs(self, expert_computed_tokens, weights_by_eid_order):
        return expert_computed_tokens * weights_by_eid_order.unsqueeze(-1)
    
    def restore_token_order(self, weighted_expert_computed_tokens):
        moe_outputs = weighted_expert_computed_tokens[self.inv_perm]
        return moe_outputs