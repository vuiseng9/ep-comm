import torch
import torch.nn as nn
import torch.distributed as dist
from nvtx import annotate as nvtx_annotate


from models.token_shuffler.local import LocalTokenShuffler

from .glu import GroupedGluMLP
from .token_shuffler import EPBackend

class RandomNormalSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input: torch.Tensor):
        return input.normal_()
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output
    
class MoE(nn.Module):
    def __init__(self, E, K, EP, D, Dff,
                ep_backend=EPBackend.LOCAL, lbgamma=0.0, lbforce=False):
        super().__init__()
        assert E >= K, "E must be larger than or equal to K"
        assert E >= 1, "E must be larger than or equal to 1"
        assert K >= 1, "K must be larger than or equal to 1"
        self.E = E # num of experts (global)
        self.K = K # num of activated experts (global)
        self.EP = EP
        assert E % EP == 0, "E must be divisible by EP"
        assert E >= EP, "E must be larger than or equal to EP"
        self.EPR = E // EP # num of experts per rank
        self.D = D
        self.Dff = Dff

        self.load_per_expert = None
        self.lbforce = lbforce
        self.lbgamma = lbgamma
        if self.lbgamma > 0.0:
            self.register_buffer("lbbias", torch.zeros(1, self.E))
            self.lb_target = 1/self.E # even (balanced) load across experts

        self.norm = nn.RMSNorm(D)
        self.router = nn.Linear(D, E, bias=False) # map D to E experts
        self.mlp = GroupedGluMLP(self.EPR, D, Dff) # local expert
        
        self.token_shuffler = None
        if isinstance(ep_backend, str):
            ep_backend = EPBackend(ep_backend)

        match ep_backend:
            case EPBackend.LOCAL: # LocalTokenShuffler
                assert self.EP == 1, f"{ep_backend.shuffler_cls.__name__} requires EP == 1"

                self.token_shuffler = ep_backend.shuffler_cls(
                    self.mlp, self.E, self.K)
                
            case EPBackend.HOST_NCCL: # HostNcclTokenShuffler
                assert self.EP > 1, f"{ep_backend.shuffler_cls.__name__} requires EP > 1"

                self.token_shuffler = ep_backend.shuffler_cls(
                    self.mlp, self.E, self.K, ep_group=dist.group.WORLD)
            
            case EPBackend.NAIVE_SYMM: # NaiveSymmMemTokenShuffler
                assert self.EP > 1, f"{ep_backend.shuffler_cls.__name__} requires EP > 1"

                self.token_shuffler = ep_backend.shuffler_cls(
                    self.mlp, self.E, self.K, ep_group=dist.group.WORLD)

            case EPBackend.POOLED_SYMM: # PooledSymmMemTokenShuffler
                assert self.EP > 1, f"{ep_backend.shuffler_cls.__name__} requires EP > 1"

                self.token_shuffler = ep_backend.shuffler_cls(
                    self.mlp, self.E, self.K, ep_group=dist.group.WORLD)

            case EPBackend.ZEROCOPY_SYMM: # ZeroCopySymmMemTokenShuffler
                assert self.EP > 1, f"{ep_backend.shuffler_cls.__name__} requires EP > 1"

                self.token_shuffler = ep_backend.shuffler_cls(
                    self.mlp, self.E, self.K, ep_group=dist.group.WORLD)
                self.mlp.set_symm_mem_pool_fn(self.token_shuffler.use_symm_mem_pool_ctx_fn)

            case _:
                raise ValueError(f"Unknown ep_backend: {ep_backend}")
        

    def extra_repr(self):
        return f"(ep={self.EP}, e{self.E}:k{self.K}, e_local={self.EPR}) " \
               f"\033[35m{self.token_shuffler.__class__.__name__}\033[0m"

    def update_lbbias(self):
        assert self.lbgamma > 0, "illegal call, lbgamma is not > 0"
        if self.load_per_expert is None:
            pass # first iteration
        else:
            self.lbbias.data -= self.lbgamma * (self.load_per_expert - self.lb_target)

    @nvtx_annotate("fw.MoE", color="lime")    
    def forward(self, x):
        residual = x
        B, S, D = x.shape

        # Flatten batch and length
        # [B*S, D]
        _x = self.norm(x).reshape(-1, D)
        
        # Routing Assignment ────────────────────────────────────────────────
        router_logits = self.router(_x)

        if self.lbforce:
            # overwrite router logits with random normal value,
            # resulting in balanced routing, for perf use
            router_logits = RandomNormalSTE.apply(router_logits)

        if self.lbgamma > 0:
            # --- biasing for load balance (DeepSeekv3)
            router_scores = router_logits.sigmoid()
            # lbbias does not participate in training,
            # only used for expert assignment
            self.update_lbbias()
            _, k_eids = torch.topk(router_scores + self.lbbias, k=self.K)
            k_weights = torch.gather(router_scores, dim=-1, index=k_eids)
            k_weights = k_weights / (k_weights.sum(dim=-1, keepdim=True) + 1e-9)
            
            self.load_per_expert = k_eids.clone().detach().reshape(-1).bincount(minlength=self.E)/(B*S*self.K)
        else:
            # --- load balance penalty
            router_probs = router_logits.softmax(dim=-1)
            k_probs, k_eids = torch.topk(router_probs, k=self.K)
            k_weights = k_probs / (k_probs.sum(dim=-1, keepdim=True) + 1e-9)

            self.load_per_expert = k_eids.clone().detach().reshape(-1).bincount(minlength=self.E)/(B*S*self.K)
            self.layer_lbloss = (router_probs.mean(dim=0)*self.load_per_expert).sum()
        
        # Dispatch -> Expert Compute -> Combine ──────────────────────────────
        # NOTE: expert compute is wrapped within the shuffler, 
        # the module is passed during shuffer instantiation
        moe_outputs = self.token_shuffler(_x, k_eids, k_weights).view(B*S, self.K, D)
                    
        return residual + moe_outputs.sum(dim=1).reshape(B, S, D)

