from abc import ABC, abstractmethod
from torch import nn
import torch.distributed as dist

class MoETokenShuffler(ABC):
    def __init__(self, module_of_experts: nn.Module, E, K, ep_group=None):
        self.experts = module_of_experts
        self.E = E
        self.K = K
        self.ep_group = ep_group
        self.EP = 1
        if ep_group is not None:
            self.EP = dist.get_world_size(group=ep_group)
        assert self.E % self.EP == 0, "E must be divisible by EP"
        self.EPR = self.E // self.EP

    @abstractmethod
    def __call__(self):
        """Run the full MoE token shuffle pipeline. """
        ...

    @abstractmethod
    def permute_for_dispatch(self):
        """Arrange local tokens so EP dispatch can send them to the right EP ranks. """
        ...

    @abstractmethod
    def dispatch(self):
        """Exchange tokens across EP ranks. """
        ...

    @abstractmethod
    def permute_for_expert_compute(self):
        """Arrange received tokens so each local expert gets a contiguous chunk. """
        ...

    @abstractmethod
    def expert_compute(self):
        """Run the expert MLPs. """
        ...

    @abstractmethod
    def unpermute_for_combine(self):
        """Undo the expert-compute layout so outputs are ready to be sent back. """
        ...

    @abstractmethod
    def combine(self):
        """Exchange expert outputs back to the original token ranks. """
        ...

    @abstractmethod
    def apply_router_probs(self):
        """Return combined tokens weighted by their routing probabilities."""
        ...

    @abstractmethod
    def restore_token_order(self):
        """Put outputs back into the model's original token order. """
        ...

    