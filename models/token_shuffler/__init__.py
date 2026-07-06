import os
from enum import Enum

from .local import LocalTokenShuffler
from .host_nccl import HostNcclTokenShuffler
from .symm_mem_naive import NaiveSymmMemTokenShuffler
from .symm_mem_pool import PooledSymmMemTokenShuffler
from .symm_mem_zero_copy import ZeroCopySymmMemTokenShuffler

class EPBackend(Enum):
    LOCAL = "local" # local compute, no cross-rank communication
    HOST_NCCL = "host_nccl" # host-initiated NCCL backend, torch's all_to_all_single
    NAIVE_SYMM = "naive_symm" # baseline PyTorch Symmetric Memory path
    POOLED_SYMM = "pooled_symm" # use symm mem pool
    ZEROCOPY_SYMM = "zerocopy_symm"  # pooled SymmMem path with zero-copy layout

    @property
    def shuffler_cls(self):
        return {
            EPBackend.LOCAL: LocalTokenShuffler,
            EPBackend.HOST_NCCL: HostNcclTokenShuffler,
            EPBackend.NAIVE_SYMM: NaiveSymmMemTokenShuffler, 
            EPBackend.POOLED_SYMM: PooledSymmMemTokenShuffler,
            EPBackend.ZEROCOPY_SYMM: ZeroCopySymmMemTokenShuffler,
        }[self]
    
    @staticmethod
    def choices():
        return set([backend.value for backend in EPBackend])

__all__ = [
    "LocalTokenShuffler", 
    "HostNcclTokenShuffler", 
    "NaiveSymmMemTokenShuffler", 
    "PooledSymmMemTokenShuffler", 
    "ZeroCopySymmMemTokenShuffler", 
    "EPBackend"
]