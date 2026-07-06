import os
import torch
import torch.distributed as dist
from dataclasses import dataclass


def yellow(s: str) -> str: return f"\033[33m{s}\033[0m"
def cyan(s: str) -> str:   return f"\033[36m{s}\033[0m"
def green(s: str) -> str:  return f"\033[32m{s}\033[0m"
def red(s: str) -> str:    return f"\033[31m{s}\033[0m"
def blue(s: str) -> str:   return f"\033[34m{s}\033[0m"
def pink(s: str) -> str:   return f"\033[35m{s}\033[0m"


def rank_print(msg, all_ranks=False):
    rank = dist.get_rank()
    if all_ranks or rank == 0:
        print(f"|R{rank}| {msg}", flush=True)


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))

    is_dist = world_size > 1
    return is_dist, rank, world_size, local_rank, local_world_size


@dataclass
class Meter:
    """Tracks last value and running arithmetic mean (no history)."""
    n: int = 0
    last: float = None
    avg: float = None

    def update(self, x: float) -> None:
        self.last = x
        if self.avg is None:
            self.avg = x
            self.n = 1
            return
        self.n += 1
        self.avg += (x - self.avg) / self.n

    def reset(self) -> None:
        self.n = 0
        self.last = None
        self.avg = None