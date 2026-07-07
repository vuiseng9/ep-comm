from contextlib import nullcontext, contextmanager

import torch
import torch.distributed as dist
import nvtx

from utils import rank_print, get_dist_info
from utils import yellow, red, green, cyan

class _NsysTrainStep:
    """for a single train step"""
    def __init__(self, profiler: "NsysTrainProfiler", step: int):
        self._profiler = profiler
        self._step = step

    def range(self, stage: str):
        return self._profiler._range(self._step, stage)

class NsysTrainProfiler:
    def __init__(self, start: int, end: int, train_step_label: str = "train_step_range", domain: str = None):

        self.enabled = start is not None and end is not None
        if self.enabled:
            assert end >= start, f"nsys end step ({end}) must be >= start step ({start})"

        self.start = start
        self.end = end
        self._emit_ctx = None
        self.domain = domain
        self.train_step_label = train_step_label
        self.is_dist, self.rank, _, _, _ = get_dist_info()    
        

    @contextmanager
    def step(self, global_step: int):
        if self.enabled and global_step == self.start:
            rank_print(f"-- {cyan(self.__class__.__name__)} {green('starts')} @ step {yellow(str(global_step))}.")
            torch.cuda.profiler.start()
            self._emit_ctx = torch.autograd.profiler.emit_nvtx()
            self._emit_ctx.__enter__()
        try:
            yield _NsysTrainStep(self, global_step)
        finally:
            if self.is_dist and self.rank == 0:
                dist.barrier() # rank 0 wait here.

            if self.enabled and global_step == self.end:
                rank_print(f"-- {cyan(self.__class__.__name__)} {red('ends')} @ step {yellow(str(global_step))}.")
                self._emit_ctx.__exit__(None, None, None)
                self._emit_ctx = None
                torch.cuda.profiler.stop()

            if self.is_dist and self.rank != 0:
                dist.barrier() # non zero rank signals all have stopped.

    def _range(self, step: int, stage: str):
        if not self.enabled or step < self.start or step > self.end:
            return nullcontext()
        match stage:
            case 'full':
                # marking one complete training step
                return nvtx.annotate(message=f"{self.train_step_label} @ step_{step}",
                                     domain=self.domain,
                                     color="gold")
            case 'data':
                return nvtx.annotate(message="tr-data",
                                     domain=self.domain,
                                     color="darkblue")
            case 'fwd':
                return nvtx.annotate(message="tr-forward",
                                     domain=self.domain,
                                     color="green")
            case 'bwd':
                return nvtx.annotate(message="tr-backward",
                                     domain=self.domain,
                                     color="purple")
            case 'opt':
                return nvtx.annotate(message="tr-optimize",
                                     domain=self.domain,
                                     color="red")
            case _:
                raise ValueError(f"unsupported stage: {stage}")