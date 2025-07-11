import time
from contextlib import contextmanager
from typing import Iterator, Optional

import torch


@contextmanager
def create_random_seed_context(seed: Optional[int] = None) -> Iterator[None]:
    try:
        g_state = torch.random.get_rng_state()
        g_cuda_states = torch.cuda.random.get_rng_state_all()
        if seed is None:
            torch.random.seed()
        else:
            torch.random.manual_seed(seed)
        yield None
    finally:
        torch.random.set_rng_state(g_state)
        torch.cuda.random.set_rng_state_all(g_cuda_states)

class _Profiler:
    def __init__(self) -> None:
        self._duration = None

    @property
    def duration(self) -> float:
        assert self._duration is not None
        return self._duration

@contextmanager
def create_profiler() -> Iterator[_Profiler]:
    try:
        start_time = time.time()
        profiler = _Profiler()
        yield profiler
    finally:
        end_time = time.time()
        profiler._duration = end_time - start_time
