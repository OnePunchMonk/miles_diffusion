"""Hold a colocated train rank's first large GPU allocation until it fits next to the rollout engines.

Startup overlaps train init with engine boot. The train side only needs room for one parameter shard
plus the largest full tensor (the broadcast from rank 0 moves one tensor at a time) plus a margin.
What it waits for is decided by the driver and handed over as an opaque ``gpu_ready`` handle (a Ray
ObjectRef): engines initialized, or engines offloaded. With --startup-engine-reserve-gib, an upper
bound on what one engine rank holds, a rank whose budget fits next to that reserve does not wait.
"""

from __future__ import annotations

import logging
import os
import time

import ray
import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)

GIB = 1 << 30


def required_bytes(model: torch.nn.Module, shard_world_size: int, margin_bytes: int) -> tuple[int, int, int]:
    total = 0
    largest = 0
    for p in model.parameters():
        nbytes = p.numel() * p.element_size()
        total += nbytes
        largest = max(largest, nbytes)
    shard = -(-total // shard_world_size)
    return shard, largest, shard + largest + margin_bytes


class GpuReady:
    """What a train rank waits on before its first large GPU allocation, as handed over by the driver.

    Holds the Ray ObjectRef inside an object: Ray resolves refs passed as top-level task arguments
    before the task starts, which would hold the whole train init until the engines are ready.
    """

    def __init__(self, ref):
        self._ref = ref

    def wait(self) -> None:
        ray.get(self._ref)


def _wait_free(need: int, timeout: float) -> int:
    start = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0]
        if free >= need or time.time() - start > timeout:
            return free
        time.sleep(0.25)


def _all_ranks_agree(ok: bool) -> bool:
    # Every rank must take the same branch, or one rank waits while another allocates.
    if not dist.is_initialized():
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=_gloo_group())
    return bool(flag.item())


def wait_for_gpu_budget(
    model: torch.nn.Module, *, shard_world_size: int, gpu_ready=None, engine_reserve_gib: float | None = None
) -> None:
    margin = int(float(os.environ.get("MILES_STARTUP_GATE_MARGIN_GIB", "4")) * GIB)
    timeout = float(os.environ.get("MILES_STARTUP_GATE_TIMEOUT_S", "900"))
    shard, largest, need = required_bytes(model, shard_world_size, margin)
    if engine_reserve_gib is not None:
        # Free memory already excludes whatever the engines have allocated, so subtracting the full
        # reserve can only over-count: if this fits, the engines' remaining load fits too.
        free = torch.cuda.mem_get_info()[0]
        if _all_ranks_agree(free - int(engine_reserve_gib * GIB) >= need):
            return
    if gpu_ready is not None:
        gpu_ready.wait()
    free = _wait_free(need, timeout)
    if free < need:
        raise RuntimeError(
            f"startup gate: only {free / GIB:.1f} GiB free, need {need / GIB:.1f} GiB "
            f"(shard {shard / GIB:.1f} + largest tensor {largest / GIB:.2f} + margin {margin / GIB:.1f}). "
            "If --startup-engine-reserve-gib is set, the engines may not fit next to the train shard; "
            "unset it to offload the engines during train init."
        )


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()
