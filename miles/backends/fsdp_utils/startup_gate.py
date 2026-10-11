"""Hold the train side's first large GPU allocation until the colocated engines leave enough room.

Startup overlaps train init with engine boot, and the engines stay resident through it. Before
its first large GPU allocation a colocated train rank waits for engine init (so it cannot grab
memory the engine load is about to need), then checks a memory budget: one parameter shard plus
the largest full tensor (the broadcast from rank 0 moves one tensor at a time) plus a margin.
If that does not fit next to the engines, the engines are offloaded once, as the serial startup
did, and the driver onloads them again before the first weight sync.
"""

from __future__ import annotations

import logging
import os
import time

import ray
import torch
import torch.distributed as dist

import miles.utils.startup_timing as st

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


def _wait_free(need: int, timeout: float) -> int:
    start = time.time()
    while True:
        free = torch.cuda.mem_get_info()[0]
        if free >= need or time.time() - start > timeout:
            return free
        time.sleep(0.25)


def _all_ranks_agree(ok: bool) -> bool:
    # Every rank must take the same branch, or one rank offloads while another allocates.
    if not dist.is_initialized():
        return ok
    flag = torch.tensor([int(ok)], dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN, group=_gloo_group())
    return bool(flag.item())


def wait_for_gpu_budget(
    model: torch.nn.Module, *, shard_world_size: int, rollout_manager, engine_reserve_gib: float | None = None
) -> None:
    margin = int(float(os.environ.get("MILES_STARTUP_GATE_MARGIN_GIB", "4")) * GIB)
    timeout = float(os.environ.get("MILES_STARTUP_GATE_TIMEOUT_S", "900"))
    shard, largest, need = required_bytes(model, shard_world_size, margin)
    with st.step(
        "train.budget_gate",
        need_gib=f"{need / GIB:.1f}",
        shard_gib=f"{shard / GIB:.1f}",
        largest_gib=f"{largest / GIB:.2f}",
    ):
        if engine_reserve_gib is not None:
            # Free memory already excludes whatever the engine has allocated, so subtracting the full reserve
            # can only over-count: if this fits, the engine's remaining load fits too.
            free = torch.cuda.mem_get_info()[0]
            reserve = int(engine_reserve_gib * GIB)
            if _all_ranks_agree(free - reserve >= need):
                st.mark("train.budget_gate.free", free_gib=f"{free / GIB:.1f}", engines_resident=-1)
                return
        with st.step("train.budget_gate.wait_engines"):
            ray.get(rollout_manager.wait_engines_ready.remote())
        free = torch.cuda.mem_get_info()[0]
        fits = _all_ranks_agree(free >= need)
        if not fits:
            logger.info(
                "startup gate: %.1f GiB free next to the engines, need %.1f GiB; offloading the engines",
                free / GIB,
                need / GIB,
            )
            with st.step("train.budget_gate.offload_engines"):
                ray.get(rollout_manager.boot_offload_once.remote())
            free = _wait_free(need, timeout)
            if free < need:
                raise RuntimeError(
                    f"startup gate: only {free / GIB:.1f} GiB free after offloading the engines, need "
                    f"{need / GIB:.1f} GiB (shard {shard / GIB:.1f} + largest tensor {largest / GIB:.2f} "
                    f"+ margin {margin / GIB:.1f})"
                )
        st.mark("train.budget_gate.free", free_gib=f"{free / GIB:.1f}", engines_resident=int(fits))


def _gloo_group():
    from miles.utils.distributed_utils import get_gloo_group

    return get_gloo_group()
