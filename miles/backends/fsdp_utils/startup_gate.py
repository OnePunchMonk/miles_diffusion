"""Hold the train side's first large GPU allocation until the colocated engines leave enough room.

Replaces a phase-based gate ("wait until the engines booted and offloaded") with a memory budget:
the train GPU phase only needs one parameter shard plus the largest full tensor (the broadcast
from rank 0 moves one tensor at a time) plus a margin. Sizes come from the module's parameter
shapes, so no weights are read.
"""

from __future__ import annotations

import logging
import os
import time

import torch
import torch.distributed as dist

import miles.utils.startup_timing as st

logger = logging.getLogger(__name__)

GIB = 1 << 30


def budget_gate_enabled() -> bool:
    return os.environ.get("MILES_STARTUP_BUDGET_GATE") == "1"


def required_bytes(model: torch.nn.Module, shard_world_size: int, margin_bytes: int) -> tuple[int, int, int]:
    total = 0
    largest = 0
    for p in model.parameters():
        nbytes = p.numel() * p.element_size()
        total += nbytes
        largest = max(largest, nbytes)
    shard = -(-total // shard_world_size)
    return shard, largest, shard + largest + margin_bytes


def wait_for_gpu_budget(model: torch.nn.Module, *, shard_world_size: int, rollout_manager=None) -> None:
    """Block until free GPU memory covers the init peak; with MILES_STARTUP_GATE_REQUIRE_ENGINES=1 also
    wait for engine init to finish (guards the race where train allocates before the engines load)."""
    margin = int(float(os.environ.get("MILES_STARTUP_GATE_MARGIN_GIB", "4")) * GIB)
    timeout = float(os.environ.get("MILES_STARTUP_GATE_TIMEOUT_S", "900"))
    shard, largest, need = required_bytes(model, shard_world_size, margin)
    with st.step(
        "train.budget_gate",
        need_gib=f"{need / GIB:.1f}",
        shard_gib=f"{shard / GIB:.1f}",
        largest_gib=f"{largest / GIB:.2f}",
    ):
        if os.environ.get("MILES_STARTUP_GATE_REQUIRE_ENGINES") == "1" and rollout_manager is not None:
            import ray

            ray.get(rollout_manager.wait_engines_ready.remote())
        start = time.time()
        while True:
            free = torch.cuda.mem_get_info()[0]
            if free >= need:
                break
            if time.time() - start > timeout:
                raise RuntimeError(
                    f"startup gate: only {free / GIB:.1f} GiB free after {timeout:.0f}s, need {need / GIB:.1f} GiB "
                    f"(shard {shard / GIB:.1f} + largest tensor {largest / GIB:.2f} + margin {margin / GIB:.1f})"
                )
            time.sleep(0.25)
        st.mark("train.budget_gate.free", free_gib=f"{free / GIB:.1f}")
    if dist.is_initialized():
        # Ranks gate on their own device; keep them in step so the broadcast does not start half-gated.
        dist.barrier()
