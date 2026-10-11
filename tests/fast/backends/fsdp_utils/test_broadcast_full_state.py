"""broadcast_full_state_to_fsdp gives every rank rank 0's params and buffers (CPU, gloo, FSDP2)."""

import os

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


class _Net(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.a = torch.nn.Linear(8, 16)
        self.b = torch.nn.Linear(16, 4, bias=False)
        self.register_buffer("persistent", torch.arange(6, dtype=torch.float32))
        self.register_buffer("table", torch.arange(5, dtype=torch.float32) * 0.5, persistent=False)


def _reference():
    torch.manual_seed(0)
    net = _Net()
    with torch.no_grad():
        net.persistent.mul_(3)
        net.table.add_(7)
    return net


def _worker(rank, world, port, out):
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    try:
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.fsdp import fully_shard

        from miles.backends.fsdp_utils.checkpoint import broadcast_full_state_to_fsdp

        if rank == 0:
            model = _reference()
            full_state = model.state_dict()
        else:
            from accelerate import init_empty_weights

            with init_empty_weights(include_buffers=False):
                model = _Net()  # params on meta, buffers hold constructor values, not rank 0's
            full_state = {}
        mesh = init_device_mesh("cpu", (world,))
        fully_shard(model, mesh=mesh)
        broadcast_full_state_to_fsdp(model, full_state, cpu_offload=False, device="cpu")
        ref = _reference()
        params_ok = all(
            torch.equal(p.full_tensor(), dict(ref.named_parameters())[n]) for n, p in model.named_parameters()
        )
        buffers_ok = all(torch.equal(b, dict(ref.named_buffers())[n]) for n, b in model.named_buffers())
        out[rank] = (params_ok, buffers_ok)
    finally:
        dist.destroy_process_group()


def test_every_rank_gets_rank0_params_and_buffers():
    pytest.importorskip("accelerate")
    world = 2
    port = 29500 + os.getpid() % 1000
    with mp.Manager() as manager:
        out = manager.dict()
        mp.spawn(_worker, args=(world, port, out), nprocs=world, join=True)
        assert dict(out) == {0: (True, True), 1: (True, True)}
