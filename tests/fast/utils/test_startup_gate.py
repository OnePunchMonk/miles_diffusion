import subprocess
import sys

import pytest
import torch

from miles.backends.fsdp_utils import startup_gate


class _Remote:
    def __init__(self, fn):
        self.fn = fn

    def remote(self):
        return self.fn()


class FakeRolloutManager:
    def __init__(self, on_offload=None):
        self.calls = []
        self.on_offload = on_offload
        self.wait_engines_ready = _Remote(lambda: self.calls.append("wait") or True)
        self.boot_offload_once = _Remote(self._offload)

    def _offload(self):
        self.calls.append("offload")
        if self.on_offload:
            self.on_offload()
        return True


@pytest.fixture
def no_ray(monkeypatch):
    monkeypatch.setattr(startup_gate.ray, "get", lambda x: x)


def _free(monkeypatch, value):
    state = {"free": value}
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (state["free"], 1 << 41), raising=False)
    return state


def test_required_bytes_is_shard_plus_largest_plus_margin():
    model = torch.nn.Sequential(torch.nn.Linear(10, 10, bias=False), torch.nn.Linear(10, 40, bias=False))
    total = (100 + 400) * 4
    shard, largest, need = startup_gate.required_bytes(model, shard_world_size=2, margin_bytes=1000)
    assert shard == total // 2
    assert largest == 400 * 4
    assert need == shard + largest + 1000


def test_required_bytes_works_on_meta_params():
    with torch.device("meta"):
        model = torch.nn.Linear(8, 8, bias=False)
    assert startup_gate.required_bytes(model, 1, 0)[2] == 8 * 8 * 4 * 2


def test_gate_keeps_engines_resident_when_it_fits(monkeypatch, no_ray):
    _free(monkeypatch, 1 << 40)
    rm = FakeRolloutManager()
    startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1, rollout_manager=rm)
    assert rm.calls == ["wait"]


def test_gate_offloads_engines_when_it_does_not_fit(monkeypatch, no_ray):
    state = _free(monkeypatch, 1)
    rm = FakeRolloutManager(on_offload=lambda: state.update(free=1 << 40))
    startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1, rollout_manager=rm)
    assert rm.calls == ["wait", "offload"]


def test_gate_raises_when_offload_does_not_free_enough(monkeypatch, no_ray):
    _free(monkeypatch, 1)
    monkeypatch.setenv("MILES_STARTUP_GATE_TIMEOUT_S", "0.3")
    rm = FakeRolloutManager()
    with pytest.raises(RuntimeError, match="startup gate"):
        startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1, rollout_manager=rm)


def test_reserve_lets_train_go_before_engines_when_it_fits(monkeypatch, no_ray):
    _free(monkeypatch, 100 << 30)
    rm = FakeRolloutManager()
    startup_gate.wait_for_gpu_budget(
        torch.nn.Linear(4, 4), shard_world_size=1, rollout_manager=rm, engine_reserve_gib=60
    )
    assert rm.calls == []


def test_reserve_falls_back_to_waiting_for_engines_when_it_does_not_fit(monkeypatch, no_ray):
    _free(monkeypatch, 100 << 30)
    rm = FakeRolloutManager()
    startup_gate.wait_for_gpu_budget(
        torch.nn.Linear(4, 4), shard_world_size=1, rollout_manager=rm, engine_reserve_gib=99
    )
    assert rm.calls == ["wait"]


def test_picklable_args_drops_only_unpicklable_values():
    import argparse
    import pickle

    from miles.backends.sglang_diffusion_utils.sglang_diffusion_engine import _picklable_args

    args = argparse.Namespace(hf_checkpoint="m", sglang_tp_size=2, hook=lambda: None)
    copy = _picklable_args(args)
    assert vars(copy) == {"hf_checkpoint": "m", "sglang_tp_size": 2}
    pickle.dumps(copy)


def test_engine_module_does_not_import_sglang():
    code = (
        "import sys; import miles.backends.sglang_diffusion_utils.sglang_diffusion_engine, miles.ray.rollout; "
        "print('sglang' in sys.modules)"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_train_ranks_are_created_together_then_get_rank0_address(monkeypatch):
    import argparse
    from types import SimpleNamespace

    from miles.ray import actor_group

    events = []

    class FakeActor:
        def __init__(self, rank):
            self.rank = rank
            self.get_master_addr_and_port = SimpleNamespace(
                remote=lambda: events.append(("get_addr", self.rank)) or ("10.0.0.1", 29500)
            )
            self.set_master_addr_and_port = SimpleNamespace(
                remote=lambda addr, port: events.append(("set_addr", self.rank, addr, port))
            )

    class FakeRemoteClass:
        def options(self, **kwargs):
            return SimpleNamespace(
                remote=lambda world_size, rank, addr, port: events.append(("create", rank, addr)) or FakeActor(rank)
            )

    monkeypatch.setattr(actor_group.ray, "remote", lambda **kwargs: (lambda cls: FakeRemoteClass()))
    monkeypatch.setattr(actor_group.ray, "get", lambda x: x)
    monkeypatch.setattr(actor_group, "PlacementGroupSchedulingStrategy", lambda **kwargs: None)
    args = argparse.Namespace(train_env_vars={}, deterministic_mode=False, train_backend="fsdp")

    actor_group.RayTrainGroup(args, num_nodes=1, num_gpus_per_node=3, pg=(object(), [0, 1, 2], [0, 1, 2]))

    assert events == [
        ("create", 0, None),
        ("create", 1, None),
        ("create", 2, None),
        ("get_addr", 0),
        ("set_addr", 1, "10.0.0.1", 29500),
        ("set_addr", 2, "10.0.0.1", 29500),
    ]


def test_prespawned_workers_get_args_pipes_and_env_at_start(monkeypatch):
    import multiprocessing as mp

    from miles.backends.sglang_diffusion_utils.scheduler_prespawn import prespawn

    workers = prespawn(2, target="tests.fast.utils.prespawn_fake_target:run")
    monkeypatch.setenv("PRESPAWN_TEST_ENV", "set-after-prespawn")
    readers, procs = [], []
    for rank, worker in enumerate(workers):
        reader, writer = mp.Pipe(duplex=False)
        proc = worker(args=(rank, writer, {"rank": rank}), name=f"w{rank}")
        proc.start()
        writer.close()
        readers.append(reader)
        procs.append(proc)
    got = sorted(reader.recv() for reader in readers)
    for proc in procs:
        proc.join(30)
    assert got == [(0, {"rank": 0}, "set-after-prespawn"), (1, {"rank": 1}, "set-after-prespawn")]
    assert [proc.exitcode for proc in procs] == [0, 0]


def test_cancelled_prespawned_worker_exits_cleanly():
    from miles.backends.sglang_diffusion_utils.scheduler_prespawn import prespawn

    (worker,) = prespawn(1, target="tests.fast.utils.prespawn_fake_target:run")
    worker.cancel()
    worker._process.join(30)
    assert worker._process.exitcode == 0
