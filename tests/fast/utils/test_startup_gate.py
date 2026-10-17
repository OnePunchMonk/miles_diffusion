import subprocess
import sys

import pytest
import torch

from miles.backends.fsdp_utils import startup_gate


class _Handle:
    """Stands in for the driver's GpuReady handle: records the wait, optionally frees memory."""

    def __init__(self, name, waited, on_wait=None):
        self.name, self.waited, self.on_wait = name, waited, on_wait

    def wait(self):
        self.waited.append(self.name)
        if self.on_wait:
            self.on_wait()


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


def test_gate_waits_for_the_handle_then_passes_when_it_fits(monkeypatch):
    _free(monkeypatch, 1 << 40)
    waited = []
    startup_gate.wait_for_gpu_budget(
        torch.nn.Linear(4, 4), shard_world_size=1, gpu_ready=_Handle("engines-offloaded", waited)
    )
    assert waited == ["engines-offloaded"]


def test_gate_passes_once_the_handle_frees_memory(monkeypatch):
    state = _free(monkeypatch, 1)
    waited = []
    handle = _Handle("engines-offloaded", waited, on_wait=lambda: state.update(free=1 << 40))
    startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1, gpu_ready=handle)
    assert waited == ["engines-offloaded"]


def test_gate_raises_when_it_never_fits(monkeypatch):
    _free(monkeypatch, 1)
    monkeypatch.setenv("MILES_STARTUP_GATE_TIMEOUT_S", "0.3")
    with pytest.raises(RuntimeError, match="startup gate"):
        startup_gate.wait_for_gpu_budget(
            torch.nn.Linear(4, 4), shard_world_size=1, gpu_ready=_Handle("engines-ready", [])
        )


def test_reserve_lets_train_go_before_the_engines_when_it_fits(monkeypatch):
    _free(monkeypatch, 100 << 30)
    waited = []
    startup_gate.wait_for_gpu_budget(
        torch.nn.Linear(4, 4), shard_world_size=1, gpu_ready=_Handle("engines-ready", waited), engine_reserve_gib=60
    )
    assert waited == []


def test_reserve_that_does_not_fit_falls_back_to_waiting(monkeypatch):
    _free(monkeypatch, 100 << 30)
    waited = []
    startup_gate.wait_for_gpu_budget(
        torch.nn.Linear(4, 4), shard_world_size=1, gpu_ready=_Handle("engines-ready", waited), engine_reserve_gib=99
    )
    assert waited == ["engines-ready"]


def test_driver_picks_what_train_waits_for():
    import argparse

    from miles.ray import placement_group as pg

    class _RM:
        wait_engines_ready = type("R", (), {"remote": staticmethod(lambda: "engines-ready")})
        offload = type("R", (), {"remote": staticmethod(lambda: "engines-offloaded")})

    def args(**kw):
        base = dict(offload_rollout=True, serial_startup=False, startup_engine_reserve_gib=None)
        return argparse.Namespace(**{**base, **kw})

    assert pg._startup_gpu_ready(args(offload_rollout=False), _RM) is None
    assert pg._startup_gpu_ready(args(serial_startup=True), _RM) is None
    assert pg._startup_gpu_ready(args(), _RM)._ref == "engines-offloaded"
    assert pg._startup_gpu_ready(args(startup_engine_reserve_gib=72), _RM)._ref == "engines-ready"
    assert pg.engines_stay_resident_at_startup(args(startup_engine_reserve_gib=72))
    assert not pg.engines_stay_resident_at_startup(args())


def test_gpu_ready_reaches_the_task_unresolved_and_the_task_starts_first():
    """Ray resolves top-level ref args before a task runs; GpuReady must not let that happen."""
    import time

    ray = pytest.importorskip("ray")
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, log_to_driver=False)
    try:

        @ray.remote
        class Engines:
            def ready(self, delay):
                time.sleep(delay)
                return time.time()

        @ray.remote
        class Train:
            def init(self, gpu_ready):
                started = time.time()
                unresolved = isinstance(gpu_ready._ref, ray.ObjectRef)
                gpu_ready.wait()
                return started, unresolved

        engines, train = Engines.remote(), Train.remote()
        ray.get([engines.__ray_ready__.remote(), train.__ray_ready__.remote()])  # workers up first
        ready = startup_gate.GpuReady(engines.ready.remote(8.0))
        started, unresolved = ray.get(train.init.remote(ready))
        resolved = ray.get(ready._ref)
        assert unresolved
        assert started < resolved - 4.0  # train init began well before the engines were ready
    finally:
        ray.shutdown()


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
