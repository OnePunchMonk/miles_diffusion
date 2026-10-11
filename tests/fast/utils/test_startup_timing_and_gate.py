import subprocess
import sys
import textwrap

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


def test_probe_lines_round_trip_through_parser(tmp_path):
    log = tmp_path / "run.log"
    script = textwrap.dedent(
        """
        import miles.utils.startup_timing as st
        st.set_role("driver")
        with st.step("driver.x", k=1):
            pass
        st.mark("driver.startup_done")
        """
    )
    out = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, check=True).stdout
    log.write_text(out)
    import importlib.util

    spec = importlib.util.spec_from_file_location("parse_startup_timing", "scripts/parse_startup_timing.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    events = mod.parse(str(log))
    names = [e["event"] for e in events]
    assert "driver.x.begin" in names and "driver.x.end" in names and "driver.startup_done" in names
    assert next(e for e in events if e["event"] == "driver.x.end")["dur"] >= 0


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
