import socket
import subprocess
import sys
import textwrap
import threading
import time

import pytest
import torch

import miles.utils.startup_timing as st
from miles.backends.fsdp_utils import startup_gate
from miles.ray import rollout


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


def test_budget_gate_flag(monkeypatch):
    monkeypatch.delenv("MILES_STARTUP_BUDGET_GATE", raising=False)
    assert not startup_gate.budget_gate_enabled()
    monkeypatch.setenv("MILES_STARTUP_BUDGET_GATE", "1")
    assert startup_gate.budget_gate_enabled()


def test_gate_times_out_when_budget_never_fits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 2), raising=False)
    monkeypatch.setenv("MILES_STARTUP_GATE_TIMEOUT_S", "0.3")
    with pytest.raises(RuntimeError, match="startup gate"):
        startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1)


def test_gate_passes_when_budget_fits(monkeypatch):
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1 << 40, 1 << 41), raising=False)
    startup_gate.wait_for_gpu_budget(torch.nn.Linear(4, 4), shard_world_size=1)


def test_wait_port_open_returns_once_listening():
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    port = srv.getsockname()[1]
    threading.Timer(0.3, srv.listen).start()

    class Alive:
        def is_alive(self):
            return True

    t0 = time.time()
    rollout._wait_port_open("127.0.0.1", port, Alive(), timeout=5)
    assert 0.2 < time.time() - t0 < 3
    srv.close()


def test_wait_port_open_gives_up_when_process_dies():
    class Dead:
        def is_alive(self):
            return False

    t0 = time.time()
    rollout._wait_port_open("127.0.0.1", 9, Dead(), timeout=5)
    assert time.time() - t0 < 1


def test_pick_router_addr_respects_external_router():
    args = type("A", (), {"sglang_router_ip": "10.0.0.1", "sglang_router_port": 1234})()
    rollout._pick_router_addr(args)
    assert (args.sglang_router_ip, args.sglang_router_port) == ("10.0.0.1", 1234)
    assert args._router_external


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
