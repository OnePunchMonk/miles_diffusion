"""Start sglang-d scheduler workers before the server process imports sglang.

A scheduler worker spends most of its start-up importing torch, transformers, diffusers and sglang,
and the server process imports the same stack before it starts the workers. Starting the workers
first lets both imports run in parallel. Each worker imports its target, then waits for the
arguments the server would have spawned it with.

Keep this module stdlib only: it runs before anything heavy is imported.
"""

import importlib
import multiprocessing as mp
import os

DEFAULT_TARGET = "sglang.multimodal_gen.runtime.managers.gpu_worker:run_scheduler_process"


def _worker_main(conn, target: str) -> None:
    import miles.utils.startup_timing as st  # stdlib only

    st.set_role(f"prespawn{os.getpid()}")
    module_name, func_name = target.split(":")
    with st.step("prespawn.worker_import"):
        func = getattr(importlib.import_module(module_name), func_name)
    msg = conn.recv()
    st.mark("prespawn.worker_got_args")
    conn.close()
    if msg is None:
        return
    env, args = msg
    # Match the environment the server process had when it would have spawned this worker.
    os.environ.clear()
    os.environ.update(env)
    func(*args)


class PrespawnedProcess:
    """Stands in for ``mp.Process(target=..., args=args)``: ``start()`` hands the args to a waiting worker."""

    def __init__(self, process, conn, args, name):
        self._process = process
        self._conn = conn
        self._args = args
        self.name = name

    def start(self) -> None:
        self._conn.send((dict(os.environ), self._args))
        self._conn.close()

    def __getattr__(self, attr):
        return getattr(self._process, attr)


def prespawn(num_workers: int, target: str = DEFAULT_TARGET) -> list:
    """Start ``num_workers`` waiting workers; returns factories ``factory(args=..., name=...)``."""
    ctx = mp.get_context("spawn")
    factories = []
    for _ in range(num_workers):
        reader, writer = ctx.Pipe(duplex=False)
        process = ctx.Process(target=_worker_main, args=(reader, target), daemon=True)
        process.start()
        reader.close()
        factories.append(
            lambda args, name, process=process, writer=writer: PrespawnedProcess(process, writer, args, name)
        )
    return factories
