"""Detached worker for proxy_exec's blocking ops; not an MCP server.

Usage (through the launcher, ``_shared/launcher.launch_argv``):
    python servers/_launch.py mimir.servers.proxy._proxy_job --job-dir <path>

Reads ``<job-dir>/job.json`` (``{"op": ..., "args": {...}}``), runs the op and writes
its reply to ``<job-dir>/result.json``. See ``_ops/exec_jobs.py``.
"""

from __future__ import annotations

import argparse
import os
import signal
import traceback
from typing import Callable



def _interrupt(_signum, _frame) -> None:
    # Raised through the solver call, which then kills the solver's process group.
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="proxy_exec job worker")
    parser.add_argument("--job-dir", required=True)
    job_dir = os.path.abspath(parser.parse_args().job_dir)

    from .._shared.responses import err
    from .._shared.atomic import write_json
    from ._lib.store import _read_json
    from ._ops import references, suites
    from ._ops.exec_jobs import JOB_FILE, RESULT_FILE

    ops: dict[str, Callable[..., dict]] = {
        "reference":        references.create,
        "suite":            suites.run_suite,
        "benchmark_create": suites.benchmark_create,
    }
    job = _read_json(os.path.join(job_dir, JOB_FILE), {}) or {}
    op = job.get("op", "")
    signal.signal(signal.SIGTERM, _interrupt)
    print(f"[proxy_job] op={op} job_dir={job_dir}", flush=True)
    try:
        if op not in ops:
            result = err(f"Unknown job op '{op}'.")
        else:
            result = ops[op](**(job.get("args") or {}))
    except KeyboardInterrupt:
        result = err(f"proxy_exec(op='{op}') was cancelled.")
    except Exception as exc:
        traceback.print_exc()
        result = err(f"proxy_exec(op='{op}') failed: {type(exc).__name__}: {exc}")
    write_json(os.path.join(job_dir, RESULT_FILE), result, indent=2)
    print(f"[proxy_job] status={result.get('status')}", flush=True)


if __name__ == "__main__":
    main()
