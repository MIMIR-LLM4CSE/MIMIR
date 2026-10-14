"""Detached worker for proxy_exec's blocking ops; not an MCP server.

Usage:
    python _proxy_job.py --job-dir <path>

Reads ``<job-dir>/job.json`` (``{"op": ..., "args": {...}}``), runs the op and writes
its reply to ``<job-dir>/result.json``. See ``_ops/exec_jobs.py``.
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import traceback

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _p in (os.path.join(_THIS_DIR, "..", "_shared"), _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def _interrupt(_signum, _frame) -> None:
    # Raised through the solver call, which then kills the solver's process group.
    raise KeyboardInterrupt


def main() -> None:
    parser = argparse.ArgumentParser(description="proxy_exec job worker")
    parser.add_argument("--job-dir", required=True)
    job_dir = os.path.abspath(parser.parse_args().job_dir)

    from responses import err
    from _lib.store import _read_json, _write_json_atomic
    from _ops import references, suites
    from _ops.exec_jobs import JOB_FILE, RESULT_FILE

    ops = {
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
    _write_json_atomic(os.path.join(job_dir, RESULT_FILE), result)
    print(f"[proxy_job] status={result.get('status')}", flush=True)


if __name__ == "__main__":
    main()
