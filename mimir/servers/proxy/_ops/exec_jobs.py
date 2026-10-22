"""proxy_exec's long ops (reference, suite, benchmark_create) as detached worker processes
(``_proxy_job.py``), awaited like ``proxy_eval(op='run')`` and handed to the watcher.
"""

from __future__ import annotations

from .._lib import layout
import os
import sys

from ....shared.launcher import launch_argv
from . import with_next, err, ok
from .eval_common import NEXT_DETACHED
from .eval_session import RUN_WAIT_BUDGET_S, await_terminal_state
from .._lib.procs import (
    launch_detached,
    run_log_path,
    new_run_dir,
    read_log,
    run_state,
    background_descriptor,
)
from .._lib.layout import read_json
from ....shared.atomic import write_json

_JOB_WORKER = "mimir.servers.proxy._proxy_job"
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
# The blocking tool's registered name, so the client can address the channel off the row.
_DIVERT_CHANNEL = "proxy_exec"


def jobs_dir() -> str:
    return os.path.join(layout.cache_dir(), "exec_jobs")


def is_job_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, JOB_FILE))


def _result(job_dir: str) -> dict | None:
    result = read_json(os.path.join(job_dir, RESULT_FILE))
    return result if isinstance(result, dict) else None


def job_state(job_dir: str) -> dict:
    """``run_state`` for a job: finished means a result was written, and it is ok."""
    st = run_state(job_dir)
    if st["state"] not in ("running", "pending"):
        result = _result(job_dir)
        st["state"] = "done" if result and result.get("status") == "ok" else "crashed"
    return st


def _no_result(job_dir: str, op: str) -> dict:
    tail = read_log(job_dir)[-1500:]
    return err(f"proxy_exec(op='{op}') ended without writing a result.",
               hint=f"Read its log at {run_log_path(job_dir)}. Last output:\n{tail}")


def status(job_dir: str) -> dict:
    """What a watcher polls, and what it reads once the job is over: the op's own reply."""
    op = (read_json(os.path.join(job_dir, JOB_FILE), {}) or {}).get("op", "")
    st = job_state(job_dir)
    payload = {"run_id": job_dir, "run_dir": job_dir, "op": op,
               "log": run_log_path(job_dir), **st}
    if st["state"] in ("running", "pending"):
        return ok(with_next(payload, NEXT_DETACHED))
    result = _result(job_dir)
    if result is None:
        payload["error"] = _no_result(job_dir, op)["error"]
        payload["log_tail"] = read_log(job_dir)[-1500:]
        return ok(with_next(payload, f"read the log at {run_log_path(job_dir)}"))
    payload["result"] = result
    return ok(with_next(payload, result.get("next_step") or result.get("hint") or ""))


def _descriptor(job_dir: str, op: str) -> dict:
    poll = {"tool": "proxy_runs", "args": {"op": "status", "run_id": job_dir}}
    return background_descriptor(job_dir, kind=f"proxy-{op}",
                                 status_op=poll, summary_op=poll,
                                 cancel_op={"tool": "proxy_exec",
                                            "args": {"op": "cancel", "run_id": job_dir}})


async def run_awaited(op: str, args: dict, background: bool = False,
                      wait_s: float = RUN_WAIT_BUDGET_S) -> dict:
    """Launch *op* with *args* in a worker; answer with its reply, or detach."""
    job_dir = new_run_dir(jobs_dir(), tag_suffix=op)
    write_json(os.path.join(job_dir, JOB_FILE), {"op": op, "args": args}, indent=2)
    # The tests (and any embedder) repoint the store in-process; the worker must follow.
    env = {**os.environ, "MIMIR_PROXY_BENCH_DIR": layout.cache_dir()}
    pid = launch_detached([sys.executable, *launch_argv(_JOB_WORKER, "--job-dir", job_dir)],
                           job_dir, log_file=run_log_path(job_dir), env=env)

    if background:
        note = f"proxy_exec(op='{op}') detached; this call returns before it ends."
    else:
        state, reason = await await_terminal_state(
            job_dir, wait_s, _DIVERT_CHANNEL, os.path.basename(job_dir), pid,
            state_fn=job_state, label=f"proxy {op}")
        if state is not None:
            result = _result(job_dir)
            if result is None:
                return _no_result(job_dir, op)
            result.setdefault("job_dir", job_dir)
            return result
        note = (
            "The user moved this job to the background while it was still going. It "
            "was not killed and it did not fail: it continues, and you are resumed "
            "with its result when it ends."
            if reason == "diverted" else
            f"Still running after {int(wait_s)}s — handed to the background watcher."
        )

    return ok(with_next({
        "op":             op,
        "job_dir":        job_dir,
        "pid":            pid,
        "log":            run_log_path(job_dir),
        "note":           note,
        "background_job": _descriptor(job_dir, op),
    }, NEXT_DETACHED))
