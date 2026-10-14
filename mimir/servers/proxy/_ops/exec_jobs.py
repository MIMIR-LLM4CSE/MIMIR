"""proxy_exec's blocking ops (reference, suite, benchmark_create) run as detached jobs.

Each of them runs the proxy to completion — up to an hour for a reference, two for a
suite. Run inside the tool call, that hit the client's tool wall long before the work
was done, and a synchronous tool also held the server's event loop the whole time. The
op now runs in a worker process of its own (``_proxy_job.py``) that writes its reply to
``result.json``; the call waits for it as ``proxy_eval(op='run')`` does, and hands it to
the background watcher when asked to up front, when the user diverts it, or when the
wait budget runs out.
"""

from __future__ import annotations

import os
import sys

from _ops import _PROXY_DIR, _with_next, err, ok
from _ops.eval_session import _NEXT_DETACHED, _RUN_WAIT_BUDGET_S, _await_terminal_state
from _lib import store
from _lib.procs import (
    _launch_detached, _log_path, _new_run_dir, _read_log, _run_state,
    background_descriptor,
)
from _lib.store import _read_json, _write_json_atomic

_JOB_WORKER = os.path.join(_PROXY_DIR, "_proxy_job.py")
JOB_FILE = "job.json"
RESULT_FILE = "result.json"
# The blocking tool's registered name, so the client can address the channel off the row.
_DIVERT_CHANNEL = "proxy_exec"


def jobs_dir() -> str:
    return os.path.join(store.cache_dir(), "exec_jobs")


def is_job_dir(path: str) -> bool:
    return os.path.isfile(os.path.join(path, JOB_FILE))


def _result(job_dir: str) -> dict | None:
    result = _read_json(os.path.join(job_dir, RESULT_FILE))
    return result if isinstance(result, dict) else None


def job_state(job_dir: str) -> dict:
    """``_run_state`` for a job: finished means a result was written, and it is ok."""
    st = _run_state(job_dir)
    if st["state"] not in ("running", "pending"):
        result = _result(job_dir)
        st["state"] = "done" if result and result.get("status") == "ok" else "crashed"
    return st


def _no_result(job_dir: str, op: str) -> dict:
    tail = _read_log(job_dir)[-1500:]
    return err(f"proxy_exec(op='{op}') ended without writing a result.",
               hint=f"Read its log at {_log_path(job_dir)}. Last output:\n{tail}")


def status(job_dir: str) -> dict:
    """What a watcher polls, and what it reads once the job is over: the op's own reply."""
    op = (_read_json(os.path.join(job_dir, JOB_FILE), {}) or {}).get("op", "")
    st = job_state(job_dir)
    payload = {"run_id": job_dir, "run_dir": job_dir, "op": op,
               "log": _log_path(job_dir), **st}
    if st["state"] in ("running", "pending"):
        return ok(_with_next(payload, _NEXT_DETACHED))
    result = _result(job_dir)
    if result is None:
        payload["error"] = _no_result(job_dir, op)["error"]
        payload["log_tail"] = _read_log(job_dir)[-1500:]
        return ok(_with_next(payload, f"read the log at {_log_path(job_dir)}"))
    payload["result"] = result
    return ok(_with_next(payload, result.get("next_step") or result.get("hint") or ""))


def _descriptor(job_dir: str, op: str) -> dict:
    poll = {"tool": "proxy_runs", "args": {"op": "status", "run_id": job_dir}}
    return background_descriptor(job_dir, kind=f"proxy-{op}",
                                 status_op=poll, summary_op=poll)


async def run_awaited(op: str, args: dict, background: bool = False,
                      wait_s: float = _RUN_WAIT_BUDGET_S) -> dict:
    """Launch *op* with *args* in a worker; answer with its reply, or detach."""
    job_dir = _new_run_dir(jobs_dir(), tag_suffix=op)
    _write_json_atomic(os.path.join(job_dir, JOB_FILE), {"op": op, "args": args})
    # The tests (and any embedder) repoint the store in-process; the worker must follow.
    env = {**os.environ, "MIMIR_PROXY_BENCH_DIR": store.cache_dir()}
    pid = _launch_detached([sys.executable, _JOB_WORKER, "--job-dir", job_dir],
                           job_dir, log_file=_log_path(job_dir), env=env)

    if background:
        note = f"proxy_exec(op='{op}') detached; this call returns before it ends."
    else:
        state, reason = await _await_terminal_state(
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

    return ok(_with_next({
        "op":             op,
        "job_dir":        job_dir,
        "pid":            pid,
        "log":            _log_path(job_dir),
        "note":           note,
        "background_job": _descriptor(job_dir, op),
    }, _NEXT_DETACHED))
