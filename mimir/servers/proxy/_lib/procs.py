"""Process and run-state helpers for the proxy server.

Owns everything about a run directory's *liveness*: pid/starttime bookkeeping
(with PID-recycling and zombie detection), Slurm job state, the detached-launch
/ sbatch-submit / cancel lifecycle, and log access.  Pure storage layout lives
in ``store``; this module only adds process semantics on top of it.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import time
from datetime import datetime, timezone

from responses import err

from _lib import store

_MAX_LOG = 256 * 1024   # bytes


# ── run-directory files ───────────────────────────────────────────────────────

def _log_path(run_dir: str) -> str:
    return os.path.join(run_dir, "stdout.log")

def _pid_path(run_dir: str) -> str:
    return os.path.join(run_dir, "pid")

def _pid_starttime_path(run_dir: str) -> str:
    return os.path.join(run_dir, "pid_starttime")

def _slurm_id_path(run_dir: str) -> str:
    return os.path.join(run_dir, "slurm_job_id")


def _read_int_file(path: str) -> int | None:
    if os.path.isfile(path):
        try:
            return int(open(path).read().strip())
        except (ValueError, OSError):
            pass
    return None


def _read_pid(run_dir: str) -> int | None:
    return _read_int_file(_pid_path(run_dir))


def _read_slurm_id(run_dir: str) -> int | None:
    return _read_int_file(_slurm_id_path(run_dir))


def _read_log(run_dir: str, max_bytes: int = _MAX_LOG) -> str:
    """Return up to *max_bytes* of the run's stdout.log ('' if unreadable)."""
    lp = _log_path(run_dir)
    if os.path.isfile(lp):
        try:
            with open(lp, errors="replace") as fh:
                return fh.read(max_bytes)
        except OSError:
            pass
    return ""


# ── process state ─────────────────────────────────────────────────────────────

def _read_proc_starttime(pid: int) -> int | None:
    """Return the starttime field from /proc/{pid}/stat (Linux only).

    Returns None on non-Linux or any read/parse failure.
    Used to detect PID recycling.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        idx = data.rfind(")")
        if idx < 0:
            return None
        fields = data[idx + 2:].split()
        return int(fields[19])
    except (OSError, IndexError, ValueError):
        return None


def _is_running(pid: int, expected_starttime: int | None = None) -> bool:
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    # A zombie has already exited — only its unreaped table entry remains
    # (the launcher never wait()s on detached children), so it must count as
    # finished or run states would stay "running" until the entry is reaped.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        idx = data.rfind(")")
        if idx >= 0 and data[idx + 2:].split()[0] == "Z":
            return False
    except (OSError, IndexError):
        pass
    # Always check starttime for PID validation (avoid PID reuse bugs)
    current = _read_proc_starttime(pid)
    if expected_starttime is not None:
        if current is not None and current != expected_starttime:
            return False
    elif current is None:
        return False
    return True


def _squeue_state(job_id: int) -> str:
    """Query squeue; return 'running'|'pending'|'done'|'crashed'|'unknown'."""
    try:
        res = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, timeout=10,
        )
        state = res.stdout.strip().upper()
        if state in ("RUNNING", "COMPLETING"):
            return "running"
        if state in ("PENDING", "CONFIGURING", "REQUEUED"):
            return "pending"
        if state in ("COMPLETED",):
            return "done"
        if state:
            return "crashed"
        return "done"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"


def _run_state(run_dir: str) -> dict:
    """Return state dict with keys: state, pid, slurm_job_id, elapsed_s."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    job_id = _read_slurm_id(run_dir)
    if job_id is not None:
        slurm_state = _squeue_state(job_id)
        if slurm_state == "done" and not os.path.isfile(metrics_path):
            slurm_state = "crashed"
        state = slurm_state
    else:
        pid = _read_pid(run_dir)
        if pid:
            expected_st = _read_int_file(_pid_starttime_path(run_dir))
            if _is_running(pid, expected_st):
                state = "running"
            elif os.path.isfile(metrics_path):
                state = "done"
            else:
                state = "crashed"
        elif os.path.isfile(metrics_path):
            state = "done"
        else:
            state = "crashed"

    elapsed: float | None = None
    sf = os.path.join(run_dir, "start_time")
    if os.path.isfile(sf):
        try:
            elapsed = round(time.time() - float(open(sf).read().strip()), 1)
        except (ValueError, OSError):
            pass

    return {
        "state":        state,
        "pid":          _read_pid(run_dir) if job_id is None else None,
        "slurm_job_id": job_id,
        "elapsed_s":    elapsed,
    }


# ── run lifecycle ─────────────────────────────────────────────────────────────

def _new_run_dir(base_dir: str, tag_suffix: str = "") -> str:
    """Create a timestamped run directory under *base_dir* with a start_time file."""
    os.makedirs(base_dir, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if tag_suffix:
        tag += f"_{tag_suffix}"
    run_dir = os.path.join(base_dir, tag)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "start_time"), "w") as fh:
        fh.write(str(time.time()))
    return run_dir


def _write_run_config(run_dir: str, config: dict) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


def _launch_detached(argv: list[str], run_dir: str, log_file: str | None = None) -> int:
    """Spawn *argv* in a new session; record pid + pid_starttime; return the pid.

    When *log_file* is given, stdout/stderr are redirected into it; otherwise
    the child manages its own output (e.g. the local-run wrapper script).
    """
    kwargs: dict = {"close_fds": True, "start_new_session": True}
    if log_file:
        kwargs["stdout"] = open(log_file, "w")
        kwargs["stderr"] = subprocess.STDOUT
    proc = subprocess.Popen(argv, **kwargs)
    with open(_pid_path(run_dir), "w") as fh:
        fh.write(str(proc.pid))
    st = _read_proc_starttime(proc.pid)
    if st is not None:
        with open(_pid_starttime_path(run_dir), "w") as fh:
            fh.write(str(st))
    return proc.pid


def _submit_sbatch(
    run_dir: str, script: str, *, local_alternative: str = "",
) -> tuple[int | None, dict | None]:
    """Write batch_script.sh, submit it, record the job id.

    Returns ``(job_id, None)`` on success or ``(None, err_response)`` where
    *err_response* is a ready-to-return ``err()`` dict.  *local_alternative*
    names the local-run call suggested when sbatch is unavailable.
    """
    batch_path = os.path.join(run_dir, "batch_script.sh")
    with open(batch_path, "w") as fh:
        fh.write(script)
    os.chmod(batch_path, 0o755)

    try:
        res = subprocess.run(["sbatch", batch_path],
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=30)
    except FileNotFoundError:
        hint = "Slurm is not available."
        if local_alternative:
            hint += f" Use {local_alternative} instead."
        return None, err("sbatch not found.", hint=hint)
    except subprocess.TimeoutExpired:
        return None, err("sbatch timed out.", hint="Check Slurm availability on this host.")

    if res.returncode != 0:
        return None, err(f"sbatch failed: {res.stderr.strip()}",
                         hint="Check partition name, account, and resource limits.")
    m = re.search(r"(\d+)", res.stdout)
    if not m:
        return None, err(f"Could not parse job ID from sbatch output: {res.stdout.strip()}")
    job_id = int(m.group(1))
    with open(_slurm_id_path(run_dir), "w") as fh:
        fh.write(str(job_id))
    return job_id, None


def _cancel_run(run_dir: str) -> dict:
    """Cancel the process owning *run_dir*: scancel for Slurm, else SIGTERM→SIGKILL.

    Returns a plain payload dict (an ``"error"`` key signals failure); callers
    wrap it in ``ok()``/``err()``.
    """
    rs = _run_state(run_dir)
    if rs["state"] not in ("running", "pending"):
        return {"state": rs["state"], "note": "Run is not active."}

    job_id = _read_slurm_id(run_dir)
    if job_id is not None:
        try:
            res = subprocess.run(["scancel", str(job_id)],
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                 text=True, timeout=15)
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            return {"error": f"scancel failed: {exc}"}
        if res.returncode != 0:
            return {"error": f"scancel returned {res.returncode}: {res.stderr.strip()}"}
        return {"cancelled": "slurm", "job_id": job_id}

    pid = _read_pid(run_dir)
    if not pid:
        return {"note": "No PID found; run may have already ended."}
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        return {"note": "Process already gone."}
    except PermissionError as exc:
        return {"error": f"Cannot signal PID {pid}: {exc}"}

    for _ in range(10):
        time.sleep(0.5)
        if not _is_running(pid):
            return {"cancelled": "sigterm", "pid": pid}
    try:
        os.kill(pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    return {"cancelled": "sigkill", "pid": pid}


def _validate_slurm_args(
    partition: str, gpus: int, cpus_per_task: int, mem: str, wall_time: str,
) -> str | None:
    """Return an error message for bad Slurm resource args, or None when valid."""
    if not partition.strip():
        return "partition is required."
    if gpus < 0 or cpus_per_task < 1:
        return "gpus must be >= 0 and cpus_per_task >= 1."
    if not re.fullmatch(r"\d+[KMGTP]?", mem.upper()):
        return "Invalid mem format. Use values like '32G', '64000M'."
    if not re.fullmatch(r"\d+(-\d{1,2}(:\d{2}(:\d{2})?)?|(:\d{2}){0,2})", wall_time):
        return "Invalid wall_time format. Use HH:MM:SS or D-HH:MM:SS."
    return None


# ── active-run symlinks ───────────────────────────────────────────────────────

def _update_active_link(proxy_name: str, run_dir: str) -> None:
    store._atomic_symlink(store._active_link(proxy_name), run_dir)


def _update_opt_active_link(proxy_name: str, run_dir: str) -> None:
    store._atomic_symlink(store._opt_active_link(proxy_name), run_dir)


def _opt_active_run_dir(proxy_name: str) -> str | None:
    link = store._opt_active_link(proxy_name)
    session_dir = store._opt_session_runs_dir(proxy_name)
    if os.path.islink(link):
        target = os.readlink(link)
        if not os.path.isabs(target):
            target = os.path.join(session_dir, target)
        if os.path.isdir(target):
            return target
    return None
