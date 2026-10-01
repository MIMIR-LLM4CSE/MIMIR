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

import build_progress
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

def _build_slurm_id_path(run_dir: str) -> str:
    """The build job of a split build/run submission, when there is one.

    ``slurm_job_id`` stays the *run* job — it is the one whose end is the run's end,
    so every watcher keeps following it. This is the other half of the chain, written
    only when the build was sent somewhere of its own.
    """
    return os.path.join(run_dir, "build_slurm_job_id")

def _build_log_path(run_dir: str) -> str:
    return os.path.join(run_dir, "build.log")

def _phase_path(run_dir: str) -> str:
    return os.path.join(run_dir, "phase.json")


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


def _read_build_slurm_id(run_dir: str) -> int | None:
    return _read_int_file(_build_slurm_id_path(run_dir))


def _read_text_head(path: str, max_bytes: int = _MAX_LOG) -> str:
    """Return up to *max_bytes* from the start of *path* ('' if unreadable)."""
    if os.path.isfile(path):
        try:
            with open(path, errors="replace") as fh:
                return fh.read(max_bytes)
        except OSError:
            pass
    return ""


def _read_log(run_dir: str, max_bytes: int = _MAX_LOG) -> str:
    """Return up to *max_bytes* of the run's stdout.log ('' if unreadable)."""
    return _read_text_head(_log_path(run_dir), max_bytes)


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
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
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


def _build_percent(run_dir: str,
                   max_bytes: int = build_progress.TAIL_BYTES) -> float | None:
    """The build's own most recent percentage, or None if it does not print one.

    Parsed rather than tracked, because the only process that could track it — the
    runner — is blocked inside the build for its whole duration. None is a real
    answer, not a failure: plenty of build commands say nothing about how far along
    they are, and showing 0% for one of those would invent a fact.

    A trailing 100% is kept: the runner's phase already says the build is the current
    step, so it cannot be a finished build standing over later work.
    """
    text = build_progress.read_tail(_build_log_path(run_dir), max_bytes)
    found = build_progress.parse(text, drop_finished=False)
    return found[0] if found else None


def _run_progress(run_dir: str) -> dict:
    """What the run is doing right now: its phase, and a percentage when one exists.

    The phase comes from the sidecar the runner writes at each of its own boundaries
    — it alone knows how many builds and how many cases there are. The percentage is
    read from the build log, and only while building: a measurement phase has no
    percentage to report, and carrying the build's last one into it would leave a bar
    frozen at 98% for the rest of the run.

    Best-effort, like every other read here: an absent or corrupt sidecar means the
    run simply does not say what it is doing.
    """
    try:
        with open(_phase_path(run_dir), encoding="utf-8") as fh:
            phase = json.load(fh)
    except (OSError, ValueError):
        return {}
    if not isinstance(phase, dict):
        return {}
    out: dict = {}
    text = phase.get("text")
    if isinstance(text, str) and text:
        out["phase"] = text
    kind = phase.get("kind")
    if isinstance(kind, str) and kind:
        out["phase_kind"] = kind
    if kind == "build":
        percent = _build_percent(run_dir)
        if percent is not None:
            out["percent"] = percent
    return out


def _build_job_failed(run_dir: str) -> bool:
    """Whether the split-off build job has already reported a failure.

    Read from build.json rather than from the job's exit state: the build job writes
    its verdict there before exiting, and a status a caller can act on ("which proxy,
    and how") only exists in the report.
    """
    from _lib import build as build_mod
    report = build_mod.read_report(run_dir)
    if not report:
        return False
    return report.get("status") not in ("ok", "skipped")


def _run_state(run_dir: str) -> dict:
    """Return state dict with keys: state, pid, slurm_job_id, elapsed_s."""
    metrics_path = os.path.join(run_dir, "metrics.json")
    job_id = _read_slurm_id(run_dir)
    build_job_id = _read_build_slurm_id(run_dir)
    if job_id is not None:
        slurm_state = _squeue_state(job_id)
        if slurm_state == "done" and not os.path.isfile(metrics_path):
            slurm_state = "crashed"
        # A split submission holds the run job PENDING behind the build. When the
        # build has already failed, that pending job is never going to start — Slurm
        # will get around to killing it, but "eventually" is not an answer for
        # something being waited on, and the run is over eitherway.
        if (build_job_id is not None and slurm_state in ("pending", "running")
                and _build_job_failed(run_dir)):
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

    # Merged here rather than left to each caller: the blocking wait, the status op
    # and the detached watcher all read their run through this one function, so a
    # progress fact added here reaches every one of them at once.
    out = {
        "state":        state,
        "pid":          _read_pid(run_dir) if job_id is None else None,
        "slurm_job_id": job_id,
        "elapsed_s":    elapsed,
        **_run_progress(run_dir),
    }
    if build_job_id is not None:
        out["build_slurm_job_id"] = build_job_id
    return out


def background_descriptor(run_dir: str, *, kind: str,
                          status_op: dict, summary_op: dict) -> dict:
    """Data-only handle a client watcher polls to completion.

    The client loop holds no tool names of its own: they arrive here, in the ops the
    watcher is to call. Without such a handle a detached run is a job nothing is
    watching — the model is told to monitor it by hand, ends its turn because there is
    nothing left to do, and is never woken when the job lands.
    """
    return {
        "server":     "proxy",
        "run_dir":    run_dir,
        "job_key":    os.path.basename(run_dir),
        "kind":       kind,
        "status_op":  status_op,
        "summary_op": summary_op,
    }


# ── run lifecycle ─────────────────────────────────────────────────────────────

def _new_run_dir(base_dir: str, tag_suffix: str = "") -> str:
    """Create a timestamped run directory under *base_dir* with a start_time file."""
    os.makedirs(base_dir, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if tag_suffix:
        tag += f"_{tag_suffix}"
    # The tag is second-resolution, so two runs started inside the same second used to
    # be handed the same directory (exist_ok=True) — one run's log, metrics and state
    # overwriting the other's, and both answering to the one job_key a watcher dedups
    # on, which left the second run unwatched and its finish unreported. A fresh
    # directory is claimed instead, by creating it: the check and the claim are one
    # step, so two submissions racing cannot both win the same name.
    run_dir = os.path.join(base_dir, tag)
    for n in range(2, 1000):
        try:
            os.makedirs(run_dir)
            break
        except FileExistsError:
            run_dir = os.path.join(base_dir, f"{tag}_{n}")
    else:
        os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "start_time"), "w") as fh:
        fh.write(str(time.time()))
    return run_dir


def _write_run_config(run_dir: str, config: dict) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


def _update_run_config(run_dir: str, updates: dict) -> None:
    """Merge *updates* into an existing run config, preserving every other key."""
    from _lib.store import _read_json
    cfg = _read_json(os.path.join(run_dir, "config.json"), {}) or {}
    cfg.update(updates)
    _write_run_config(run_dir, cfg)


def _launch_detached(argv: list[str], run_dir: str, log_file: str | None = None) -> int:
    """Spawn *argv* in a new session; record pid + pid_starttime; return the pid.

    When *log_file* is given, stdout/stderr are redirected into it; otherwise
    the child manages its own output (e.g. the local-run wrapper script).
    """
    # stdin is never inherited: it is this server's MCP protocol pipe, and a child that
    # reads it eats the client's JSON-RPC traffic.
    kwargs: dict = {"close_fds": True, "start_new_session": True,
                    "stdin": subprocess.DEVNULL}
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
    script_name: str = "batch_script.sh", id_file: str = "slurm_job_id",
) -> tuple[int | None, dict | None]:
    """Write the batch script, submit it, record the job id.

    Returns ``(job_id, None)`` on success or ``(None, err_response)`` where
    *err_response* is a ready-to-return ``err()`` dict.  *local_alternative*
    names the local-run call suggested when sbatch is unavailable.

    *script_name* and *id_file* exist because a split build/run submission puts two
    jobs in one run directory and neither may overwrite the other's script or id.
    Their defaults are the single-job names, so every existing caller is unchanged.
    """
    batch_path = os.path.join(run_dir, script_name)
    with open(batch_path, "w") as fh:
        fh.write(script)
    os.chmod(batch_path, 0o755)

    try:
        res = subprocess.run(["sbatch", batch_path],
                             stdin=subprocess.DEVNULL,
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
    with open(os.path.join(run_dir, id_file), "w") as fh:
        fh.write(str(job_id))
    return job_id, None


def _scancel(job_id: int) -> str | None:
    """Cancel one Slurm job; returns an error message, or None when it worked."""
    try:
        res = subprocess.run(["scancel", str(job_id)],
                             stdin=subprocess.DEVNULL,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, timeout=15)
    except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
        return f"scancel failed: {exc}"
    if res.returncode != 0:
        return f"scancel returned {res.returncode}: {res.stderr.strip()}"
    return None


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
        # Both halves of a split submission, build first: cancelling only the run job
        # would leave the build compiling for a run that no longer exists, on
        # allocation hours nobody is going to look at.
        build_job_id = _read_build_slurm_id(run_dir)
        for jid in ([build_job_id] if build_job_id is not None else []) + [job_id]:
            error = _scancel(jid)
            if error:
                return {"error": error}
        payload = {"cancelled": "slurm", "job_id": job_id}
        if build_job_id is not None:
            payload["build_job_id"] = build_job_id
        return payload

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


# What a Slurm feature expression or node list is made of: names, the boolean
# operators Slurm reads (&|), grouping, and the counted forms (*+). Nothing here is
# passed to a shell — it lands in a ``#SBATCH`` directive — which is exactly why the
# character that matters most is the newline: one of those in the value and the rest
# of it becomes a directive line of the caller's choosing.
_SLURM_TOKEN_RE = re.compile(r"[A-Za-z0-9_,:.&|()\[\]*+-]+")


def _validate_slurm_token(value: str, field: str) -> str | None:
    """Return why *value* cannot go in a #SBATCH directive, or None when it can."""
    if not value:
        return None
    if not _SLURM_TOKEN_RE.fullmatch(value):
        return (f"Invalid {field}: {value!r}. Use a plain Slurm feature expression "
                f"or node list, e.g. 'a100', 'bigmem&avx512', 'node[01-04]'.")
    return None


# A comment is free text, so the token expression above would reject most real ones
# (spaces, '/', '#'). What still cannot appear is a control character: a newline ends
# the directive line and makes the rest of the value directives of the caller's
# choosing, and sacct/squeue render the field on one line either way. The cap is
# Slurm's own practical limit on the field.
_SLURM_COMMENT_MAX = 512


def _validate_slurm_comment(value: str, field: str = "comment") -> str | None:
    """Return why *value* cannot be a --comment, or None when it can."""
    if not value:
        return None
    if len(value) > _SLURM_COMMENT_MAX:
        return (f"{field} is too long ({len(value)} chars). "
                f"Keep it under {_SLURM_COMMENT_MAX}.")
    bad = [ch for ch in value if ord(ch) < 0x20 or ord(ch) == 0x7F]
    if bad:
        return (f"Invalid {field}: control characters are not allowed "
                f"(found {bad[0]!r}). Keep it to a single line of plain text.")
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
