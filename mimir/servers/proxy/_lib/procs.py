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

from ..._shared import build_progress
from ..._shared.responses import err
from ..._shared.slurm_site import (
    WATCHER_FILE as _WATCHER_FILE,
    record_job_dir as _record_job_dir,
    HOST_FILE as _SLURM_HOST_FILE,
    LOCAL as _SLURM_LOCAL,
    exec_argv as _slurm_exec_argv,
    hosts as _slurm_hosts,
    owned_job_hosts as _owned_job_hosts,
    owned_jobs as _owned_slurm_jobs,
    owns_job as _owns_slurm_job,
    record_host as _record_slurm_host,
)
from ..._shared.state_paths import active_session_id, session_owns

from ..._shared.atomic import replace_symlink
from ..._shared.processes import is_running, proc_starttime
from . import store

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

# The session that started the run: run directories are shared by the whole workspace.
_OWNER_FILE = "owner_session"


def _run_owner(run_dir: str) -> str | None:
    """The session that started *run_dir*, or None when the run recorded none.

    None is "the run recorded nothing", not "nobody": it was started before MIMIR
    recorded owners, or by something that is not MIMIR, and neither is a claim this
    session can make. An *empty* record is a claim — it is what a run started with no
    conversation at all (the CLI) writes, and :func:`session_owns` matches it.
    """
    try:
        with open(os.path.join(run_dir, _OWNER_FILE)) as fh:
            return fh.read().strip()
    except OSError:
        return None


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


def _slurm_host_path(run_dir: str) -> str:
    return os.path.join(run_dir, _SLURM_HOST_FILE)


def _read_slurm_host(run_dir: str) -> str | None:
    """Which scheduler this run was submitted to, or None when it did not say.

    None is not "local": a run directory from before the host was written beside the
    id cannot answer, and reading the silence as this machine would poll the wrong
    scheduler and call a running job finished. :func:`_locate_slurm_job` asks every
    reachable one instead.
    """
    try:
        with open(_slurm_host_path(run_dir)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def _write_slurm_host(run_dir: str, host: str) -> None:
    """Record the scheduler beside the job id. Best effort — a cache, not a binding."""
    try:
        with open(_slurm_host_path(run_dir), "w") as fh:
            fh.write(host)
    except OSError:
        pass


# Where this run's Slurm job is filed as a job *of the workspace*. A run is both: its
# own state machine here, and a job in a queue that every reader of ``slurm_site`` has
# to be able to account for. Kept as a pointer rather than re-derived, because the
# descriptor's directory is named by the moment it was made and nothing else knows it.
_WORKSPACE_JOB_DIR = "workspace_job_dir"


def _record_workspace_job(run_dir: str, job_id: int, host: str) -> None:
    """File this run's Slurm job where the workspace accounts for its jobs.

    Without it the run is a job nothing is watching to every reader of ``slurm_site``:
    the queue listing calls it unwatched, the conclude-phase reminder names its id, and
    the remedy offered — adopt it — puts a second watcher on a run this server is
    already polling, so one run ends and wakes its conversation twice. Best effort: a
    run that could not be filed is still a run, and the submission has already happened.
    """
    try:
        job_dir = _record_job_dir(str(job_id), **{_SLURM_HOST_FILE: host})
        with open(os.path.join(run_dir, _WORKSPACE_JOB_DIR), "w") as fh:
            fh.write(job_dir)
    except OSError:
        pass


def _workspace_job_dir(run_dir: str) -> str:
    try:
        with open(os.path.join(run_dir, _WORKSPACE_JOB_DIR)) as fh:
            return fh.read().strip()
    except OSError:
        return ""


def _locate_slurm_job(run_dir: str, job_id: int) -> tuple[str, str]:
    """``(host, state)`` for *job_id*: what the run dir says, else whoever knows it.

    The sweep is what keeps a run trackable without being owned by a scheduler, and
    the host it finds is written down so the next poll — and these are polls, on a
    timer, for the life of the run — is one call again.
    """
    hint = _read_slurm_host(run_dir)
    if hint is not None:
        state = _squeue_state(job_id, host=hint)
        if state != "unknown":
            return hint, state
    for host in _slurm_hosts():
        if host == hint:
            continue
        state = _squeue_state(job_id, host=host)
        if state != "unknown":
            _write_slurm_host(run_dir, host)
            return host, state
    return hint or _SLURM_LOCAL, "unknown"


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

def _slurm_run(argv: list[str], timeout: int, stdin: str | None = None, *,
               host: str = _SLURM_LOCAL):
    """Run one Slurm command on *host*'s scheduler, and hand back the completed call.

    Through the shared seam (``_shared/slurm_site``) rather than straight to
    ``subprocess``, because the scheduler is not always on the host MIMIR runs on: on a
    login node whose schedulers live elsewhere, a bare ``sbatch`` is "not found" and
    these submissions simply could not be made. One owner for the question, so the three
    servers that submit cannot disagree about which hosts are reachable — and locally
    the argv is still exec'd as it stands, with no shell, which is what these commands
    are built as arguments for.

    *stdin* feeds the command its own input, which is how a batch script reaches
    ``sbatch`` without any file having to exist on the host that runs it.
    """
    return subprocess.run(
        _slurm_exec_argv(argv, host=host, piped=stdin is not None),
        input=stdin,
        stdin=None if stdin is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
    )


def _squeue_state(job_id: int, *, host: str = _SLURM_LOCAL) -> str:
    """Query squeue on *host*; return 'running'|'pending'|'done'|'crashed'|'unknown'."""
    try:
        res = _slurm_run(["squeue", "-j", str(job_id), "-h", "-o", "%T"], 10,
                         host=host)
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
    from . import build as build_mod
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
        # Wherever it is: a run's scheduler is read from its own directory, so the
        # ~15 callers of this keep asking the same question and get the right answer
        # for a job on any of them.
        _, slurm_state = _locate_slurm_job(run_dir, job_id)
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
            if is_running(pid, expected_st):
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
    if state == "running":
        tail = build_progress.last_lines(_log_path(run_dir))
        if tail:
            out["log_tail"] = tail
    return out


def background_descriptor(run_dir: str, *, kind: str,
                          status_op: dict, summary_op: dict) -> dict:
    """Data-only handle a client watcher polls to completion.

    The client loop holds no tool names of its own: they arrive here, in the ops the
    watcher is to call. Without such a handle a detached run is a job nothing is
    watching — the model is told to monitor it by hand, ends its turn because there is
    nothing left to do, and is never woken when the job lands.
    """
    descriptor = {
        "server":     "proxy",
        "run_dir":    run_dir,
        "job_key":    os.path.basename(run_dir),
        "kind":       kind,
        "status_op":  status_op,
        "summary_op": summary_op,
    }
    # Recorded beside the job id when this run is a Slurm job, so a window that closes
    # and comes back re-arms *this* watcher rather than a stock poll beside it: two
    # watchers on one run are two wakes for one result.
    job_dir = _workspace_job_dir(run_dir)
    if job_dir:
        try:
            with open(os.path.join(job_dir, _WATCHER_FILE), "w") as fh:
                json.dump(descriptor, fh)
        except OSError:
            pass
    return descriptor


# ── run lifecycle ─────────────────────────────────────────────────────────────

def _new_run_dir(base_dir: str, tag_suffix: str = "") -> str:
    """Create a timestamped run directory under *base_dir* with a start_time file."""
    os.makedirs(base_dir, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    if tag_suffix:
        tag += f"_{tag_suffix}"
    # The tag is second-resolution, so two runs started inside the same second would
    # share a directory and a job_key. A fresh directory is claimed by creating it: the
    # check and the claim are one step, so two submissions racing cannot both win.
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
    with open(os.path.join(run_dir, _OWNER_FILE), "w") as fh:
        fh.write(active_session_id())
    return run_dir


def _write_run_config(run_dir: str, config: dict) -> None:
    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(config, fh, indent=2)


def _update_run_config(run_dir: str, updates: dict) -> None:
    """Merge *updates* into an existing run config, preserving every other key."""
    from .store import _read_json
    cfg = _read_json(os.path.join(run_dir, "config.json"), {}) or {}
    cfg.update(updates)
    _write_run_config(run_dir, cfg)


def _launch_detached(argv: list[str], run_dir: str, log_file: str | None = None,
                     env: dict | None = None) -> int:
    """Spawn *argv* in a new session; record pid + pid_starttime; return the pid.

    When *log_file* is given, stdout/stderr are redirected into it; otherwise
    the child manages its own output (e.g. the local-run wrapper script).
    """
    # stdin is never inherited: it is this server's MCP protocol pipe, and a child that
    # reads it eats the client's JSON-RPC traffic.
    kwargs: dict = {"close_fds": True, "start_new_session": True,
                    "stdin": subprocess.DEVNULL}
    if env is not None:
        kwargs["env"] = env
    if log_file:
        kwargs["stdout"] = open(log_file, "w")
        kwargs["stderr"] = subprocess.STDOUT
    proc = subprocess.Popen(argv, **kwargs)
    with open(_pid_path(run_dir), "w") as fh:
        fh.write(str(proc.pid))
    st = proc_starttime(proc.pid)
    if st is not None:
        with open(_pid_starttime_path(run_dir), "w") as fh:
            fh.write(str(st))
    return proc.pid


def _submit_sbatch(
    run_dir: str, script: str, *, local_alternative: str = "",
    script_name: str = "batch_script.sh", id_file: str = "slurm_job_id",
    host: str = _SLURM_LOCAL,
) -> tuple[int | None, dict | None]:
    """Write the batch script, submit it, record the job id.

    Returns ``(job_id, None)`` on success or ``(None, err_response)`` where
    *err_response* is a ready-to-return ``err()`` dict.  *local_alternative*
    names the local-run call suggested when sbatch is unavailable.

    *script_name* and *id_file* exist because a split build/run submission puts two
    jobs in one run directory and neither may overwrite the other's script or id.
    Their defaults are the single-job names, so every existing caller is unchanged.

    *host* is the scheduler to submit to, written beside the id so every later poll of
    this run goes straight to it.
    """
    batch_path = os.path.join(run_dir, script_name)
    with open(batch_path, "w") as fh:
        fh.write(script)
    os.chmod(batch_path, 0o755)

    try:
        # The script goes on sbatch's stdin, not as a path: it is kept on disk for a
        # reader asking what was submitted, but nothing of ours then has to exist on
        # the host running the scheduler.
        res = _slurm_run(["sbatch"], 30, stdin=script, host=host)
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
    _write_slurm_host(run_dir, host)
    if host:
        # A submission that succeeded is proof this host reaches that scheduler.
        _record_slurm_host(host)
    # Only the run job carries the pointer: a split build/run submission files both, so
    # neither is reported as unaccounted for, but the run is the one whose end is the
    # run's end and therefore the one a watcher is re-armed on.
    if id_file == "slurm_job_id":
        _record_workspace_job(run_dir, job_id, host)
    else:
        try:
            _record_job_dir(str(job_id), **{_SLURM_HOST_FILE: host})
        except OSError:
            pass
    return job_id, None


def _scancel(job_id: int, *, host: str = _SLURM_LOCAL,
             owned: dict[str, set[str]] | None = None) -> str | None:
    """Cancel one Slurm job on *host*; returns an error message, or None when it worked.

    Refused unless this session or one of its sub-agents submitted it there. *owned* is
    the ownership map when the caller has already walked for one (see
    :func:`slurm_site.owned_jobs`); the check is made here either way, because this is
    the only thing every cancellation goes through.
    """
    if not _owns_slurm_job(str(job_id), host, owned):
        return (f"Slurm job {job_id} was not submitted by this conversation or its "
                "sub-agents; it is not this conversation's to cancel.")
    try:
        res = _slurm_run(["scancel", str(job_id)], 15, host=host)
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
        # Walked for once and handed to every check below: each of them asks about a
        # job of the same session, and a walk apiece re-read every descriptor in the
        # tree to answer the same question.
        owned = _owned_slurm_jobs()
        # The scheduler it was submitted to, from this session's own record: a sweep
        # could land on the same id elsewhere, which is somebody else's job.
        hosts = _owned_job_hosts(str(job_id), owned)
        if not hosts:
            return {"error": f"Slurm job {job_id} of {run_dir} was not submitted by "
                              "this conversation or its sub-agents; it is not this "
                              "conversation's to cancel."}
        if len(hosts) > 1:
            return {"error": f"Slurm job {job_id} of {run_dir} was submitted here on "
                             "several schedulers (" + ", ".join(sorted(hosts))
                             + "); cancel it with slurm_cancel(scheduler=...)."}
        host = next(iter(hosts))
        jids = ([build_job_id] if build_job_id is not None else []) + [job_id]
        # All or nothing: never leave half of a split submission cancelled.
        foreign = [jid for jid in jids if not _owns_slurm_job(str(jid), host, owned)]
        if foreign:
            return {"error": f"Slurm job {foreign[0]} was not submitted by this "
                              "conversation or its sub-agents; nothing was cancelled."}
        for jid in jids:
            error = _scancel(jid, host=host, owned=owned)
            if error:
                return {"error": error}
        payload = {"cancelled": "slurm", "job_id": job_id}
        if build_job_id is not None:
            payload["build_job_id"] = build_job_id
        return payload

    owner = _run_owner(run_dir)
    if owner is None:
        return {"error": f"{run_dir} records no session that started it, so there is "
                          "nothing here that says it is this conversation's to stop. "
                          "Ask the user to stop it where it was started."}
    if not session_owns(owner):
        return {"error": f"{run_dir} was started by another conversation; it is not "
                          "this conversation's to stop."}
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
        if not is_running(pid):
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
    replace_symlink(store._active_link(proxy_name), run_dir)


def _update_opt_active_link(proxy_name: str, run_dir: str) -> None:
    replace_symlink(store._opt_active_link(proxy_name), run_dir)


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
