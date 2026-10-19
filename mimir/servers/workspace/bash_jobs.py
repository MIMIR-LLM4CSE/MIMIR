"""Shell jobs: every ``bash_run`` spawned into its own session, output to a log on disk.

A job directory (``<STATE_DIR>/sessions/<sid>/jobs/``) records pid, start time and the
exit code a trap writes. Blocking runs use an ``ephemeral`` directory deleted on return;
detached ones keep it. Validation is ``bash_run``'s; ``&`` stays refused.
"""

import json
import os
import re
import shlex
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from typing import Any

from ...shared import build_progress
from ...shared.jobs import EXIT_CODE, META, shell_descriptor
from ...shared.responses import err, ok
from ...shared.processes import is_running, proc_starttime

def jobs_root() -> str:
    """This session's job root, resolved per call (the session-less fallback can move)."""
    from ...shared.state_paths import session_state_dir
    return os.path.join(session_state_dir(), "jobs")

# A job key is used as a directory name and echoed back by the client watcher, so it is
# generated here rather than accepted from a caller, and validated on the way back in.
_JOB_KEY_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}$")

# How long a job gets to exit on SIGTERM before SIGKILL. A build interrupted mid-link
# leaves a mess either way; the wait is for the common case of a process that handles
# the signal and stops cleanly.
_TERM_GRACE_S = 5.0

# Popen handles for children we deliberately stopped waiting on (a diverted run).
# Held only so the garbage collector does not warn about a still-running child; the
# zombie one of these leaves behind is the case ``is_running`` already reads.
ABANDONED: list[subprocess.Popen] = []

# Set once per server process, so the sweep below costs one listdir per process
# rather than one per command.
_swept = False


def _job_dir(job_key: str, run_dir: str = "") -> str:
    """This job's directory: *run_dir* when given (a sub-agent's run), else derived from the session."""
    return run_dir or os.path.join(jobs_root(), job_key)


def _path(job_key: str, name: str, run_dir: str = "") -> str:
    return os.path.join(_job_dir(job_key, run_dir), name)


def job_dir(job_key: str) -> str:
    """This job's directory. What a descriptor carries as its ``run_dir``."""
    return _job_dir(job_key)


def log_path(job_key: str, run_dir: str = "") -> str:
    return _path(job_key, "run.log", run_dir)


def err_path(job_key: str, run_dir: str = "") -> str:
    """Where a split-stream job's stderr goes; absent for a merged one."""
    return _path(job_key, "run.err", run_dir)


def tee_path(job_key: str, run_dir: str = "") -> str:
    """Where a piped build's output is copied, so its count stays readable.

    Written only when :func:`launch` spliced a ``tee`` in; see
    ``build_progress.tee_command``. Never what the caller is shown — that stays the
    pipeline's own output in ``run.log``, which is what it asked for.
    """
    return _path(job_key, "progress.log", run_dir)


def progress_path(job_key: str, run_dir: str = "") -> str:
    """The log a progress read should look at: the tee's copy when there is one.

    A job whose output was not piped has no copy, and its run.log already holds
    everything the build printed.
    """
    teed = tee_path(job_key, run_dir)
    return teed if os.path.exists(teed) else log_path(job_key, run_dir)


def _read(job_key: str, name: str, default: str = "", run_dir: str = "") -> str:
    try:
        with open(_path(job_key, name, run_dir)) as fh:
            return fh.read().strip()
    except OSError:
        return default


def _new_job_key() -> str:
    """A sortable, collision-free key: UTC stamp plus four random hex."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    while True:
        key = f"{stamp}-{os.urandom(2).hex()}"
        if not os.path.exists(_job_dir(key)):
            return key


def valid_key(job_key: str) -> bool:
    return bool(_JOB_KEY_RE.match(job_key or ""))


def _wrap(script: str, rc_path: str) -> str:
    """*script* under traps that record how it ended in *rc_path*.

    EXIT records ``$?`` (an epilogue misses ``exit 3``); the signal traps record 128+signum,
    since a signalled bash runs EXIT with ``$?`` = 0. Built here, so it is outside the
    validator's no-substitution rule.
    """
    rc = shlex.quote(rc_path)
    return (
        f"__mimir_rc() {{ printf %s \"$1\" > {rc}; }}\n"
        "trap '__mimir_rc $?' EXIT\n"
        "trap '__mimir_rc 143; exit 143' TERM\n"
        "trap '__mimir_rc 130; exit 130' INT\n"
        "trap '__mimir_rc 129; exit 129' HUP\n"
        f"{script}\n"
    )


def _write_meta(job_key: str, **fields) -> None:
    """Write one job's ``meta.json`` (the one writer: launch, adoption, promotion)."""
    meta = {"job_key": job_key, "started_at": time.time(), **fields}
    with open(_path(job_key, META), "w") as fh:
        json.dump(meta, fh, indent=2)


def launch(command: str, cwd: str, env: dict, preamble: str = "",
           output_targets: list[str] | None = None, *,
           split_stderr: bool = False, ephemeral: bool = False, scheduler: str = "") -> dict:
    """Spawn *command* in its own session; return its key, pid, log path and ``proc`` handle.

    *output_targets*: the files it redirects to; *split_stderr*: stderr apart (blocking run);
    *ephemeral*: a scratch buffer; *scheduler*: the host whose Slurm it runs through ssh.
    ``proc`` is never written to ``meta.json``.
    """
    _sweep_once()
    job_key = _new_job_key()
    os.makedirs(_job_dir(job_key), exist_ok=True)

    rc_path = _path(job_key, EXIT_CODE)
    # A build piped into a filter gets a second copy of its output, taken before the
    # filter, or nothing here would see its progress until it had finished. The
    # command the caller wrote still runs, and still decides the exit status.
    effective = build_progress.tee_command(command, tee_path(job_key)) or command
    script = _wrap(preamble + effective, rc_path)

    log = log_path(job_key)
    err_fh = open(err_path(job_key), "w") if split_stderr else None
    try:
        with open(log, "w") as log_fh:
            proc = subprocess.Popen(
                ["bash", "--noprofile", "--norc", "-c", script],
                cwd=cwd,
                env=env,
                # Never the server's own stdin: that is the MCP protocol pipe, and a job
                # that reads it (``ssh`` without ``-n``, ``cat``) eats the client's
                # JSON-RPC traffic — swallowing requests whole, or desynchronising the
                # framing, while the server still looks healthy.
                stdin=subprocess.DEVNULL,
                stdout=log_fh,
                stderr=err_fh if err_fh is not None else subprocess.STDOUT,
                close_fds=True,
                # Its own session: the job must outlive the call that started it, and a
                # stop must be able to signal the whole tree rather than just the shell.
                start_new_session=True,
            )
    finally:
        if err_fh is not None:
            err_fh.close()

    extra: dict[str, Any] = {"split_stderr": bool(split_stderr), "ephemeral": bool(ephemeral)}
    if scheduler:
        extra["scheduler"] = scheduler
    if effective != command:
        # What bash was actually handed, kept next to what was asked for: a later
        # reader must be able to see that the two differ and how.
        extra["effective_command"] = effective
    _write_meta(job_key, command=command, cwd=cwd, pid=proc.pid,
                pid_starttime=proc_starttime(proc.pid),
                output_targets=list(output_targets or ()), **extra)
    return {"job_key": job_key, "pid": proc.pid, "log": log,
            "job_dir": _job_dir(job_key), "proc": proc}


def adopt(pid: int, command: str, cwd: str, log: str = "") -> dict | None:
    """Record a running process this server did not start, so it is watched; None if not running.

    No exit-code trap can be added to it, so its end is ``unknown``. *log* is recorded as
    its output target.
    """
    starttime = proc_starttime(pid)
    if starttime is None or not is_running(pid, starttime):
        return None
    _sweep_once()
    job_key = _new_job_key()
    os.makedirs(_job_dir(job_key), exist_ok=True)
    # ``adopted`` says why there will be no exit code, for every reader of this
    # directory and for the note ``state`` gives once the process is gone. The flags a
    # launch sets — ``split_stderr``, ``ephemeral`` — are absent rather than false:
    # every reader asks with ``get``, and an adoption knows neither.
    _write_meta(job_key, command=command, cwd=cwd, pid=int(pid),
                pid_starttime=starttime, output_targets=[log] if log else [],
                adopted=True)
    return {"job_key": job_key, "pid": int(pid), "log": log,
            "job_dir": _job_dir(job_key)}


# Share of a finished run's budget given to its beginning; the end gets the rest.
_ENDS_HEAD_SHARE = 0.25


def _clip_ends(fh, size: int, max_bytes: int) -> str:
    """The beginning and the end of an open binary file, with the middle elided.

    Cut on line boundaries, so neither half starts or stops mid-line, and the marker
    says how much was left out.
    """
    head_bytes = int(max_bytes * _ENDS_HEAD_SHARE)
    tail_bytes = max_bytes - head_bytes
    head = fh.read(head_bytes)
    head = head[:head.rfind(b"\n") + 1] or head
    fh.seek(size - tail_bytes)
    tail = fh.read()
    newline = tail.find(b"\n")
    if 0 <= newline < len(tail) - 1:
        tail = tail[newline + 1:]
    omitted = size - len(head) - len(tail)
    marker = f"\n[... {omitted} bytes of output omitted ...]\n\n".encode()
    return (head + marker + tail).decode("utf-8", errors="replace")


def _clip(path: str, max_bytes: int, keep: str) -> tuple[str, bool]:
    """*path*'s content, clipped to *max_bytes* from whichever end matters."""
    try:
        size = os.path.getsize(path)
        if size > max_bytes and keep == "ends":
            with open(path, "rb") as fh:
                return _clip_ends(fh, size, max_bytes), True
        with open(path, errors="replace") as fh:
            if size > max_bytes and keep == "tail":
                fh.seek(size - max_bytes)
                return fh.read(), True
            return fh.read(max_bytes), size > max_bytes
    except OSError:
        return "", False


def streams(job_key: str, max_bytes: int, keep: str = "ends") -> tuple[str, str, bool]:
    """The job's stdout and stderr so far, and whether either was clipped.

    *keep* is "ends" for a run that finished — the first error explains the rest, and
    the last lines say how it ended — and "tail" for one still going, where the
    frontier is the whole point.
    """
    out, out_cut = _clip(log_path(job_key), max_bytes, keep)
    errs, err_cut = _clip(err_path(job_key), max_bytes, keep)
    return out, errs, out_cut or err_cut


def promote(job_key: str, reason: str) -> None:
    """Turn a blocking run's scratch directory into a handle somebody now holds."""
    meta = _meta(job_key)
    if not meta:
        return
    meta.update({"ephemeral": False, "diverted": True, "divert_reason": reason,
                 "diverted_at": time.time()})
    try:
        # Its own ``started_at`` and ``job_key`` survive: they are already in *meta*,
        # and a later keyword wins over the writer's default.
        _write_meta(job_key, **{k: v for k, v in meta.items() if k != "job_key"})
    except OSError:
        pass


def discard(job_key: str) -> None:
    """Delete a finished blocking run's directory — nobody was given its key."""
    shutil.rmtree(_job_dir(job_key), ignore_errors=True)


def _sweep_once() -> None:
    """Remove cold ephemeral job dirs left by a dead server, once per process (never detached ones)."""
    global _swept
    if _swept:
        return
    _swept = True
    try:
        keys = [k for k in os.listdir(jobs_root()) if valid_key(k)]
    except OSError:
        return
    for key in keys:
        meta = _meta(key)
        if not meta.get("ephemeral"):
            continue
        if time.time() - float(meta.get("started_at") or 0) < 3600:
            continue
        if is_running(int(meta.get("pid") or 0), meta.get("pid_starttime")):
            continue
        discard(key)


def _meta(job_key: str, run_dir: str = "") -> dict:
    try:
        with open(_path(job_key, META, run_dir)) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def state(job_key: str, run_dir: str = "") -> dict:
    """The job's state: running, done (exit 0), crashed (non-zero), or unknown (no own status).

    Unknown covers a missing exit code and a recorded signal (128+signum).
    """
    meta = _meta(job_key, run_dir)
    if not meta:
        return {"state": "unknown", "job_key": job_key, "error": "No such job."}

    rc_raw = _read(job_key, EXIT_CODE, run_dir=run_dir)
    alive = is_running(int(meta.get("pid") or 0), meta.get("pid_starttime"))

    payload = {
        "job_key": job_key,
        "command": meta.get("command", ""),
        "pid": meta.get("pid"),
        "log": log_path(job_key, run_dir),
        "elapsed_s": round(time.time() - float(meta.get("started_at") or 0), 1),
    }
    if meta.get("scheduler"):
        # The process here is the ssh session; the work is that host's Slurm job.
        payload["scheduler"] = meta["scheduler"]
    if alive and not rc_raw:
        payload["state"] = "running"
        # The build's own count, when its output carries one — the client watcher
        # passes phase/percent on without knowing what produced them.
        progress = build_progress.from_file(progress_path(job_key, run_dir))
        if progress:
            payload["percent"], payload["phase"] = progress
        tail = build_progress.last_lines(progress_path(job_key, run_dir))
        if tail:
            payload["log_tail"] = tail
        return payload
    if rc_raw.isdigit() or (rc_raw.startswith("-") and rc_raw[1:].isdigit()):
        rc = int(rc_raw)
        payload["returncode"] = rc
        if rc > 128:
            payload["state"] = "unknown"
            payload["signal"] = rc - 128
            payload["note"] = (f"Terminated by signal {rc - 128} — it was stopped or "
                               "killed rather than returning a status of its own. "
                               "Whatever it had written to disk is still there.")
            return payload
        payload["state"] = "done" if rc == 0 else "crashed"
        return payload
    payload["state"] = "unknown"
    payload["note"] = (
        # An adopted process never had the traps that write a status, so "it was
        # killed" would be a reading of the absence that the absence does not
        # support — it is just as likely to have finished cleanly.
        "The process has ended. Nothing recorded a status for it: it was adopted "
        "rather than started here, so it never had the exit-code trap. Its own "
        "output is what says how it went."
        if meta.get("adopted") else
        "The process is gone and wrote no exit code — it was killed from outside "
        "(SIGKILL, OOM, host restart) rather than returning.")
    return payload


def output(job_key: str, max_bytes: int, running_max_bytes: int | None = None,
           run_dir: str = "") -> dict:
    """The tail of the job's log, with its state; ``note`` names where output went if elsewhere.

    *running_max_bytes* caps the tail mid-run (a progress check); a finished job gets the
    full *max_bytes*.
    """
    payload = state(job_key, run_dir)
    if payload.get("error"):
        return payload
    if running_max_bytes is not None and payload.get("state") == "running":
        max_bytes = min(max_bytes, running_max_bytes)
    meta = _meta(job_key, run_dir)
    log = log_path(job_key, run_dir)
    if not os.path.exists(log):
        # An adopted job has no log of its own: nothing of ours opened the process's
        # output, so there is no file here to read. Said plainly, because the generic
        # answer below — "Log unreadable" — reads as a job whose output was lost, and
        # the output is not lost: it is wherever the process is writing it.
        targets = meta.get("output_targets") or []
        payload["output"] = ""
        where = (f"its output goes to {', '.join(targets)}" if targets else
                 "where its output goes is known only to the process itself")
        note = ("This job has no log of its own — it was adopted rather than started "
                f"here, and {where}.")
        payload["note"] = f"{payload['note']} {note}" if payload.get("note") else note
        return payload
    try:
        size = os.path.getsize(log)
        with open(log, errors="replace") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                payload["truncated"] = True
                payload["log_bytes"] = size
            payload["output"] = fh.read()
        if meta.get("split_stderr"):
            # A diverted run keeps its streams apart, so the log alone is half the
            # story. Appended under a marker rather than merged blind: the reader
            # must be able to tell which stream said what.
            errs, _ = _clip(err_path(job_key, run_dir), max_bytes, "tail")
            if errs:
                payload["output"] = f"{payload['output']}\n--- stderr ---\n{errs}"
        targets = meta.get("output_targets") or []
        if size == 0 and targets:
            # A state note (a signal death) may already hold the key and explains
            # something else, so this is appended rather than substituted.
            empty = ("This job's log is empty because the command redirected its own "
                     f"output to {', '.join(targets)} — read that for what it "
                     "produced.")
            payload["note"] = f"{payload['note']} {empty}" if payload.get("note") else empty
    except OSError as exc:
        payload["output"] = ""
        payload["note"] = f"Log unreadable: {exc}"
    return payload


def listing() -> list[dict]:
    """Every known job, newest first — keys sort chronologically by construction."""
    try:
        keys = sorted((k for k in os.listdir(jobs_root()) if valid_key(k)), reverse=True)
    except OSError:
        return []
    return [state(k) for k in keys]


def stop(job_key: str) -> dict:
    """SIGTERM the job's process group, escalating to SIGKILL.

    The group, not the pid: the shell forks, and signalling only the shell leaves the
    compiler it launched running — the same reason ``proc_run`` kills groups on timeout.
    """
    payload = state(job_key)
    if payload.get("error"):
        return payload
    if payload["state"] != "running":
        payload["note"] = "Job is not running; nothing to stop."
        return payload

    pid = int(payload["pid"])
    try:
        pgid = os.getpgid(pid)
    except (ProcessLookupError, PermissionError):
        payload["note"] = "Process already gone."
        return payload

    for sig, deadline in ((signal.SIGTERM, _TERM_GRACE_S), (signal.SIGKILL, 1.0)):
        try:
            os.killpg(pgid, sig)
        except (ProcessLookupError, PermissionError):
            break
        waited = 0.0
        while waited < deadline:
            time.sleep(0.2)
            waited += 0.2
            if state(job_key)["state"] != "running":
                out = state(job_key)
                out["stopped"] = "sigterm" if sig == signal.SIGTERM else "sigkill"
                return out
    out = state(job_key)
    out["stopped"] = "sigkill"
    return out


def resolve_run_dir(run_dir: str, job_key: str) -> tuple[str, dict | None]:
    """``(run_dir, None)`` for an explicit job directory inside this session's own state dir
    (its sub-agents' runs), else ``("", error)``.
    """
    from ...shared.state_paths import session_state_dir
    resolved = os.path.realpath(os.path.expanduser(run_dir))
    root = os.path.realpath(session_state_dir())
    if os.path.commonpath([resolved, root]) != root:
        return "", err("That run_dir is outside this session.",
                       hint="A run_dir names a job of this session or of one of its "
                            "sub-agents; leave it out to read this session's own job.")
    if os.path.basename(os.path.dirname(resolved)) != "jobs":
        return "", err("That run_dir is not a job directory.",
                       hint="A job directory sits under a 'jobs/' directory and is "
                            "named like the handle bash_run returned.")
    if not valid_key(os.path.basename(resolved)):
        return "", err("That run_dir is not named like a job.",
                       hint="Its last segment is the handle bash_run returned.")
    if job_key and os.path.basename(resolved) != job_key:
        return "", err("That run_dir names a different job than job_key does.",
                       hint="Pass the pair as the descriptor handed them over, or "
                            "leave run_dir out.")
    return resolved, None


def with_live_descriptors(payload: dict, jobs: list[dict], run_dir: str = "") -> dict:
    """Attach a watcher handle (``background_jobs``, plural) for every job still running.

    Asking about a run is what lets a client that lost its watchers put them back.
    """
    live = [shell_descriptor(j["job_key"],
                            run_dir or job_dir(j["job_key"]))
            for j in jobs
            if isinstance(j, dict) and j.get("state") == "running" and j.get("job_key")]
    if live:
        payload["background_jobs"] = live
    return payload


def answer(payload: dict) -> dict:
    """Wrap a job-layer answer in ``ok()``, unless it carries ``error`` (a key naming no job)."""
    reason = payload.pop("error", "")
    if reason:
        return err(reason, hint="Use bash_job(op='list') to see this session's jobs.",
                   **payload)
    return ok(payload)
