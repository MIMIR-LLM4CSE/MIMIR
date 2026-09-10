"""Detached shell jobs behind ``bash_run(background=True)``.

``bash_run`` blocks its turn and is capped at 300s, which is the right shape for a
search, a test suite or a short build. It is the wrong shape for the work that
actually motivated this: a dependency tree that takes two hours to compile. The tool's
own docstring already told the model to "submit the run as a background job" past the
cap — a handle that did not exist outside Slurm, so on a host with no controller the
advice named a capability nothing could provide, and the model improvised chunked
``curl -C -`` resumes instead.

What a job is: the same validated command, spawned into its own session with its
output redirected to a log, plus a small directory recording pid, start time and — the
part that makes a terminal state knowable — the exit code the shell wrote on its way
out. Nothing here relaxes ``bash_run``'s validation: backgrounding is a parameter on a
command that has already passed the same path checks and denylists, which is why ``&``
stays refused. Detaching is the server's job, not a shell operator the caller supplies.

The job directory lives under a fixed cache root (``trusted_read_roots``) so the log is
readable with the ordinary file tools while the run is still going.

Every ``bash_run`` comes through here now, detached or not: a blocking call launches a
job and waits on it. That is what lets a run be abandoned without being lost, since its
output is on disk rather than in a pipe nobody is left to drain. The two uses want
different things from a job directory, and ``ephemeral`` is which — a transient output
buffer for a blocking run, deleted the moment it returns, or a durable handle for a
detached one, which only ``sweep`` may ever touch and only while still ephemeral.
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

JOBS_ROOT = os.path.expanduser("~/.cache/mimir_bash/jobs")

# A job key is used as a directory name and echoed back by the client watcher, so it is
# generated here rather than accepted from a caller, and validated on the way back in.
_JOB_KEY_RE = re.compile(r"^[0-9]{8}T[0-9]{6}Z-[0-9a-f]{4}$")

# How long a job gets to exit on SIGTERM before SIGKILL. A build interrupted mid-link
# leaves a mess either way; the wait is for the common case of a process that handles
# the signal and stops cleanly.
_TERM_GRACE_S = 5.0

# Popen handles for children we deliberately stopped waiting on (a diverted run).
# Held only so the garbage collector does not warn about a still-running child; the
# zombie one of these leaves behind is the case ``_is_running`` already reads.
_ABANDONED: list[subprocess.Popen] = []

# Set once per server process, so the sweep below costs one listdir per process
# rather than one per command.
_swept = False


def _job_dir(job_key: str) -> str:
    return os.path.join(JOBS_ROOT, job_key)


def _path(job_key: str, name: str) -> str:
    return os.path.join(_job_dir(job_key), name)


def log_path(job_key: str) -> str:
    return _path(job_key, "run.log")


def err_path(job_key: str) -> str:
    """Where a split-stream job's stderr goes; absent for a merged one."""
    return _path(job_key, "run.err")


def _read(job_key: str, name: str, default: str = "") -> str:
    try:
        with open(_path(job_key, name)) as fh:
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


def _proc_starttime(pid: int) -> int | None:
    """Field 22 of /proc/<pid>/stat — pins a pid to *this* process.

    A pid alone is ambiguous once it has been recycled, and a job directory outlives
    the process it describes. Recording the start time makes "is it still running"
    answerable without mistaking a stranger's pid for the job's.
    """
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        # The comm field is parenthesized and may contain spaces: split after its close.
        return int(data[data.rfind(")") + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _is_running(pid: int, expected_starttime: int | None) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    if expected_starttime is not None:
        actual = _proc_starttime(pid)
        if actual is not None and actual != expected_starttime:
            return False  # pid recycled — a different process wears it now
    # Nothing reaps a detached child, so its table entry lingers as a zombie after it
    # exits. Left as "running", a finished job would never reach a terminal state.
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        return data[data.rfind(")") + 2:].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _wrap(script: str, rc_path: str) -> str:
    """The command, under a trap that records how it ended.

    The exit code is what separates 'done' from 'crashed', and a detached child has
    nobody to wait() on it — so the shell writes its own status before leaving. A trap
    rather than an epilogue after the command: an epilogue is never reached by a script
    ending in `exit 3`, which reported a plain non-zero run as 'unknown'. The EXIT trap
    fires on that path too, and `$?` inside it is the status the shell is leaving with.

    Built here, never supplied by the caller, which is why its substitution is not
    subject to the command validator's no-substitution rule (cf. the module preamble).
    The signal traps are not redundant with EXIT. Signalled through its process group,
    bash still runs its EXIT trap, and `$?` there was 0 — so a stopped run recorded a
    clean exit and read as 'done'. A killed build reported as a successful one is the
    worst answer this module could give, so a signal records 128+signum and the state
    reader treats that range as "did not return a status of its own".

    A command that installs its own EXIT trap replaces this one and will read as
    'unknown' — correct, since nothing then records what it returned.
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


def launch(command: str, cwd: str, env: dict, preamble: str = "",
           output_targets: list[str] | None = None, *,
           split_stderr: bool = False, ephemeral: bool = False) -> dict:
    """Spawn *command* in its own session; return its key, pid, log path and handle.

    *output_targets* are the files the command redirects its own output to, taken from
    the parse the validator already performed. Recorded so an empty log can be told
    apart from output that went somewhere else, without guessing after the fact.

    *split_stderr* gives stderr its own file. A detached job merges the two, which is
    what a tailed log wants; a blocking run must keep them apart, because the payload
    it builds classifies the failure from stderr alone and the UI gives it its own
    pane. *ephemeral* marks a job dir as a blocking run's scratch buffer rather than a
    handle somebody was handed — see ``discard`` and ``sweep``.

    ``proc`` is in the returned dict but never in ``meta.json``: a Popen is a handle
    valid inside this process only, not state a later reader could act on. A blocking
    caller polls it for the authoritative exit status; a detaching one drops it.
    """
    _sweep_once()
    job_key = _new_job_key()
    os.makedirs(_job_dir(job_key), exist_ok=True)

    rc_path = _path(job_key, "exit_code")
    script = _wrap(preamble + command, rc_path)

    log = log_path(job_key)
    err_fh = open(err_path(job_key), "w") if split_stderr else None
    try:
        with open(log, "w") as log_fh:
            proc = subprocess.Popen(
                ["bash", "--noprofile", "--norc", "-c", script],
                cwd=cwd,
                env=env,
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

    meta = {
        "job_key": job_key,
        "command": command,
        "cwd": cwd,
        "pid": proc.pid,
        "pid_starttime": _proc_starttime(proc.pid),
        "started_at": time.time(),
        "output_targets": list(output_targets or ()),
        "split_stderr": bool(split_stderr),
        "ephemeral": bool(ephemeral),
    }
    with open(_path(job_key, "meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    return {"job_key": job_key, "pid": proc.pid, "log": log,
            "job_dir": _job_dir(job_key), "proc": proc}


def _clip(path: str, max_bytes: int, keep: str) -> tuple[str, bool]:
    """*path*'s content, clipped to *max_bytes* from whichever end matters."""
    try:
        size = os.path.getsize(path)
        with open(path, errors="replace") as fh:
            if size > max_bytes and keep == "tail":
                fh.seek(size - max_bytes)
                return fh.read(), True
            return fh.read(max_bytes), size > max_bytes
    except OSError:
        return "", False


def streams(job_key: str, max_bytes: int, keep: str = "head") -> tuple[str, str, bool]:
    """The job's stdout and stderr so far, and whether either was clipped.

    *keep* is "head" for a run that finished — the first error is what explains the
    rest — and "tail" for one still going, where the frontier is the whole point.
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
        with open(_path(job_key, "meta.json"), "w") as fh:
            json.dump(meta, fh, indent=2)
    except OSError:
        pass


def discard(job_key: str) -> None:
    """Delete a finished blocking run's directory — nobody was given its key."""
    shutil.rmtree(_job_dir(job_key), ignore_errors=True)


def _sweep_once() -> None:
    """Remove ephemeral leftovers, once per server process.

    A blocking run deletes its own directory on the way out, so anything ephemeral
    still here belongs to a run that died with the server. Only ephemeral dirs, and
    only cold ones: a detached job's directory is the only record of it and is never
    swept, however old.
    """
    global _swept
    if _swept:
        return
    _swept = True
    try:
        keys = [k for k in os.listdir(JOBS_ROOT) if valid_key(k)]
    except OSError:
        return
    for key in keys:
        meta = _meta(key)
        if not meta.get("ephemeral"):
            continue
        if time.time() - float(meta.get("started_at") or 0) < 3600:
            continue
        if _is_running(int(meta.get("pid") or 0), meta.get("pid_starttime")):
            continue
        discard(key)


def _meta(job_key: str) -> dict:
    try:
        with open(_path(job_key, "meta.json")) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def state(job_key: str) -> dict:
    """The job's state in the vocabulary the client watcher polls for.

    running — the process is alive. done — it exited 0. crashed — it exited non-zero.
    unknown — it was killed rather than having returned: either no exit code was
    recorded at all (SIGKILL, an OOM, a host restart) or the recorded one is a signal
    (128+signum, written by the signal traps). It is reported as its own answer rather
    than folded into 'done' or 'crashed', because there is no status of the command's
    own to act on — and reporting a stopped build as 'done' would be a lie the caller
    cannot detect.
    """
    meta = _meta(job_key)
    if not meta:
        return {"state": "unknown", "job_key": job_key, "error": "No such job."}

    rc_raw = _read(job_key, "exit_code")
    alive = _is_running(int(meta.get("pid") or 0), meta.get("pid_starttime"))

    payload = {
        "job_key": job_key,
        "command": meta.get("command", ""),
        "pid": meta.get("pid"),
        "log": log_path(job_key),
        "elapsed_s": round(time.time() - float(meta.get("started_at") or 0), 1),
    }
    if alive and not rc_raw:
        payload["state"] = "running"
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
    payload["note"] = ("The process is gone and wrote no exit code — it was killed "
                       "from outside (SIGKILL, OOM, host restart) rather than "
                       "returning.")
    return payload


def output(job_key: str, max_bytes: int) -> dict:
    """The tail of the job's log, with its state.

    The tail rather than the head: a long build's interesting part — the error, the
    last target — is at the end, and the head is configure noise.

    An empty log says nothing on its own: plenty of long jobs — an extraction, a copy,
    a quiet install — legitimately print nothing, and annotating those would put a
    caveat on the ordinary case. So the note is attached only where the command is
    *known* to have sent its output elsewhere, which the validator's own parse already
    established (``write_targets``), and it names the file instead of describing the
    possibility. A run that simply stayed silent is reported as silent, uncommented.

    Under ``note``: ``hint`` is reserved to error payloads and ok() drops it from this
    one, so a message put there never reaches the caller.
    """
    payload = state(job_key)
    if payload.get("error"):
        return payload
    meta = _meta(job_key)
    log = log_path(job_key)
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
            errs, _ = _clip(err_path(job_key), max_bytes, "tail")
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
        keys = sorted((k for k in os.listdir(JOBS_ROOT) if valid_key(k)), reverse=True)
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
