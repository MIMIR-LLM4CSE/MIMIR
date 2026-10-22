"""A detached run as the client and every server see it.

The descriptor a watcher polls, the words that end a run, the files a run's directory
holds, and the two payload keys that carry descriptors. The client imports this module;
nothing here imports the client. Slurm's states are read in ``slurm_site``.
"""
from __future__ import annotations

import json
import os
from typing import Any

# ── Files a run's directory holds ────────────────────────────────────────────
EXIT_CODE = "exit_code"            # written by a shell job's trap as it leaves
META = "meta.json"                 # a shell job's pid, command and start time
SLURM_JOB_ID = "slurm_job_id"      # a submission's id, written once at submission
SLURM_STATE = "slurm_state"        # a submission's terminal word, written by a poll
WATCHER_FILE = "watcher.json"      # the descriptor of a run its own server watches
REPORTED = "reported"              # the run's ending was handed to a turn
WAKE_BASELINE = ".wake_baseline"   # a session's runs were scanned once already

# ── States ───────────────────────────────────────────────────────────────────
#: The words that end a run. ``unknown`` is not a failure: the run stopped being
#: trackable, and no outcome was observed. Any other word means still going.
TERMINAL = frozenset({"done", "crashed", "unknown"})


def is_terminal(state: str | None) -> bool:
    return state in TERMINAL


def exit_code_state(code: int | None) -> str:
    """A shell exit code as a run state: 0 done, a signal (128+n) or none unknown, else crashed."""
    if code is None or code > 128:
        return "unknown"
    return "done" if code == 0 else "crashed"


def record_terminal_state(job_dir: str, state: str) -> None:
    """Write *state* into *job_dir* once, at the first poll that sees it end. Best-effort.

    Write-once: a later poll may read ``unknown`` past the scheduler's retention.
    """
    if not job_dir or not is_terminal(state):
        return
    try:
        with open(os.path.join(job_dir, SLURM_STATE), "x") as fh:
            fh.write(state)
    except OSError:
        pass


def recorded_terminal_state(job_dir: str) -> str:
    """The terminal state recorded in *job_dir*, or ""."""
    try:
        with open(os.path.join(job_dir, SLURM_STATE)) as fh:
            word = fh.read().strip()
    except OSError:
        return ""
    return word if is_terminal(word) else ""


# ── Descriptors ──────────────────────────────────────────────────────────────
#: The payload key of a run this call launched (honoured for a BACKGROUNDABLE tool).
LAUNCHED = "background_job"
#: The payload key of runs still going that a call hands over: a status answer, an
#: adoption, a sub-agent's leftovers. Needs no capability.
HANDED_OVER = "background_jobs"


def op(tool: str, **args: Any) -> dict:
    """A tool call a watcher makes: ``{"tool", "args"}``."""
    return {"tool": tool, "args": args}


def descriptor(*, server: str, kind: str, job_key: str, status_op: dict,
               summary_op: dict | None = None, cancel_op: dict | None = None,
               run_dir: str = "") -> dict:
    """The handle a watcher polls: *status_op* until a terminal state, then *summary_op*.

    *cancel_op* is the call that stops the run, offered to the user beside it; a run
    without one cannot be stopped from outside the conversation.
    """
    out: dict[str, Any] = {"server": server, "kind": kind, "job_key": job_key}
    if run_dir:
        out["run_dir"] = run_dir
    out["status_op"] = status_op
    if summary_op:
        out["summary_op"] = summary_op
    if cancel_op:
        out["cancel_op"] = cancel_op
    return out


def _adopted(job_dir: str) -> bool:
    """Whether the shell job in *job_dir* is a process this server did not start."""
    try:
        with open(os.path.join(job_dir, META)) as fh:
            return bool(json.load(fh).get("adopted"))
    except (OSError, ValueError, AttributeError):
        return False


def shell_descriptor(job_key: str, job_dir: str) -> dict:
    """A detached shell job. Its directory travels in the op args, so a job of a
    sub-agent's session still resolves from the parent.

    An adopted process gets no cancel op: its process group is not one this server
    made, and stopping the group could take the user's own shell with it.
    """
    cancel = (None if _adopted(job_dir)
              else op("bash_job_stop", job_key=job_key, run_dir=job_dir))
    return descriptor(server="bash", kind="shell-command", job_key=job_key, run_dir=job_dir,
                      status_op=op("bash_job", job_key=job_key, run_dir=job_dir),
                      summary_op=op("bash_job", op="output", job_key=job_key,
                                    run_dir=job_dir),
                      cancel_op=cancel)


def slurm_descriptor(job_id: str) -> dict:
    """A Slurm job of this workspace, polled by the HPC server."""
    return descriptor(server="hpc", kind="slurm-batch", job_key=job_id,
                      status_op=op("slurm_job_status", job_id=job_id),
                      cancel_op=op("slurm_cancel", job_id=job_id, confirm=True))


def job_key_of(desc: dict) -> str:
    """The key a run is known by: its ``job_key``, else its ``run_dir``."""
    return str(desc.get("job_key") or desc.get("run_dir") or "")


def is_rewatchable(desc: Any) -> bool:
    """Whether *desc* names a run a watcher can poll: a key and a status tool."""
    if not isinstance(desc, dict) or not job_key_of(desc):
        return False
    status_op = desc.get("status_op")
    return isinstance(status_op, dict) and bool(status_op.get("tool"))


def launched(payload: Any) -> dict | None:
    """The descriptor of the run *payload* says it launched, or None."""
    found = payload.get(LAUNCHED) if isinstance(payload, dict) else None
    return found if isinstance(found, dict) else None


def handed_over(payload: Any) -> list[dict]:
    """The pollable descriptors of runs *payload* reports as still going."""
    found = payload.get(HANDED_OVER) if isinstance(payload, dict) else None
    return [d for d in found if is_rewatchable(d)] if isinstance(found, list) else []


def is_detached(payload: Any) -> bool:
    """Whether *payload* is about a run that outlives the call."""
    return launched(payload) is not None or bool(handed_over(payload))
