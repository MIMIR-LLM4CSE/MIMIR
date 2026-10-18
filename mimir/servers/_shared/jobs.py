"""A detached run as the client and every server see it.

The descriptor a watcher polls, the words that end a run, the files a run's directory
holds, the two payload keys that carry descriptors, and Slurm's states read into those
words. The client imports this module; nothing here imports the client.
"""
from __future__ import annotations

import os
import shlex
from typing import Any, Callable

from . import cpu_facts

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
               summary_op: dict | None = None, run_dir: str = "") -> dict:
    """The handle a watcher polls: *status_op* until a terminal state, then *summary_op*."""
    out: dict[str, Any] = {"server": server, "kind": kind, "job_key": job_key}
    if run_dir:
        out["run_dir"] = run_dir
    out["status_op"] = status_op
    if summary_op:
        out["summary_op"] = summary_op
    return out


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


# ── Slurm ────────────────────────────────────────────────────────────────────
_ACTIVE = ("RUNNING", "COMPLETING")
_WAITING = ("PENDING", "CONFIGURING", "REQUEUED")
# Prefixes, because Slurm qualifies some words ("CANCELLED by 1234").
_CRASH_PREFIXES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                   "BOOT_FAIL", "DEADLINE", "PREEMPTED")
_DONE_PREFIXES = ("COMPLETED",)


def _crashed(word: str) -> bool:
    return word.startswith(_CRASH_PREFIXES)


def _done(word: str) -> bool:
    return word.startswith(_DONE_PREFIXES)


def slurm_state(job_id: str, run_lines: Callable[[str], list[str]]
                ) -> tuple[str, str, list[str]]:
    """``(state, raw, nodes)`` of a Slurm job; *run_lines* runs a command on its scheduler.

    ``squeue``, then ``sacct``; nodes only while running. For an array, any live element
    keeps it live and the verdict is the worst element's. An ended word in squeue settles
    the job without sacct, all a site without accounting has.
    """
    queued, where = [], []
    for row in run_lines(f"squeue -j {shlex.quote(job_id)} -h -o '%T|%R'"):
        word, _, reason_or_node = row.partition("|")
        queued.append(word.strip())
        where.append(reason_or_node.strip())
    raw = ",".join(sorted(set(queued)))
    if any(word in _ACTIVE for word in queued):
        nodes = sorted({node for word, expr in zip(queued, where) if word in _ACTIVE
                        for node in cpu_facts.expand_hostlist(expr) if node})
        return "running", raw, nodes
    if queued and all(word in _WAITING for word in queued):
        return "pending", raw, []
    if queued and all(_done(w) or _crashed(w) for w in queued):
        return ("crashed" if any(_crashed(w) for w in queued) else "done"), raw, []

    accounted = run_lines(f"sacct -j {shlex.quote(job_id)} -n -X -o State")
    sraw = ",".join(sorted(set(accounted)))
    if any(_crashed(word) for word in accounted):
        return "crashed", sraw, []
    if accounted and all(_done(word) for word in accounted):
        return "done", sraw, []
    if queued:                    # in the queue under a non-standard active word
        return "running", raw, []
    if accounted:                 # some other terminal word
        return "crashed", sraw, []
    return "unknown", "", []


def locate_slurm_job(hint: str | None, hosts: list[str],
                     state_on: Callable[[str], tuple[str, str, list[str]]]
                     ) -> tuple[str | None, tuple[str, str, list[str]]]:
    """``(host, state_on(host))`` for the first scheduler that knows the job, its recorded
    *hint* first; ``(None, unknown)`` when none does."""
    order = ([hint] if hint is not None else []) + [h for h in hosts if h != hint]
    for host in order:
        found = state_on(host)
        if found[0] != "unknown":
            return host, found
    return None, ("unknown", "", [])
