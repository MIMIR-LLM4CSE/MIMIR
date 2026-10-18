"""Detached sub-agents and how many run at once: the job table a caller polls, and the
slots that cap concurrent children."""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time
from .._shared.jobs import descriptor, op
from . import spawn_limits
from . import spawn_user_channel
from . import spawn_worktree

JOBS: dict[str, dict] = {}


JOBS_LOCK = threading.Lock()


# Enough to answer for the children of one session; past it the oldest finished entry
# goes. A run is never dropped while it is still running.
JOBS_MAX = 64


# ── How many at once ──────────────────────────────────────────────────────────
# The only bound on a fan-out (the client dispatches a step's delegations concurrently).
# Here, because this process sees a detached child for its whole life.
SUBAGENT_MAX_CONCURRENT = int(os.environ.get("MIMIR_SUBAGENT_MAX_CONCURRENT") or 3)


# How long a call waits for a slot before giving up. Working children run for hours, so
# a full house can stay full for hours; a caller that simply queued behind that would
# freeze the turn it belongs to with nothing on screen to explain it. Better to say
# what is running and let the model do something else meanwhile.
SUBAGENT_SLOT_WAIT_SECS = 20


# A threading semaphore, not an asyncio one. A detached child gives its slot back from
# its own thread, long after the loop its call ran on has gone; an asyncio semaphore
# would have to be released through that loop, and releasing through a closed loop
# silently leaks the slot until the server is restarted. This one is released from
# wherever the child happens to end.
SLOTS = threading.Semaphore(SUBAGENT_MAX_CONCURRENT)


# Who holds a slot: {session: (task, started_at)}. Kept beside the semaphore rather
# than read off JOBS, which only knows the detached ones — a refusal that listed half
# of what is running would send the caller looking for the rest.
SLOT_HOLDERS: dict[str, tuple[str, float]] = {}


# Child threads carry this prefix. A slot is released from the child's own thread,
# which can outlive the call that started it, so "no live thread with this prefix"
# is the only reliable way to know every release has already landed.
THREAD_NAME_PREFIX = "mimir-subagent-"


def take_slot(session: str, task: str) -> None:
    with JOBS_LOCK:
        SLOT_HOLDERS[session] = (task.strip().splitlines()[0][:60], time.time())


async def wait_for_slot() -> bool:
    """Take a slot, waiting up to SUBAGENT_SLOT_WAIT_SECS. False when none came free.

    The blocking acquire runs on a worker thread so this server's loop keeps answering
    while a caller queues — including the very calls that would free a slot.
    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(
        None, SLOTS.acquire, True, SUBAGENT_SLOT_WAIT_SECS)


def free_slot(session: str) -> None:
    """Give the slot back. Called once, from the child's thread, whatever happened."""
    with JOBS_LOCK:
        SLOT_HOLDERS.pop(session, None)
    SLOTS.release()


def busy_with() -> str:
    """What is holding the slots right now, for a caller that could not get one."""
    with JOBS_LOCK:
        held = list(SLOT_HOLDERS.values())
    if not held:
        return "nothing this server still remembers"
    now = time.time()
    return "; ".join(f"{task!r} ({int(now - started)}s so far)" for task, started in held)


def job_descriptor(job_key: str) -> dict:
    """The watcher handle of a detached sub-agent: its ``status_op`` and ``summary_op``."""
    return descriptor(server="agent", kind="sub-agent", job_key=job_key,
                      status_op=op("subagent_job", job_key=job_key),
                      summary_op=op("subagent_job", op="result", job_key=job_key))


def register_job(job_key: str, entry: dict) -> None:
    with JOBS_LOCK:
        if len(JOBS) >= JOBS_MAX:
            for key, old in sorted(JOBS.items(), key=lambda kv: kv[1].get("started", 0)):
                if old.get("state") != "running":
                    JOBS.pop(key, None)
                    break
        JOBS[job_key] = entry


def job_snapshot(job_key: str) -> dict | None:
    with JOBS_LOCK:
        entry = JOBS.get(job_key)
        return dict(entry) if entry else None


def settle_job(job_key: str, state: str, result: dict | None) -> None:
    with JOBS_LOCK:
        entry = JOBS.get(job_key)
        if entry is not None:
            entry["state"] = state
            entry["result"] = result
            entry["ended"] = time.time()


# What a card's own word means to a watcher, whose vocabulary is running/done/
# crashed/unknown and nothing else (the WebSocket job watcher: any other word is read as
# "still going" and polled for ever). "abandoned" maps to crashed because that is what
# it is from here — a run that ended without delivering; the card's own word travels
# beside it as `outcome`, so the report never loses what actually happened.
CARD_STATE_TO_JOB = {
    "finished": "done",
    "failed": "crashed",
    "abandoned": "crashed",
}


def card_epoch(stamp: str | None) -> float:
    """A card's timestamp as seconds, or 0 when it has none that parses.

    The cards write local time without an offset, and both ends of an elapsed time come
    from the same card — so a naive parse subtracts correctly, and a stamp that does not
    parse yields 0 rather than failing a read.
    """
    try:
        return time.mktime(time.strptime(str(stamp), "%Y-%m-%dT%H:%M:%S"))
    except (TypeError, ValueError):
        return 0.0


def entry_from_card(job_key: str) -> dict | None:
    """Rebuild a job this process forgot from its child's card, or None.

    The key is the child session's last segment. A "running" card whose pid is gone reads
    ``unknown``.
    """
    if not job_key:
        return None
    try:
        from .._shared.state_paths import active_session_id, subagents_dir
        path = os.path.join(subagents_dir(active_session_id()), job_key, "subagent.json")
        with open(path, encoding="utf-8") as fh:
            card = json.load(fh)
    except Exception:
        return None
    state = str(card.get("state") or "unknown")
    if state == "running":
        job_state = "running" if spawn_worktree.card_is_live(card) else "unknown"
    else:
        job_state = CARD_STATE_TO_JOB.get(state, "unknown")
    return {
        "state": job_state,
        "outcome": state,
        "started": card_epoch(card.get("started_at")),
        "ended": card_epoch(card.get("ended_at")) or None,
        "budget": card.get("budget_secs", spawn_limits.SUBAGENT_HARD_CAP_SECS),
        "task": card.get("task", ""),
        "session": card.get("session", ""),
        "model": card.get("model", ""),
        "tools": card.get("tools", []),
        "phase": "",
        "from_card": True,
        "result": None if job_state == "running" else {
            "answer": card.get("answer", ""),
            "completed": bool(card.get("completed")),
            "files_read": [],
            "files_written": card.get("files_written", []),
            "blocked_by_mode": [],
            "workspace": card.get("workspace", {}),
        },
    }


def job_state(entry: dict) -> str:
    """The state the watcher acts on: running, done, crashed — or unknown past the budget.

    A child that overruns keeps going in its own thread, so "unknown" is the honest
    answer: the run was let go, and what it had done is still reported.
    """
    if entry.get("state") != "running":
        return entry["state"]
    if entry.get("from_card"):
        # Read off a card, not off this process's clock: its pid said the run is still
        # being written, and there is no start time here to time it out against.
        return "running"
    # Time the child spent in front of the user is not time it spent working. Without
    # this, a run parked on its own approval card ages out of its budget and is
    # reported lost to the watcher — which treats "unknown" as terminal — while the
    # card it is waiting on is still on screen.
    working = (time.time() - entry.get("started", 0)) - spawn_user_channel.paused_secs(entry.get("session", ""))
    if working > entry.get("budget", spawn_limits.SUBAGENT_HARD_CAP_SECS):
        return "unknown"
    return "running"
