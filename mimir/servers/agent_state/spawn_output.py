"""What a sub-agent's run shows its caller: its events compacted and forwarded as MCP
progress, a heartbeat while it is quiet, and an activity log on disk."""
from __future__ import annotations

import json
import os
import queue
import sys
import time
from typing import Any, Callable
from .._shared.atomic import write_text
from mcp.server.fastmcp import Context
from . import spawn_jobs

CHILD_QUEUE_MAX = 256      # child events buffered between two forwarding ticks


MAX_EVENTS_PER_TICK = 8    # forwarded per tick, so a tool storm cannot hog the loop


MAX_EVENTS_PER_RUN = 500   # ceiling per child run; past it only the trailer is sent


POLL_SECS = 0.05


# The child's activity log: enough to see what it has been doing, bounded so a long
# run cannot fill a disk with rows nobody will scroll back to.
ACTIVITY_MAX_BYTES = 256_000


ACTIVITY_KEEP_LINES = 400


# A child in a model turn emits nothing at all, and the caller's row has no way to
# tell that from a hung run. Past this silence, say it is still there.
HEARTBEAT_SECS = 20.0


stdout_silenced = False


def silence_stdout_once() -> None:
    """Point stdout at stderr, once, on first use — never at import time.

    The stdio transport wraps ``sys.stdout.buffer`` once at startup; rebinding
    before that would make it wrap *stderr* and the server would go mute. Doing it
    afterwards is harmless (the transport holds its own reference), and it is never
    restored: two concurrent children would restore each other's saved value.
    """
    global stdout_silenced
    if stdout_silenced:
        return
    sys.stdout = sys.stderr
    stdout_silenced = True


def clip(value: Any, limit: int) -> str:
    text = "" if value is None else str(value)
    return text if len(text) <= limit else text[: limit - 1] + "…"


def compact_event(ev: dict) -> dict | None:
    """One child engine event in wire form, or None when it is not worth forwarding.

    The child's tool activity travels, and its step-level status lines with it:
    tokens, thinking and diffs would cost far more than they show, and the caller
    renders the rest as ordinary tool rows. Keys are short because each event rides
    in a progress notification.

    Status matters because a step is not one model call. An empty turn is retried, a
    nudge is answered, a checklist refresh re-asks — each up to the child's whole
    per-step ceiling, none of them a tool call. Dropped, they left a gap in the log
    with nothing in it, and a child spending four minutes that way was indistinguishable
    from a child that had hung. The loop already says what it is doing; forwarding it
    is the difference between reading that and guessing at it.
    """
    kind = ev.get("type")
    if kind == "tool_call":
        return {
            "v": 1, "t": "tc",
            "i": str(ev.get("id") or ""),
            "n": str(ev.get("name") or ""),
            "l": clip(ev.get("label"), 120),
            "d": clip(ev.get("detail"), 160),
            # The row's work family and the child model's own sentence about the call —
            # what a child row shows, exactly as a top-level one does. `w` for the
            # second: `d` is already the argument preview.
            "k": clip(ev.get("kind"), 16),
            "w": clip(ev.get("doing"), 120),
        }
    if kind == "tool_result":
        out = {
            "v": 1, "t": "tr",
            "i": str(ev.get("id") or ""),
            "ok": bool(ev.get("ok")),
            "s": clip(ev.get("summary"), 160),
        }
        ms = ev.get("duration_ms")
        if isinstance(ms, (int, float)):
            out["ms"] = int(ms)
        if isinstance(ev.get("target"), dict):
            out["f"] = ev["target"]
        return out
    if kind == "status":
        text = clip(ev.get("text"), 160).strip()
        # The loop emits a blank status as a spacer for a terminal that has rows to
        # separate. This log has none, and a blank line here reads as a lost event.
        return {"v": 1, "t": "st", "s": text} if text else None
    return None


def activity_path(session: str) -> str:
    from .._shared.state_paths import ACTIVITY_FILENAME, session_dir
    return os.path.join(session_dir(session), ACTIVITY_FILENAME)


def append_activity(session: str, event: dict) -> None:
    """Add one line to this child's activity log, trimming it when it grows.

    The log is how anyone watches a sub-agent work. The progress channel only exists
    while the call is open — a detached child would otherwise be invisible until it
    lands — and a file is read the same way by both kinds of child, by the panel, and
    after the fact. Best effort: a child's work must not fail over its own log.
    """
    if not session:
        return
    try:
        path = activity_path(session)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps({**event, "at": time.time()}, separators=(",", ":")) + "\n")
            trim = fh.tell() > ACTIVITY_MAX_BYTES
        if trim:
            with open(path, "r", encoding="utf-8") as fh:
                lines = fh.readlines()[-ACTIVITY_KEEP_LINES:]
            write_text(path, "".join(lines))
    except Exception:
        pass


def make_child_sink(q: queue.Queue, counters: dict,
                     job_key: str = "", session: str = "") -> Callable[[dict], None]:
    """The child's event_callback: enqueue, never block, never raise.

    Raising here would put the child's engine back on emit()'s print fallback —
    straight into the JSON-RPC pipe — so a full queue drops the event and counts it.

    Three readers, one sink. The caller's progress channel gets every event while the
    call is open; the child's activity log gets them all regardless, which is what lets
    anyone watch a detached child; and a detached job records its last tool row as its
    phase, the only sign of life the watcher can pass on once the call has returned.
    """
    def _sink(ev: dict) -> None:
        try:
            compact = compact_event(ev)
            if compact is None:
                return
            append_activity(session, compact)
            if job_key and compact.get("t") == "tc":
                with spawn_jobs.JOBS_LOCK:
                    entry = spawn_jobs.JOBS.get(job_key)
                    if entry is not None:
                        entry["phase"] = compact.get("l") or compact.get("n") or ""
            q.put_nowait(compact)
        except Exception:
            counters["dropped"] = counters.get("dropped", 0) + 1
    return _sink


async def report(ctx: Context | None, state: dict, payload: dict) -> None:
    """Send one event to the caller. A dead pipe must not fail the child's run."""
    if ctx is None:
        return
    state["sent"] = state.get("sent", 0) + 1
    try:
        await ctx.report_progress(
            progress=state["sent"],
            message=json.dumps(payload, separators=(",", ":")),
        )
    except Exception:
        pass


async def forward_pending(ctx: Context | None, q: queue.Queue, state: dict) -> None:
    """Drain a bounded slice of the child's events to the caller."""
    for _ in range(MAX_EVENTS_PER_TICK):
        try:
            ev = q.get_nowait()
        except queue.Empty:
            return
        if state.get("sent", 0) >= MAX_EVENTS_PER_RUN:
            state["dropped"] = state.get("dropped", 0) + 1
            continue
        state["last_activity"] = time.monotonic()
        await report(ctx, state, ev)


async def maybe_heartbeat(ctx: Context | None, state: dict, started: float) -> None:
    """Say the child is alive when it has been silent long enough to look hung.

    Silence is the normal shape of a model turn — no tool call, nothing to forward —
    and a delegated turn can be minutes of it. Heartbeats are exempt from the per-run
    event ceiling: the one moment the caller most needs a sign of life is the run that
    already spent its budget on child steps.
    """
    now = time.monotonic()
    if now - state.get("last_activity", started) < HEARTBEAT_SECS:
        return
    state["last_activity"] = now
    await report(ctx, state, {"v": 1, "t": "hb", "s": int(now - started)})
