"""Polling a detached run to a terminal state, for every caller that has to.

Three places need this loop: the WebSocket worker's watcher task, which holds a run
across turns; the in-turn fallback in :mod:`background`, used where no watcher exists;
and a sub-agent's own watcher, which is a watcher on a loop that lives only as long as
the child. What they share is not the schedule — each picks its own, because a run held
across turns can afford to ask rarely and a turn waiting on one cannot — but what a poll
*means*:

* four words end a run (:data:`TERMINAL`), and every other answer is "still going";
* a status answer is an envelope with text blocks and any appended annotation after it,
  so it is read with ``parse_tool_payload`` and never with a bare ``json.loads``;
* a tick that learned nothing new is a tick that says nothing, which is why what it
  learned is handed to the caller rather than announced from here;
* a status op that stops being readable is given up as lost rather than polled for ever,
  because an honest "I lost track of it" reaches the user and another silent tick does
  not.

Shape-driven, like the descriptors it serves: nothing here names a tool. The op to call
travels in the descriptor, and whatever that op chose to report about progress is passed
on without being understood.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ..protocol.tool_result import parse_tool_payload

logger = logging.getLogger(__name__)

# The four words that end a run. Anything else the status op says is "still going", so a
# descriptor's own vocabulary stays its own business — see the translation table in
# ``server_spawn_agent._CARD_STATE_TO_JOB`` for what speaking it costs.
TERMINAL = frozenset({"done", "crashed", "unknown"})

# Consecutive status probes that come back without a state we recognise before a
# poll gives up. A probe that stopped working — a policy violation, a dead server,
# a renamed op — is indistinguishable from a running job if we only look for a
# terminal state, and a watcher that keeps polling one leaves the agent waiting on a
# wake that can never come. Five ticks is past the exponential backoff's early, short
# intervals, so a single transient failure never trips it.
UNREADABLE_POLL_LIMIT = 5


# How long one status probe may take before it is given up on.
#
# There is no deadline anywhere beneath this: ``session.call_tool`` takes none, and the
# MCP sessions are built without a read timeout, so a status op that stops answering —
# a wedged server, a scheduler that hangs, a queue command that never returns — blocks
# the watcher for ever. That is the one failure that genuinely loses a run: the task
# stays in ``_bg_jobs``, so the agent is never released and never woken, the bulletins
# keep repeating a status frozen at the moment it stopped, and nothing anywhere says
# the run is no longer being watched. A watcher that dies at least says so.
#
# Generous on purpose: a status op is a cheap read, and this is the line between "slow"
# and "never", not a budget to work inside. A tick that crosses it is counted as
# unreadable, which is a case this loop already has — ``UNREADABLE_POLL_LIMIT`` of
# them in a row and the run is reported as unknown, with the reason — so a stuck status
# op ends as an honest "I lost track of it" instead of silence.
STATUS_CALL_TIMEOUT = 60.0

# What a poll keeps out of a status answer, for whoever reports the run to somebody: a
# check-in, a progress line, a badge on the node it landed on. A list of *field names*
# rather than a parser per tool, which is what keeps this shape-driven — a status op that
# reports none of them yields nothing here, and one that starts reporting a node needs no
# change on this side. Names only, never a tool: ``nodes``/``scheduler`` come from the
# cluster ops, ``pid``/``command`` from the shell one, and neither knows about the other.
STATUS_FIELDS = ("state", "phase", "percent", "scheduler", "nodes", "node", "pid",
                 "command", "elapsed_s", "job_id", "log_tail")

# Signs in a running job's last lines that it went wrong while still running. Names a
# category, never a tool: whatever the job is, these read the same in its output.
_LOG_ALERTS = (
    ("non-finite value", re.compile(r"(?<![\w.])[-+]?(?:nan|inf(?:inity)?)(?!\w)", re.I)),
    ("overflow", re.compile(r"\boverflow\b", re.I)),
    ("traceback", re.compile(r"Traceback \(most recent call last\)")),
    ("fatal signal", re.compile(
        r"segmentation fault|floating point exception|bus error|core dumped", re.I)),
    ("out of memory", re.compile(r"out of memory|oom-kill|MemoryError|std::bad_alloc",
                                 re.I)),
)


def log_alerts(lines: Any) -> dict[str, str]:
    """``{category: first matching line}`` for the alert signs in *lines*."""
    found: dict[str, str] = {}
    if not isinstance(lines, list):
        return found
    for line in lines:
        for name, pattern in _LOG_ALERTS:
            if name not in found and pattern.search(str(line)):
                found[name] = str(line)
    return found


def first_line(value: Any) -> str:
    """The first line of an error payload, for a one-line reason. "" when there is none."""
    text = str(value or "").strip()
    return text.splitlines()[0][:200] if text else ""


def kept_fields(payload: dict | None) -> dict:
    """The :data:`STATUS_FIELDS` a status answer carried, and nothing else."""
    return {k: (payload or {})[k] for k in STATUS_FIELDS if k in (payload or {})}


@dataclass(frozen=True)
class Tick:
    """What one poll learned, for a caller to do something with.

    ``answered`` separates "the op said nothing useful" from "the op said nothing at
    all": a raise or a timeout is not an answer, and a caller that reports progress to
    somebody usually wants to stay quiet about a tick it never got.

    ``fields`` is whatever of :data:`STATUS_FIELDS` this answer carried, so a caller that
    has to say where a run is does not re-read the payload to find out.
    """

    state: str
    phase: str
    percent: float | None
    answered: bool
    reason: str
    fields: dict


async def poll_until_terminal(
    run_tool: Callable[..., Awaitable[Any]],
    status_op: dict,
    *,
    interval: float,
    max_interval: float,
    call_timeout: float | None = None,
    unreadable_limit: int | None = None,
    max_polls: int | None = None,
    on_tick: Callable[[Tick], None] | None = None,
    label: str = "",
    cancel_flag: Any = None,
) -> tuple[str, str]:
    """Poll *status_op* until the run it names reaches a terminal state.

    Returns ``(state, reason)``; *reason* is a single line on why there is no state, and
    is meaningful only for ``"unknown"`` — a run that finished has nothing to explain.

    The probes are made with observations off by every caller, because a watcher tick is
    the client asking a question on its own account, not a step the model took; recording
    one as a step would credit the agent with work it did not do.

    Args:
        run_tool: The agent's tool runner, called ``(tool, args, record_observations=False)``.
        status_op: ``{"tool": ..., "args": {...}}`` off the descriptor. A missing tool
            name returns at once — there is nothing to poll, and a caller must not be
            left holding a promise it cannot keep.
        interval, max_interval: The first wait and the ceiling; each wait is 1.5× the
            last. The first poll happens after *interval*, never immediately: the run was
            launched moments ago and has nothing to say yet.
        call_timeout: Deadline on each probe. ``None`` waits as long as the tool takes,
            which is right only where the caller is itself inside a turn.
        unreadable_limit: How many consecutive unusable answers end the run as
            ``"unknown"``. ``None`` keeps polling, for a caller whose run cannot outlive
            it anyway.
        max_polls: A safety bound for a caller nothing will ever cancel. ``None`` for a
            task somebody can cancel.
        on_tick: Called with a :class:`Tick` per non-terminal poll.
        label: Names the run in this module's logs.
        cancel_flag: A ``threading.Event`` the user's Stop sets. Checked after every
            wait, so a caller polling inside a turn stops with the turn instead of
            outliving it; raises ``asyncio.CancelledError``.
    """
    status_tool = (status_op or {}).get("tool")
    if not status_tool:
        return "", "the descriptor names no status op"

    args = dict((status_op or {}).get("args") or {})
    state = "running"
    unreadable = 0
    polls = 0
    while max_polls is None or polls < max_polls:
        polls += 1
        await asyncio.sleep(interval)
        if cancel_flag is not None and cancel_flag.is_set():
            raise asyncio.CancelledError("Cancelled by user")
        interval = min(interval * 1.5, max_interval)
        answered = True
        reason = ""
        payload: dict | None = {}
        try:
            call = run_tool(status_tool, args, record_observations=False)
            raw = await (asyncio.wait_for(call, timeout=call_timeout)
                         if call_timeout is not None else call)
            payload = parse_tool_payload(raw) if isinstance(raw, str) else (raw or {})
            state = str((payload or {}).get("state") or "")
            if not state:
                reason = (first_line((payload or {}).get("error"))
                          or f"'{status_tool}' returned no state")
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            # Named rather than lumped in with the failures below: "it never answered"
            # and "it answered with an error" are the two things a reader most needs to
            # tell apart, and only this knows which it saw.
            payload, answered, state = {}, False, ""
            reason = (f"'{status_tool}' did not answer within "
                      f"{(call_timeout or 0):.0f}s")
            logger.warning("background job %r: its status op %r did not answer within "
                           "%.0fs", label, status_tool, call_timeout or 0)
        except Exception as exc:
            payload, answered, state = {}, False, ""
            reason = f"'{status_tool}' raised {type(exc).__name__}"
        if state in TERMINAL:
            return state, reason
        if on_tick is not None:
            percent = (payload or {}).get("percent")
            on_tick(Tick(
                state=state,
                phase=str((payload or {}).get("phase") or ""),
                percent=percent if isinstance(percent, (int, float)) else None,
                answered=answered,
                reason=reason,
                fields=kept_fields(payload),
            ))
        if state:
            unreadable = 0   # 'running', or any state the descriptor's op owns
            continue
        # Not a state anybody can act on. Retried a few times in case it is transient,
        # then given up as unknown.
        unreadable += 1
        if unreadable_limit is not None and unreadable >= unreadable_limit:
            return "unknown", reason
    return state, f"'{status_tool}' never reached a terminal state"


async def read_summary(
    run_tool: Callable[..., Awaitable[Any]],
    summary_op: dict,
    *,
    call_timeout: float | None = None,
    label: str = "",
) -> dict:
    """What the run recorded, for the wake that announces it. ``{}`` when there is none.

    Under the same deadline as a status probe, and here it matters more: the run is
    already over, so a summary op that never answers holds back the wake itself. Better
    a wake with nothing in it than no wake — which is why every failure here is a ``{}``
    and not a raise.

    Shared by the watchers, which want the payload as a mapping to put on an event. The
    in-turn wait keeps its own read: it folds the summary into a tool result as the text
    it arrived as, and parsing it to a dict only to render it back would lose whatever
    the server wrote around it.
    """
    tool = (summary_op or {}).get("tool")
    if not tool:
        return {}
    try:
        call = run_tool(tool, dict((summary_op or {}).get("args") or {}),
                        record_observations=False)
        raw = await (asyncio.wait_for(call, timeout=call_timeout)
                     if call_timeout is not None else call)
        # parse_tool_payload, not json.loads: a tool result is an envelope, its text
        # blocks and any annotation appended after them, and a bare load reads that as
        # one broken document. A summary lost here reaches the model as "it recorded no
        # result of its own" — the wake still lands, emptied of what it was carrying.
        return parse_tool_payload(raw) if isinstance(raw, str) else (raw or {})
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("background job %r finished, but its summary op %r did not "
                       "answer within %.0fs; waking it without one",
                       label, tool, call_timeout or 0)
        return {}
    except Exception:
        return {}
