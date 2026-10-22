"""Polling a detached run to a terminal state: the one loop every watcher uses.

Three words end a run (:data:`TERMINAL`); answers are read with ``parse_tool_payload``;
a status op that stays unreadable ends as lost. The op travels in the descriptor.
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from ...protocol.tool_result import parse_tool_payload
from ....shared.jobs import TERMINAL

logger = logging.getLogger(__name__)


# Consecutive status probes that come back without a state we recognise before a
# poll gives up. A probe that stopped working — a policy violation, a dead server,
# a renamed op — is indistinguishable from a running job if we only look for a
# terminal state, and a watcher that keeps polling one leaves the agent waiting on a
# wake that can never come. Five ticks is past the exponential backoff's early, short
# intervals, so a single transient failure never trips it.
UNREADABLE_POLL_LIMIT = 5


# How long one status probe may take. Nothing beneath has a deadline, so without this a
# wedged status op blocks the watcher for ever; a tick past it counts as unreadable, and
# ``UNREADABLE_POLL_LIMIT`` of those report the run as unknown.
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
    """What one poll learned: whether the op answered, and the :data:`STATUS_FIELDS` it carried."""

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
    wake: asyncio.Event | None = None,
) -> tuple[str, str]:
    """Poll *status_op* until its run is terminal; ``(state, reason)``, *reason* for "unknown".

    Waits *interval* first, then 1.5× up to *max_interval*. *call_timeout*,
    *unreadable_limit* and *max_polls* bound it; *on_tick* sees each :class:`Tick`;
    *cancel_flag* (the user's Stop) raises ``CancelledError``; *wake* cuts a wait short.
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
        await _wait(interval, wake)
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


async def _wait(interval: float, wake: asyncio.Event | None) -> None:
    """Sleep *interval*, or less when *wake* is set; *wake* is cleared for the next wait."""
    if wake is None:
        await asyncio.sleep(interval)
        return
    try:
        await asyncio.wait_for(wake.wait(), timeout=interval)
    except asyncio.TimeoutError:
        pass
    wake.clear()


async def read_summary(
    run_tool: Callable[..., Awaitable[Any]],
    summary_op: dict,
    *,
    call_timeout: float | None = None,
    label: str = "",
) -> dict:
    """What the run recorded, as a mapping for the wake; ``{}`` on any failure."""
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
        return (parse_tool_payload(raw) or {}) if isinstance(raw, str) else (raw or {})
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("background job %r finished, but its summary op %r did not "
                       "answer within %.0fs; waking it without one",
                       label, tool, call_timeout or 0)
        return {}
    except Exception:
        logger.debug("read_summary: a best-effort step failed and was skipped", exc_info=True)
        return {}
