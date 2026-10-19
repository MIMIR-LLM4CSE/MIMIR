"""A tool call that launched a detached run, as the agent loop handles it.

The tool returns a ``background_job`` descriptor. With a front-end watcher the loop
registers it and tells the model to yield; without one (the CLI) it waits for the run
in-turn, polling the descriptor's status op with no model calls. It also emits the
``open_editor`` event when a result asks for it.
"""
from __future__ import annotations

import json
import logging

from ...protocol.event_sink import emit
from ....shared import jobs
from ...tools.capabilities import BACKGROUNDABLE, has_cap
from ...protocol.tool_result import parse_tool_payload
from . import poll
from .watcher import RunWatcher, Watch
from ...core.ports import AgentPort

logger = logging.getLogger(__name__)


def maybe_emit_open_editor(result_text: str) -> None:
    """Emit ``open_editor`` for a result carrying ``open_in_editor: true`` and an absolute ``path``."""
    payload = parse_tool_payload(result_text) if isinstance(result_text, str) else None
    if not isinstance(payload, dict) or not payload.get("open_in_editor"):
        return
    path = payload.get("path")
    if isinstance(path, str) and path.strip():
        emit({"type": "open_editor", "path": path})


def detect_background_job(name: str, result_text: str, agent: AgentPort) -> dict | None:
    """Return a ``background_job`` descriptor from a BACKGROUNDABLE tool result, else None.

    Shape-driven like ``maybe_emit_open_editor`` — no tool name is hard-coded, and
    the payload is read through ``parse_tool_payload`` for the same reason: a launched
    run must not stop being watched because something was appended after its JSON.
    """
    if not has_cap(name, BACKGROUNDABLE, agent.tool_caps):
        return None
    return jobs.launched(parse_tool_payload(result_text) if isinstance(result_text, str) else None)


def maybe_rewatch_background_jobs(name: str, result_text: str, agent: AgentPort) -> str:
    """Put watchers on the in-flight runs a result reported — a status answer's, or
    the one an adopter just handed over.

    Silent about the ones already watched: the hook dedups on ``job_key`` and reports
    success either way, so the note names what was re-armed only when that is news.
    """
    # Plural means "still going", not "this call launched it": asking about a run is how
    # tracking resumes after a restart.
    descriptors = jobs.handed_over(
        parse_tool_payload(result_text) if isinstance(result_text, str) else None)
    if not descriptors:
        return result_text
    hook = getattr(agent, "register_background_job", None)
    if not hook:
        return result_text   # CLI: no watcher to re-arm, and nothing was promised
    known = set()
    watched = getattr(agent, "watched_background_jobs", None)
    if callable(watched):
        try:
            known = {jobs.job_key_of(d) for d in (watched() or []) if isinstance(d, dict)}
        except Exception:
            known = set()
    rearmed = []
    for descriptor in descriptors:
        job_key = jobs.job_key_of(descriptor)
        if job_key in known:
            continue
        try:
            if hook(descriptor):
                rearmed.append(job_key)
        except Exception:
            logger.warning("re-watch registration raised for job %r", job_key,
                           exc_info=True)
    if not rearmed:
        return result_text
    keys = ", ".join(f"'{k}'" for k in rearmed)
    note = (
        f"\n\n[background] Now tracking {keys}. A watcher resumes you "
        "automatically when each one finishes, so do NOT poll their status and do NOT "
        "sleep waiting for them. Carry on with whatever else there is to do, or end "
        "your turn saying what you are waiting for."
    )
    return result_text + note


def maybe_register_background_job(
    name: str, result_text: str, agent: AgentPort, descriptor: dict | None = None,
) -> tuple[str, bool]:
    """Register a watcher for a detached run; ``(result, registered)``.

    Through the optional ``agent.register_background_job`` hook; the "end your turn" note is
    added only when a watcher holds the run. Best-effort, and logged.
    """
    if descriptor is None:
        descriptor = detect_background_job(name, result_text, agent)
    if descriptor is None:
        return result_text, False
    job_key = jobs.job_key_of(descriptor) or "?"
    hook = getattr(agent, "register_background_job", None)
    if not hook:
        # CLI: there is no watcher to register with. Not a failure — the caller waits
        # the run out in-turn instead.
        logger.debug("no background-job watcher hook; job %r will be awaited in-turn",
                     job_key)
        return result_text, False
    try:
        registered = bool(hook(descriptor))
    except Exception:
        logger.warning("background watcher registration raised for job %r — "
                       "falling back to an in-turn await", job_key, exc_info=True)
        return result_text, False
    if not registered:
        logger.warning("background watcher registration declined job %r — "
                       "falling back to an in-turn await", job_key)
        return result_text, False
    # Order matters: waiting is the *last* resort, not the first instruction. Read
    # the other way round, a model with plenty left to do would stop dead on a
    # two-hour build rather than get on with the rest of the task.
    note = (
        f"\n\n[background] This run is now tracked as background job '{job_key}'. "
        "A watcher resumes you automatically with its results when it completes, so "
        "do NOT poll its status and do NOT sleep waiting for it. If there is other "
        "useful work in this task, carry on with it now. If the only thing left is "
        "waiting for this run, end your turn and say what you are waiting for."
    )
    return result_text + note, True


class _InTurnWait(RunWatcher):
    """A run waited out by the turn that launched it: the CLI's, which has no watcher."""

    # Tighter than a watcher's: the turn is held until it ends.
    first_interval = 3.0
    max_interval = 20.0

    def __init__(self, agent: AgentPort) -> None:
        super().__init__()
        self._agent = agent
        self.event: dict = {}

    def tool_runner(self):
        return self._agent.run_tool

    def cancel_flag(self):
        return self._agent.turn.cancel_flag

    def on_tick(self, job_key: str, watch: Watch, tick: poll.Tick,
                session_id: str | None) -> None:
        # Only what the op actually answered: a probe that raised is a tick nobody
        # asked about, and "background run: " with nothing after it is noise in the
        # one place the caller is watching this run from.
        if tick.answered:
            emit({"type": "status", "text": f"background run: {tick.state}"})

    def deliver(self, event: dict) -> None:
        self.event = event


async def await_background_job(descriptor: dict, agent: AgentPort, result_text: str) -> str:
    """Wait out a detached run in-turn without model calls, then append its summary.

    For the CLI, which has no watcher: a :class:`RunWatcher` followed by the turn itself,
    with the watcher's limits, stopped by the user's Stop.
    """
    if not (descriptor.get("status_op") or {}).get("tool"):
        return result_text
    waiter = _InTurnWait(agent)
    await waiter.follow(descriptor)
    note = f"\n\n[background:awaited] Run reached state '{waiter.event.get('state', '')}'."
    summary = waiter.event.get("summary")
    if summary:
        note += f" Results:\n{json.dumps(summary, indent=2, default=str)}"
    return result_text + note
