"""A sub-agent's own watcher for the detached runs it launched: the conversation's
:class:`RunWatcher`, delivering to the child.

A finished run and a check-in reach the child as steers (``poll_steer``), in
``wake_text``'s and ``checkin_steer``'s words.
The child's loop is its own: only :meth:`ChildWatcher.handover` may be called from another
thread; :meth:`ChildWatcher.close` cancels, on the child's loop.
"""
from __future__ import annotations

import logging
from collections import deque

from mimir.client.query_engine.jobs import RunWatcher, checkin_steer, wake_text

logger = logging.getLogger(__name__)


class ChildWatcher(RunWatcher):
    """Holds a sub-agent's detached runs, and answers for them when it ends.

    Constructed per child, on the child's own loop, and installed on its agent as the
    conversation's watcher is, plus the steer channel its resume arrives through.
    """

    # A child asks more often than the parent's watcher does. Its runs are steps of a task
    # it is holding open against a clock, so hearing a minute late costs it a minute of
    # budget; the parent's watcher holds a run across turns and can afford to be patient.
    first_interval = 3.0
    max_interval = 20.0
    call_timeout = 60.0
    unreadable_limit = 5

    def __init__(self, *, agent, origin: dict, on_activity=None) -> None:
        super().__init__()
        # The agent itself, resolved to its tool runner only when a watch polls: asking
        # at construction would make a watcher impossible to install on an agent that
        # has not finished being one.
        self._agent = agent
        self._origin = dict(origin or {})
        self._on_activity = on_activity
        # Written by the watch tasks and drained by the agent loop, both on the child's
        # loop, so the deque needs no lock of its own.
        self._steers: deque[str] = deque()
        self._handed_over = False

    # ── RunWatcher ──────────────────────────────────────────────────────────────

    def tool_runner(self):
        return getattr(self._agent, "run_tool", None)

    def accepts(self) -> bool:
        return not self._handed_over

    def session_for(self, owner: str | None) -> str | None:
        return None   # the child is the only reader

    def deliver(self, event: dict) -> None:
        with self._lock:
            handed_over = self._handed_over
        if handed_over:
            # Its caller has already been told these runs are the parent's. Steering the
            # child now would have two agents answering for one run.
            return
        if event.get("type") == "job_checkin":
            self._steers.append(checkin_steer(event))
            return
        self._steers.append(wake_text(event))
        if self._on_activity is not None:
            try:
                self._on_activity({"type": "status", "text": f"background job "
                                   f"{event.get('job_key')}: {event.get('state')}"})
            except Exception:
                pass   # a child's activity line must not fail its run

    # ── the steer channel and the handover ───────────────────────────────────────

    def drain_steers(self) -> list[str]:
        """Pop every wake waiting for the child. The child's ``poll_steer``."""
        out = []
        while self._steers:
            out.append(self._steers.popleft())
        return out

    def handover(self) -> list[dict]:
        """The runs still going, as descriptors the parent can poll; stops accepting new ones.

        Called from the caller's thread, cancels nothing. Each names the child it came
        from; its ops already name the run's directory, so they resolve from the parent.
        """
        with self._lock:
            if self._handed_over:
                # Idempotent: the caller out of patience and the child's own finally both
                # reach this, and a second list would have the parent adopt the runs twice.
                return []
            self._handed_over = True
            live = [dict(w.descriptor) for w in self.watches.values() if not w.task.done()]
        for descriptor in live:
            descriptor.setdefault("origin", dict(self._origin))
        return live

    async def close(self) -> None:
        """Drop the watches, from the child's own loop, in its ``finally``.

        A task still polling when ``asyncio.run`` returns is the one way a child's thread
        can be left hanging, so this is not optional housekeeping.
        """
        with self._lock:
            self._handed_over = True
        await super().close()
