"""A one-worker stand-in for ``AgentPool``, for tests that build a bare ``Connection``.

``Connection.worker`` is a property: it resolves the agent of the conversation on screen
through the pool, there being one agent per conversation. A test that only needs "a
worker" cannot assign to that property, and should not have to care how the lookup works.

This gives it the simple shape — one worker, whatever session is active — while the code
under test goes through the real seam.

Pass ``workers={session_id: worker}`` instead where *which* agent was addressed is the
thing under test. One worker answering for every session cannot tell a wake routed to
the conversation that launched the job from one routed to whatever is on screen, and a
stand-in that cannot fail the way production fails is not testing the seam, only the
call.
"""
from __future__ import annotations

import tempfile
from typing import Any, Iterator

from mimir.client.core.settings import read_setting
from mimir.client.session.conversation import Conversations
from mimir.client.session.store import SessionStore
from mimir.client.ui.ws.pool.bus import EventBus
from mimir.client.ui.ws.pool.wakes import WakeRouter
from mimir.client.ui.ws.pool.pool import AgentPool


class FakePool:
    def __init__(self, worker: Any = None, *, active: str | None = None,
                 workers: dict[str, Any] | None = None,
                 idle_reasons: list[str] | None = None,
                 store: SessionStore | None = None) -> None:
        self.worker = worker
        self.workers = dict(workers or {})
        self.active_session_id = active
        self.model = getattr(worker, "model", "test-model")
        self.cap = 3
        # What this process would still owe once the asker has gone. Non-empty by
        # default: the ordinary case under test is a conversation with work in it, and
        # a stand-in that reported "nothing to do" would have every detachment take the
        # warning path. Pass ``idle_reasons=[]`` for the case where there is nothing.
        self.idle_reasons = (["session s1 has a turn in flight"]
                             if idle_reasons is None else list(idle_reasons))
        self.server_idle_ttl = 7200.0
        self.queued: list[tuple[str, Any]] = []
        self.closed: list[str] = []
        # The real conversations, router and bus over this stand-in: what carries a
        # worker's output into a conversation and on to a session is the seam under
        # test in most of these cases, so faking it would test the call and not the
        # seam. Tests drive the bus with ``bus.pump_once()``.
        # In a directory of its own unless the test hands one in: a stand-in that saved
        # into the real state directory would leave conversations in the user's panel.
        if store is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="mimir-fakepool-")
            store = SessionStore(self._tmp.name)
        self.conversations = Conversations(store)
        self.store = store
        self.wakes = WakeRouter(self)
        self.bus = EventBus(self, commit=self.commit_turn, durable=self.wakes.consume,
                             steered=self.wakes.steered)

    # The pool's own: how a conversation is written is not what a stand-in may vary.
    commit_turn = AgentPool.commit_turn
    submit_turn = AgentPool.submit_turn
    save_conversation = AgentPool.save_conversation

    def _for(self, session_id: str | None) -> Any:
        """The agent of *session_id*: the per-session table when there is one."""
        if self.workers:
            return self.workers.get(session_id)
        return self.worker

    # ── what Connection reads ───────────────────────────────────────────────────

    def get(self, session_id: str | None) -> Any:
        return self._for(session_id)

    def worker_or_detached(self, session_id: str | None) -> Any:
        return self._for(session_id) or self.worker

    def items(self) -> Iterator[tuple[str, Any]]:
        if self.workers:
            return iter(list(self.workers.items()))
        if self.worker is None:
            return iter(())
        return iter([(self.active_session_id or "s1", self.worker)])

    def __contains__(self, session_id: object) -> bool:
        if self.workers:
            return session_id in self.workers
        return self.worker is not None

    def __len__(self) -> int:
        if self.workers:
            return len(self.workers)
        return 0 if self.worker is None else 1

    def is_busy(self, session_id: str | None) -> bool:
        """Busy *with that conversation's* turn.

        The real pool looks the session up and asks its own worker, and a worker only ever
        runs its own session's turn. This stand-in has one worker for every session, so it
        emulates that lookup by comparing against the turn the worker says it is running;
        without that it would report every conversation as busy whenever any one of them
        is.
        """
        worker = self._for(session_id)
        busy = getattr(worker, "is_busy", None)
        if worker is None or busy is None or not busy():
            return False
        if self.workers:
            return True   # that session's own agent said so
        running = getattr(worker, "_query_session_id", None)
        return running is None or session_id is None or running == session_id

    def idle_report(self, *, excluding: Any = None) -> dict:
        """Why this process is still needed, from the list the test set up.

        The real one asks the workers, the queue and the job directories on disk; what
        callers do with it is decide whether there is anything worth leaving running,
        and that is what this stands in for.
        """
        return {"idle": not self.idle_reasons, "reasons": list(self.idle_reasons)}

    def is_parked(self, session_id: str | None) -> bool:
        human = getattr(self._for(session_id), "human", None)
        return human is not None and human.awaiting_user()

    def set_active(self, session_id: str | None) -> None:
        self.active_session_id = session_id

    def queued_position(self, session_id: str) -> int | None:
        return None

    # ── building / releasing ──────────────────────────────────────────────────

    async def worker_for(self, session_id: str, *, on_wait: Any = None) -> Any:
        return self._for(session_id) or self.worker

    def enqueue(self, session_id: str, submit: Any) -> int:
        self.queued.append((session_id, submit))
        return len(self.queued)

    def drop_queued(self, session_id: str) -> None:
        self.queued = [e for e in self.queued if e[0] != session_id]

    async def pump(self) -> list[str]:
        return []

    async def close(self, session_id: str) -> None:
        self.closed.append(session_id)

    async def aclose_all(self) -> None:
        pass

    def ensure_reaper(self) -> None:
        pass

    def ensure_pump(self) -> None:
        pass

    # ── settings ──────────────────────────────────────────────────────────────

    def ui_state(self, session_id: str | None) -> dict:
        w = self.worker
        setting = getattr(w, "setting", None)

        def read(name: str, default: str) -> str:
            return default if setting is None else setting(name, default)

        return {
            "model": self.model,
            "thinking": w.get_thinking_profile() if hasattr(w, "get_thinking_profile") else {},
            "temperature": w.get_temperature_state() if hasattr(w, "get_temperature_state") else {},
            "agent_ready": w.agent_ready() if hasattr(w, "agent_ready") else False,
            "context_mode": read("context_mode", "compact"),
            "enforcement": read("enforcement", "light"),
            "approval_mode": read("approval_mode", "manual"),
        }

    def setting(self, name: str, session_id: str | None) -> Any:
        source = getattr(self._for(session_id) or self.worker, "setting", None)
        return read_setting(None, name) if source is None else source(name)

    def apply_setting(self, name: str, value: Any) -> list[str]:
        method = getattr(self.worker, "apply_setting", None)
        return [] if method is None else [r for r in [method(name, value)] if r]

    def apply_toggle(self, kind: str, name: str, enabled: bool) -> None:
        method = getattr(self.worker, f"set_{kind}_enabled", None)
        if method is not None:
            method(name, enabled)

    def set_non_interactive(self) -> None:
        method = getattr(self.worker, "set_non_interactive", None)
        if method is not None:
            method(True)

    def set_model(self, model: str) -> list[str]:
        self.model = model
        return self.apply_setting("model", model)
