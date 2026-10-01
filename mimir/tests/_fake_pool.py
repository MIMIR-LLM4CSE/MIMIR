"""A one-worker stand-in for ``_AgentPool``, for tests that build a bare ``_Session``.

``_Session.worker`` is a property now: it resolves the agent of the conversation on
screen through the pool, because there is one agent per conversation rather than one for
all of them. A test that used to set ``sess.worker = w`` cannot assign to a property, and
should not need to care how the lookup works.

This gives those tests the old shape back — one worker, whatever session is active —
while the code under test goes through the real seam.
"""
from __future__ import annotations

from typing import Any, Iterator


class FakePool:
    def __init__(self, worker: Any = None, *, active: str | None = None) -> None:
        self.worker = worker
        self.active_session_id = active
        self.model = getattr(worker, "model", "test-model")
        self.cap = 3
        self.queued: list[tuple[str, Any]] = []
        self.closed: list[str] = []

    # ── what _Session reads ───────────────────────────────────────────────────

    def get(self, session_id: str | None) -> Any:
        return self.worker

    def worker_or_detached(self, session_id: str | None) -> Any:
        return self.worker

    def items(self) -> Iterator[tuple[str, Any]]:
        if self.worker is None:
            return iter(())
        return iter([(self.active_session_id or "s1", self.worker)])

    def __contains__(self, session_id: object) -> bool:
        return self.worker is not None

    def __len__(self) -> int:
        return 0 if self.worker is None else 1

    def is_busy(self, session_id: str | None) -> bool:
        return bool(self.worker is not None and self.worker.is_busy())

    def is_parked(self, session_id: str | None) -> bool:
        return getattr(self.worker, "_pending_prompt", None) is not None

    def set_active(self, session_id: str | None) -> None:
        self.active_session_id = session_id
        if self.worker is not None:
            self.worker.active_session_id = session_id

    def queued_position(self, session_id: str) -> int | None:
        return None

    # ── building / releasing ──────────────────────────────────────────────────

    async def worker_for(self, session_id: str, *, on_wait: Any = None) -> Any:
        return self.worker

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

    # ── settings ──────────────────────────────────────────────────────────────

    def ui_state(self, session_id: str | None) -> dict:
        w = self.worker
        return {
            "model": self.model,
            "thinking": w.get_thinking_profile() if hasattr(w, "get_thinking_profile") else {},
            "temperature": w.get_temperature_state() if hasattr(w, "get_temperature_state") else {},
            "agent_ready": w.agent_ready() if hasattr(w, "agent_ready") else False,
            "context_mode": w.get_context_mode() if hasattr(w, "get_context_mode") else "compact",
            "enforcement": w.get_enforcement() if hasattr(w, "get_enforcement") else "strict",
            "approval_mode": w.get_approval_mode() if hasattr(w, "get_approval_mode") else "manual",
        }

    def apply_setting(self, name: str, *args: Any) -> list[Any]:
        if self.worker is None:
            return []
        method = getattr(self.worker, name, None)
        return [] if method is None else [method(*args)]

    def set_model(self, model: str) -> list[str]:
        self.model = model
        return [r for r in self.apply_setting("set_model", model) if r]
