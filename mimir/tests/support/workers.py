"""Workers for tests: a real ``AgentWorker`` with its fields, its human bridge and its
job watcher, and no thread, servers or agent unless a test gives it one."""
from __future__ import annotations

from typing import Any

from mimir.client.ui.ws.worker.worker import AgentWorker


def bare_worker(model: str = "test-model", session_id: str | None = "s1", *,
                agent: Any = None) -> AgentWorker:
    worker = AgentWorker(model, session_id, start=False)
    worker._agent = agent
    return worker
