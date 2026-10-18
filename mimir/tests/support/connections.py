"""A bare ``Connection`` for tests that drive one handler, not a whole connection.

A ``Connection`` showing a conversation already: the one on screen is the pool's own
object for *active*, created if the pool has none, exactly as a connection holds it.
"""
from __future__ import annotations

import json
from typing import Any

from mimir.client.ui.ws.connection.connection import Connection


class FakeWS:
    """A socket that keeps what is sent, decoded."""

    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


def bare_session(pool: Any, *, active: str | None = "s1", ws: Any = None,
                 history: list[dict] | None = None) -> Connection:
    sess = Connection(ws if ws is not None else FakeWS(), pool)
    sess.view.active_session_id = active
    if active is not None:
        sess.view.conv = pool.conversations.get(active) or pool.conversations.new(active)
        if history is not None:
            sess.view.conv.history = list(history)
            sess.view.conv.record.llm_history_full = list(history)
            sess.view.conv.submitted_len = len(history)
    return sess
