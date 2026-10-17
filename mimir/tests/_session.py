"""A bare ``Connection`` for tests that drive one handler, not a whole connection.

``Connection.__init__`` wants a socket and opens nothing else, but most tests want a
session already showing a conversation, with only the fields the handler under test
reads. This builds that: the conversation on screen is the pool's own object for
*active*, created if the pool has none, exactly as a connection holds it.
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
    sess = object.__new__(Connection)
    sess.ws = ws if ws is not None else FakeWS()
    sess.pool = pool
    sess.store = pool.conversations.store
    sess._active_session_id = active
    sess.conv = None
    if active is not None:
        sess.conv = pool.conversations.get(active) or pool.conversations.new(active)
        if history is not None:
            sess.conv.history = list(history)
            sess.conv.record.llm_history_full = list(history)
            sess.conv.submitted_len = len(history)
    sess._rendered_seq = 0
    sess._stale_prompt_ids = set()
    sess._delete_refused = set()
    sess._live_rows = {}
    sess._sent_progress = {}
    sess._foreign_withheld = {}
    sess._last_context_usage = None
    sess._summary_task = None
    sess._sub = None
    sess._panel_task = None
    sess._panel_again = None
    return sess
