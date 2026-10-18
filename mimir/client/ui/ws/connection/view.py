"""What one connection shows, and the socket it shows it on.

``ViewState`` is the connection's own state: the conversation on screen, how far the
client has rendered it, and the bookkeeping a screen needs. ``Sender`` is the socket,
seen from a handler: a frame, a line of chrome, a send scheduled for later.
"""
from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from typing import Any

from ....session.conversation import Conversation
from ..pool.bus import Subscription

logger = logging.getLogger(__name__)


@dataclass
class ViewState:
    # The session on screen, and the pool's own object for it, shared with everything
    # else that writes it. This connection only shows it.
    active_session_id: str | None = None
    conv: Conversation | None = None
    # How far into the journal the client's rendered transcript goes, as the client
    # last said. The watermark a re-attach replays from.
    rendered_seq: int = 0
    # This connection's view of the stream, opened in ``Connection.run``. The pump owns
    # the draining; this is only where our share of it arrives.
    sub: Subscription | None = None
    # Events withheld because they belong to another conversation, per session. The one
    # quiet-chat cause that is otherwise invisible — see ``Screen.is_foreign_event``.
    foreign_withheld: dict[str, int] = field(default_factory=dict)
    # Tool rows of a running turn that publish a run channel, per session, as
    # {call id: tool name}. The name is the channel (see ``run_channel``), so this is
    # what lets a divert click name the run the user is looking at, and what the
    # progress poller walks. Per session: two conversations can each have a divertible
    # run in flight, and one map between them would let a click detach the other's.
    live_rows: dict[str, dict[str, str]] = field(default_factory=dict)
    # Last (phase, percent) pushed per call id, so a tick that learned nothing sends
    # nothing.
    sent_progress: dict[str, dict[str, tuple]] = field(default_factory=dict)
    # Ids of deferred cards the user moved past without answering. Their card may still
    # be on screen; an answer to one must not reach the next live prompt.
    stale_prompt_ids: set[str] = field(default_factory=set)
    # Sessions whose deletion was refused once because work of theirs was running.
    # Asking again goes through: the second click is the confirmation.
    delete_refused: set[str] = field(default_factory=set)
    # Last `used_tokens` pushed, so the mid-turn refresh sends a frame only when the
    # number moved.
    last_context_usage: int | None = None
    # Per session: the client's transcript as last applied, by id, and which revision
    # of which run of full-then-deltas it is.
    transcript_sync: dict[str, dict] = field(default_factory=dict)

    def rows_for(self, session_id: str | None) -> dict[str, str]:
        """The divertible tool rows of one conversation's running turn."""
        return self.live_rows.setdefault(session_id or "", {})

    def forget_row(self, call_id: str, session_id: str | None = None) -> None:
        """Drop a finished row from the divert/progress bookkeeping."""
        sid = session_id if session_id is not None else self.active_session_id
        self.rows_for(sid).pop(call_id, None)
        self.sent_progress.setdefault(sid or "", {}).pop(call_id, None)

    def forget_session(self, session_id: str) -> None:
        """Drop everything kept per conversation, for one that is gone."""
        for per_session in (self.live_rows, self.sent_progress, self.foreign_withheld,
                            self.transcript_sync):
            per_session.pop(session_id, None)


class Sender:
    """The socket, as the handlers write to it."""

    def __init__(self, ws: Any) -> None:
        self.ws = ws

    async def send(self, payload: dict) -> None:
        """One frame. Raises when the client has gone."""
        await self.ws.send(json.dumps(payload, default=str))

    async def try_send(self, payload: dict) -> bool:
        """One frame, or False when the client has gone. Never raises."""
        try:
            await self.send(payload)
            return True
        except Exception:
            logger.debug("send: the client went away mid-send", exc_info=True)
            return False

    async def notify(self, text: str) -> None:
        """One line of chrome, outside any conversation. Never raises."""
        await self.try_send({"type": "output", "text": f"  {text}\n"})

    @staticmethod
    def schedule(coro: Any) -> None:
        """Fire a send without awaiting it, for callers that are not coroutines."""
        try:
            asyncio.get_running_loop().create_task(coro)
        except RuntimeError:
            coro.close()
