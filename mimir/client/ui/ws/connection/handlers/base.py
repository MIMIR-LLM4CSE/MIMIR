"""What every handler group of a connection is built with."""
from __future__ import annotations

from ...pool.pool import AgentPool
from ..screen import Screen
from ..view import Sender, ViewState


class Handlers:
    """A group of inbound message handlers. ``MESSAGES`` maps each message type it
    answers to the method that serves it."""

    MESSAGES: dict[str, str] = {}

    def __init__(self, view: ViewState, out: Sender, pool: AgentPool, screen: Screen) -> None:
        self.view = view
        self.out = out
        self.pool = pool
        self.screen = screen
