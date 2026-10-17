"""What the turn in flight, and the last one that finished, leave on the agent.

The loop writes it; the front-ends read it (context usage mid-turn, the messages a
finished turn produced) and set ``cancel_flag`` when the user presses Stop.
"""
from __future__ import annotations
from typing import Any

import threading
from dataclasses import dataclass, field


@dataclass
class TurnState:
    # Set by the user's Stop; the backend checks it on every streamed chunk.
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    # The running query's messages, system message first; None between turns.
    live_messages: list[dict[str, Any]] | None = None
    # The message that opened the running turn, so its start survives compaction.
    opening_message: dict[str, Any] | None = None
    # The last finished query's messages, system message excluded.
    last_full_messages: list[dict[str, Any]] = field(default_factory=list)
    # Where the last finished turn's own messages begin in ``last_full_messages``, or
    # None when its opening message did not survive compaction.
    last_turn_start: int | None = None
    # Read-only tool results of this query, by call; emptied when a query starts.
    tool_cache: dict[str, Any] = field(default_factory=dict)
    # Questions set aside for the user while the turn ran on, and the turn they parked.
    deferred_prompts: list[dict[str, Any]] = field(default_factory=list)
    deferred_turn: dict[str, Any] | None = None
