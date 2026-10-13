"""Writing a finished turn into its session file, with or without a socket.

No socket and no session object: everything it needs arrives as an argument — the
store, the session id, the answer event, the private extras the pump took off it, where
the turn began, and the context mode. That is what lets the pump call it with nobody
attached, which is the whole point. A turn's ``answer`` carries its history, and if the
only thing that unpacks it needs a client, a turn that finishes while nobody is looking
runs, costs its tokens and vanishes.

The announcing is deliberately elsewhere. Notifying the user, refreshing the session
list and flushing the wakes that piled up are things a *connection* does; a commit that
insisted on them could not run without one.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .session_store import StaleSessionWrite

logger = logging.getLogger(__name__)


@dataclass
class CommitResult:
    """What a commit did, for whoever wants to announce it."""

    session_id: str
    title: str
    deferred: dict | None
    wrote_full_history: bool


def turn_messages(full: list[dict], submitted: int | None = None,
                  start: Any = None) -> list[dict]:
    """The slice of *full* this turn produced — what the archive has yet to record.

    The boundary comes from the loop, which is the only place it is knowable. The
    length submitted is not an index into *full* once the in-turn budget pass has
    rewritten the list: it evicts old tool results, replaces the middle with a summary
    and then repairs the assistant↔tool pairing those break. Every such rewrite shifts
    the prefix, and a stale boundary re-archives whatever it shifted past or drops
    whatever it shifted over — cutting through an assistant↔tool pair on the way, which
    is how a record ends up holding tool results with no call in front of them.

    Falls back to the submitted length, and then to the answer alone, when the loop
    cannot place the boundary — a turn long enough to have its own opening message
    summarized away. Better a turn recorded by its answer than a record quietly
    interleaved with a copy of an older one.
    """
    if not isinstance(start, int):
        start = submitted
    if not isinstance(start, int):
        return full[-1:] if full else []
    added = full[start:] if len(full) > start else []
    return added or full[-1:]


def commit_answer(store, session_id: str, ev: dict, extras: dict | None = None, *,
                  submitted_len: int | None = None,
                  context_mode: str = "full") -> CommitResult | None:
    """Write *ev* — a finished turn's answer — into *session_id*'s stored session.

    Returns what it did, or None when there was nothing to write or the store would
    not have it. Never raises: it runs on the pump, which serves every conversation,
    and one session's unwritable file must not stop the rest from being recorded.

    *context_mode* is the mode of the agent that ran **this** turn, passed in rather
    than read off whichever worker is on screen: that is the wrong worker whenever the
    turn belongs to another conversation, which for a detached turn is always.
    """
    if not session_id:
        return None
    try:
        session = store.load_session(session_id)
    except Exception:
        logger.warning("commit: session %s could not be loaded", session_id,
                       exc_info=True)
        return None

    extras = extras or {}
    full = extras.get("_full")
    start = extras.get("_turn_start")

    if full is not None and context_mode == "full":
        added = turn_messages(full, submitted_len, start=start)
        session.llm_history_full.extend(dict(m) for m in added)
        session.llm_history = list(full)
        wrote_full = True
    else:
        # No transcript to place, or a mode that does not keep one: the answer alone is
        # the honest record of the turn.
        answer_msg = {"role": "assistant", "content": ev.get("text", "")}
        session.llm_history.append(answer_msg)
        session.llm_history_full.append(dict(answer_msg))
        wrote_full = False

    if ev.get("text"):
        session.display_messages.append(
            {"role": "agent", "kind": "text", "text": ev.get("text", "")})

    deferred = extras.get("_deferred")
    if deferred:
        session.pending_interaction = deferred

    try:
        store.save_session(session)
    except StaleSessionWrite:
        # Somebody wrote this session since it was loaded, so this copy's history is
        # older than what is on disk. Said out loud rather than swallowed: the turn
        # whose answer this was is the thing that goes missing.
        logger.warning("commit: session %s was written by someone else; this turn's "
                       "answer was not stored", session_id)
        return None
    except Exception:
        logger.warning("commit: session %s would not save", session_id, exc_info=True)
        return None

    return CommitResult(session_id=session_id, title=session.title or session_id,
                        deferred=deferred, wrote_full_history=wrote_full)
