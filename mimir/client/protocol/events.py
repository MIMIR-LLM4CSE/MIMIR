"""The events a turn produces, and the sets of them that decide where each one goes.

Every event is a dict with a ``type``. The types listed here are the ones the session
layer routes on: what ends a turn, what must reach a turn even with nobody watching,
what is kept for the replay only. Named once, so the producer and every consumer agree.
"""
from __future__ import annotations

# The event types that end a turn. An ``error`` is a conclusion as much as an ``answer``
# is — the turn is over either way — and conflating "concluded" with "succeeded" would
# leave a session that failed looking busy for ever.
CONCLUSIVE_EVENTS = frozenset({"answer", "error"})

# Events that describe a detached run rather than a turn. Everything else a worker
# queues — its tokens, its rows, its answer — is debris once the socket drawing it is
# gone; these two are the reason the pool consumes the bus at all. They have to reach a
# turn whether or not anyone is attached, so both consumers key off this one set.
DURABLE_EVENTS = frozenset({"job_complete", "job_checkin"})

# Recorded, never delivered live: ``assistant_text`` aggregates a streamed prose block a
# live client already has as deltas. It is what a replay shows of the prose.
REPLAY_ONLY_EVENTS = frozenset({"assistant_text"})

# What a finished turn carries on its ``answer`` event for the session layer, and only
# for it: the turn's history (``_full`` from ``_turn_start``), the length of the history
# it was submitted with, the context mode it ran in, its deferral record, and the
# steering it ended without reading. Taken off the event before it is journaled or sent
# (``ui/ws/pool/bus``): none of it is part of the wire protocol.
ANSWER_EXTRA_KEYS = ("_full", "_turn_start", "_deferred", "_submitted_len",
                     "_context_mode", "_unconsumed_steer")


def pop_answer_extras(ev: dict) -> dict:
    """Take the session-layer payload off an answer event, and return it."""
    if ev.get("type") != "answer":
        return {}
    return {k: ev.pop(k) for k in ANSWER_EXTRA_KEYS if k in ev}
