"""One conversation in memory: the one copy every writer goes through.

A conversation used to have two writers. The connection showing it held its history in
memory and saved that; the pool, when nobody was connected, loaded the stored copy,
wrote a turn into it and saved that. Two copies of one conversation had to be
arbitrated — a claim on every answer and every finished run, a grace window during
which the socket had first refusal, a reload when the pool had won, a revision check
on save for when the arbitration failed — and each of those was a place a turn could
be recorded twice or lost.

Now there is one copy. :class:`Conversations` hands out the same :class:`Conversation`
object to whoever asks for a session id, the connection and the pool alike, and every
change goes through it and is saved from it. They all run on the server's one event
loop, so nothing about sharing it needs a lock.
"""
from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from ..protocol.messages import reconcile_tool_pairs
from ..query_engine.history import carries_compaction_summary
from .store import FullSession, SessionMeta, SessionStore

logger = logging.getLogger(__name__)


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


def _resume_window(record: FullSession) -> list[dict]:
    """The window the model resumes on, from what was stored.

    A stored window that carries a compaction summary *is* the conversation: the
    handoff note stands in for everything that was cut, at a fraction of the tokens.
    Reloading the untrimmed record over it threw that work away and handed the model a
    context that was full before the user had typed anything. Absent a summary the
    window is only the tail a front-trim left behind, so the untrimmed record is the
    faithful resume: its prefix was never summarized anywhere.

    Each message is copied, not just the list: the budgeting pass rewrites messages in
    place (force-fit rewrites ``content``, the argument digest ``tool_calls``), and
    sharing the dicts made those writes reach into the untrimmed record. Repaired, since
    a stored window can start on an orphaned tool result or end on a call whose result
    never landed, and strict tokenizers reject both.
    """
    full = record.llm_history_full or record.llm_history
    resumed = (record.llm_history if carries_compaction_summary(record.llm_history)
               else full)
    window = [dict(m) for m in resumed]
    try:
        return reconcile_tool_pairs(window)
    except Exception:
        logger.warning("conversation %s: its stored window could not be repaired",
                       record.id, exc_info=True)
        return window


class Conversation:
    """A conversation: what the model works on, what the user sees, what is owed.

    ``history`` is the working window — trimmed and compacted to fit the budget.
    ``history_full`` is the same conversation never trimmed, the record a resume
    reloads. ``display`` is the transcript the user sees. ``submitted_len`` is how much
    of ``history`` the running turn inherited, which is what tells its own messages from
    the prefix when it answers.

    The rest is what the conversation is owed between turns, kept here because it
    belongs to the conversation and not to whichever process or window is looking:
    finished runs not yet handed to a turn, the check-in held back while it worked, and
    the history entries of steering typed into the turn now running.
    """

    def __init__(self, record: FullSession, *, saved: bool) -> None:
        self.record = record
        self.saved = saved
        self.history: list[dict] = _resume_window(record) if saved else []
        if not record.llm_history_full:
            # The untrimmed record is always the archive. Sessions saved before it
            # existed only have the trimmed window, which is still the best they have.
            record.llm_history_full = list(record.llm_history)
        self.submitted_len = len(self.history)
        # Finished runs whose wake no turn has taken in yet, each remembering whether
        # the user has already been shown it: ``{"ev": job_complete, "told": bool}``.
        self.pending_wakes: list[dict] = []
        # The last bulletin this conversation was not interrupted with.
        self.held_checkin: dict | None = None
        # The history entries written for steering the running turn was sent, held by
        # identity: a steer the run never read is moved to the end of the history,
        # where the turn that answers it starts, and matching its text instead could
        # move an older, identical user turn.
        self.live_steer: list[dict] = []

    # ── What it is ────────────────────────────────────────────────────────────

    @property
    def id(self) -> str:
        return self.record.id

    @property
    def title(self) -> str:
        return self.record.title or ""

    @property
    def history_full(self) -> list[dict]:
        return self.record.llm_history_full

    @property
    def display(self) -> list[dict]:
        return self.record.display_messages

    @display.setter
    def display(self, messages: list[dict]) -> None:
        self.record.display_messages = messages

    @property
    def pending_interaction(self) -> dict | None:
        return self.record.pending_interaction

    @pending_interaction.setter
    def pending_interaction(self, value: dict | None) -> None:
        self.record.pending_interaction = value

    def meta(self) -> SessionMeta:
        return self.record.meta()

    # ── Changing it ───────────────────────────────────────────────────────────

    def add_user_message(self, text: str, *, show: bool = False) -> None:
        """A turn's opening message: to the window and the record, and — when *show* —
        as a bubble. A wake or a check-in is addressed to the model, not to a reader;
        the bubble that announces it is the caller's to write."""
        self.history.append({"role": "user", "content": text})
        self.history_full.append({"role": "user", "content": text})
        if show:
            self.display.append({"role": "user", "kind": "text", "text": text})
        self.submitted_len = len(self.history)

    def add_steer(self, text: str) -> None:
        """Steering typed into the running turn.

        Not added to ``history_full``: it comes back inside the turn's own messages
        when the answer lands, and recording it here too would double it.
        """
        entry = {"role": "user", "content": text}
        self.history.append(entry)
        self.live_steer.append(entry)
        self.display.append({"role": "user", "kind": "text", "text": text})

    def take_unread_steer(self, texts: list[str]) -> None:
        """Drop the live copies of the steering the finished turn never read.

        They were written in the middle of the history, where they were typed. The turn
        that answers them starts at the end, and left in place as well they read to the
        model as said twice. Matched among this turn's own entries only, which keeps the
        match off an older, identical user turn. Called on every answer: it is also
        where the finished turn's steering stops being live.
        """
        stale, self.live_steer = self.live_steer, []
        wanted = set(texts)
        unread = [e for e in stale if e.get("content") in wanted]
        if unread:
            self.history = [m for m in self.history if not any(m is e for e in unread)]

    def note(self, text: str) -> None:
        """A line this layer writes into the transcript itself (a run finished, the
        process is stopping) — never sent by the client, so never in its copy."""
        self.display.append({"role": "system", "kind": "text", "text": text})

    def commit_answer(self, ev: dict, extras: dict) -> bool:
        """Write a finished turn into the conversation. True when its full transcript
        was kept rather than the answer alone.

        In full-context mode the structured transcript is kept — tool calls, results
        and answer, chain-of-thought stripped — so the model recalls the tools it ran.
        Only what the turn itself produced goes to the record: the loop may have
        trimmed or compacted the prefix it inherited, and that prefix is exactly what
        the untrimmed record exists to hold. Copied message by message, since a later
        turn's budgeting rewrites messages in place.

        *extras* are the private keys the worker put on the answer: the transcript,
        where the turn began, how much it was handed, the mode of the agent that ran
        it (not of whichever is on screen), and a deferred interaction.
        """
        full = extras.get("_full")
        context_mode = extras.get("_context_mode") or "full"
        submitted = extras.get("_submitted_len")
        if not isinstance(submitted, int):
            submitted = self.submitted_len
        if full is not None and context_mode == "full":
            added = turn_messages(full, submitted, start=extras.get("_turn_start"))
            self.history_full.extend(dict(m) for m in added)
            self.history = list(full)
            wrote_full = True
        else:
            # No transcript to place, or a mode that does not keep one: the answer
            # alone is the honest record of the turn.
            answer = {"role": "assistant", "content": ev.get("text", "")}
            self.history.append(answer)
            self.history_full.append(dict(answer))
            wrote_full = False
        if ev.get("text"):
            self.display.append({"role": "agent", "kind": "text", "text": ev["text"]})
        if extras.get("_deferred"):
            self.pending_interaction = extras["_deferred"]
        return wrote_full

    def to_record(self) -> FullSession:
        """The record to save: the window as it now stands, the title if it has none."""
        record = self.record
        record.llm_history = list(self.history)
        if not record.title:
            # From the first thing the user said.
            for msg in record.display_messages:
                if msg.get("role") == "user":
                    record.title = (msg.get("text") or msg.get("content", ""))[:60].strip()
                    break
        record.preview = record.title[:80]
        record.updated_at = datetime.now(timezone.utc).isoformat()
        return record


class Conversations:
    """Every conversation of this process, one object each.

    The one thing that loads and saves them. A conversation is loaded the first time
    it is asked for and kept: the next asker must get the same object, or there are two
    copies again.
    """

    def __init__(self, store: SessionStore | None = None) -> None:
        self.store = store or SessionStore()
        self._open: dict[str, Conversation] = {}

    def new(self, session_id: str | None = None) -> Conversation:
        """A conversation not on disk yet. It is saved once it has something in it:
        saving on creation left an empty "New session" behind every reconnect."""
        record = self.store.new_session()
        if session_id:
            record.id = session_id
        conversation = Conversation(record, saved=False)
        self._open[conversation.id] = conversation
        return conversation

    def get(self, session_id: str | None) -> Conversation | None:
        """The conversation *session_id*, or None if there is no such conversation."""
        if not session_id:
            return None
        conversation = self._open.get(session_id)
        if conversation is not None:
            return conversation
        try:
            record = self.store.load_session(session_id)
        except FileNotFoundError:
            return None
        except Exception:
            logger.warning("conversation %s could not be loaded", session_id,
                           exc_info=True)
            return None
        conversation = Conversation(record, saved=True)
        self._open[session_id] = conversation
        return conversation

    def save(self, conversation: Conversation) -> bool:
        """Write *conversation* down. False, logged, when the store would not have it:
        a caller on the pump serves every conversation and must not be stopped by one."""
        try:
            self.store.save_session(conversation.to_record())
        except Exception:
            logger.warning("conversation %s would not save", conversation.id,
                           exc_info=True)
            return False
        conversation.saved = True
        return True

    def unsaved(self) -> list[SessionMeta]:
        """The conversations that exist only here yet, for a listing to show."""
        return [c.meta() for c in self._open.values() if not c.saved]

    def forget(self, session_id: str) -> None:
        """Drop a conversation from memory (and only from memory)."""
        self._open.pop(session_id, None)

    def delete(self, session_id: str) -> None:
        self.forget(session_id)
        self.store.delete_session(session_id)
