"""Writing a conversation: a finished turn, a submitted one, a save.

Every write goes through the pool, so a conversation has one writer whichever
connection or wake produced the turn."""
from __future__ import annotations

import logging
from typing import Callable

from ..runtime import todo_file_for_session
from ..worker.worker import AgentWorker
from ....context.todo import load_todo_deps, load_todo_items
from ....session.conversation import Conversation, Conversations

logger = logging.getLogger(__name__)


class Writer:
    """The one writer of every conversation of the pool.

    *worker_of* finds a conversation's live worker, whose agent state a save records;
    *after_turn* carries a conversation on once its turn is written
    (``WakeRouter.after_turn``).
    """

    def __init__(self, conversations: Conversations,
                 worker_of: Callable[[str | None], AgentWorker | None],
                 after_turn: Callable[[Conversation, dict], list[str]]) -> None:
        self.conversations = conversations
        self._worker_of = worker_of
        self._after_turn = after_turn

    def commit_turn(self, ev: dict, extras: dict) -> dict:
        """Write a finished turn into its conversation, and carry on from there.

        Called by the pump for every answer, before any socket sees it, whether or not
        one is attached: there is one conversation object, so there is one writer, and
        a connection showing the conversation renders what is already written. Writing
        it down is half of a turn landing; the other half is carrying on
        (``WakeRouter.after_turn``), and a process that did only the first half would
        answer the step that finished and stop there.

        Returns what a connection needs to say about it: the conversation's title, the
        interaction it is now waiting on, and the steering that was given a turn.
        """
        session_id = ev.get("session_id")
        conversation = self.conversations.get(session_id)
        if conversation is None:
            logger.warning("commit: the answer of session %s has no conversation to go "
                           "to", session_id)
            return {}
        conversation.commit_answer(ev, extras)
        self.save_conversation(conversation)
        try:
            steering = self._after_turn(conversation, extras)
        except Exception:
            logger.warning("session %s could not be carried on after its turn",
                           session_id, exc_info=True)
            steering = []
        return {"title": conversation.title or session_id,
                "deferred": extras.get("_deferred"),
                "steer_turn": steering}

    def submit_turn(self, conversation: Conversation, worker: AgentWorker,
                    text: str) -> bool:
        """Open a turn of *conversation* with *text* and queue it on its own agent.

        The message is written down before the turn is queued, so a reconnect while it
        runs reloads it; when it cannot be, nothing is queued and the message is taken
        back out. Its answer comes back through :meth:`commit_turn`.
        """
        conversation.add_user_message(text)
        if not self.save_conversation(conversation):
            conversation.history.pop()
            conversation.history_full.pop()
            logger.warning("no turn started for session %s: it would not save",
                           conversation.id)
            return False
        worker.submit_query(text, list(conversation.history), session_id=conversation.id)
        return True

    def save_conversation(self, conversation: Conversation) -> bool:
        """Write *conversation* down, with what its agent and its checklist hold now.

        The agent's carried context and mode come from its live worker; with none, the
        conversation keeps what it was saved with. The checklist is read off disk,
        where the agent keeps it.
        """
        record = conversation.record
        worker = self._worker_of(conversation.id)
        if worker is not None:
            try:
                record.carry_context = worker.export_agent_state().get("carry_context", {})
                record.context_mode = worker.setting("context_mode", record.context_mode)
            except Exception:
                logger.warning("session %s: its agent's state could not be read",
                               conversation.id, exc_info=True)
        todo_file = todo_file_for_session(conversation.id)
        try:
            record.todos = load_todo_items(todo_file)
            record.todo_deps = load_todo_deps(todo_file)
        except Exception:
            logger.warning("session %s: its checklist could not be read",
                           conversation.id, exc_info=True)
        return self.conversations.save(conversation)
