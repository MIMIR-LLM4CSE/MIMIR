"""One conversation, one copy: what writes a turn into it, with or without a socket.

The pump commits every answer into its conversation before any socket sees it, and the
connection showing the conversation renders what is already written. Pinned here: the
commit itself (``Conversation.commit_answer``), the registry that hands every asker the
same object (``Conversations``), and the pump doing the writing whether or not anyone
is attached.
"""
import queue as _queue
import tempfile
import unittest
from unittest import mock

from mimir.client.session import store as session_store
from mimir.client.session.conversation import Conversation, Conversations, turn_messages
from mimir.client.session.store import SessionStore, StaleSessionWrite
from mimir.client.ui.ws import transcript_log
from mimir.client.ui.ws.human_bridge import HumanBridge
from mimir.client.ui.ws.ws_worker import _AgentWorker


def _bare_worker(session_id: str) -> _AgentWorker:
    w = object.__new__(_AgentWorker)
    w.human = HumanBridge(w)
    w.out_q = _queue.Queue()
    w.session_id = session_id
    w._query_session_id = None
    w._agent = None
    w._current_task = None
    w._query_q = _queue.Queue()
    return w


class _CommitCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        for target, attr in ((session_store, "STATE_DIR"),
                             (transcript_log, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.store = SessionStore()
        self.conversations = Conversations(self.store)

    def _session(self, **fields):
        s = self.store.new_session()
        for k, v in fields.items():
            setattr(s, k, v)
        self.store.save_session(s)
        return s

    def _commit(self, session_id, ev, extras):
        conversation = self.conversations.get(session_id)
        wrote_full = conversation.commit_answer(ev, extras)
        self.assertTrue(self.conversations.save(conversation))
        return wrote_full


class CommitTests(_CommitCase):
    def test_an_answer_lands_in_the_session_file(self):
        s = self._session(llm_history=[{"role": "user", "content": "q"}],
                          llm_history_full=[{"role": "user", "content": "q"}])
        full = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "done"}]

        wrote_full = self._commit(
            s.id, {"type": "answer", "text": "done", "session_id": s.id},
            {"_full": full, "_turn_start": 1, "_submitted_len": 1,
             "_context_mode": "full"})

        self.assertTrue(wrote_full)
        stored = self.store.load_session(s.id)
        self.assertEqual(stored.llm_history, full)
        self.assertEqual(stored.llm_history_full[-1]["content"], "done")
        self.assertEqual([m["text"] for m in stored.display_messages], ["done"])

    def test_a_deferral_is_stored_as_the_card_to_put_back(self):
        s = self._session()
        self._commit(s.id, {"type": "answer", "text": "waiting", "session_id": s.id},
                     {"_deferred": {"kind": "calls", "prompt": {"type": "approval"}}})
        stored = self.store.load_session(s.id)
        self.assertEqual(stored.pending_interaction["kind"], "calls")

    def test_a_non_full_context_mode_records_the_answer_alone(self):
        # Nothing to place and no window to restore: the answer is the honest record.
        s = self._session()
        full = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "done"}]
        wrote_full = self._commit(
            s.id, {"type": "answer", "text": "done", "session_id": s.id},
            {"_full": full, "_turn_start": 1, "_submitted_len": 1,
             "_context_mode": "compact"})

        self.assertFalse(wrote_full)
        stored = self.store.load_session(s.id)
        self.assertEqual([m["content"] for m in stored.llm_history], ["done"])

    def test_no_transcript_records_the_answer_alone_too(self):
        s = self._session()
        self.assertFalse(self._commit(
            s.id, {"type": "answer", "text": "done", "session_id": s.id}, {}))
        self.assertEqual(
            [m["content"] for m in self.store.load_session(s.id).llm_history], ["done"])

    def test_an_empty_answer_leaves_the_chat_alone(self):
        # A silent turn adds no bubble; the history still closes.
        s = self._session()
        self._commit(s.id, {"type": "answer", "text": "", "session_id": s.id}, {})
        stored = self.store.load_session(s.id)
        self.assertEqual(stored.display_messages, [])
        self.assertEqual(stored.llm_history[-1]["role"], "assistant")


class ConversationsTests(_CommitCase):
    def test_every_asker_gets_the_same_object(self):
        # Two copies of one conversation is what had to be arbitrated before.
        s = self._session()
        self.assertIs(self.conversations.get(s.id), self.conversations.get(s.id))

    def test_a_conversation_that_is_not_there_is_none(self):
        self.assertIsNone(self.conversations.get("never-existed"))
        self.assertIsNone(self.conversations.get(""))

    def test_a_new_conversation_is_not_on_disk_until_saved(self):
        conversation = self.conversations.new()
        self.assertFalse(self.store.session_exists(conversation.id))
        self.assertEqual([m.id for m in self.conversations.unsaved()], [conversation.id])
        conversation.add_user_message("hello", show=True)
        self.assertTrue(self.conversations.save(conversation))
        self.assertTrue(self.store.session_exists(conversation.id))
        self.assertEqual(self.store.load_session(conversation.id).title, "hello")
        self.assertEqual(self.conversations.unsaved(), [])

    def test_deleting_forgets_it_too(self):
        s = self._session()
        self.conversations.get(s.id)
        self.conversations.delete(s.id)
        self.assertIsNone(self.conversations.get(s.id))

    def test_a_resume_starts_from_the_untrimmed_record(self):
        s = self._session(llm_history=[{"role": "user", "content": "tail"}],
                          llm_history_full=[{"role": "user", "content": "head"},
                                            {"role": "assistant", "content": "a"},
                                            {"role": "user", "content": "tail"}])
        self.assertEqual(len(Conversation(self.store.load_session(s.id), saved=True).history), 3)

    def test_unread_steering_moves_to_the_end_by_identity(self):
        conversation = self.conversations.new()
        conversation.add_user_message("same words")
        conversation.add_steer("same words")
        conversation.history.append({"role": "assistant", "content": "a"})
        conversation.take_unread_steer(["same words"])
        # The older, identical turn stays; only the live copy goes.
        self.assertEqual([m["content"] for m in conversation.history],
                         ["same words", "a"])
        self.assertEqual(conversation.live_steer, [])


class ThePumpCommitsEveryAnswerTests(_CommitCase):
    """One conversation object, so one writer: the pump, attached or not."""

    def _pool_with_bus(self, session_id: str):
        from mimir.tests._fake_pool import FakePool
        worker = _bare_worker(session_id)
        pool = FakePool(worker, workers={session_id: worker})
        pool.conversations = self.conversations
        pool.store = self.store
        return pool, worker

    def _answer(self, session_id: str) -> dict:
        return {"type": "answer", "text": "done", "session_id": session_id,
                "_full": [{"role": "assistant", "content": "done"}],
                "_turn_start": 0, "_submitted_len": 0, "_context_mode": "full"}

    def test_it_commits_with_nobody_attached(self):
        s = self._session()
        pool, worker = self._pool_with_bus(s.id)
        worker.out_q.put(self._answer(s.id))
        pool.bus.pump_once()
        self.assertEqual(
            [m["text"] for m in self.store.load_session(s.id).display_messages],
            ["done"])

    def test_it_commits_while_a_client_is_attached_too(self):
        # The connected view no longer writes the turn: it renders what this wrote.
        s = self._session()
        pool, worker = self._pool_with_bus(s.id)
        sub = pool.bus.subscribe()
        worker.out_q.put(self._answer(s.id))
        pool.bus.pump_once()
        self.assertEqual(
            [m["text"] for m in self.store.load_session(s.id).display_messages],
            ["done"])
        _ev, extras = sub.queue.get_nowait()
        self.assertEqual(extras["_committed"]["title"], s.id)

    def test_a_cancelled_turn_closes_the_history_as_well(self):
        # It was committed when a view held the conversation and dropped when none
        # did; the user's message then stood unanswered in the history.
        s = self._session(llm_history=[{"role": "user", "content": "q"}])
        pool, worker = self._pool_with_bus(s.id)
        worker.out_q.put({**self._answer(s.id), "text": "[Cancelled]", "cancelled": True,
                          "_full": None})
        pool.bus.pump_once()
        self.assertEqual(self.store.load_session(s.id).llm_history[-1]["role"],
                         "assistant")

    def test_the_answer_is_committed_once_not_twice(self):
        s = self._session()
        pool, worker = self._pool_with_bus(s.id)
        worker.out_q.put(self._answer(s.id))
        pool.bus.pump_once()
        pool.bus.pump_once()          # a second tick has nothing left to see
        self.assertEqual(
            len(self.store.load_session(s.id).display_messages), 1)


class StaleWriteTests(_CommitCase):
    """A save built on a superseded copy is refused, not silently applied."""

    def test_a_second_writer_that_loaded_earlier_is_refused(self):
        s = self._session(title="first")
        mine = self.store.load_session(s.id)
        theirs = self.store.load_session(s.id)

        theirs.title = "theirs"
        self.store.save_session(theirs)          # moves the revision on

        mine.title = "mine"
        with self.assertRaises(StaleSessionWrite):
            self.store.save_session(mine)
        self.assertEqual(self.store.load_session(s.id).title, "theirs")

    def test_saving_the_same_object_repeatedly_is_fine(self):
        # The revision is carried back onto a successful save, so an owner that holds
        # one session and writes it as the turn progresses is never refused.
        s = self._session()
        for i in range(5):
            s.title = f"round {i}"
            self.store.save_session(s)
        self.assertEqual(self.store.load_session(s.id).title, "round 4")

    def test_a_failed_write_does_not_consume_the_revision(self):
        s = self._session()
        with mock.patch("json.dump", side_effect=OSError("disk full")):
            with self.assertRaises(OSError):
                self.store.save_session(s)
        s.title = "after the failure"
        self.store.save_session(s)               # must not be refused
        self.assertEqual(self.store.load_session(s.id).title, "after the failure")

    def test_a_refused_write_is_reported_rather_than_lost_quietly(self):
        s = self._session()
        conversation = self.conversations.get(s.id)
        self.store.save_session(self.store.load_session(s.id))   # someone else wrote

        with self.assertLogs("mimir.client.session.conversation", "WARNING") as logs:
            self.assertFalse(self.conversations.save(conversation))
        self.assertIn("would not save", "\n".join(logs.output))

    def test_a_session_file_written_before_revisions_existed_still_writes(self):
        # The migration case: a file with no ``rev`` key reads as 0 and the guard, which
        # only refuses a disk revision *ahead* of the copy in hand, lets it through.
        import json
        import os
        s = self._session(title="from an older build")
        path = os.path.join(self._tmp.name, "sessions", f"{s.id}.json")
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        data.pop("rev", None)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)

        loaded = self.store.load_session(s.id)
        self.assertEqual(loaded.rev, 0)
        loaded.title = "written by this build"
        self.store.save_session(loaded)
        self.assertEqual(self.store.load_session(s.id).title, "written by this build")


class TurnMessagesTests(unittest.TestCase):
    """Which slice of the transcript a turn added, when the boundary is unreliable."""

    def test_the_loops_boundary_wins(self):
        full = [{"i": 0}, {"i": 1}, {"i": 2}]
        self.assertEqual(turn_messages(full, 1, start=2), [{"i": 2}])

    def test_the_submitted_length_is_the_fallback(self):
        full = [{"i": 0}, {"i": 1}, {"i": 2}]
        self.assertEqual(turn_messages(full, 1, start=None), [{"i": 1}, {"i": 2}])

    def test_with_no_boundary_at_all_the_answer_alone_is_recorded(self):
        # Better a turn recorded by its answer than a record interleaved with a copy
        # of an older one.
        full = [{"i": 0}, {"i": 1}]
        self.assertEqual(turn_messages(full, None, start=None), [{"i": 1}])

    def test_a_boundary_past_the_end_falls_back_to_the_answer(self):
        full = [{"i": 0}, {"i": 1}]
        self.assertEqual(turn_messages(full, None, start=9), [{"i": 1}])

    def test_an_empty_transcript_yields_nothing(self):
        self.assertEqual(turn_messages([], None, start=None), [])


if __name__ == "__main__":
    unittest.main()
