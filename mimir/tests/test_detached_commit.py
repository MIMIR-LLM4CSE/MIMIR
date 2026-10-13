"""A turn that finishes while nobody is attached still gets written down.

Recording a turn's result from inside a connection would mean a run that finishes while
the user is away costs its tokens and vanishes. These tests pin the committer: it takes
a store, a session id and the answer, has no socket and no session object, and the pump
calls it when ``bus.attached()`` is zero.
"""
import queue as _queue
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws import session_store, transcript_log
from mimir.client.ui.ws.session_store import SessionStore, StaleSessionWrite
from mimir.client.ui.ws.turn_commit import commit_answer, turn_messages
from mimir.client.ui.ws.ws_worker import _AgentWorker


def _bare_worker(session_id: str) -> _AgentWorker:
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w.session_id = session_id
    w._query_session_id = None
    return w


class _FakePool:
    def __init__(self, workers: dict) -> None:
        self._workers = workers

    def items(self):
        return list(self._workers.items())


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

    def _session(self, **fields):
        s = self.store.new_session()
        for k, v in fields.items():
            setattr(s, k, v)
        self.store.save_session(s)
        return s


class CommitWithNoSocketTests(_CommitCase):
    def test_an_answer_lands_in_the_session_file(self):
        s = self._session(llm_history=[{"role": "user", "content": "q"}],
                          llm_history_full=[{"role": "user", "content": "q"}])
        full = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "done"}]

        result = commit_answer(
            self.store, s.id,
            {"type": "answer", "text": "done", "session_id": s.id},
            {"_full": full, "_turn_start": 1},
            submitted_len=1, context_mode="full")

        self.assertIsNotNone(result)
        self.assertTrue(result.wrote_full_history)
        stored = self.store.load_session(s.id)
        self.assertEqual(stored.llm_history, full)
        self.assertEqual(stored.llm_history_full[-1]["content"], "done")
        self.assertEqual([m["text"] for m in stored.display_messages], ["done"])

    def test_a_deferral_is_stored_as_the_card_to_put_back(self):
        s = self._session()
        result = commit_answer(
            self.store, s.id,
            {"type": "answer", "text": "waiting", "session_id": s.id},
            {"_deferred": {"kind": "calls", "prompt": {"type": "approval"}}},
            context_mode="full")

        self.assertEqual(result.deferred["kind"], "calls")
        stored = self.store.load_session(s.id)
        self.assertEqual(stored.pending_interaction["kind"], "calls")

    def test_a_non_full_context_mode_records_the_answer_alone(self):
        # Nothing to place and no window to restore: the answer is the honest record.
        s = self._session()
        full = [{"role": "user", "content": "q"},
                {"role": "assistant", "content": "done"}]
        result = commit_answer(
            self.store, s.id,
            {"type": "answer", "text": "done", "session_id": s.id},
            {"_full": full, "_turn_start": 1},
            submitted_len=1, context_mode="compact")

        self.assertFalse(result.wrote_full_history)
        stored = self.store.load_session(s.id)
        self.assertEqual([m["content"] for m in stored.llm_history], ["done"])

    def test_no_transcript_records_the_answer_alone_too(self):
        s = self._session()
        result = commit_answer(
            self.store, s.id,
            {"type": "answer", "text": "done", "session_id": s.id},
            {}, context_mode="full")
        self.assertFalse(result.wrote_full_history)
        self.assertEqual(
            [m["content"] for m in self.store.load_session(s.id).llm_history], ["done"])

    def test_a_session_that_is_gone_is_not_an_error(self):
        self.assertIsNone(commit_answer(self.store, "never-existed",
                                        {"type": "answer", "text": "x"}, {}))

    def test_no_session_id_writes_nothing(self):
        self.assertIsNone(commit_answer(self.store, "", {"type": "answer"}, {}))

    def test_an_empty_answer_leaves_the_chat_alone(self):
        # A cancelled or silent turn adds no bubble; the history still closes.
        s = self._session()
        commit_answer(self.store, s.id,
                      {"type": "answer", "text": "", "session_id": s.id}, {})
        self.assertEqual(self.store.load_session(s.id).display_messages, [])


class ThePumpCommitsOnlyWhenNobodyIsAttachedTests(_CommitCase):
    """Two writers of one file lose history silently, so only one may be live."""

    def _pool_with_bus(self, session_id: str):
        from mimir.client.ui.ws.ws_pool import _AgentPool
        pool = object.__new__(_AgentPool)
        pool.store = self.store
        worker = _bare_worker(session_id)
        pool._workers = {session_id: worker}
        pool.items = lambda: [(session_id, worker)]
        from mimir.client.ui.ws.event_bus import _EventBus
        pool.bus = _EventBus(pool, commit=pool._commit_turn)
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

    def test_it_stays_out_of_the_way_while_a_client_is_attached(self):
        # The connected session writes this itself, as it always has.
        s = self._session()
        pool, worker = self._pool_with_bus(s.id)
        pool.bus.subscribe()
        worker.out_q.put(self._answer(s.id))
        pool.bus.pump_once()
        self.assertEqual(self.store.load_session(s.id).display_messages, [])

    def test_a_cancelled_turn_is_not_committed(self):
        s = self._session()
        pool, worker = self._pool_with_bus(s.id)
        worker.out_q.put({**self._answer(s.id), "cancelled": True})
        pool.bus.pump_once()
        self.assertEqual(self.store.load_session(s.id).display_messages, [])

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

    def test_the_committer_reports_a_refused_write_rather_than_losing_it_quietly(self):
        s = self._session()
        stale = self.store.load_session(s.id)
        self.store.save_session(self.store.load_session(s.id))

        with mock.patch.object(SessionStore, "load_session", return_value=stale):
            with self.assertLogs("mimir.client.ui.ws.turn_commit", "WARNING") as logs:
                result = commit_answer(
                    self.store, s.id,
                    {"type": "answer", "text": "done", "session_id": s.id}, {})
        self.assertIsNone(result)
        self.assertIn("written by someone else", "\n".join(logs.output))

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
