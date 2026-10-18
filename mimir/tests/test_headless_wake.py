"""A run that finishes with no window open still gets its turn.

The pool's consumer appends the instruction to the stored conversation, queues the
turn on its agent, and settles the run. The marker records delivery.
"""
import json
import os
import queue as _queue
import tempfile
import unittest

from mimir.client.ui.ws.worker.human import HumanBridge
from mimir.client.session import journal
from mimir.client.session.jobs_on_disk import scan_session
from mimir.client.session.conversation import Conversations
from mimir.client.session.store import SessionStore
from mimir.client.ui.ws.pool.pool import AgentPool
from mimir.tests.support.pools import pool_with
from mimir.tests.support.state import pin_state_dir


class _Worker:
    """Only what the headless consumer asks of a worker."""

    def __init__(self, session_id: str, *, busy: bool = False, prompt=None) -> None:
        self.human = HumanBridge(self)
        self.out_q = _queue.Queue()
        self.session_id = session_id
        self.active_session_id = None
        self.query_session_id = None
        self._busy = busy
        self.human._pending_prompt = prompt
        self.queries: list[dict] = []
        self.steers: list[str] = []

    def is_busy(self) -> bool:
        return self._busy

    def has_work_pending(self) -> bool:
        return self._busy

    def submit_query(self, text, history, session_id=None) -> None:
        self.queries.append({"text": text, "history": list(history),
                             "session_id": session_id})

    def submit_steer(self, text: str) -> None:
        self.steers.append(text)

    def drain(self) -> list:
        out = []
        while not self.out_q.empty():
            out.append(self.out_q.get_nowait())
        return out


class _HeadlessCase(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        pin_state_dir(self, self._tmp.name)
        self.addCleanup(self._tmp.cleanup)
        self.store = SessionStore()

    def _pool(self, workers: dict | None = None) -> AgentPool:
        return pool_with(Conversations(self.store), workers)

    def _session(self, question: str = "launch the run"):
        s = self.store.new_session()
        s.llm_history = [{"role": "user", "content": question}]
        s.llm_history_full = [{"role": "user", "content": question}]
        self.store.save_session(s)
        return s

    def _baseline(self, session_id: str) -> None:
        path = os.path.join(self._tmp.name, "sessions", session_id)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, ".wake_baseline"), "w") as fh:
            fh.write("0")

    def _ended_job(self, session_id: str, job_key: str, exit_code: int = 0) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "watch_first_cost.sh",
                       "pid": 1, "pid_starttime": 1}, fh)
        with open(os.path.join(job_dir, "exit_code"), "w") as fh:
            fh.write(str(exit_code))
        self._baseline(session_id)

    def _complete(self, session_id: str, job_key: str = "j1", **fields) -> dict:
        ev = {"type": "job_complete", "job_key": job_key, "server": "bash",
              "kind": "shell-command", "state": "done", "session_id": session_id,
              "summary": {"output": "PREMIER COUT APPARU\n0\t7.2151675"},
              "status_op": {"tool": "bash_job", "args": {"job_key": job_key}}}
        ev.update(fields)
        return ev

    def _checkin(self, session_id: str, job_key: str = "j1") -> dict:
        return {"type": "job_checkin", "session_id": session_id,
                "jobs": [{"job_key": job_key, "kind": "shell-command",
                          "state": "running", "phase": "", "percent": None}]}


class AWakeStartsATurnTests(_HeadlessCase):
    def test_a_finished_run_is_handed_to_its_conversation(self):
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._complete(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertEqual(w.queries[0]["session_id"], s.id)
        self.assertIn("finished", w.queries[0]["text"])
        # What the run recorded travels with it: a wake that drops the payload makes the
        # agent go and ask for what it was already told.
        self.assertIn("7.2151675", w.queries[0]["text"])

    def test_the_turn_is_given_the_conversation_it_belongs_to(self):
        # Not the history of whatever was last on screen: the run belongs to the
        # conversation that launched it, and that conversation lives on disk.
        s = self._session("validate J0 when the first cost appears")
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._complete(s.id))

        handed = w.queries[0]["history"]
        self.assertEqual(handed[0]["content"], "validate J0 when the first cost appears")
        self.assertIn("finished", handed[-1]["content"])

    def test_the_wake_is_written_into_the_stored_conversation(self):
        # The turn has to be answerable later by a process that never saw this event,
        # and `_commit_turn` writes its answer against the history stored here.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.wakes.consume(self._complete(s.id))

        stored = self.store.load_session(s.id)
        self.assertEqual(stored.llm_history[-1]["role"], "user")
        self.assertIn("finished", stored.llm_history[-1]["content"])
        self.assertIn("finished", stored.llm_history_full[-1]["content"])

    def test_the_user_finds_the_wake_on_their_return(self):
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.wakes.consume(self._complete(s.id))

        stored = self.store.load_session(s.id)
        self.assertTrue(stored.display_messages[-1]["text"].startswith("🔔"))

    def test_the_wake_is_journaled_where_the_turn_begins(self):
        # A window opening later replays the journal. Without this it finds a turn with
        # no question in front of it.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.wakes.consume(self._complete(s.id))

        events = journal.read_since(s.id, 0)[0]
        self.assertEqual([e["type"] for e in events], ["job_wake"])
        self.assertEqual(events[0]["job"], "j1")


class ItSettlesWhatItDeliversTests(_HeadlessCase):
    def test_a_delivered_run_is_left_out_of_the_next_scan(self):
        s = self._session()
        self._ended_job(s.id, "j1")
        pool = self._pool({s.id: _Worker(s.id)})
        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])

        pool.wakes.consume(self._complete(s.id))
        self.assertEqual(scan_session(s.id), [])

    def test_a_run_it_could_not_deliver_stays_owed(self):
        # No agent for the conversation: nothing can run the turn. The run must remain
        # unsettled so the next worker built, or the next connection, announces it.
        s = self._session()
        self._ended_job(s.id, "j1")
        pool = self._pool()
        pool.wakes.consume(self._complete(s.id))

        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])

    def test_a_steered_run_stays_owed(self):
        # A steer is only known to have been read when the loop says so. Re-telling a
        # run is recoverable; losing one is not.
        s = self._session()
        self._ended_job(s.id, "j1")
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._complete(s.id))

        self.assertEqual(len(w.steers), 1)
        self.assertEqual(w.queries, [])
        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])


class WhoseJobItIsTests(_HeadlessCase):
    def test_an_attached_client_does_not_start_a_second_turn(self):
        # Two consumers starting a turn for one finished run is the duplicate no marker
        # can catch. There is one consumer: the client is told, it does not act.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        w.out_q.put(self._complete(s.id))
        pool.bus.subscribe()
        pool.bus.pump_once()

        self.assertEqual(len(w.queries), 1)

    def test_the_pump_hands_it_over_when_nobody_is_attached(self):
        # The wiring, end to end: a watcher puts the event on the worker's queue and
        # the turn starts, with no socket anywhere in the path.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        w.out_q.put(self._complete(s.id))
        pool.bus.pump_once()

        self.assertEqual(len(w.queries), 1)

    def test_a_turn_still_starts_for_a_conversation_that_is_not_on_screen(self):
        s = self._session()
        other = self._session("something else")
        w = _Worker(s.id)
        pool = self._pool({s.id: w, other.id: _Worker(other.id)})
        pool.active_session_id = other.id
        pool.wakes.consume(self._complete(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertEqual(w.queries[0]["session_id"], s.id)

    def test_an_event_with_no_conversation_is_dropped(self):
        pool = self._pool({"s1": _Worker("s1")})
        pool.wakes.consume(self._complete(None))
        self.assertEqual(pool._workers["s1"].queries, [])


class BulletinsTests(_HeadlessCase):
    def test_a_check_in_starts_its_short_turn(self):
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._checkin(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertIn("Background check-in", w.queries[0]["text"])

    def test_a_check_in_leaves_no_bubble(self):
        # A conversation checked on twenty times must not reopen on twenty blocks of an
        # instruction addressed to the model.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.wakes.consume(self._checkin(s.id))

        stored = self.store.load_session(s.id)
        self.assertEqual(stored.display_messages, [])

    def test_a_check_in_never_interrupts_a_turn(self):
        # A bulletin carries "nothing to report". Steering that into a turn makes the
        # agent answer about the job instead of the work it was asked for.
        s = self._session()
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._checkin(s.id))

        self.assertEqual(w.queries, [])
        self.assertEqual(w.steers, [])

    def test_a_check_in_never_interrupts_a_card(self):
        s = self._session()
        w = _Worker(s.id, prompt={"type": "approval"})
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._checkin(s.id))

        self.assertEqual(w.queries, [])


class TheNightThatWasLostTests(_HeadlessCase):
    """The sequence this consumer exists for, start to finish.

    A conversation answers, its window closes, and three minutes later the run it was
    waiting on finishes. Everything up to the journal already worked; what did not
    happen was the turn.
    """

    def test_a_run_that_ends_after_the_window_closes_resumes_the_conversation(self):
        s = self._session("watch for the first cost, then validate J0")
        w = _Worker(s.id)
        pool = self._pool({s.id: w})

        # The last answer of the evening, with a client still attached.
        sub = pool.bus.subscribe()
        w.out_q.put({"type": "answer", "text": "the watcher is polling",
                     "session_id": s.id})
        pool.bus.pump_once()
        self.assertEqual(w.queries, [])

        # The window closes.
        sub.close()
        self.assertEqual(pool.bus.attached(), 0)

        # The watcher, still on the worker's loop, reports the run.
        self._ended_job(s.id, "j1")
        w.out_q.put(self._complete(s.id))
        pool.bus.pump_once()

        # The turn starts there and then, rather than sixteen hours later.
        self.assertEqual(len(w.queries), 1)
        self.assertIn("7.2151675", w.queries[0]["text"])
        self.assertEqual(scan_session(s.id), [])

    def test_the_whole_turn_happens_with_nothing_attached(self):
        # Asked, answered and written down, with no socket in the path at any point:
        # `consume_durable_event` submits it and `_commit_turn` — the committer that
        # already existed for this case — writes the answer back.
        s = self._session("watch for the first cost, then validate J0")
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        self._ended_job(s.id, "j1")
        w.out_q.put(self._complete(s.id))
        pool.bus.pump_once()

        wake = w.queries[0]["text"]
        full = list(w.queries[0]["history"]) + [
            {"role": "assistant", "content": "J0 = 7.2151675, matches the reference"}]
        w.out_q.put({"type": "answer", "text": "J0 validated", "session_id": s.id,
                     "_full": full, "_turn_start": len(full) - 1,
                     "_submitted_len": len(w.queries[0]["history"]),
                     "_context_mode": "full"})
        pool.bus.pump_once()

        stored = self.store.load_session(s.id)
        self.assertEqual(stored.llm_history, full)
        self.assertIn(wake, [m.get("content") for m in stored.llm_history_full])
        self.assertEqual(stored.llm_history_full[-1]["content"],
                         "J0 = 7.2151675, matches the reference")

    def test_the_process_stays_up_while_the_wake_is_owed(self):
        # The other half of the same failure: with nothing left to do, the server
        # stopped two hours after the job ended — owing a turn it had never started.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        w.out_q.put({"type": "answer", "text": "waiting", "session_id": s.id})
        pool.bus.pump_once()
        self.assertTrue(pool.lifecycle.idle_report()["idle"])

        self._ended_job(s.id, "j1")
        report = pool.lifecycle.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("nobody has taken in" in r for r in report["reasons"]),
                        report["reasons"])


if __name__ == "__main__":
    unittest.main()
