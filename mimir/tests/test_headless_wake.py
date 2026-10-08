"""A run that finishes with no window open still gets its turn.

A watcher outlives every connection: it is a task on a worker's loop, and the pool
refuses to release a worker that holds one. So a detached run reports in whether or not
a panel is attached, and the pump journals what it says. What the journal cannot do is
*act* on it — and for a background run, acting on it is the whole point. A job that
finished at 15:39 with the window shut was recorded and never handed to the conversation
waiting on it: the agent resumed sixteen hours later, when somebody opened the panel and
asked where things were.

These tests pin the consumer that closes that gap. It belongs to the pool rather than to
a socket, it is handed the event by the pump, and it does the three things a wake needs:
appends the instruction to the stored conversation, queues the turn on that
conversation's own agent, and settles the run so no later scan re-announces it.

The companion invariant lives in :mod:`test_job_rearm` and :mod:`test_reattach_replay`:
the marker records *delivery*. A wake that nothing took in is re-announced on the next
scan, which is what makes the two paths safe to have at once.
"""
import json
import os
import queue as _queue
import tempfile
import unittest
from unittest import mock

from mimir.client.ui.ws import job_scan, session_store, transcript_log
from mimir.client.ui.ws.event_bus import _EventBus
from mimir.client.ui.ws.job_scan import scan_session
from mimir.client.ui.ws.session_store import SessionStore
from mimir.client.ui.ws.ws_pool import _AgentPool


class _Worker:
    """Only what the headless consumer asks of a worker."""

    def __init__(self, session_id: str, *, busy: bool = False, prompt=None) -> None:
        self.out_q = _queue.Queue()
        self.session_id = session_id
        self.active_session_id = None
        self._query_session_id = None
        self._busy = busy
        self._pending_prompt = prompt
        self.unattended_since = None
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
        for target, attr in ((session_store, "STATE_DIR"),
                             (transcript_log, "_MIMIR_DIR_WS"),
                             (job_scan, "_MIMIR_DIR_WS")):
            patcher = mock.patch.object(target, attr, self._tmp.name)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)
        self.store = SessionStore()

    def _pool(self, workers: dict | None = None) -> _AgentPool:
        pool = object.__new__(_AgentPool)
        pool.active_session_id = None
        pool._workers = dict(workers or {})
        pool._last_use = {}
        pool._queue = []
        pool._building = {}
        pool.stop_requested = None
        pool.server_idle_ttl = 100.0
        pool._idle_since = None
        pool._wakes_pending = {}
        pool._held_checkin = {}
        pool.store = self.store
        pool.bus = _EventBus(pool, commit=pool._commit_turn,
                             durable=pool.consume_durable_event)
        return pool

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
        pool.consume_durable_event(self._complete(s.id))

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
        pool.consume_durable_event(self._complete(s.id))

        handed = w.queries[0]["history"]
        self.assertEqual(handed[0]["content"], "validate J0 when the first cost appears")
        self.assertIn("finished", handed[-1]["content"])

    def test_the_wake_is_written_into_the_stored_conversation(self):
        # The turn has to be answerable later by a process that never saw this event,
        # and `_commit_turn` writes its answer against the history stored here.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.consume_durable_event(self._complete(s.id))

        stored = self.store.load_session(s.id)
        self.assertEqual(stored.llm_history[-1]["role"], "user")
        self.assertIn("finished", stored.llm_history[-1]["content"])
        self.assertIn("finished", stored.llm_history_full[-1]["content"])

    def test_the_user_finds_the_wake_on_their_return(self):
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.consume_durable_event(self._complete(s.id))

        stored = self.store.load_session(s.id)
        self.assertTrue(stored.display_messages[-1]["text"].startswith("🔔"))

    def test_the_wake_is_journaled_where_the_turn_begins(self):
        # A window opening later replays the journal. Without this it finds a turn with
        # no question in front of it.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.consume_durable_event(self._complete(s.id))

        events = transcript_log.read_since(s.id, 0)[0]
        self.assertEqual([e["type"] for e in events], ["job_wake"])
        self.assertEqual(events[0]["job"], "j1")


class ItSettlesWhatItDeliversTests(_HeadlessCase):
    def test_a_delivered_run_is_left_out_of_the_next_scan(self):
        s = self._session()
        self._ended_job(s.id, "j1")
        pool = self._pool({s.id: _Worker(s.id)})
        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])

        pool.consume_durable_event(self._complete(s.id))
        self.assertEqual(scan_session(s.id), [])

    def test_a_run_it_could_not_deliver_stays_owed(self):
        # No agent for the conversation: nothing can run the turn. The run must remain
        # unsettled so the next worker built, or the next connection, announces it.
        s = self._session()
        self._ended_job(s.id, "j1")
        pool = self._pool()
        pool.consume_durable_event(self._complete(s.id))

        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])

    def test_a_steered_run_stays_owed(self):
        # A steer is only known to have been read when the loop says so. Re-telling a
        # run is recoverable; losing one is not.
        s = self._session()
        self._ended_job(s.id, "j1")
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.consume_durable_event(self._complete(s.id))

        self.assertEqual(len(w.steers), 1)
        self.assertEqual(w.queries, [])
        self.assertEqual([j.job_key for j in scan_session(s.id)], ["j1"])


class WhoseJobItIsTests(_HeadlessCase):
    def test_an_attached_client_routes_it_instead(self):
        # Two consumers starting a turn for one finished run is the duplicate no marker
        # can catch — neither has written it yet. A subscriber present means a
        # `_Session` is about to do this against the history it holds on screen.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        w.out_q.put(self._complete(s.id))
        pool.bus.subscribe()
        pool.bus.pump_once()

        self.assertEqual(w.queries, [])

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
        pool.consume_durable_event(self._complete(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertEqual(w.queries[0]["session_id"], s.id)

    def test_an_event_with_no_conversation_is_dropped(self):
        pool = self._pool({"s1": _Worker("s1")})
        pool.consume_durable_event(self._complete(None))
        self.assertEqual(pool._workers["s1"].queries, [])


class BulletinsTests(_HeadlessCase):
    def test_a_check_in_starts_its_short_turn(self):
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.consume_durable_event(self._checkin(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertIn("Background check-in", w.queries[0]["text"])

    def test_a_check_in_leaves_no_bubble(self):
        # A conversation checked on twenty times must not reopen on twenty blocks of an
        # instruction addressed to the model.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool.consume_durable_event(self._checkin(s.id))

        stored = self.store.load_session(s.id)
        self.assertEqual(stored.display_messages, [])

    def test_a_check_in_never_interrupts_a_turn(self):
        # A bulletin carries "nothing to report". Steering that into a turn makes the
        # agent answer about the job instead of the work it was asked for.
        s = self._session()
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.consume_durable_event(self._checkin(s.id))

        self.assertEqual(w.queries, [])
        self.assertEqual(w.steers, [])

    def test_a_check_in_never_interrupts_a_card(self):
        s = self._session()
        w = _Worker(s.id, prompt={"type": "approval"})
        pool = self._pool({s.id: w})
        pool.consume_durable_event(self._checkin(s.id))

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
        self.assertTrue(pool.idle_report()["idle"])

        self._ended_job(s.id, "j1")
        report = pool.idle_report()
        self.assertFalse(report["idle"])
        self.assertTrue(any("nobody has taken in" in r for r in report["reasons"]),
                        report["reasons"])


if __name__ == "__main__":
    unittest.main()
