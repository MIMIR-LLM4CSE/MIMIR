"""Who takes a finished run's wake in — and that somebody always does.

A finished background run is owed a turn of the conversation that launched it. It used
to have two possible consumers — the socket's drain loop, against the conversation it
held on screen, and the pool, against the one on disk — and a claim deciding which one
acted. There is one conversation object now, so there is one consumer: the pump hands
every wake to the pool's ``WakeRouter`` inside the tick that drained it, whether or not
a window is open, and a view only reports what it did.

These tests pin that: a wake is taken in within its tick, attached or not; a turn that
lands carries on with what it left unread; and an agent owed anything is not released.
"""
import asyncio
import json
import os
import queue as _queue
import tempfile
import unittest
from unittest import mock

from mimir.client.session import store as session_store
from mimir.client.session.conversation import Conversations
from mimir.client.session.store import SessionStore
from mimir.client.ui.ws.human_bridge import HumanBridge
from mimir.client.ui.ws import job_scan, transcript_log
from mimir.client.ui.ws.event_bus import _EventBus
from mimir.client.ui.ws.wake_router import WakeRouter
from mimir.client.ui.ws.ws_pool import _AgentPool


class _Worker:
    """Only what the headless consumer asks of a worker."""

    def __init__(self, session_id: str, *, busy: bool = False) -> None:
        self.human = HumanBridge(self)
        self.out_q = _queue.Queue()
        self.session_id = session_id
        self.active_session_id = None
        self._query_session_id = None
        self._busy = busy
        self.queries: list[dict] = []
        self.steers: list[str] = []
        self.watched: list[str] = []
        self._bg_jobs: dict = {}

    def is_busy(self) -> bool:
        return self._busy

    def has_work_pending(self) -> bool:
        return self._busy

    def submit_query(self, text, history, session_id=None) -> None:
        self.queries.append({"text": text, "session_id": session_id})

    def submit_steer(self, text: str) -> None:
        self.steers.append(text)

    def watched_job_keys(self) -> list[str]:
        return list(self.watched)

    def drain(self) -> list:
        out = []
        while not self.out_q.empty():
            out.append(self.out_q.get_nowait())
        return out


class _WakeCase(unittest.TestCase):
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
        pool._reaper = None
        pool.stop_requested = None
        pool.server_idle_ttl = 100.0
        pool._idle_since = None
        pool.conversations = Conversations(self.store)
        pool.store = self.store
        pool.wakes = WakeRouter(pool)
        pool.bus = _EventBus(pool, commit=pool.commit_turn, durable=pool.wakes.consume,
                             steered=pool.wakes.steered)
        return pool

    def _session(self):
        s = self.store.new_session()
        s.llm_history = [{"role": "user", "content": "launch the run"}]
        s.llm_history_full = list(s.llm_history)
        self.store.save_session(s)
        return s

    def _ended_job(self, session_id: str, job_key: str = "j1") -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "train.sh",
                       "pid": 1, "pid_starttime": 1}, fh)
        with open(os.path.join(job_dir, "exit_code"), "w") as fh:
            fh.write("0")
        path = os.path.join(self._tmp.name, "sessions", session_id)
        with open(os.path.join(path, ".wake_baseline"), "w") as fh:
            fh.write("0")

    @staticmethod
    def _complete(session_id: str, job_key: str = "j1") -> dict:
        return {"type": "job_complete", "job_key": job_key, "server": "bash",
                "kind": "shell-command", "state": "done", "session_id": session_id,
                "summary": {"output": "training finished"},
                "status_op": {"tool": "bash_job", "args": {"job_key": job_key}}}


class NobodyAttachedTests(_WakeCase):
    def test_a_wake_is_taken_in_within_the_tick_that_drained_it(self):
        # The detached case, and the latency it had when it was decided by a test on
        # the subscriber list: no view to grant first refusal to, so no waiting.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        w.out_q.put(self._complete(s.id))

        pool.bus.pump_once()

        self.assertEqual(len(w.queries), 1)
        self.assertIn("finished", w.queries[0]["text"])


class AnAttachedViewChangesNothingTests(_WakeCase):
    """A subscription is a view: it is told what happened, it decides nothing.

    When a view had first refusal, a subscription outliving the loop that read it — a
    drain loop that returned on a failed send — stood in for a consumer nobody was
    running, and the run waited out a grace window, or the night.
    """

    def test_the_wake_is_taken_in_within_the_tick_all_the_same(self):
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.bus.subscribe()          # subscribed, and nothing ever reads it
        w.out_q.put(self._complete(s.id))

        pool.bus.pump_once()

        self.assertEqual(len(w.queries), 1)
        self.assertIn("finished", w.queries[0]["text"])

    def test_the_view_is_told_a_turn_started(self):
        # So the chat can mark itself busy for a turn nobody pressed send for.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        sub = pool.bus.subscribe()
        w.out_q.put(self._complete(s.id))
        pool.bus.pump_once()

        ev, extras = sub.queue.get_nowait()
        self.assertEqual(ev["type"], "job_complete")
        self.assertTrue(extras["_started"])

    def test_the_run_is_settled_as_it_is_delivered(self):
        s = self._session()
        self._ended_job(s.id)
        pool = self._pool({s.id: _Worker(s.id)})
        pool.bus.subscribe()
        self.assertEqual([j.job_key for j in job_scan.scan_session(s.id)], ["j1"])
        pool._workers[s.id].out_q.put(self._complete(s.id))

        pool.bus.pump_once()

        self.assertEqual(job_scan.scan_session(s.id), [])

    def test_a_run_announced_twice_starts_one_turn(self):
        # The watcher that held it and a scan that found it unspoken-for can both
        # speak before either delivers.
        s = self._session()
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._complete(s.id))
        pool.wakes.consume(self._complete(s.id))
        self.assertEqual(len(w.steers), 1)


class CheckInsGoTheSameWayTests(_WakeCase):
    def test_a_bulletin_is_taken_in_with_a_view_attached(self):
        # The other half of the same silence: a run nobody is reassured about reads
        # exactly like one that died.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool.bus.subscribe()
        w.out_q.put({"type": "job_checkin", "session_id": s.id,
                     "jobs": [{"job_key": "j1", "kind": "shell-command",
                               "state": "running", "phase": "", "percent": None}]})
        pool.bus.pump_once()

        self.assertEqual(len(w.queries), 1)
        self.assertIn("still", w.queries[0]["text"].lower())


class ATurnLandingCarriesTheWorkOnTests(_WakeCase):
    """What a connection does the moment an answer lands, with none attached.

    This is the difference between a chain of steps that runs overnight and one that
    stops at its first link. The training finishes, its wake starts a turn, that turn
    launches the FWI — and the run that finishes while it is still working is steered
    into it. If that turn never reads the steer, a connection would start a turn for it
    on the spot; a detached process that only wrote the answer down would stop there,
    with a finished run and nobody ever told.
    """

    def _steered(self, pool, worker, s, job_key="j1"):
        """One wake handed to a turn already running."""
        worker._busy = True
        before = len(worker.steers)
        pool.wakes.consume(self._complete(s.id, job_key))
        self.assertEqual(len(worker.steers), before + 1,
                         "it was not steered into the turn already running")
        worker._busy = False

    @staticmethod
    def _answer(session_id, unread=()):
        return ({"type": "answer", "text": "launched the FWI", "session_id": session_id},
                {"_unconsumed_steer": list(unread)})

    def test_a_steered_wake_the_turn_never_read_gets_its_own_turn(self):
        from mimir.client.ui.ws.job_wakes import wake_text

        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        self._steered(pool, w, s)
        never_read = wake_text(self._complete(s.id))

        ev, extras = self._answer(s.id, unread=[never_read])
        pool.commit_turn(ev, extras)

        self.assertEqual(len(w.queries), 1,
                         "the run that finished during the turn was never answered for")
        self.assertIn("finished", w.queries[0]["text"])

    def test_a_steered_wake_the_turn_did_read_is_not_told_twice(self):
        s = self._session()
        self._ended_job(s.id)
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        self._steered(pool, w, s)

        ev, extras = self._answer(s.id, unread=[])      # the loop took it in
        pool.commit_turn(ev, extras)

        self.assertEqual(w.queries, [], "a run already dealt with was told again")
        self.assertEqual(job_scan.scan_session(s.id), [],
                         "a run the turn answered for was left owed")

    def test_several_runs_that_finished_during_the_turn_arrive_as_one_turn(self):
        from mimir.client.ui.ws.job_wakes import wake_text

        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        self._steered(pool, w, s, "j1")
        self._steered(pool, w, s, "j2")
        unread = [wake_text(self._complete(s.id, "j1")),
                  wake_text(self._complete(s.id, "j2"))]

        ev, extras = self._answer(s.id, unread=unread)
        pool.commit_turn(ev, extras)

        self.assertEqual(len(w.queries), 1, "one turn apiece, not one for the burst")
        self.assertIn("j1", w.queries[0]["text"])
        self.assertIn("j2", w.queries[0]["text"])

    def test_the_user_finds_a_steered_wake_on_their_return(self):
        # Equivalence is also what the conversation looks like afterwards: a wake
        # handed to a running turn leaves its bubble where it happened, so coming back
        # shows why the agent changed course.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        self._steered(pool, pool._workers[s.id], s)

        stored = self.store.load_session(s.id)
        self.assertTrue(stored.display_messages[-1]["text"].startswith("🔔"))
        self.assertEqual([e["type"] for e in transcript_log.read_since(s.id, 0)[0]],
                         ["job_wake"])

    def test_a_bulletin_held_back_is_delivered_when_the_turn_lands(self):
        s = self._session()
        w = _Worker(s.id, busy=True)
        w.watched = ["j1"]
        pool = self._pool({s.id: w})
        pool.wakes.consume(
            {"type": "job_checkin", "session_id": s.id,
             "jobs": [{"job_key": "j1", "state": "running"}]})
        self.assertEqual(w.queries, [], "a bulletin interrupted a running turn")

        w._busy = False
        pool.commit_turn(*self._answer(s.id))

        self.assertEqual(len(w.queries), 1)
        self.assertIn("still", w.queries[0]["text"].lower())

    def test_a_bulletin_is_dropped_when_its_runs_have_since_finished(self):
        # Their completion wakes say everything it would, and better.
        s = self._session()
        w = _Worker(s.id, busy=True)
        w.watched = ["j1"]
        pool = self._pool({s.id: w})
        pool.wakes.consume(
            {"type": "job_checkin", "session_id": s.id,
             "jobs": [{"job_key": "j1", "state": "running"}]})

        w._busy = False
        w.watched = []
        pool.commit_turn(*self._answer(s.id))

        self.assertEqual(w.queries, [])
        self.assertIsNone(pool.conversations.get(s.id).held_checkin)


class AWakeNeverArrivesForAClosedAgentTests(_WakeCase):
    """The invariant that replaces a rebuild.

    An agent that has delivered its final answer with nothing of its outstanding is
    *meant* to be closed: no wake can come for it, because every emitter that goes
    through the bus runs on a worker's own loop. So rather than machinery to rebuild an
    agent for a wake, the guarantee is that the agent is still there — ``releasable``
    refuses to close one while a job of its is watched, a wake of its is owed, its
    output is undrained.

    What is left for the unreachable case is honesty: say so, keep the wake for the
    conversation's next turn, and leave the run unsettled on disk.
    """

    def test_the_whole_handover_happens_while_the_agent_is_unreleasable(self):
        # The window the clauses exist for: the watcher drops its job the instant it
        # reports it, and the conversation has been idle for hours by then.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool._last_use[s.id] = 0.0
        w._bg_jobs = {}                       # the watcher has just let go
        w.out_q.put(self._complete(s.id))
        self.assertFalse(pool.releasable(s.id), "releasable with its news unread")

        pool.bus.pump_once()
        self.assertEqual(len(w.queries), 1)

    def test_a_wake_for_a_closed_agent_leaves_the_run_owed(self):
        # The unreachable case, kept honest rather than handled: nothing is invented,
        # and the run stays unsettled so the next agent built announces it.
        s = self._session()
        self._ended_job(s.id)
        pool = self._pool()                   # no agent for this conversation
        with self.assertLogs("mimir.client.ui.ws.wake_router", "WARNING") as caught:
            pool.wakes.consume(self._complete(s.id))
        self.assertIn("no agent", "".join(caught.output))
        self.assertEqual([j.job_key for j in job_scan.scan_session(s.id)], ["j1"])
        # Kept for the conversation's next turn, not dropped.
        self.assertEqual(len(pool.conversations.get(s.id).pending_wakes), 1)


class _NoWS:
    """A socket that accepts sends and carries no inbound messages."""

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send(self, payload) -> None:
        self.sent.append(payload)

    def __aiter__(self):
        async def _empty():
            if False:
                yield ""
        return _empty()


class AnAgentOwedAWakeIsNotIdleTests(_WakeCase):
    """What "idle for ten minutes" may not be allowed to mean.

    Idleness is measured from the last time the pool was asked for the worker — not
    from anything the run is doing — so a conversation waiting on a twenty-four-hour
    job has been "idle" for twenty-four hours. What protects it is not the clock but
    ``releasable``: a watched job makes it unreleasable outright.

    The gap this closes is one step later. A watcher drops its job the instant it
    reports it finished, which leaves the agent releasable for the moment between the
    report and the turn that answers it — and the sweep that looks every thirty seconds
    is looking at a conversation whose idle clock has been running for hours.
    """

    def test_an_agent_watching_a_run_is_never_released(self):
        s = self._session()
        w = _Worker(s.id)
        w._bg_jobs = {"j1": object()}
        pool = self._pool({s.id: w})
        pool._last_use[s.id] = 0.0            # idle since the epoch
        self.assertFalse(pool.releasable(s.id))

    def test_an_agent_owed_a_wake_is_not_released_either(self):
        # The run has finished and been reported; the turn that answers it has not
        # started. Releasing here is releasing the agent the wake is addressed to.
        s = self._session()
        w = _Worker(s.id, busy=True)
        pool = self._pool({s.id: w})
        pool.wakes.consume(self._complete(s.id))   # steered, still owed
        w._busy = False
        pool._last_use[s.id] = 0.0
        self.assertFalse(pool.releasable(s.id))

    def test_an_agent_whose_last_word_is_unread_is_not_released(self):
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        pool._last_use[s.id] = 0.0
        w.out_q.put(self._complete(s.id))     # the completion event, not yet drained
        self.assertFalse(pool.releasable(s.id))

        pool.bus.pump_once()                  # drained, and the turn started
        self.assertEqual(len(w.queries), 1)

    def test_an_agent_with_nothing_outstanding_still_gives_up_its_slot(self):
        # The clauses above must not add up to a pool that never releases anything.
        s = self._session()
        pool = self._pool({s.id: _Worker(s.id)})
        pool._last_use[s.id] = 0.0
        self.assertTrue(pool.releasable(s.id))


class AViewThatEndsReleasesWhatItHoldsTests(_WakeCase):
    """``run`` subscribes before the handshake; it must unsubscribe however it leaves.

    Every step of that handshake sends on a socket that can close under it — a window
    shut during the replay of a long detached run raises there — so an exit before the
    read loop is the ordinary case rather than the exotic one. A subscription left
    behind is the server's standing evidence that somebody is watching, and under it
    nothing is committed and no finished run is handed to its conversation for the
    whole life of the process.
    """

    def _view(self, pool, worker):
        from mimir.client.ui.ws.ws_session import _Session

        async def _noop(*_a, **_k):
            return None

        sess = object.__new__(_Session)
        sess.ws = _NoWS()
        sess.pool = pool
        sess.store = self.store
        sess.worker_for_test = worker
        sess.conv = None
        sess._active_session_id = None
        sess._rendered_seq = 0
        sess._summary_task = None
        sess._greeting = lambda: {"type": "ready", "agent_ready": True}
        sess._purge_empty_sessions = lambda: None
        sess._send_sessions_list = _noop
        sess._send_toggles = _noop
        sess._load_session = _noop
        sess._create_new_session = _noop
        sess._send_replay = _noop
        sess._restore_detached_autonomy = _noop
        sess._resend_parked_prompt = _noop
        sess._send_served_models = _noop
        sess._drain_loop = _noop
        sess._report_ended_jobs = _noop
        return sess

    def test_a_handshake_that_raises_does_not_leave_the_server_attended(self):
        pool = self._pool()
        sess = self._view(pool, None)

        async def _boom():
            raise ConnectionResetError("the window closed during the replay")

        sess._send_replay = _boom

        with self.assertRaises(ConnectionResetError):
            asyncio.run(sess.run())
        self.assertEqual(pool.bus.attached(), 0)

    def test_a_greeting_that_cannot_be_sent_does_not_either(self):
        pool = self._pool()
        sess = self._view(pool, None)

        async def _boom(_payload):
            raise ConnectionResetError("gone")

        sess.ws.send = _boom

        asyncio.run(sess.run())
        self.assertEqual(pool.bus.attached(), 0)

    def test_a_view_that_ran_and_ended_releases_it(self):
        pool = self._pool()
        sess = self._view(pool, None)
        asyncio.run(sess.run())
        self.assertEqual(pool.bus.attached(), 0)

    def test_a_run_finishing_after_that_is_still_answered_for(self):
        # The view is gone, and the pool takes the wake in within the tick that
        # drained it — as it would have with the view there.
        s = self._session()
        w = _Worker(s.id)
        pool = self._pool({s.id: w})
        sess = self._view(pool, w)

        async def _boom():
            raise ConnectionResetError("the window closed during the replay")

        sess._send_replay = _boom
        with self.assertRaises(ConnectionResetError):
            asyncio.run(sess.run())

        w.out_q.put(self._complete(s.id))
        pool.bus.pump_once()
        self.assertEqual(len(w.queries), 1)


if __name__ == "__main__":
    unittest.main()
