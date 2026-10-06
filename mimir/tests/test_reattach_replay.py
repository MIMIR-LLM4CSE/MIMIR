"""Coming back to a run that kept going — once each, in order, with no gap.

The journal's ``seq`` is the watermark. A client says how far its rendered transcript
reaches, and a re-attach sends back only what came after. The whole no-duplicate /
no-drop argument is an *ordering*: the subscription is opened before the handshake, so
nothing produced since can be missed; the journal is then read from the watermark and
sent; and only afterwards is the subscription's gate raised to where the replay ended,
so events produced *during* the read — which are in both places — collapse to one copy.
"""
import json
import queue as _queue
import tempfile
import unittest
from unittest import mock

import os

from mimir.client.ui.ws import job_scan, transcript_log
from mimir.client.ui.ws.ws_session import _Session
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests._fake_pool import FakePool


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, payload: str) -> None:
        self.sent.append(json.loads(payload))


def _bare_worker(session_id: str = "s1") -> _AgentWorker:
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w.session_id = session_id
    w._query_session_id = None
    return w


class _ReplayCase(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(transcript_log, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

        self.worker = _bare_worker("s1")
        self.pool = FakePool(self.worker, active="s1")
        self.sess = object.__new__(_Session)
        self.sess.ws = _FakeWS()
        self.sess.pool = self.pool
        self.sess._active_session_id = "s1"
        self.sess._rendered_seq = 0
        self.sess._sub = None

    def _produce(self, *texts: str) -> None:
        """Let the agent emit, with nobody attached, and the pump record it."""
        for text in texts:
            self.worker.out_q.put({"type": "output", "text": text})
        self.pool.bus.pump_once()

    def _attach(self) -> None:
        self.sess._sub = self.pool.bus.subscribe()

    def _frames(self) -> list[dict]:
        return [m for m in self.sess.ws.sent if m.get("type") == "replay"]

    def _replayed(self) -> list[str]:
        return [e["text"] for f in self._frames() for e in f["events"]]


class WatermarkTests(_ReplayCase):
    async def test_everything_is_replayed_to_a_client_that_has_seen_nothing(self):
        self._produce("a", "b", "c")
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self._replayed(), ["a", "b", "c"])
        self.assertEqual(self._frames()[-1]["through_seq"], 3)

    async def test_only_what_came_after_the_watermark_is_replayed(self):
        self._produce("a", "b", "c", "d")
        self.sess._rendered_seq = 2
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self._replayed(), ["c", "d"])

    async def test_a_client_that_has_seen_everything_is_sent_nothing(self):
        self._produce("a", "b")
        self.sess._rendered_seq = 2
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self._frames(), [])

    async def test_a_session_with_no_journal_replays_nothing(self):
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self._frames(), [])

    async def test_the_gate_moves_even_when_there_is_nothing_to_replay(self):
        # Otherwise a live event this connection was already handed is sent twice.
        self._produce("a", "b")
        self.sess._rendered_seq = 2
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self.sess._sub.min_seq, 2)


class NoDuplicateNoGapTests(_ReplayCase):
    async def test_the_gate_lands_where_the_replay_ended(self):
        self._produce("a", "b", "c")
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self.sess._sub.min_seq, 3)

    async def test_an_already_replayed_event_is_not_delivered_live_as_well(self):
        # The overlap: the subscription was open while the journal was being read, so
        # it holds the same events the replay just sent.
        self._attach()
        self._produce("a", "b")           # queued on the subscription AND journaled
        await self.sess._send_replay()
        delivered = []
        while not self.sess._sub.queue.empty():
            ev, _extras = self.sess._sub.queue.get_nowait()
            if self.sess._sub.wants(ev):
                delivered.append(ev["text"])
        self.assertEqual(self._replayed(), ["a", "b"])
        self.assertEqual(delivered, [], "an event was both replayed and delivered live")

    async def test_an_event_produced_after_the_replay_still_arrives_live(self):
        self._produce("a")
        self._attach()
        await self.sess._send_replay()
        self._produce("b")
        live = [ev["text"] for ev, _x in
                [self.sess._sub.queue.get_nowait()
                 for _ in range(self.sess._sub.queue.qsize())]
                if self.sess._sub.wants(ev)]
        self.assertEqual(live, ["b"])

    async def test_nothing_produced_between_subscribing_and_reading_is_lost(self):
        # The gap the ordering rules out: subscribe first, so output in this window is
        # held by the subscription even though the journal read has not happened yet.
        self._attach()
        self._produce("during")
        await self.sess._send_replay()
        self.assertIn("during", self._replayed())


class FramingTests(_ReplayCase):
    async def test_a_long_run_is_sent_in_several_frames(self):
        with mock.patch("mimir.client.ui.ws.ws_session._REPLAY_CHUNK", 2):
            self._produce("a", "b", "c", "d", "e")
            self._attach()
            await self.sess._send_replay()
        frames = self._frames()
        self.assertEqual([len(f["events"]) for f in frames], [2, 2, 1])
        self.assertEqual([f["more"] for f in frames], [True, True, False])
        self.assertEqual([f["through_seq"] for f in frames], [2, 4, 5])
        self.assertEqual(self._replayed(), ["a", "b", "c", "d", "e"])

    async def test_a_capped_replay_keeps_the_end_and_says_it_truncated(self):
        # What a returning user needs is the end of the run; an unbounded replay into
        # the webview's reducer is how a twelve-hour absence wedges the panel.
        with mock.patch("mimir.client.ui.ws.ws_session._REPLAY_MAX_EVENTS", 3):
            self._produce("a", "b", "c", "d", "e")
            self._attach()
            await self.sess._send_replay()
        self.assertEqual(self._replayed(), ["c", "d", "e"])
        self.assertTrue(self._frames()[0]["truncated"])

    async def test_every_frame_names_its_session(self):
        self._produce("a")
        self._attach()
        await self.sess._send_replay()
        self.assertEqual(self._frames()[0]["session_id"], "s1")

    async def test_a_socket_that_dies_mid_replay_leaves_the_gate_alone(self):
        # The gate must not advance past what the client actually received, or the rest
        # is lost to both the replay and the live stream.
        self._produce("a", "b")
        self._attach()

        async def _boom(_payload):
            raise ConnectionResetError("gone")

        self.sess.ws.send = _boom
        await self.sess._send_replay()        # must not raise
        self.assertEqual(self.sess._sub.min_seq, 0)


class WhatIsNotReplayedTests(_ReplayCase):
    async def test_streamed_deltas_are_absent_but_their_aggregate_is_not(self):
        self.worker.out_q.put({"type": "token", "text": "tok"})
        self.worker.out_q.put({"type": "thinking_end", "tokens": 12})
        self.worker.out_q.put({"type": "answer", "text": "done"})
        self.pool.bus.pump_once()
        self._attach()
        await self.sess._send_replay()
        kinds = [e["type"] for f in self._frames() for e in f["events"]]
        self.assertEqual(kinds, ["thinking_end", "answer"])


class EndedJobsAreReportedOnAttachTests(unittest.IsolatedAsyncioTestCase):
    """Coming back is enough — nobody has to ask where the build got to.

    The run survived; the promise to report it did not. A connection arriving scans the
    descriptors and hands what ended to the ordinary wake routing, so a run reported
    late is indistinguishable from one reported on time.
    """

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(job_scan, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

        self.sess = object.__new__(_Session)
        self.sess.ws = _FakeWS()
        self.sess._active_session_id = "s1"
        self.handled: list[dict] = []

        async def _handle(ev, steer=None):
            self.handled.append(ev)
            return False

        self.sess._handle_job_complete = _handle

    def _baseline(self, session_id: str) -> None:
        """This session has been looked at before — the ordinary state.

        Only the very first scan establishes a baseline rather than reporting, and that
        case has its own tests below.
        """
        path = os.path.join(self._tmp.name, "sessions", session_id, "jobs")
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, ".wake_baseline"), "w") as fh:
            fh.write("0")

    def _ended_job(self, session_id: str, job_key: str, exit_code: int) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir, exist_ok=True)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "make -j8", "pid": 1,
                       "pid_starttime": 1}, fh)
        with open(os.path.join(job_dir, "exit_code"), "w") as fh:
            fh.write(str(exit_code))
        self._baseline(session_id)

    def _live_job(self, session_id: str, job_key: str) -> None:
        job_dir = os.path.join(self._tmp.name, "sessions", session_id, "jobs", job_key)
        os.makedirs(job_dir)
        with open(os.path.join(job_dir, "meta.json"), "w") as fh:
            json.dump({"job_key": job_key, "command": "sleep 9999",
                       "pid": os.getpid(),
                       "pid_starttime": job_scan._proc_starttime(os.getpid())}, fh)

    async def test_a_run_that_ended_while_away_is_handed_to_the_wake_routing(self):
        self._ended_job("s1", "j1", 0)
        await self.sess._report_ended_jobs()
        self.assertEqual(len(self.handled), 1)
        ev = self.handled[0]
        self.assertEqual(ev["type"], "job_complete")
        self.assertEqual(ev["state"], "done")
        self.assertEqual(ev["session_id"], "s1")

    async def test_a_run_of_another_conversation_is_reported_to_that_conversation(self):
        # A two-hour build outlives the conversation on screen, and its result belongs
        # to the one that asked for it.
        self._ended_job("s2", "theirs", 0)
        await self.sess._report_ended_jobs()
        self.assertEqual(self.handled[0]["session_id"], "s2")

    async def test_a_run_still_going_is_left_to_the_worker_that_will_watch_it(self):
        self._live_job("s1", "j1")
        await self.sess._report_ended_jobs()
        self.assertEqual(self.handled, [])

    async def test_a_second_connection_does_not_report_it_again(self):
        self._ended_job("s1", "j1", 0)
        await self.sess._report_ended_jobs()
        self.handled.clear()
        await self.sess._report_ended_jobs()
        self.assertEqual(self.handled, [])

    async def test_a_crash_keeps_its_exit_code_in_the_summary(self):
        self._ended_job("s1", "j1", 3)
        await self.sess._report_ended_jobs()
        self.assertEqual(self.handled[0]["state"], "crashed")
        self.assertEqual(self.handled[0]["summary"]["exit_code"], 3)

    async def test_the_first_look_at_a_session_reports_nothing(self):
        # An existing workspace has every build it ever ran on disk with no marker —
        # job directories are never swept. Reporting that history would wake a
        # conversation per historical build on the first attach after upgrading.
        for i in range(4):
            job_dir = os.path.join(self._tmp.name, "sessions", "s1", "jobs", f"old-{i}")
            os.makedirs(job_dir)
            with open(os.path.join(job_dir, "meta.json"), "w") as fh:
                json.dump({"job_key": f"old-{i}", "command": "make", "pid": 1,
                           "pid_starttime": 1}, fh)
            with open(os.path.join(job_dir, "exit_code"), "w") as fh:
                fh.write("0")

        await self.sess._report_ended_jobs()
        self.assertEqual(self.handled, [])
        self.assertTrue(job_scan.has_baseline("s1"))

    async def test_a_job_that_ends_after_the_first_look_is_reported(self):
        await self.sess._report_ended_jobs()      # establishes the baseline
        self._ended_job("s1", "j1", 0)
        await self.sess._report_ended_jobs()
        self.assertEqual([e["job_key"] for e in self.handled], ["j1"])

    async def test_an_unreadable_state_dir_is_not_fatal(self):
        with mock.patch.object(job_scan, "scan_all_sessions",
                               side_effect=OSError("gone")):
            await self.sess._report_ended_jobs()   # must not raise
        self.assertEqual(self.handled, [])


class TheStoredWatermarkTests(unittest.IsolatedAsyncioTestCase):
    """The client's statement of what it has rendered, kept across connections."""

    def setUp(self) -> None:
        self.sess = object.__new__(_Session)
        self.sess.ws = _FakeWS()
        self.sess._active_session_id = "s1"
        self.sess._rendered_seq = 0
        self.sess._display_messages = []
        self.sess._autosave_session = lambda messages: None

    async def test_the_transcript_carries_the_watermark_forward(self):
        await self.sess._handle_transcript(
            {"session_id": "s1", "messages": [{"role": "user", "text": "hi"}],
             "through_seq": 7})
        self.assertEqual(self.sess._rendered_seq, 7)

    async def test_the_watermark_only_ever_moves_forward(self):
        # A webview that reopened on less than we hold must not shorten it either.
        self.sess._rendered_seq = 9
        await self.sess._handle_transcript(
            {"session_id": "s1", "messages": [{"role": "user", "text": "hi"}],
             "through_seq": 3})
        self.assertEqual(self.sess._rendered_seq, 9)

    async def test_a_transcript_without_a_watermark_is_still_accepted(self):
        # An older client sends none; the watermark simply does not advance, and the
        # next attach replays more than it needed to.
        await self.sess._handle_transcript(
            {"session_id": "s1", "messages": [{"role": "user", "text": "hi"}]})
        self.assertEqual(self.sess._rendered_seq, 0)
        self.assertEqual(len(self.sess._display_messages), 1)


if __name__ == "__main__":
    unittest.main()
