"""The pump records with nobody attached — the premise of a detached run.

Everything that persists used to live in ``_Session._drain_loop``, created per
WebSocket connection: a socket that dropped stopped the journal, left ``out_q`` growing
and lost a finished turn's answer. These tests pin the inversion — the pump drains and
journals whether or not a client is listening, ``seq`` has exactly one writer, and a
subscriber that falls behind is told it has a gap rather than stalling the process.
"""
import asyncio
import queue as _queue
import tempfile
import time
import unittest
from unittest import mock

from mimir.client.ui.ws import event_bus, transcript_log
from mimir.client.ui.ws.event_bus import _EventBus
from mimir.client.ui.ws.ws_worker import _AgentWorker


def _bare_worker(session_id: str | None = None) -> _AgentWorker:
    """A worker with only the fields ``drain()`` touches (no agent, no loop)."""
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


class _BusCase(unittest.TestCase):
    """Each test gets its own state dir, so journals never leak between tests."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        patcher = mock.patch.object(transcript_log, "_MIMIR_DIR_WS", self._tmp.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self._tmp.cleanup)

    def _lines(self, session_id: str) -> list[dict]:
        events, _truncated = transcript_log.read_since(session_id, 0)
        return events


class PumpRecordsWhenDetachedTests(_BusCase):
    def test_journals_every_event_with_no_subscriber(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        for i in range(100):
            w.out_q.put({"type": "output", "text": f"line {i}\n"})

        self.assertEqual(bus.attached(), 0)
        self.assertEqual(bus.pump_once(), 100)

        written = self._lines("s1")
        self.assertEqual(len(written), 100)
        self.assertEqual([r["seq"] for r in written], list(range(1, 101)))
        self.assertTrue(w.out_q.empty())

    def test_stream_deltas_are_not_journaled_and_leave_no_hole(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "output", "text": "a\n"})
        w.out_q.put({"type": "token", "text": "tok"})
        w.out_q.put({"type": "output", "text": "b\n"})
        bus.pump_once()

        written = self._lines("s1")
        self.assertEqual([r["type"] for r in written], ["output", "output"])
        self.assertEqual([r["seq"] for r in written], [1, 2])

    def test_a_skipped_event_reaches_a_live_client_unstamped(self):
        # Live-only by design: it must still be delivered, but it carries no seq
        # because nothing replays it.
        async def run():
            w = _bare_worker("s1")
            bus = _EventBus(_FakePool({"s1": w}))
            sub = bus.subscribe()
            w.out_q.put({"type": "token", "text": "tok"})
            bus.pump_once()
            ev, _extras = sub.queue.get_nowait()
            self.assertEqual(ev["type"], "token")
            self.assertNotIn("seq", ev)

        asyncio.run(run())

    def test_private_answer_keys_reach_neither_journal_nor_wire(self):
        async def run():
            w = _bare_worker("s1")
            committed = []
            bus = _EventBus(_FakePool({"s1": w}),
                            commit=lambda ev, extras: committed.append((ev, extras)))
            sub = bus.subscribe()
            w.out_q.put({
                "type": "answer", "text": "done",
                "_full": [{"role": "user", "content": "q"}],
                "_turn_start": 0,
                "_deferred": {"kind": "calls"},
            })
            bus.pump_once()

            on_wire, carried = sub.queue.get_nowait()
            for key in ("_full", "_turn_start", "_deferred"):
                self.assertNotIn(key, on_wire)
            # Beside the event, though — the session that owns the turn needs them.
            self.assertEqual(carried["_turn_start"], 0)
            written = self._lines("s1")
            self.assertEqual(len(written), 1)
            for key in ("_full", "_turn_start", "_deferred"):
                self.assertNotIn(key, written[0])

            # They are not dropped — they reach the committer, which is what writes
            # the session file when no attached view claims the answer. This one is
            # subscribed, so the committer is offered it once that first refusal
            # expires: it must still arrive carrying them.
            self.assertEqual(committed, [])
            bus._sweep_unclaimed(now=time.monotonic() + event_bus._CLAIM_GRACE + 1.0)
            self.assertEqual(len(committed), 1)
            _ev, extras = committed[0]
            self.assertEqual(extras["_turn_start"], 0)
            self.assertEqual(extras["_deferred"], {"kind": "calls"})

        asyncio.run(run())

    def test_a_failing_commit_does_not_stop_the_fan_out(self):
        async def run():
            w = _bare_worker("s1")

            def boom(_ev, _extras):
                raise RuntimeError("disk full")

            bus = _EventBus(_FakePool({"s1": w}), commit=boom)
            sub = bus.subscribe()
            w.out_q.put({"type": "answer", "text": "done"})
            bus.pump_once()
            self.assertEqual(sub.queue.get_nowait()[0]["type"], "answer")

        asyncio.run(run())


class SeqHasOneWriterTests(_BusCase):
    def test_two_sessions_get_independent_gapless_runs(self):
        w1, w2 = _bare_worker("s1"), _bare_worker("s2")
        bus = _EventBus(_FakePool({"s1": w1, "s2": w2}))
        for i in range(5):
            w1.out_q.put({"type": "output", "text": f"a{i}\n"})
            w2.out_q.put({"type": "output", "text": f"b{i}\n"})
        bus.pump_once()

        self.assertEqual([r["seq"] for r in self._lines("s1")], [1, 2, 3, 4, 5])
        self.assertEqual([r["seq"] for r in self._lines("s2")], [1, 2, 3, 4, 5])

    def test_a_client_line_shares_the_session_seq_run(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "output", "text": "a\n"})
        bus.pump_once()
        bus.record_client_event("s1", {"type": "query", "text": "go"})
        w.out_q.put({"type": "output", "text": "b\n"})
        bus.pump_once()

        written = self._lines("s1")
        self.assertEqual([r["type"] for r in written], ["output", "query", "output"])
        self.assertEqual([r["seq"] for r in written], [1, 2, 3])
        self.assertEqual(bus.last_seq("s1"), 3)

    def test_seq_resumes_across_a_restart_rather_than_restarting(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "output", "text": "a\n"})
        bus.pump_once()
        # A second bus over the same state dir — what a restarted server is.
        w2 = _bare_worker("s1")
        bus2 = _EventBus(_FakePool({"s1": w2}))
        w2.out_q.put({"type": "output", "text": "b\n"})
        bus2.pump_once()

        self.assertEqual([r["seq"] for r in self._lines("s1")], [1, 2])


class SubscriberTests(_BusCase):
    def test_the_watermark_filters_what_the_client_already_rendered(self):
        async def run():
            w = _bare_worker("s1")
            bus = _EventBus(_FakePool({"s1": w}))
            sub = bus.subscribe()
            sub.rendered_through("s1", 2)
            for i in range(4):
                w.out_q.put({"type": "output", "text": f"{i}\n"})
            bus.pump_once()

            got = []
            while not sub.queue.empty():
                got.append(sub.queue.get_nowait()[0]["seq"])
            self.assertEqual(got, [3, 4])

        asyncio.run(run())

    def test_an_overflowing_subscriber_is_gapped_not_blocking(self):
        async def run():
            w = _bare_worker("s1")
            bus = _EventBus(_FakePool({"s1": w}))
            sub = bus.subscribe()
            sub.queue = asyncio.Queue(maxsize=3)
            for i in range(10):
                w.out_q.put({"type": "output", "text": f"{i}\n"})

            # The pump completes: a client that cannot keep up never holds it.
            self.assertEqual(bus.pump_once(), 10)
            self.assertTrue(sub.gapped)
            self.assertEqual(sub.queue.qsize(), 3)
            # It kept the END of the stream, which is what a client needs most; the
            # beginning is what the journal replays best.
            tail = [sub.queue.get_nowait()[0]["seq"] for _ in range(3)]
            self.assertEqual(tail, [8, 9, 10])
            # And nothing was lost from the record.
            self.assertEqual(len(self._lines("s1")), 10)

        asyncio.run(run())

    def test_a_filter_fences_a_subscriber_to_its_sessions(self):
        async def run():
            w1, w2 = _bare_worker("s1"), _bare_worker("s2")
            bus = _EventBus(_FakePool({"s1": w1, "s2": w2}))
            sub = bus.subscribe(lambda ev: ev.get("session_id") == "s1")
            w1.out_q.put({"type": "output", "text": "mine\n"})
            w2.out_q.put({"type": "output", "text": "theirs\n"})
            bus.pump_once()

            got = []
            while not sub.queue.empty():
                got.append(sub.queue.get_nowait()[0]["text"])
            self.assertEqual(got, ["mine\n"])

        asyncio.run(run())

    def test_unsubscribing_stops_delivery_and_leaves_the_journal_alone(self):
        async def run():
            w = _bare_worker("s1")
            bus = _EventBus(_FakePool({"s1": w}))
            sub = bus.subscribe()
            sub.close()
            self.assertEqual(bus.attached(), 0)
            w.out_q.put({"type": "output", "text": "a\n"})
            bus.pump_once()
            self.assertTrue(sub.queue.empty())
            self.assertEqual(len(self._lines("s1")), 1)

        asyncio.run(run())


class ReadSinceTests(_BusCase):
    def test_reads_only_what_came_after_the_watermark(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        for i in range(6):
            w.out_q.put({"type": "output", "text": f"{i}\n"})
        bus.pump_once()

        events, truncated = transcript_log.read_since("s1", 4)
        self.assertEqual([e["seq"] for e in events], [5, 6])
        self.assertFalse(truncated)

    def test_a_limit_keeps_the_most_recent_and_says_it_truncated(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        for i in range(10):
            w.out_q.put({"type": "output", "text": f"{i}\n"})
        bus.pump_once()

        events, truncated = transcript_log.read_since("s1", 0, limit=3)
        self.assertEqual([e["seq"] for e in events], [8, 9, 10])
        self.assertTrue(truncated)

    def test_no_journal_reads_as_empty_rather_than_raising(self):
        events, truncated = transcript_log.read_since("never-existed", 0)
        self.assertEqual(events, [])
        self.assertFalse(truncated)


class ConcludedTests(_BusCase):
    """What counts as finished — the positive criterion the idle shutdown rests on.

    "Not busy" is not "has finished": a worker is also not busy between queries, after
    a turn that broke, and while it waits behind a card. An idle predicate built on the
    absence of noise would stop a server whose turn merely paused, so the bus records
    the conclusion itself.
    """

    def test_an_answer_marks_the_session_concluded(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        self.assertIsNone(bus.concluded_at("s1"))
        w.out_q.put({"type": "answer", "text": "done"})
        bus.pump_once()
        self.assertIsNotNone(bus.concluded_at("s1"))
        self.assertEqual(bus.concluded_since("s1", now=bus.concluded_at("s1") + 5), 5)

    def test_an_error_concludes_too(self):
        # The turn is over either way; calling only a success a conclusion would leave
        # a failed session looking busy for ever.
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "error", "text": "backend gone"})
        bus.pump_once()
        self.assertIsNotNone(bus.concluded_at("s1"))

    def test_output_alone_never_counts_as_concluded(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        for i in range(5):
            w.out_q.put({"type": "output", "text": f"{i}\n"})
        bus.pump_once()
        self.assertIsNone(bus.concluded_at("s1"))
        self.assertIsNone(bus.concluded_since("s1"))

    def test_a_parked_card_is_not_a_conclusion(self):
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "approval", "id": "a1", "tool": "bash_run"})
        bus.pump_once()
        self.assertIsNone(bus.concluded_at("s1"))

    def test_producing_again_clears_the_conclusion(self):
        # The clock restarts rather than letting a stale conclusion stand: this is what
        # makes "idle for N" mean idle throughout, not idle at some point.
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "answer", "text": "done"})
        bus.pump_once()
        self.assertIsNotNone(bus.concluded_at("s1"))
        w.out_q.put({"type": "output", "text": "a new turn speaks\n"})
        bus.pump_once()
        self.assertIsNone(bus.concluded_at("s1"))

    def test_a_job_waking_a_session_clears_its_conclusion(self):
        # A background job that finishes wakes the conversation that launched it, so a
        # session that had concluded is working again — and must stop counting as idle.
        w = _bare_worker("s1")
        bus = _EventBus(_FakePool({"s1": w}))
        w.out_q.put({"type": "answer", "text": "launched the build"})
        bus.pump_once()
        w.out_q.put({"type": "job_complete", "job_key": "build", "state": "done"})
        bus.pump_once()
        self.assertIsNone(bus.concluded_at("s1"))

    def test_conclusions_are_tracked_per_session(self):
        w1, w2 = _bare_worker("s1"), _bare_worker("s2")
        bus = _EventBus(_FakePool({"s1": w1, "s2": w2}))
        w1.out_q.put({"type": "answer", "text": "done"})
        w2.out_q.put({"type": "output", "text": "still going\n"})
        bus.pump_once()
        self.assertIsNotNone(bus.concluded_at("s1"))
        self.assertIsNone(bus.concluded_at("s2"))


class PumpLoopTests(_BusCase):
    def test_the_loop_keeps_pumping_after_a_failing_tick(self):
        async def run():
            w = _bare_worker("s1")
            pool = _FakePool({"s1": w})
            bus = _EventBus(pool, interval=0.01)

            calls = {"n": 0}
            real_collect = bus._collect

            def flaky():
                calls["n"] += 1
                if calls["n"] == 1:
                    raise RuntimeError("transient")
                return real_collect()

            bus._collect = flaky
            bus.start()
            w.out_q.put({"type": "output", "text": "a\n"})
            for _ in range(50):
                await asyncio.sleep(0.01)
                if self._lines("s1"):
                    break
            await bus.aclose()

            self.assertGreaterEqual(calls["n"], 2)
            self.assertEqual(len(self._lines("s1")), 1)

        asyncio.run(run())

    def test_aclose_is_safe_when_never_started(self):
        async def run():
            bus = _EventBus(_FakePool({}))
            await bus.aclose()

        asyncio.run(run())


if __name__ == "__main__":
    unittest.main()
