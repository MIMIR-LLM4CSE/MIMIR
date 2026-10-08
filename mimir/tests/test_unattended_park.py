"""An unanswered card with nobody there is set aside, not waited out for ever.

The wait behind an approval has no timeout, deliberately: nothing may proceed because
the user was slow. That is right while somebody is there and a deadlock once nobody is
— a detached run meets its first sensitive tool, holds the worker thread, and the pool
then reaps the agent out from under it.

So the criterion added is *attached*, not *elapsed*. These tests pin both halves: that a
card with somebody there still waits for ever (which `test_approval_wait` also pins,
unmodified), and that one with nobody there is deferred — through the park/placeholder/
resume mechanism that already existed and had no caller.
"""
import asyncio
import os
import queue as _queue
import threading
import time
import types
import unittest
from unittest import mock

from mimir.client.ui.ws.event_bus import _EventBus
from mimir.client.ui.ws.ws_worker import _AgentWorker, _detach_grace


def _worker(session_id: str = "s1") -> _AgentWorker:
    """A worker with only what the wait touches: no agent loop, no MCP servers."""
    w = object.__new__(_AgentWorker)
    w.out_q = _queue.Queue()
    w._approval_q = _queue.Queue()
    w._question_q = _queue.Queue()
    w.session_id = session_id
    w._query_session_id = session_id
    w.active_session_id = None
    w._pending_prompt = {"type": "approval", "id": "a1", "tool": "bash_run"}
    w._pending_questions = []
    w._defer = threading.Event()
    w._preanswer = None
    w.unattended_since = None
    w.has_deferral = False
    w._agent = types.SimpleNamespace(
        _cancel_flag=threading.Event(), _deferred_prompts=None, _deferred_turn=None,
        non_interactive=False,
    )
    return w


class _FakePool:
    def __init__(self, workers: dict) -> None:
        self._workers = workers

    def items(self):
        return list(self._workers.items())


class GracePeriodTests(unittest.TestCase):
    def test_the_default_leaves_room_for_a_window_reload(self):
        # Reloading a VS Code window closes and reopens the socket; parking every card
        # on that blink would put the user's own question away under their nose.
        self.assertGreaterEqual(_detach_grace(), 5.0)

    def test_it_is_configurable(self):
        with mock.patch.dict(os.environ, {"MIMIR_DETACH_GRACE": "2.5"}):
            self.assertEqual(_detach_grace(), 2.5)

    def test_nonsense_falls_back_rather_than_raising(self):
        with mock.patch.dict(os.environ, {"MIMIR_DETACH_GRACE": "soon"}):
            self.assertEqual(_detach_grace(), 30.0)


class UnattendedPredicateTests(unittest.TestCase):
    def test_a_worker_with_a_client_is_never_past_grace(self):
        w = _worker()
        w.unattended_since = None
        self.assertFalse(w._unattended_past_grace())

    def test_within_the_grace_it_is_not_yet_unattended(self):
        w = _worker()
        w.unattended_since = time.monotonic()
        self.assertFalse(w._unattended_past_grace())

    def test_past_the_grace_it_is(self):
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        self.assertTrue(w._unattended_past_grace())

    def test_a_worker_the_pump_has_not_reached_assumes_somebody_is_there(self):
        # A real state, not only a test's: the safe reading of no information is that
        # the user is present, because being wrong the other way puts a card away.
        w = _worker()
        del w.unattended_since
        self.assertFalse(w._unattended_past_grace())


class TheWaitTests(unittest.TestCase):
    def test_a_card_with_somebody_there_still_waits(self):
        w = _worker()
        w.unattended_since = None
        done: list = []
        t = threading.Thread(
            target=lambda: done.append(w._await_response(w._approval_q)), daemon=True)
        t.start()
        t.join(0.6)
        self.assertTrue(t.is_alive(), "the wait gave up while a client was attached")
        self.assertEqual(done, [])
        w._approval_q.put({"choice": "y"})
        t.join(2)
        self.assertEqual(done, [{"choice": "y"}])

    def test_a_card_with_nobody_there_is_set_aside(self):
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        self.assertIsNone(w._await_response(w._approval_q))
        # Deferred, not cancelled: the record of what it was holding up is kept, which
        # is what lets the answer resume the turn rather than restart it.
        self.assertTrue(w._deferring())
        self.assertTrue(w.has_deferral)
        self.assertEqual(
            [p["prompt"]["id"] for p in w._agent._deferred_prompts], ["a1"])

    def test_the_card_is_let_go_of_so_nothing_resends_it(self):
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        w._await_response(w._approval_q)
        self.assertIsNone(w._pending_prompt)

    def test_a_wait_with_its_own_deadline_is_left_alone(self):
        # A question carries five minutes and a documented meaning for running out of
        # them ("go ahead with what you recommend"). Overriding that here would change
        # an answer the model has already been promised.
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        from mimir.client.ui.ws.ws_worker import TIMED_OUT
        self.assertIs(w._await_response(w._question_q, timeout=0.05), TIMED_OUT)
        self.assertFalse(w.has_deferral)

    def test_an_answer_already_in_hand_wins_over_the_grace(self):
        # The queue gets its turn first, always. Consulting the grace before reading it
        # discards a reply that crossed with the grace elapsing — the user answered into
        # a void.
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        w._approval_q.put({"choice": "y"})
        self.assertEqual(w._await_response(w._approval_q), {"choice": "y"})
        self.assertFalse(w.has_deferral, "set a card aside that had been answered")

    def test_an_answer_landing_during_the_poll_also_wins(self):
        # The same race one tick later: the grace has elapsed, the queue is empty when
        # first read, and the reply arrives while the poll is still open.
        w = _worker()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        threading.Timer(0.05, lambda: w._approval_q.put({"choice": "n"})).start()
        self.assertEqual(w._await_response(w._approval_q), {"choice": "n"})

    def test_answering_the_deferral_clears_the_debt(self):
        w = _worker()
        w._query_q = _queue.Queue()
        w._query_event = threading.Event()
        w.unattended_since = time.monotonic() - (_detach_grace() + 1)
        w._await_response(w._approval_q)
        self.assertTrue(w.has_deferral)
        w.submit_resume({"query": "go", "kind": "calls"}, {"choice": "y"}, [], "s1")
        self.assertFalse(w.has_deferral)


class ThePumpPublishesAttachmentTests(unittest.TestCase):
    """The worker thread cannot ask the loop, so the loop tells it."""

    def test_an_attached_bus_reports_nobody_waiting(self):
        async def run():
            w = _worker()
            bus = _EventBus(_FakePool({"s1": w}))
            bus.subscribe()
            bus.pump_once()
            self.assertIsNone(w.unattended_since)
        asyncio.run(run())

    def test_a_bus_with_no_subscriber_starts_the_clock(self):
        w = _worker()
        bus = _EventBus(_FakePool({"s1": w}))
        bus.pump_once()
        self.assertIsNotNone(w.unattended_since)

    def test_the_clock_does_not_restart_on_every_tick(self):
        # Otherwise the grace never elapses and the card waits for ever anyway.
        w = _worker()
        bus = _EventBus(_FakePool({"s1": w}))
        bus.pump_once()
        first = w.unattended_since
        time.sleep(0.02)
        bus.pump_once()
        self.assertEqual(w.unattended_since, first)

    def test_a_client_arriving_stops_the_clock(self):
        async def run():
            w = _worker()
            bus = _EventBus(_FakePool({"s1": w}))
            bus.pump_once()
            self.assertIsNotNone(w.unattended_since)
            bus.subscribe()
            bus.pump_once()
            self.assertIsNone(w.unattended_since)
        asyncio.run(run())

    def test_a_client_leaving_restarts_it(self):
        async def run():
            w = _worker()
            bus = _EventBus(_FakePool({"s1": w}))
            sub = bus.subscribe()
            bus.pump_once()
            sub.close()
            bus.pump_once()
            self.assertIsNotNone(w.unattended_since)
        asyncio.run(run())

    def test_it_is_published_on_ticks_that_move_nothing(self):
        # The thing being reported is the absence of events, so it cannot be carried
        # by one.
        w = _worker()
        bus = _EventBus(_FakePool({"s1": w}))
        self.assertEqual(bus.pump_once(), 0)
        self.assertIsNotNone(w.unattended_since)


class TheReaperTests(unittest.TestCase):
    """A deferred turn has no pending card, so nothing else would protect it."""

    def _pool(self, worker):
        from mimir.client.ui.ws.ws_pool import _AgentPool
        pool = object.__new__(_AgentPool)
        pool.active_session_id = None
        pool._workers = {"s1": worker}
        pool._last_use = {"s1": 0.0}
        pool._wakes_pending = {}
        return pool

    def _releasable_worker(self):
        w = _worker()
        w._pending_prompt = None
        w.has_work_pending = lambda: False
        w._bg_jobs = {}
        return w

    def test_an_idle_worker_is_releasable(self):
        w = self._releasable_worker()
        self.assertTrue(self._pool(w).releasable("s1"))

    def test_one_holding_a_deferral_is_not(self):
        w = self._releasable_worker()
        w.has_deferral = True
        self.assertFalse(self._pool(w).releasable("s1"),
                         "released an agent the user still owes an answer")

    def test_a_parked_card_still_protects_it(self):
        w = self._releasable_worker()
        w._pending_prompt = {"id": "a1"}
        self.assertFalse(self._pool(w).releasable("s1"))


class NonInteractiveTests(unittest.TestCase):
    def test_the_agent_is_told_there_is_no_terminal(self):
        # Read by the policy engine's _is_interactive_session, which otherwise has to
        # infer it from a tty — and a log file is not a tty but also not proof that
        # nobody is reachable.
        w = _worker()
        self.assertFalse(w._agent.non_interactive)
        w.set_non_interactive()
        self.assertTrue(w._agent.non_interactive)

    def test_a_worker_with_no_agent_yet_is_not_an_error(self):
        w = _worker()
        w._agent = None
        w.set_non_interactive()


if __name__ == "__main__":
    unittest.main()
