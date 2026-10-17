"""The agent pool: one worker per conversation, built late and let go.

One ``AgentWorker`` per conversation, which is what lets a turn keep running in one
nobody is reading. Three properties make that affordable, and each has a failure mode
worth pinning:

* **lazily** — a worker is built on its conversation's first query. Build it eagerly and
  every conversation pays the backend wait plus ~19 server spawns whether it asks anything
  or not.
* **capped** — past the cap a turn waits for a slot. Evict a working conversation to make
  room instead, and a visible wait becomes silently lost work.
* **released** — an idle worker is closed, servers included. Skip that and a few hours of
  use strands a full set of subprocesses per conversation ever opened.

Pure-Python: the workers here are stubs, because what is under test is the bookkeeping,
not the agent.
"""
from __future__ import annotations

import asyncio
import threading
import time
import unittest

from mimir.client.ui.ws.worker.human import HumanBridge
from mimir.client.ui.ws.pool import pool as ws_pool
from mimir.client.ui.ws.pool.pool import AgentPool
from mimir.client.ui.ws.worker.outbox import Outbox


class _StubWorker:
    """Stands in for a built worker, recording what the pool did to it."""

    built = 0

    def __init__(self, model: str, session_id: str | None = None) -> None:
        self.human = HumanBridge(self)
        type(self).built += 1
        self.model = model
        self.session_id = session_id
        self.closed = False
        self.busy = False
        self.queued = False
        self._bg_jobs: dict = {}
        self.human._pending_prompt = None
        self.applied: list[tuple] = []
        self.out_q = Outbox()

    def drain(self) -> list:
        out = []
        while not self.out_q.empty():
            out.append(self.out_q.get_nowait())
        return out

    def is_busy(self) -> bool:
        return self.busy

    def watched_job_keys(self) -> list:
        return list(self._bg_jobs)

    def has_work_pending(self) -> bool:
        return self.busy or self.queued

    def aclose(self) -> None:
        self.closed = True

    def apply_setting(self, name: str, value) -> str:
        self.applied.append((name, value))
        return ""

    def set_enabled(self, kind: str, name: str, enabled: bool) -> None:
        self.applied.append((kind, name, enabled))

    # ── the pre-agent stand-in (see AgentWorker.detached) ────────────────────

    @classmethod
    def detached(cls, model: str) -> "_StubWorker":
        worker = cls(model)
        cls.built -= 1   # the stand-in is not a built agent
        worker.detached_stub = True
        return worker

    def setting(self, name: str, default=None):
        # As a worker with no agent yet answers: the caller's default first.
        if default is not None:
            return default
        return {"context_mode": "compact", "enforcement": "strict",
                "approval_mode": "manual"}[name]

    def get_thinking_profile(self) -> dict:
        return {"mechanism": "kwarg", "levels": [], "can_disable": True}

    def get_temperature_state(self) -> dict:
        return {"supported": True, "value": None}

    def agent_ready(self) -> bool:
        return not getattr(self, "detached_stub", False)


class _PoolCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _StubWorker.built = 0
        self._orig = ws_pool.AgentWorker
        ws_pool.AgentWorker = _StubWorker
        # The stub stands in for the detached worker too: building the real one is cheap
        # but is not what these tests are about.
        self.addAsyncCleanup(self._restore)

    async def _restore(self) -> None:
        ws_pool.AgentWorker = self._orig

    def _pool(self, cap: int = 2, idle_ttl: float = 1000.0) -> AgentPool:
        pool = AgentPool("test-model", cap=cap, idle_ttl=idle_ttl)
        self.addAsyncCleanup(pool.aclose_all)
        return pool


class LazyBuildTests(_PoolCase):
    async def test_nothing_is_built_until_a_conversation_asks(self) -> None:
        self._pool()
        self.assertEqual(_StubWorker.built, 0)

    async def test_the_worker_is_built_for_its_own_session(self) -> None:
        pool = self._pool()
        worker = await pool.worker_for("s1")
        self.assertEqual(worker.session_id, "s1")
        self.assertIs(pool.get("s1"), worker)

    async def test_asking_twice_builds_once(self) -> None:
        pool = self._pool()
        first = await pool.worker_for("s1")
        second = await pool.worker_for("s1")
        self.assertIs(first, second)
        self.assertEqual(_StubWorker.built, 1)

    async def test_two_queries_arriving_together_build_one_agent(self) -> None:
        """Two sends in the same tick must not each start a build: that is two sets of ~19
        servers for one conversation, one of which nothing would ever reap."""
        pool = self._pool()
        both = await asyncio.gather(pool.worker_for("s1"), pool.worker_for("s1"))
        self.assertIs(both[0], both[1])
        self.assertEqual(_StubWorker.built, 1)

    async def test_the_waiting_notice_fires_only_for_a_real_build(self) -> None:
        pool = self._pool()
        calls: list[int] = []
        await pool.worker_for("s1", on_wait=lambda: calls.append(1))
        self.assertEqual(len(calls), 1)
        await pool.worker_for("s1", on_wait=lambda: calls.append(1))
        self.assertEqual(len(calls), 1, "announced a build for a worker that exists")

    async def test_a_failed_build_does_not_poison_the_pool(self) -> None:
        pool = self._pool()

        def _boom(model, session_id=None):
            raise RuntimeError("backend never answered")

        ws_pool.AgentWorker = _boom
        with self.assertRaises(RuntimeError):
            await pool.worker_for("s1")
        self.assertNotIn("s1", pool)
        # And the conversation can try again once the backend is up.
        ws_pool.AgentWorker = _StubWorker
        self.assertIsNotNone(await pool.worker_for("s1"))


class CapAndQueueTests(_PoolCase):
    async def test_the_cap_is_respected(self) -> None:
        pool = self._pool(cap=2)
        await pool.worker_for("s1")
        await pool.worker_for("s2")
        for sid in ("s1", "s2"):
            pool.get(sid).busy = True
        self.assertIsNone(await pool.worker_for("s3"))
        self.assertEqual(len(pool), 2)

    async def test_an_idle_worker_gives_up_its_slot(self) -> None:
        pool = self._pool(cap=1)
        first = await pool.worker_for("s1")
        second = await pool.worker_for("s2")
        self.assertIsNotNone(second)
        self.assertTrue(first.closed, "the idle worker's servers were left running")
        self.assertNotIn("s1", pool)

    async def test_a_queued_turn_is_admitted_when_a_slot_frees(self) -> None:
        pool = self._pool(cap=1)
        held = await pool.worker_for("s1")
        held.busy = True
        self.assertIsNone(await pool.worker_for("s2"))

        ran: list[str] = []
        self.assertEqual(pool.enqueue("s2", lambda w: ran.append(w.session_id)), 1)
        self.assertEqual(pool.queued_position("s2"), 1)

        self.assertEqual(await pool.pump(), [], "admitted while the slot was still held")
        held.busy = False
        self.assertEqual(await pool.pump(), ["s2"])
        self.assertEqual(ran, ["s2"])
        self.assertIsNone(pool.queued_position("s2"))

    async def test_a_deleted_conversation_takes_its_queued_turn_with_it(self) -> None:
        pool = self._pool(cap=1)
        (await pool.worker_for("s1")).busy = True
        ran: list[str] = []
        pool.enqueue("s2", lambda w: ran.append("s2"))
        pool.drop_queued("s2")
        pool.get("s1").busy = False
        self.assertEqual(await pool.pump(), [])
        self.assertEqual(ran, [])


    def _slow_builds(self, pool) -> None:
        async def _build(session_id):
            await asyncio.sleep(0.01)
            return _StubWorker(pool.model, session_id=session_id)
        pool._build = _build

    async def test_builds_in_flight_count_against_the_cap(self) -> None:
        # Counted on built workers only, two sessions starting together both saw a
        # free slot and the pool ran past its cap.
        pool = self._pool(cap=1)
        self._slow_builds(pool)
        got = await asyncio.gather(pool.worker_for("s1"), pool.worker_for("s2"))
        self.assertEqual(sum(w is not None for w in got), 1)
        self.assertEqual(len(pool), 1)

    async def test_a_turn_dropped_while_its_worker_builds_never_runs(self) -> None:
        # pump() popped the head of the line after awaiting the build; a drop in that
        # window replaced the line, and the deleted session's turn ran anyway.
        pool = self._pool(cap=1)
        self._slow_builds(pool)
        ran: list[str] = []
        pool.enqueue("s2", lambda w: ran.append("s2"))
        pumping = asyncio.ensure_future(pool.pump())
        await asyncio.sleep(0)
        pool.drop_queued("s2")
        self.assertEqual(await pumping, [])
        self.assertEqual(ran, [])


class ReleaseTests(_PoolCase):
    async def test_an_idle_worker_past_its_ttl_is_released(self) -> None:
        pool = self._pool(idle_ttl=0.0)
        worker = await pool.worker_for("s1")
        self.assertEqual(await pool.release_idle(), ["s1"])
        self.assertTrue(worker.closed)

    async def test_a_busy_worker_is_never_released(self) -> None:
        pool = self._pool(idle_ttl=0.0)
        (await pool.worker_for("s1")).busy = True
        self.assertEqual(await pool.release_idle(), [])

    async def test_a_worker_with_a_queued_turn_is_never_released(self) -> None:
        """Submitted but not yet picked up: releasing here kills the turn's servers.

        The turn then starts on a closed agent and every tool call it makes fails with
        ``ClosedResourceError``. ``is_busy`` is still False in that gap, which is why
        the pool asks ``has_work_pending``.
        """
        pool = self._pool(idle_ttl=0.0)
        worker = await pool.worker_for("s1")
        worker.queued = True
        self.assertEqual(await pool.release_idle(), [])
        self.assertFalse(worker.closed)

    async def test_a_queued_turn_also_blocks_eviction_for_a_slot(self) -> None:
        """The same answer on the path that evicts to make room, not on the TTL."""
        pool = self._pool(cap=1)
        (await pool.worker_for("s1")).queued = True
        self.assertIsNone(await pool.worker_for("s2"),
                          "evicted a conversation whose turn was about to start")

    async def test_a_worker_watching_a_job_is_never_released(self) -> None:
        """Its loop hosts the watcher; evicting it loses the wake the build is owed."""
        pool = self._pool(idle_ttl=0.0)
        (await pool.worker_for("s1"))._bg_jobs = {"k": object()}
        self.assertEqual(await pool.release_idle(), [])

    async def test_a_worker_parked_on_a_card_is_never_released(self) -> None:
        """The turn is waiting on a person, with no timeout, by design."""
        pool = self._pool(idle_ttl=0.0)
        (await pool.worker_for("s1")).human._pending_prompt = {"id": "a"}
        self.assertEqual(await pool.release_idle(), [])

    async def test_the_conversation_on_screen_is_never_released(self) -> None:
        pool = self._pool(idle_ttl=0.0)
        await pool.worker_for("s1")
        pool.set_active("s1")
        self.assertEqual(await pool.release_idle(), [])

    async def test_a_slot_is_not_taken_from_a_conversation_mid_task(self) -> None:
        """The decision this encodes: wait visibly rather than lose work silently."""
        pool = self._pool(cap=1)
        held = await pool.worker_for("s1")
        held.human._pending_prompt = {"id": "waiting on the user"}
        self.assertIsNone(await pool.worker_for("s2"))
        self.assertFalse(held.closed)

    async def test_closing_a_conversation_closes_its_agent(self) -> None:
        pool = self._pool()
        worker = await pool.worker_for("s1")
        await pool.close("s1")
        self.assertTrue(worker.closed)
        self.assertNotIn("s1", pool)

    async def test_shutting_down_closes_everything(self) -> None:
        pool = self._pool(cap=3)
        workers = [await pool.worker_for(f"s{i}") for i in range(3)]
        await pool.aclose_all()
        self.assertTrue(all(w.closed for w in workers))
        self.assertEqual(len(pool), 0)

    async def test_a_slow_close_does_not_hold_up_another_conversation(self) -> None:
        """A close waits on the MCP servers going away — seconds. Held under the pool's
        lock, it kept every other conversation from having its agent built."""
        pool = self._pool(cap=3)
        slow = await pool.worker_for("s1")
        closing = threading.Event()

        def _slow_close() -> None:
            closing.set()
            time.sleep(0.5)
            slow.closed = True

        slow.aclose = _slow_close
        close = asyncio.create_task(pool.close("s1"))
        while not closing.is_set():
            await asyncio.sleep(0.01)
        started = time.monotonic()
        await pool.worker_for("s2")
        self.assertLess(time.monotonic() - started, 0.3)
        await close
        self.assertTrue(slow.closed)


class SettingsTests(_PoolCase):
    async def test_a_setting_reaches_every_live_conversation(self) -> None:
        pool = self._pool(cap=2)
        a, b = await pool.worker_for("s1"), await pool.worker_for("s2")
        pool.apply_setting("mode", "plan")
        self.assertIn(("mode", "plan"), a.applied)
        self.assertIn(("mode", "plan"), b.applied)

    async def test_a_setting_survives_into_the_next_agent_built(self) -> None:
        """The user chooses these long before an agent exists to receive them."""
        pool = self._pool()
        pool.apply_setting("mode", "plan")
        pool.apply_setting("thinking", False)
        later = await pool.worker_for("s1")
        self.assertIn(("mode", "plan"), later.applied)
        self.assertIn(("thinking", False), later.applied)

    async def test_the_last_value_of_a_setting_is_the_one_replayed(self) -> None:
        pool = self._pool()
        pool.apply_setting("mode", "plan")
        pool.apply_setting("mode", "agent")
        later = await pool.worker_for("s1")
        self.assertEqual([c for c in later.applied if c[0] == "mode"],
                         [("mode", "agent")])

    async def test_every_toggle_made_before_the_first_agent_is_replayed(self) -> None:
        # Recorded by method name, only the last of two switches survived.
        pool = self._pool()
        pool.apply_toggle("server", "web", False)
        pool.apply_toggle("server", "github", False)
        later = await pool.worker_for("s1")
        self.assertIn(("server", "web", False), later.applied)
        self.assertIn(("server", "github", False), later.applied)

    async def test_a_rejection_from_a_live_worker_is_reported(self) -> None:
        pool = self._pool()
        worker = await pool.worker_for("s1")
        worker.apply_setting = lambda name, value: "not ready"
        self.assertEqual(pool.apply_setting("mode", "plan"), ["not ready"])

    async def test_an_invalid_value_is_refused_before_it_is_recorded(self) -> None:
        # Recorded unchecked, it was only refused at replay, on an agent being built.
        pool = self._pool()
        errors = pool.apply_setting("mode", "nonsense")
        self.assertEqual(len(errors), 1)
        self.assertIn("Invalid mode", errors[0])
        later = await pool.worker_for("s1")
        self.assertEqual([c for c in later.applied if c[0] == "mode"], [])

    async def test_the_greeting_reports_a_choice_made_before_any_agent(self) -> None:
        """Reporting the bare default would show "agent" to a user who picked plan — and
        the choice is real: the pool replays it as soon as an agent exists."""
        pool = AgentPool("test-model", cap=1)
        self.addAsyncCleanup(pool.aclose_all)
        pool.apply_setting("context_mode", "full")
        self.assertEqual(pool.ui_state(None)["context_mode"], "full")

    async def test_a_conversation_with_no_agent_is_not_shown_as_starting(self) -> None:
        """Its agent is built on the first query: nothing is waiting on the backend."""
        pool = self._pool()
        self.assertTrue(pool.ui_state("s1")["agent_ready"])
        worker = await pool.worker_for("s1")
        worker.agent_ready = lambda: False
        self.assertFalse(pool.ui_state("s1")["agent_ready"])

    async def test_the_screen_pointer_is_the_pools_alone(self) -> None:
        pool = self._pool(cap=2)
        a = await pool.worker_for("s1")
        pool.set_active("s2")
        self.assertEqual(pool.active_session_id, "s2")
        self.assertFalse(hasattr(a, "active_session_id"),
                         "a worker was told what is on screen")


class AddressingASessionTests(_PoolCase):
    async def test_busy_and_parked_are_asked_per_conversation(self) -> None:
        pool = self._pool(cap=2)
        a, b = await pool.worker_for("s1"), await pool.worker_for("s2")
        a.busy = True
        b.human._pending_prompt = {"id": "x"}
        self.assertTrue(pool.is_busy("s1"))
        self.assertFalse(pool.is_busy("s2"))
        self.assertTrue(pool.is_parked("s2"))
        self.assertFalse(pool.is_parked("s1"))

    async def test_a_conversation_with_no_agent_is_neither(self) -> None:
        pool = self._pool()
        self.assertFalse(pool.is_busy("never-asked"))
        self.assertFalse(pool.is_parked("never-asked"))


if __name__ == "__main__":
    unittest.main()


class AdmissionLatencyTests(_PoolCase):
    """A freed slot admits the waiting conversation at once, not on the next sweep.

    The idle sweep runs every ``_REAP_INTERVAL`` seconds, so leaving admission to it makes
    a queued conversation sit through the rest of that interval after the slot it needs
    has already freed — having just been told it is next in line.
    """

    async def test_a_turn_ending_is_enough_to_admit_the_next(self) -> None:
        pool = self._pool(cap=1)
        held = await pool.worker_for("s1")
        held.busy = True
        ran: list[str] = []
        pool.enqueue("s2", lambda w: ran.append(w.session_id))

        # Nothing is admitted while the slot is genuinely held.
        self.assertEqual(await pool.pump(), [])
        # The turn ends; the conversation is idle but nowhere near any TTL.
        held.busy = False
        self.assertLess(pool.idle_for("s1"), pool.idle_ttl)
        self.assertEqual(await pool.pump(), ["s2"])
        self.assertEqual(ran, ["s2"])

    async def test_the_session_reports_the_admission(self) -> None:
        """Said out loud: the queue notice is still on screen until something replaces it."""
        from mimir.tests.support.connections import bare_session
        pool = self._pool(cap=1)
        held = await pool.worker_for("s1")
        held.busy = True
        pool.enqueue("s2", lambda w: None)
        sess = bare_session(pool, active="s1")
        held.busy = False
        await sess._admit_waiting()
        notices = [m for m in sess.ws.sent if m.get("session_id") == "s2"]
        self.assertEqual(len(notices), 1)
        self.assertIn("starting", notices[0]["text"])

    async def test_a_turn_ending_here_or_elsewhere_admits(self) -> None:
        import types
        from unittest import mock
        from mimir.client.ui.ws.connection.router import EventRouter
        session = types.SimpleNamespace(
            _rows_for=lambda owner: {}, _sent_progress={},
            _emit_context_usage=mock.AsyncMock(), _admit_waiting=mock.AsyncMock(),
            _announce_detached_answer=mock.AsyncMock(), _foreign_withheld={})
        router = EventRouter(session)
        await router._after_error({"type": "error"}, {}, "s1")
        await router._foreign({"type": "error"}, {}, "s2")
        self.assertEqual(session._admit_waiting.await_count, 2)


class AnEventWakesThePumpTests(_PoolCase):
    """The pump used to look every 50 ms whether anything was there or not."""

    async def test_an_event_from_the_worker_thread_wakes_the_pump(self) -> None:
        import threading
        pool = self._pool()
        worker = await pool.worker_for("s1")
        pool.bus._interval = 60.0          # nothing but the nudge can wake it in time
        pool.bus._wake.clear()
        threading.Thread(target=worker.out_q.put,
                         args=({"type": "output", "text": "x"},)).start()
        await asyncio.wait_for(pool.bus._wake.wait(), 2.0)


class ShutdownClosesEverythingAtOnceTests(_PoolCase):
    """Closing waits on MCP servers going away, so in series it costs that wait per
    conversation — a minute of a process already asked to stop, while its replacement
    starts beside it."""

    async def test_every_agent_is_closed(self) -> None:
        pool = self._pool(cap=3)
        workers = [await pool.worker_for(f"s{i}") for i in range(3)]
        await pool.aclose_all()
        self.assertTrue(all(w.closed for w in workers))
        self.assertEqual(len(pool), 0)

    async def test_a_slow_close_does_not_hold_up_the_others(self) -> None:
        import time
        pool = self._pool(cap=3)
        workers = [await pool.worker_for(f"s{i}") for i in range(3)]
        for w in workers:
            original = w.aclose

            def _slow(orig=original):
                time.sleep(0.3)
                orig()

            w.aclose = _slow
        started = time.monotonic()
        await pool.aclose_all()
        elapsed = time.monotonic() - started
        self.assertTrue(all(w.closed for w in workers))
        self.assertLess(elapsed, 0.9, f"closed in series: {elapsed:.2f}s for 3 × 0.3s")

    async def test_one_failing_close_does_not_abandon_the_rest(self) -> None:
        pool = self._pool(cap=3)
        workers = [await pool.worker_for(f"s{i}") for i in range(3)]

        def _boom():
            raise RuntimeError("its loop was already gone")

        workers[1].aclose = _boom
        await pool.aclose_all()
        self.assertTrue(workers[0].closed)
        self.assertTrue(workers[2].closed)
        self.assertEqual(len(pool), 0)


class OpeningAConversationCostsNothingTests(_PoolCase):
    """Reading conversations is free; only asking one something builds its agent.

    The cap bounds how many conversations can be *working*, not how many exist or how many
    the user looks at. Building on open would spend the backend wait and ~19 server spawns
    on a conversation the user is only reading, and would have browsing a history of ten
    hit the cap on the fourth click.
    """

    async def test_the_pool_is_empty_until_something_is_asked(self) -> None:
        pool = self._pool(cap=1)
        # Whatever a session layer does on open — read its activity, look it up, point the
        # screen at it — none of it may build.
        for sid in ("s1", "s2", "s3", "s4"):
            pool.set_active(sid)
            self.assertIsNone(pool.get(sid))
            self.assertFalse(pool.is_busy(sid))
            self.assertFalse(pool.is_parked(sid))
            self.assertIsNone(pool.queued_position(sid))
            pool.ui_state(sid)
        self.assertEqual(len(pool), 0)
        self.assertEqual(_StubWorker.built, 0)

    async def test_only_an_explicit_ask_can_build_one(self) -> None:
        """Two asks build an agent: a query, and opening the drawer.

        The query is the obvious one. The other is the scientific-computing drawer being
        opened: its sections are filled by MCP servers — the clusters this workspace
        reaches, the machine, the optimisation in progress — and a conversation that has
        no agent holds none of them, so that drawer showed nothing until something had
        been asked of the model. It is one build, for the conversation already on screen,
        that its first query would have paid for anyway.

        What this still forbids is the thing the cap is about: *reading* a conversation —
        switching to it, listing it, replaying it — must never build, or browsing a
        history of ten would hit the cap on the fourth click.
        """
        import asyncio
        from mimir.tests.support.connections import bare_session
        pool = self._pool(cap=2)
        sess = bare_session(pool, active="s1")
        sess._request_panel = lambda **_kw: None   # the report itself is not under test
        await sess._handle_list_panel({"type": "list_panel"})        # a timer asking
        await asyncio.sleep(0.05)
        self.assertEqual(_StubWorker.built, 0)
        await sess._handle_list_panel({"type": "list_panel", "build": True})
        for _ in range(20):
            if _StubWorker.built:
                break
            await asyncio.sleep(0.01)
        self.assertEqual(_StubWorker.built, 1)
        self.assertIsNotNone(pool.get("s1"))

    async def test_returning_to_a_conversation_reuses_its_agent(self) -> None:
        """Its context is live, so coming back is instant and costs no rebuild."""
        pool = self._pool(cap=2)
        first = await pool.worker_for("s1")
        pool.set_active("s2")
        pool.set_active("s1")
        self.assertIs(pool.get("s1"), first)
        self.assertEqual(_StubWorker.built, 1)
