"""The agent pool: one worker per conversation, built late and let go.

One ``_AgentWorker`` per conversation, which is what lets a turn keep running in one
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
import unittest

from mimir.client.ui.ws import ws_pool
from mimir.client.ui.ws.ws_pool import _AgentPool


class _StubWorker:
    """Stands in for a built worker, recording what the pool did to it."""

    built = 0

    def __init__(self, model: str, session_id: str | None = None) -> None:
        type(self).built += 1
        self.model = model
        self.session_id = session_id
        self.active_session_id: str | None = None
        self.closed = False
        self.busy = False
        self._bg_jobs: dict = {}
        self._pending_prompt: dict | None = None
        self.applied: list[tuple] = []

    def is_busy(self) -> bool:
        return self.busy

    def aclose(self) -> None:
        self.closed = True

    def set_mode(self, mode: str) -> str:
        self.applied.append(("set_mode", mode))
        return ""

    def set_thinking(self, on: bool) -> None:
        self.applied.append(("set_thinking", on))

    def set_context_mode(self, mode: str) -> None:
        self.applied.append(("set_context_mode", mode))

    # ── the pre-agent stand-in (see _AgentWorker.detached) ────────────────────

    @classmethod
    def detached(cls, model: str) -> "_StubWorker":
        worker = cls(model)
        cls.built -= 1   # the stand-in is not a built agent
        worker.detached_stub = True
        return worker

    def get_context_mode(self) -> str:
        return "compact"

    def get_enforcement(self) -> str:
        return "strict"

    def get_approval_mode(self) -> str:
        return "manual"

    def get_thinking_profile(self) -> dict:
        return {"mechanism": "kwarg", "levels": [], "can_disable": True}

    def get_temperature_state(self) -> dict:
        return {"supported": True, "value": None}

    def agent_ready(self) -> bool:
        return not getattr(self, "detached_stub", False)


class _PoolCase(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        _StubWorker.built = 0
        self._orig = ws_pool._AgentWorker
        ws_pool._AgentWorker = _StubWorker
        # The stub stands in for the detached worker too: building the real one is cheap
        # but is not what these tests are about.
        self.addAsyncCleanup(self._restore)

    async def _restore(self) -> None:
        ws_pool._AgentWorker = self._orig

    def _pool(self, cap: int = 2, idle_ttl: float = 1000.0) -> _AgentPool:
        pool = _AgentPool("test-model", cap=cap, idle_ttl=idle_ttl)
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

        ws_pool._AgentWorker = _boom
        with self.assertRaises(RuntimeError):
            await pool.worker_for("s1")
        self.assertNotIn("s1", pool)
        # And the conversation can try again once the backend is up.
        ws_pool._AgentWorker = _StubWorker
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

    async def test_a_worker_watching_a_job_is_never_released(self) -> None:
        """Its loop hosts the watcher; evicting it loses the wake the build is owed."""
        pool = self._pool(idle_ttl=0.0)
        (await pool.worker_for("s1"))._bg_jobs = {"k": object()}
        self.assertEqual(await pool.release_idle(), [])

    async def test_a_worker_parked_on_a_card_is_never_released(self) -> None:
        """The turn is waiting on a person, with no timeout, by design."""
        pool = self._pool(idle_ttl=0.0)
        (await pool.worker_for("s1"))._pending_prompt = {"id": "a"}
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
        held._pending_prompt = {"id": "waiting on the user"}
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


class SettingsTests(_PoolCase):
    async def test_a_setting_reaches_every_live_conversation(self) -> None:
        pool = self._pool(cap=2)
        a, b = await pool.worker_for("s1"), await pool.worker_for("s2")
        pool.apply_setting("set_mode", "plan")
        self.assertIn(("set_mode", "plan"), a.applied)
        self.assertIn(("set_mode", "plan"), b.applied)

    async def test_a_setting_survives_into_the_next_agent_built(self) -> None:
        """The user chooses these long before an agent exists to receive them."""
        pool = self._pool()
        pool.apply_setting("set_mode", "plan")
        pool.apply_setting("set_thinking", False)
        later = await pool.worker_for("s1")
        self.assertIn(("set_mode", "plan"), later.applied)
        self.assertIn(("set_thinking", False), later.applied)

    async def test_the_last_value_of_a_setting_is_the_one_replayed(self) -> None:
        pool = self._pool()
        pool.apply_setting("set_mode", "plan")
        pool.apply_setting("set_mode", "agent")
        later = await pool.worker_for("s1")
        self.assertEqual([c for c in later.applied if c[0] == "set_mode"],
                         [("set_mode", "agent")])

    async def test_a_rejection_from_a_live_worker_is_reported(self) -> None:
        pool = self._pool()
        worker = await pool.worker_for("s1")
        worker.set_mode = lambda mode: "no such mode"
        self.assertEqual(pool.apply_setting("set_mode", "nonsense"), ["no such mode"])

    async def test_the_greeting_reports_a_choice_made_before_any_agent(self) -> None:
        """Reporting the bare default would show "agent" to a user who picked plan — and
        the choice is real: the pool replays it as soon as an agent exists."""
        pool = _AgentPool("test-model", cap=1)
        self.addAsyncCleanup(pool.aclose_all)
        pool.apply_setting("set_context_mode", "full")
        self.assertEqual(pool.ui_state(None)["context_mode"], "full")

    async def test_the_screen_pointer_reaches_every_worker(self) -> None:
        pool = self._pool(cap=2)
        a, b = await pool.worker_for("s1"), await pool.worker_for("s2")
        pool.set_active("s2")
        self.assertEqual(a.active_session_id, "s2")
        self.assertEqual(b.active_session_id, "s2")


class AddressingASessionTests(_PoolCase):
    async def test_busy_and_parked_are_asked_per_conversation(self) -> None:
        pool = self._pool(cap=2)
        a, b = await pool.worker_for("s1"), await pool.worker_for("s2")
        a.busy = True
        b._pending_prompt = {"id": "x"}
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
