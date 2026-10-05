"""A resumed session is accounted for as the session it was, not as a default.

Reopening a conversation whose context had been compacted showed the bar over-full,
and the first query then compacted and trimmed a history that fit the real window
perfectly well. Three separate causes, one per group below:

* the window was sized from a mode the conversation was not in, because the budget was
  read before the agent that answers for it existed,
* the mode itself was not saved with the session, so there was nothing to fall back to,
* and everything that makes the count exact — the server-measured prompt overhead and
  the history's chars-per-token — lived in the process and died with it, so the figure
  moved on its own as soon as the first answer re-measured it.
"""
from __future__ import annotations

import json
import tempfile
import unittest
from unittest.mock import patch

from mimir.client.config import constants
from mimir.client.config.constants import context_budget_for
from mimir.client.query_engine import token_calibration
from mimir.client.query_engine.backends.base import LLMBackend
from mimir.client.ui.ws.session_store import FullSession
from mimir.client.ui.ws.ws_session import _Session
from mimir.client.ui.ws.ws_worker import _AgentWorker
from mimir.tests._fake_pool import FakePool


class _NoAgentWorker:
    """What the pool hands back for a conversation that has no agent yet."""

    model = "test-model"

    def get_context_mode(self, default: str = "full") -> str:
        return default

    def context_overhead_tokens(self) -> int:
        return 0  # no agent, so the system prompt and tools are unknown

    def context_overhead_is_measured(self) -> bool:
        return False

    def live_history(self):
        return None


def _session(mode: str = "full") -> _Session:
    sess = object.__new__(_Session)
    sess.pool = FakePool(_NoAgentWorker(), active="s1")
    sess._active_session_id = "s1"
    sess._resumed_context_mode = mode
    sess.history = []
    sess.history_full = []
    return sess


class ResumedBudgetTests(unittest.TestCase):
    """The window a resumed session is measured against is its own."""

    def test_a_resumed_full_session_keeps_the_full_window(self):
        full_total, _, _, _ = context_budget_for("test-model", "full")
        total, reserved = _session("full")._ctx_budget()
        self.assertEqual(total, full_total)
        self.assertGreater(total, constants.CTX_TOTAL_COMPACT)

    def test_a_resumed_compact_session_keeps_the_compact_window(self):
        total, _ = _session("compact")._ctx_budget()
        self.assertEqual(total, constants.CTX_TOTAL_COMPACT)

    def test_a_live_agent_still_answers_for_itself(self):
        """The saved mode is a fallback, not an override: a mode switched mid-session
        must not be undone by what the session was saved with."""
        sess = _session("compact")

        class _Live(_NoAgentWorker):
            def get_context_mode(self, default: str = "full") -> str:
                return "full"

        sess.pool = FakePool(_Live(), active="s1")
        total, _ = sess._ctx_budget()
        self.assertEqual(total, context_budget_for("test-model", "full")[0])


class SavedContextModeTests(unittest.TestCase):
    def test_the_mode_round_trips_through_the_store(self):
        session = FullSession(id="s1", title="t", created_at="now", updated_at="now",
                              context_mode="compact")
        self.assertEqual(FullSession.from_dict(session.to_dict()).context_mode, "compact")

    def test_a_session_saved_before_the_field_existed_reads_as_full(self):
        """Full is what a rebuilt agent comes up in, so it is the honest assumption."""
        self.assertEqual(
            FullSession.from_dict({"id": "s1", "title": "t"}).context_mode, "full"
        )


class BudgetOrderTests(unittest.IsolatedAsyncioTestCase):
    """The budget is read after the agent exists, never before."""

    async def test_the_window_is_fitted_only_once_the_agent_is_resolved(self):
        order: list[str] = []
        sess = _session()

        async def _ensure_worker():
            order.append("agent")
            return None  # pool full: the turn is queued, which is not what this asks

        async def _fit():
            order.append("budget")

        sess._ensure_worker = _ensure_worker
        sess._fit_history_to_budget = _fit
        sess._pending_interaction = None
        sess._display_messages = []
        sess._autosave_session = lambda *_: None
        sess._stale_prompt_ids = set()
        sess.transcript = type("T", (), {"append": lambda *_: None})()
        sess.pool.enqueue = lambda *_a, **_k: 1
        sess.ws = type("WS", (), {"send": staticmethod(lambda *_: _done())})()

        async def _done():
            return None

        await sess._handle_query({"text": "carry on"})
        self.assertEqual(order, ["agent", "budget"])


class _CountingBackend(LLMBackend):
    def chat(self, model, messages, tools, thinking, streaming, options, **_):
        return {"role": "assistant", "content": "ok"}

    def list_models(self):
        return []


class RememberedCalibrationTests(unittest.TestCase):
    """What the server measured once is not re-estimated after a restart."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patcher = patch.object(constants, "STATE_DIR", self._dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        token_calibration.reset_for_tests()
        self.addCleanup(token_calibration.reset_for_tests)

    def _fresh_process(self) -> None:
        """What a restart does to the cache: the in-process copy goes, the file stays."""
        token_calibration.reset_for_tests()

    def test_a_remembered_ratio_counts_the_history_like_the_server_did(self):
        token_calibration.remember_chars_per_token("m", 5.0)
        self._fresh_process()
        # 10k characters at the measured 5.0, not at the default 4.
        self.assertEqual(_CountingBackend().count_text_tokens("m", "z" * 10_000), 2000)

    def test_this_process_own_measurement_wins_over_the_remembered_one(self):
        token_calibration.remember_chars_per_token("m", 5.0)
        backend = _CountingBackend()
        backend._calibrated_cpt["m"] = 4.0
        self.assertEqual(backend.count_text_tokens("m", "z" * 10_000), 2500)

    def test_an_overhead_comes_back_for_the_same_fixed_part(self):
        key = token_calibration.overhead_key("m", "full", "SYSTEM", "[tools]")
        token_calibration.remember_overhead(key, 36_000)
        self._fresh_process()
        same = token_calibration.overhead_key("m", "full", "SYSTEM", "[tools]")
        self.assertEqual(token_calibration.recall_overhead(same), 36_000)

    def test_a_changed_fixed_part_is_a_miss_rather_than_a_stale_hit(self):
        """A server switched on, a mode changed, a prompt edited: the measurement no
        longer describes the prompt, and an estimate is better than a wrong figure
        wearing the authority of a measurement."""
        token_calibration.remember_overhead(
            token_calibration.overhead_key("m", "full", "SYSTEM", "[tools]"), 36_000
        )
        self._fresh_process()
        for label, key in (
            ("another model", token_calibration.overhead_key("other", "full", "SYSTEM", "[tools]")),
            ("another mode", token_calibration.overhead_key("m", "compact", "SYSTEM", "[tools]")),
            ("an edited prompt", token_calibration.overhead_key("m", "full", "SYSTEM v2", "[tools]")),
            ("one more server", token_calibration.overhead_key("m", "full", "SYSTEM", "[tools, more]")),
        ):
            with self.subTest(label):
                self.assertIsNone(token_calibration.recall_overhead(key))

    def test_the_cache_does_not_grow_without_bound(self):
        for i in range(token_calibration._MAX_OVERHEAD_ENTRIES + 10):
            token_calibration.remember_overhead(
                token_calibration.overhead_key("m", "full", f"prompt {i}", ""), 1000 + i
            )
        self._fresh_process()
        entries = token_calibration._load_locked()["prompt_overhead"]
        self.assertEqual(len(entries), token_calibration._MAX_OVERHEAD_ENTRIES)
        # The ones that went are the oldest, not the ones still in use.
        newest = token_calibration.overhead_key(
            "m", "full", f"prompt {token_calibration._MAX_OVERHEAD_ENTRIES + 9}", ""
        )
        self.assertIn(newest, entries)

    def test_an_unreadable_cache_costs_nothing(self):
        with open(token_calibration._path(), "w", encoding="utf-8") as fh:
            fh.write("{not json")
        self._fresh_process()
        self.assertIsNone(token_calibration.recall_chars_per_token("m"))
        self.assertEqual(_CountingBackend().count_text_tokens("m", "z" * 10_000), 2500)


class _StubAgent:
    mode = "agent"
    context_mode = "full"
    tools = [{"name": "read_file"}]

    def build_system_content_now(self, mode: str) -> str:
        return "SYSTEM " * 100

    def advertised_tools_for_mode(self, mode: str):
        return self.tools


class MeasuredOverheadTests(unittest.TestCase):
    """What the bar is told about the overhead it is shown."""

    def setUp(self) -> None:
        self._dir = tempfile.TemporaryDirectory()
        self.addCleanup(self._dir.cleanup)
        patcher = patch.object(constants, "STATE_DIR", self._dir.name)
        patcher.start()
        self.addCleanup(patcher.stop)
        token_calibration.reset_for_tests()
        self.addCleanup(token_calibration.reset_for_tests)
        self._worker = _AgentWorker.detached("m")
        self._worker._agent = _StubAgent()

    def _overhead(self) -> int:
        with patch("mimir.client.query_engine.backends.factory.get_backend",
                   return_value=_CountingBackend()):
            return self._worker.context_overhead_tokens()

    def test_an_estimate_says_it_is_one(self):
        self.assertGreater(self._overhead(), 0)
        self.assertFalse(self._worker.context_overhead_is_measured())

    def test_a_remembered_figure_is_reported_as_measured(self):
        """It is: a server counted it, against this very prompt and tool set."""
        agent = self._worker._agent
        key = token_calibration.overhead_key(
            "m", agent.context_mode, agent.build_system_content_now(agent.mode),
            json.dumps(agent.tools),
        )
        token_calibration.remember_overhead(key, 36_000)
        token_calibration.reset_for_tests()  # a fresh process reads it off disk
        self.assertEqual(self._overhead(), 36_000)
        self.assertTrue(self._worker.context_overhead_is_measured())

    def test_a_changed_tool_set_falls_back_to_the_estimate(self):
        agent = self._worker._agent
        token_calibration.remember_overhead(
            token_calibration.overhead_key(
                "m", agent.context_mode, agent.build_system_content_now(agent.mode),
                '[{"name": "read_file"}, {"name": "write_file"}]',
            ),
            36_000,
        )
        token_calibration.reset_for_tests()
        self.assertNotEqual(self._overhead(), 36_000)
        self.assertFalse(self._worker.context_overhead_is_measured())


if __name__ == "__main__":
    unittest.main()
