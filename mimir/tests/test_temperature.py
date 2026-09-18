"""The user's sampling temperature: per model, on disk, and absent unless set.

None means the model's own: nothing is sent and the server applies the model's
generation_config. A forced low temperature sent reasoning models into repetition
loops, so the default must stay "send nothing".
"""
import asyncio
import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mimir.client.agent_core import MimirAgent
from mimir.client.config import parse_temperature, preferences
from mimir.client.ui.cli.chat_commands import handle_chat_command
# Modules, not their TestCase classes: an imported class would be collected twice.
from mimir.tests import test_session_isolation as _iso
from mimir.tests import test_thinking_depth as _depth
from mimir.tests.test_session_isolation import _bare_worker
from mimir.tests.test_thinking_depth import _LoopAgent, _tool_call


class _StateDir:
    """Point preferences.json at a throwaway directory for the duration."""

    def __enter__(self):
        self._tmp = tempfile.TemporaryDirectory()
        self._orig = preferences.STATE_DIR
        preferences.STATE_DIR = self._tmp.name
        return self._tmp.name

    def __exit__(self, *exc):
        preferences.STATE_DIR = self._orig
        self._tmp.cleanup()


class _Agent:
    """The real setter over just the model and temperature."""

    set_temperature = MimirAgent.set_temperature

    def __init__(self, model="qwen3"):
        self.model = model
        self.temperature = preferences.load_temperature(model)


class ParseTests(unittest.TestCase):
    def test_default_words_mean_none(self):
        for word in ("default", "DEFAULT", "auto", "none", "model"):
            self.assertEqual(parse_temperature(word), (True, None))

    def test_numbers_in_range(self):
        self.assertEqual(parse_temperature("0"), (True, 0.0))
        self.assertEqual(parse_temperature("0.65"), (True, 0.65))
        self.assertEqual(parse_temperature("2"), (True, 2.0))

    def test_out_of_range_and_garbage_are_refused(self):
        for raw in ("-0.1", "2.01", "hot", ""):
            self.assertFalse(parse_temperature(raw)[0], raw)


class PreferencesTests(unittest.TestCase):
    def test_round_trip_per_model(self):
        with _StateDir():
            preferences.save_temperature("a", 0.6)
            preferences.save_temperature("b", 1.1)
            self.assertEqual(preferences.load_temperature("a"), 0.6)
            self.assertEqual(preferences.load_temperature("b"), 1.1)
            self.assertIsNone(preferences.load_temperature("c"))

    def test_none_removes_the_entry(self):
        with _StateDir() as d:
            preferences.save_temperature("a", 0.6)
            preferences.save_temperature("a", None)
            self.assertIsNone(preferences.load_temperature("a"))
            with open(os.path.join(d, "preferences.json")) as f:
                self.assertNotIn("temperatures", json.load(f))

    def test_saving_toggles_keeps_temperatures_and_back(self):
        with _StateDir():
            preferences.save_temperature("a", 0.6)
            preferences.save_disabled({"strings"}, set(), set())
            self.assertEqual(preferences.load_temperature("a"), 0.6)
            preferences.save_temperature("b", 0.9)
            self.assertEqual(preferences.load_disabled()[0], {"strings"})

    def test_a_malformed_entry_reads_as_default(self):
        with _StateDir() as d:
            with open(os.path.join(d, "preferences.json"), "w") as f:
                json.dump({"temperatures": {"a": "hot", "b": True}}, f)
            self.assertIsNone(preferences.load_temperature("a"))
            self.assertIsNone(preferences.load_temperature("b"))


class AgentTests(unittest.TestCase):
    def test_starts_at_the_model_default(self):
        with _StateDir():
            self.assertIsNone(_Agent().temperature)

    def test_set_persists_and_a_new_agent_of_the_same_model_inherits_it(self):
        with _StateDir():
            _Agent("qwen3").set_temperature(0.7)
            self.assertEqual(_Agent("qwen3").temperature, 0.7)  # e.g. a sub-agent
            self.assertIsNone(_Agent("other").temperature)

    def test_out_of_range_raises_and_changes_nothing(self):
        with _StateDir():
            a = _Agent()
            with self.assertRaises(ValueError):
                a.set_temperature(3.0)
            self.assertIsNone(a.temperature)
            self.assertIsNone(preferences.load_temperature(a.model))

    def test_model_switch_reloads_that_model_s_value(self):
        with _StateDir():
            preferences.save_temperature("b", 1.2)
            a = _Agent("a")
            a.set_model = MimirAgent.set_model.__get__(a)
            with patch.dict(os.environ):
                a.set_model("b")
                self.assertEqual(a.temperature, 1.2)
                a.set_model("a")
                self.assertIsNone(a.temperature)


class LoopTests(unittest.TestCase):
    """The temperature reaches the backend only when set, and a change mid-query
    lands on the next call — like the thinking depth."""

    _script = [
        {"content": "working", "tool_calls": [_tool_call("noop")]},
        {"content": "done"},
    ]

    def _run(self, agent, on_step):
        return _depth.LiveRungChangeTests._run(self, agent, list(self._script), on_step)

    def test_unset_sends_nothing(self):
        agent = _LoopAgent()
        backend = self._run(agent, lambda: None)
        for call in backend.calls:
            self.assertNotIn("temperature", call["options"])

    def test_set_mid_query_reaches_the_next_call(self):
        agent = _LoopAgent()
        backend = self._run(agent, lambda: setattr(agent, "temperature", 0.7))
        self.assertNotIn("temperature", backend.calls[0]["options"])
        self.assertEqual(backend.calls[1]["options"]["temperature"], 0.7)


class CliTests(unittest.TestCase):
    def _cmd(self, query, temperature=None, setter=None):
        return asyncio.run(handle_chat_command(
            query=query, mode="agent", thinking=True, streaming=True, batch_mode=False,
            set_mode=lambda m: None, set_thinking=lambda b: None,
            set_streaming=lambda b: None, set_batch_mode=lambda b: None,
            temperature=temperature, set_temperature=setter,
        ))

    def test_bare_command_reports(self):
        self.assertIn("default", self._cmd("/temperature")[1])
        self.assertIn("0.6", self._cmd("/temperature", temperature=0.6)[1])

    def test_set_and_reset(self):
        got = []
        self._cmd("/temperature 0.8", setter=got.append)
        self._cmd("/temperature default", setter=got.append)
        self.assertEqual(got, [0.8, None])

    def test_invalid_value_is_a_usage_error(self):
        got = []
        handled, text = self._cmd("/temperature 5", setter=got.append)
        self.assertTrue(handled)
        self.assertIn("Usage", text)
        self.assertEqual(got, [])

    def test_a_backend_that_ignores_it_says_so(self):
        with patch.dict(os.environ, {"LLM_BACKEND": "anthropic"}):
            self.assertIn("ignores", self._cmd("/temperature")[1])


class WsCommandTests(unittest.IsolatedAsyncioTestCase):
    def _session(self, worker):
        return _iso.SessionFencingTests._session(self, worker)

    def _worker(self):
        w = _bare_worker()
        w.model = "m"
        w.state = {"supported": True, "value": None}
        w.set_temperature = lambda v: w.state.update(value=v)
        w.get_temperature_state = lambda: dict(w.state)
        return w

    async def test_set_reports_the_held_value(self):
        sess = self._session(self._worker())
        await sess._handle_command("/temperature 0.7")
        self.assertEqual(json.loads(sess.ws.sent[-1]),
                         {"type": "temperature", "supported": True, "value": 0.7})
        await sess._handle_command("/temperature default")
        self.assertIsNone(json.loads(sess.ws.sent[-1])["value"])

    async def test_invalid_value_is_an_error_and_changes_nothing(self):
        w = self._worker()
        sess = self._session(w)
        await sess._handle_command("/temperature 9")
        self.assertEqual(json.loads(sess.ws.sent[-1])["type"], "error")
        self.assertIsNone(w.state["value"])

    async def test_worker_reads_the_stored_value_before_the_agent_exists(self):
        with _StateDir():
            preferences.save_temperature("m", 0.5)
            w = _bare_worker()
            w.model = "m"
            w._agent = None
            with patch.dict(os.environ, {"LLM_BACKEND": "anthropic"}):
                self.assertEqual(w.get_temperature_state(),
                                 {"supported": False, "value": 0.5})


if __name__ == "__main__":
    unittest.main()
