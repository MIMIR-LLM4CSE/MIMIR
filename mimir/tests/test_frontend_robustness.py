"""The frontends survive what one message, one race or one session can do to them."""

from __future__ import annotations

import asyncio
import json
import unittest

from mimir.client.ui.ws.ws_session import _Session


class _FakeWS:
    def __init__(self) -> None:
        self.sent: list[dict] = []

    async def send(self, raw: str) -> None:
        self.sent.append(json.loads(raw))


class HandlerIsolationTests(unittest.TestCase):
    def test_a_failing_handler_reports_and_leaves_the_connection_up(self) -> None:
        # Raised out of _handle, the exception ended the read loop: the socket stayed
        # open and nothing the user sent afterwards was ever read.
        sess = object.__new__(_Session)
        sess.ws = _FakeWS()

        async def _boom(_msg):
            raise AttributeError("'NoneType' object has no attribute 'set_backend'")

        sess._handle_rename_session = _boom
        asyncio.run(sess._handle(json.dumps({"type": "rename_session"})))
        self.assertEqual(sess.ws.sent[-1]["type"], "error")
        self.assertIn("rename_session", sess.ws.sent[-1]["text"])


class _Pool:
    def __init__(self) -> None:
        self.turns: list[tuple[str, str]] = []
        self.worker = object()

    def get(self, session_id):
        return self.worker

    def _submit_turn(self, owner, worker, text) -> bool:
        self.turns.append((owner, text))
        return True


class SteerOwnershipTests(unittest.TestCase):
    def test_unread_steering_of_another_conversation_stays_in_it(self) -> None:
        # Addressed to "the session on screen", a switch during the answer sent one
        # conversation's unread message into the other one's history.
        sess = object.__new__(_Session)
        sess.ws = _FakeWS()
        sess.pool = _Pool()
        sess._active_session_id = "on-screen"
        live = [{"role": "user", "content": "typed on screen"}]
        sess._live_steer = list(live)
        sess.history = [{"role": "user", "content": "hello"}]
        asyncio.run(sess._flush_unconsumed_steer("elsewhere", ["and also this"]))
        self.assertEqual(sess.pool.turns, [("elsewhere", "and also this")])
        self.assertEqual(sess.history, [{"role": "user", "content": "hello"}])
        self.assertEqual(sess._live_steer, live)
        self.assertEqual(sess.ws.sent, [])



class _StandIn:
    """What ``_Session.worker`` answers before a conversation's first query."""

    _agent = None

    def __init__(self, payload: dict | None = None) -> None:
        self._payload = payload or {}

    def call_session_tool(self, tool, args):
        import concurrent.futures

        fut: concurrent.futures.Future = concurrent.futures.Future()
        fut.set_result(self._payload)
        return fut

    def get_subagent_level(self) -> str:
        return "default"


class _SettingsPool:
    def __init__(self, stand_in: _StandIn) -> None:
        self.stand_in = stand_in
        self.applied: list[tuple] = []

    def worker_or_detached(self, _sid):
        return self.stand_in

    def apply_setting(self, name, *args):
        self.applied.append((name, *args))
        return []


class CommandsBeforeTheFirstQueryTests(unittest.TestCase):
    def _session(self, payload: dict | None = None) -> _Session:
        sess = object.__new__(_Session)
        sess.ws = _FakeWS()
        sess.pool = _SettingsPool(_StandIn(payload))
        sess._active_session_id = "s1"
        return sess

    def test_memory_list_reports_an_error_as_an_error(self) -> None:
        # Read as an empty listing, a tool error said "No memories stored".
        sess = self._session({"status": "error", "error": "No agent session is running."})
        asyncio.run(sess._handle_command("/memory list"))
        self.assertEqual(sess.ws.sent[-1]["type"], "error")
        self.assertIn("No agent session", sess.ws.sent[-1]["text"])

    def test_backend_switches_with_no_agent_up(self) -> None:
        from unittest import mock
        from mimir.client.ui.ws import ws_session

        sess = self._session()
        with mock.patch.object(ws_session, "switch_backend") as switch:
            asyncio.run(sess._handle_command("/backend ray"))
        switch.assert_called_once_with("ray")
        self.assertIn(("backend", "ray"), sess.pool.applied)
        self.assertEqual(sess.ws.sent[-1]["type"], "command_output")

    def test_subagents_is_recorded_for_the_agent_built_next(self) -> None:
        from unittest import mock
        from mimir.client.config import SUBAGENT_LEVELS
        from mimir.client.ui.ws import ws_session

        level = SUBAGENT_LEVELS[-1]
        sess = self._session()
        with mock.patch.object(ws_session, "save_subagent_level") as save:
            asyncio.run(sess._handle_command(f"/subagents {level}"))
        save.assert_called_once_with(level)
        self.assertIn(("subagent_level", level), sess.pool.applied)
        self.assertEqual(sess.ws.sent[-1], {"type": "subagent_level", "level": level})



class DeletedSessionTests(unittest.TestCase):
    def test_a_deleted_conversation_leaves_no_bookkeeping_behind(self) -> None:
        sess = object.__new__(_Session)
        for name in ("_pending_wakes", "_held_checkin", "_detached_turns", "_live_rows",
                     "_sent_progress", "_foreign_withheld"):
            setattr(sess, name, {"gone": object(), "kept": object()})
        sess._forget_session_state("gone")
        for name in ("_pending_wakes", "_held_checkin", "_detached_turns", "_live_rows",
                     "_sent_progress", "_foreign_withheld"):
            self.assertEqual(set(getattr(sess, name)), {"kept"}, name)

    def test_the_pool_forgets_what_it_owed_a_deleted_conversation(self) -> None:
        from mimir.client.ui.ws.ws_pool import _AgentPool

        pool = _AgentPool("m", cap=1)
        pool._wakes_pending["gone"] = [{"ev": {}, "told": False}]
        pool._held_checkin["gone"] = {"type": "job_checkin"}
        asyncio.run(pool.close("gone"))
        self.assertNotIn("gone", pool._wakes_pending)
        self.assertNotIn("gone", pool._held_checkin)



class CliCommandFailureTests(unittest.TestCase):
    def test_a_failing_command_does_not_end_the_chat(self) -> None:
        # Outside the loop's guard, an exception from a command ended the session.
        import types
        from unittest import mock
        from mimir.client.ui.cli import chat_session

        prompts = iter(["/memory list", "quit"])
        agent = types.SimpleNamespace(
            tool_caps={}, mode="agent", thinking=False, streaming=False,
            approvals=types.SimpleNamespace(
                batch_mode=False, approval_mode="manual", trust_tool=None,
                untrust_tool=None, trusted_tools_view=lambda: []),
            context_mode="full", enforcement="strict", thinking_depth=0, temperature=None,
            apply_setting=None)
        printed: list[str] = []

        async def _boom(**_kw):
            raise RuntimeError("server gone")

        with mock.patch("builtins.input", lambda _p: next(prompts)), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))), \
             mock.patch.object(chat_session, "handle_chat_command", _boom):
            asyncio.run(chat_session.run_chat_session(agent))
        self.assertTrue(any("/memory failed" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
