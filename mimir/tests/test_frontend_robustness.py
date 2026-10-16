"""The frontends survive what one message, one race or one session can do to them."""

from __future__ import annotations

import asyncio
import json
import unittest

from mimir.client.core.settings import read_setting
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



class _SettingsPool:
    """Records what is applied, and answers with it: the pool's own contract."""

    def __init__(self, stand_in: _StandIn) -> None:
        self.stand_in = stand_in
        self.applied: list[tuple] = []

    def worker_or_detached(self, _sid):
        return self.stand_in

    def apply_setting(self, name, value):
        self.applied.append((name, value))
        return []

    def setting(self, name, _sid):
        recorded = dict(self.applied)
        return recorded[name] if name in recorded else read_setting(None, name)


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
        from mimir.client.session.commands import settings

        sess = self._session()
        with mock.patch.object(settings, "switch_backend") as switch:
            asyncio.run(sess._handle_command("/backend ray"))
        switch.assert_called_once_with("ray")
        self.assertIn(("backend", "ray"), sess.pool.applied)
        self.assertEqual(sess.ws.sent[-1]["type"], "command_output")

    def test_subagents_is_recorded_for_the_agent_built_next(self) -> None:
        from unittest import mock
        from mimir.client.config import SUBAGENT_LEVELS
        from mimir.client.session.commands import settings

        level = SUBAGENT_LEVELS[-1]
        sess = self._session()
        with mock.patch.object(settings, "save_subagent_level") as save:
            asyncio.run(sess._handle_command(f"/subagents {level}"))
        save.assert_called_once_with(level)
        self.assertIn(("subagent_level", level), sess.pool.applied)
        self.assertEqual(sess.ws.sent[-1], {"type": "subagent_level", "level": level})



class DeletedSessionTests(unittest.TestCase):
    def test_a_deleted_conversation_leaves_no_bookkeeping_behind(self) -> None:
        sess = object.__new__(_Session)
        for name in ("_live_rows", "_sent_progress", "_foreign_withheld"):
            setattr(sess, name, {"gone": object(), "kept": object()})
        sess._forget_session_state("gone")
        for name in ("_live_rows", "_sent_progress", "_foreign_withheld"):
            self.assertEqual(set(getattr(sess, name)), {"kept"}, name)

    def test_the_pool_forgets_what_it_owed_a_deleted_conversation(self) -> None:
        from mimir.client.ui.ws.ws_pool import _AgentPool

        pool = _AgentPool("m", cap=1)
        conversation = pool.conversations.new("gone")
        conversation.pending_wakes.append({"ev": {}, "told": False})
        conversation.held_checkin = {"type": "job_checkin"}
        asyncio.run(pool.close("gone"))
        # What was owed goes with the conversation object, which is forgotten.
        self.assertNotIn("gone", pool.conversations._open)



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

        async def _boom(*_a, **_kw):
            raise RuntimeError("server gone")

        with mock.patch("builtins.input", lambda _p: next(prompts)), \
             mock.patch("builtins.print", lambda *a, **k: printed.append(" ".join(map(str, a)))), \
             mock.patch.object(chat_session, "run_cli_command", _boom):
            asyncio.run(chat_session.run_chat_session(agent))
        self.assertTrue(any("/memory failed" in line for line in printed))


if __name__ == "__main__":
    unittest.main()
