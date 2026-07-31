"""Direct unit coverage for agent-loop functions that had none.

Covers ``_maybe_compact_intra_query``, ``_post_dispatch_inject``,
``_finalize_answer``, ``_run_plan_mode``, and the non-interactive
``run_agent_query`` path — driving the latter through the real ``_stream_chat``
wrapper with a ``ScriptedBackend`` so the loop's plumbing is exercised, not mocked.

Plain ``unittest`` + ``asyncio.run`` to match the rest of the suite.
"""
from __future__ import annotations

import asyncio
import contextlib
import json
import os
import tempfile
import types
import unittest
from unittest.mock import patch

import mimir.client.query_engine.agent_loop as agent_loop_module
import mimir.client.query_engine.streaming as streaming_module
import mimir.client.query_engine.history as history_module
import mimir.client.query_engine.finalize as finalize_module
import mimir.client.query_engine.dispatch as dispatch_module
import mimir.client.query_engine.plan_loop as plan_loop_module
import mimir.client.event_sink as event_sink_module
from mimir.client.context.capabilities import (
    PLAN_BLOCKED, PLAN_READONLY, TASK_PLANNING, ToolCaps,
)
from mimir.client.context.execution_context import build_execution_context, loop_control
from mimir.client.guardrails.observations import _observe_todo_flags
from mimir.client.query_engine.streaming import _to_dict
from mimir.client.tool_execution.normalizer import _make_hashable
from mimir.client.agent_core import MimirAgent
from mimir.tests._fake_backend import ScriptedBackend


def _tool_call(name: str, args: str = "{}", call_id: str = "1") -> dict:
    return {"id": call_id, "function": {"name": name, "arguments": args}}


# A minimal live registry classifying the checklist tool, mirroring what
# connect_server builds from the todo server's self-declared caps: a TASK_PLANNING
# tool whose `plan_steps` arg-role marks it as the ordered-steps checklist.
# `todo_set_plan` carries the same TASK_PLANNING capability with NO plan_steps role:
# it is the prose plan document, the thing the user actually reads and approves. Both
# forms must be in the registry or plan-mode tests cannot see the case where a model
# records its plan in prose and never produces the checklist.
_CHECKLIST_CAPS = {
    "todo_write": ToolCaps(
        name="todo_write",
        capabilities=frozenset({TASK_PLANNING}),
        arg_roles={"plan_steps": ("steps",)},
    ),
    "todo_set_plan": ToolCaps(
        name="todo_set_plan",
        capabilities=frozenset({TASK_PLANNING}),
    ),
}


def _record_plan_flags(tool_calls, agent, execution_context) -> None:
    """Set the plan flags a real dispatch would, via the production observer.

    Plan mode reads "is a plan recorded" from the execution context, which
    ``record_tool_observation`` populates on every call. A test dispatch that skipped
    this would exercise a loop that can never see a recorded plan — so the real
    observer is called here rather than a local imitation of its rules.
    """
    for tc in tool_calls:
        name = _to_dict(_to_dict(tc).get("function", {})).get("name", "")
        _observe_todo_flags(agent, name, "ok", execution_context)


class MaybeCompactIntraQueryTests(unittest.TestCase):
    def _messages(self, n_middle: int) -> list[dict]:
        msgs = [
            {"role": "system", "content": "S"},
            {"role": "user", "content": "U"},
        ]
        msgs += [{"role": "assistant", "content": f"m{i}"} for i in range(n_middle)]
        msgs += [{"role": "assistant", "content": f"tail{i}"} for i in range(4)]
        return msgs

    def test_compacts_middle_when_over_budget(self) -> None:
        messages = self._messages(n_middle=4)  # 10 total → middle = messages[2:-4]
        original_head = messages[:2]
        original_tail = messages[-4:]

        compacted = lambda middle: [{"role": "assistant", "content": "SUMMARY"}]
        with patch.object(history_module, "_INTRA_QUERY_COMPACT_TOKENS", 1):
            history_module._maybe_compact_intra_query(
                messages, "S", {}, compacted, token_counter=len,
            )

        self.assertEqual(messages[:2], original_head)
        self.assertEqual(messages[-4:], original_tail)
        self.assertEqual(messages[2:-4], [{"role": "assistant", "content": "SUMMARY"}])

    def test_noop_under_budget(self) -> None:
        messages = self._messages(n_middle=4)
        snapshot = [dict(m) for m in messages]
        called = {"n": 0}

        def _compact(middle):
            called["n"] += 1
            return []

        # Huge budget → never triggers; compact_fn must not be called.
        with patch.object(history_module, "_INTRA_QUERY_COMPACT_TOKENS", 10**9):
            history_module._maybe_compact_intra_query(
                messages, "S", {}, _compact, token_counter=len,
            )
        self.assertEqual(messages, snapshot)
        self.assertEqual(called["n"], 0)

    def test_noop_when_too_few_messages(self) -> None:
        messages = self._messages(n_middle=1)  # 7 total < 8 → never compacts
        snapshot = [dict(m) for m in messages]
        with patch.object(history_module, "_INTRA_QUERY_COMPACT_TOKENS", 1):
            history_module._maybe_compact_intra_query(
                messages, "S", {}, lambda m: [{"role": "assistant", "content": "X"}],
                token_counter=len,
            )
        self.assertEqual(messages, snapshot)

    def test_noop_when_no_compact_fn(self) -> None:
        messages = self._messages(n_middle=4)
        snapshot = [dict(m) for m in messages]
        history_module._maybe_compact_intra_query(messages, "S", {}, None, token_counter=len)
        self.assertEqual(messages, snapshot)


class PostDispatchInjectTests(unittest.TestCase):
    def test_injects_todo_reminder_after_successful_edit(self) -> None:
        messages: list[dict] = []
        ec = {
            "last_edit_success_path": "src/kernel.cu",
            "todo_written": True,
            "todo_file_path": "/tmp/todo.md",
        }
        asyncio.run(dispatch_module._post_dispatch_inject(None, messages, ec))

        self.assertEqual(ec["last_edit_success_path"], "")   # consumed on inject
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("kernel.cu", messages[0]["content"])
        self.assertIn("task checklist", messages[0]["content"])

    def test_no_reminder_without_written_todo(self) -> None:
        messages: list[dict] = []
        ec = {
            "last_edit_success_path": "src/kernel.cu",
            "todo_written": False,          # plan not written → no nudge
            "todo_file_path": "/tmp/todo.md",
        }
        asyncio.run(dispatch_module._post_dispatch_inject(None, messages, ec))
        self.assertEqual(messages, [])
        # success_path is NOT consumed when no reminder fires
        self.assertEqual(ec["last_edit_success_path"], "src/kernel.cu")

    def test_injects_repeat_corrective_and_consumes_alert(self) -> None:
        messages: list[dict] = []
        ec = {"_repeat_alert": ("code_check_file", 2)}
        asyncio.run(dispatch_module._post_dispatch_inject(None, messages, ec))
        self.assertNotIn("_repeat_alert", ec)          # consumed
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "user")
        self.assertIn("failed 2 times", messages[0]["content"])


class FinalizeAnswerTests(unittest.TestCase):
    def _agent(self):
        carry = {"n": 0}
        agent = types.SimpleNamespace(
            tool_owner={},
            _run_tool=lambda *a, **k: "{}",
            _truncate_text=lambda text, limit=600: text[:limit],
            _update_carry_context=lambda ec: carry.__setitem__("n", carry["n"] + 1),
        )
        return agent, carry

    def test_annotates_persists_and_saves_carry(self) -> None:
        agent, carry = self._agent()
        ec = {"dirty_written_files": {"a.py", "b.cu"}}
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "U"}]

        async def _noop_store(**kwargs):
            return None

        with patch.object(finalize_module, "auto_store_memory", new=_noop_store):
            result = asyncio.run(
                agent_loop_module._finalize_answer(
                    agent, "q", "Done.", ec, messages, logger=None,
                )
            )

        self.assertTrue(result.startswith("Done."))
        self.assertIn("[Verification ledger", result)
        self.assertIn("a.py", result)
        self.assertIn("b.cu", result)
        # Neither file was validated, so the ledger says so rather than merely
        # listing them (the old "[Files written this query: …]" behaviour).
        self.assertIn("NOT validated", result)
        self.assertEqual(carry["n"], 1)                      # carry context saved
        self.assertEqual(agent._last_full_messages, messages[1:])  # system stripped

    def test_no_annotation_when_nothing_written(self) -> None:
        agent, _ = self._agent()

        async def _noop_store(**kwargs):
            return None

        with patch.object(finalize_module, "auto_store_memory", new=_noop_store):
            result = asyncio.run(
                agent_loop_module._finalize_answer(
                    agent, "q", "Nothing changed.", {"dirty_written_files": set()},
                    [{"role": "system", "content": "S"}], logger=None,
                )
            )
        self.assertEqual(result, "Nothing changed.")


class RunPlanModeTests(unittest.TestCase):
    def test_emits_plan_then_delivers_answer(self) -> None:
        # Step 1: model calls todo_write. Step 2: no tool calls → content is the answer.
        backend = ScriptedBackend([
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is the plan."},
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "Here is the plan.")
        # After todo_write the loop appends the deliver-answer nudge.
        self.assertTrue(any(
            m["role"] == "user" and "final answer" in m["content"].lower()
            for m in messages
        ))
        self.assertEqual(len(backend.calls), 2)

    def test_replaying_the_plan_forever_is_cut_short(self) -> None:
        # Regression: with the plan already recorded, some models keep re-reading /
        # re-writing it and echoing its text every turn. A turn that calls tools never
        # reached the delivery branch, so the loop parroted until max_steps and the
        # user never got the approval prompt. Past the post-record budget the calls
        # are dropped and the plan is delivered.
        parrot = {"content": "Here is the plan.", "tool_calls": [_tool_call("todo_read_plan")]}
        backend = ScriptedBackend([
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            parrot, parrot, parrot, parrot, parrot,
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=30, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "Here is the plan.")
        # todo_write + the 2 tolerated post-record turns + the one that gets cut short.
        self.assertEqual(len(backend.calls), 2 + plan_loop_module._PLAN_POST_RECORD_TOOL_TURNS)
        # The dropped calls were answered so the model knows why, and the repeated
        # deliver nudge escalated instead of being sent verbatim over and over.
        self.assertTrue(any(
            m["role"] == "tool" and "already recorded" in m["content"] for m in messages
        ))
        self.assertTrue(any(
            m["role"] == "user" and m["content"] == plan_loop_module.PLAN_DELIVER_ANSWER_FIRM
            for m in messages
        ))

    def test_prose_plan_alone_is_recorded_and_delivered(self) -> None:
        # Regression (the wave2d plan-mode hang): the model records its plan as the
        # prose document and never produces the checklist. The loop used to decide
        # "is a plan recorded" by re-deriving it from tool names, counting only the
        # plan_steps-carrying checklist, so it kept telling a model whose plan was on
        # disk that it had "not yet recorded a plan" — which the model answered by
        # rewriting that same document, until max_steps, with nothing delivered.
        # The prose form now counts, and the missing checklist is asked for by name.
        prose = {"content": "", "tool_calls": [_tool_call("todo_set_plan")]}
        backend = ScriptedBackend([prose, prose, prose, prose, prose, prose])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages,
                    execution_context=build_execution_context(),
                    max_steps=30, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        # Bounded by the post-record budget, not by max_steps.
        self.assertEqual(len(backend.calls), 2 + plan_loop_module._PLAN_POST_RECORD_TOOL_TURNS)
        # It was asked for the form that was actually missing, and never told that it
        # had recorded no plan at all.
        self.assertTrue(any(
            m["role"] == "user" and m["content"] == plan_loop_module.PLAN_CHECKLIST_MISSING_NUDGE
            for m in messages
        ))
        self.assertFalse(any(
            m["role"] == "user" and "not yet recorded a plan" in m["content"]
            for m in messages
        ))

    def test_stalled_calls_are_dropped_even_with_no_prose(self) -> None:
        # The anti-parroting escape hatch used to require a non-empty answer, so a
        # model emitting tool calls and nothing else — the shape this guard exists
        # for — sailed past it and ran to max_steps.
        parrot = {"content": "", "tool_calls": [_tool_call("todo_read_plan")]}
        backend = ScriptedBackend([
            {"content": "", "tool_calls": [_tool_call("todo_write")]},
            parrot, parrot, parrot, parrot, parrot,
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages,
                    execution_context=build_execution_context(),
                    max_steps=30, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(len(backend.calls), 2 + plan_loop_module._PLAN_POST_RECORD_TOOL_TURNS)
        self.assertTrue(any(
            m["role"] == "tool" and "already recorded" in m["content"] for m in messages
        ))

    def test_prose_without_plan_is_nudged_not_accepted(self) -> None:
        # A no-tool-call turn before any plan is recorded must NOT be accepted as
        # the answer: the model narrated a plan as prose instead of calling the
        # checklist tool. The loop nudges it to record the plan, and only accepts
        # the answer once todo_write has actually run.
        backend = ScriptedBackend([
            {"content": "Here's what I'd do: step 1, step 2."},          # prose, no plan
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is the plan."},
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        # The prose turn was not accepted; the plan was recorded and delivered.
        self.assertEqual(result, "Here is the plan.")
        self.assertTrue(any(
            m["role"] == "user" and "not yet recorded a plan" in m["content"]
            for m in messages
        ))
        self.assertEqual(len(backend.calls), 3)

    def test_accept_switches_to_agent_mode(self) -> None:
        # Plan recorded + delivered, user accepts → switch to agent mode and hand
        # off to the agent loop, executing the approved plan to completion.
        backend = ScriptedBackend([
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is the plan."},
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        captured: dict = {}

        async def _fake_agent_loop(**kwargs):
            captured.update(kwargs)
            return "executed"

        async def _build_system(active_mode: str) -> str:
            return f"SYS::{active_mode}"

        mode_calls: list = []
        agent = types.SimpleNamespace(
            model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS),
            _request_user_question=lambda qs: {"answers": [{"selected": [plan_loop_module._PLAN_ACCEPT], "other_text": None}]},
            _build_system_content=_build_system,
            set_mode=lambda m: mode_calls.append(m),
        )
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(agent_loop_module, "_run_agent_loop", _fake_agent_loop):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "executed")
        self.assertEqual(captured["active_mode"], "agent")
        self.assertEqual(captured["system_content"], "SYS::agent")
        # The agent's mode is flipped globally so the switch applies everywhere.
        self.assertEqual(mode_calls, ["agent"])
        # The system message was rewritten for agent mode and the execute nudge added.
        self.assertEqual(messages[0]["content"], "SYS::agent")
        self.assertTrue(any(
            m["role"] == "user" and "APPROVED" in m["content"] for m in messages
        ))

    def test_revise_loops_back_then_accepts(self) -> None:
        # User requests changes → the plan is reworked (new todo_write) and
        # re-presented; the second review accepts and hands off to agent mode.
        backend = ScriptedBackend([
            {"content": "plan v1", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is plan v1."},   # review #1 → revise
            {"content": "plan v2", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is plan v2."},   # review #2 → accept
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _fake_agent_loop(**kwargs):
            return "executed"

        async def _build_system(active_mode: str) -> str:
            return f"SYS::{active_mode}"

        calls = {"n": 0}

        def _ask(questions):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"answers": [{"selected": [], "other_text": "add an error-handling step"}]}
            return {"answers": [{"selected": [plan_loop_module._PLAN_ACCEPT], "other_text": None}]}

        agent = types.SimpleNamespace(
            model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS),
            _request_user_question=_ask, _build_system_content=_build_system,
        )
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(agent_loop_module, "_run_agent_loop", _fake_agent_loop):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "executed")
        self.assertEqual(calls["n"], 2)
        self.assertEqual(len(backend.calls), 4)
        # The user's requested change was folded into the revision nudge.
        self.assertTrue(any(
            m["role"] == "user" and "add an error-handling step" in m["content"]
            for m in messages
        ))

    def test_reject_stops_without_executing(self) -> None:
        # User rejects the plan: no hand-off to agent mode, no re-planning — the
        # loop stops immediately with the "nothing was executed" answer.
        backend = ScriptedBackend([
            {"content": "plan v1", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is plan v1."},   # review → reject
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        async def _fake_agent_loop(**kwargs):
            raise AssertionError("agent loop must not run on a rejected plan")

        agent = types.SimpleNamespace(
            model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS),
            _request_user_question=lambda qs: {
                "answers": [{"selected": [plan_loop_module._PLAN_REJECT], "other_text": None}]
            },
        )
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "plan it"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize), \
             patch.object(agent_loop_module, "_run_agent_loop", _fake_agent_loop):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=build_execution_context(),
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, plan_loop_module.PLAN_REJECTED_ANSWER)
        self.assertEqual(len(backend.calls), 2)
        self.assertTrue(any(
            m["role"] == "user" and "REJECTED" in m["content"] for m in messages
        ))

    def test_plan_without_evidence_is_flagged_not_rejected(self) -> None:
        # Repo-touching query + zero exploration evidence: the plan is flagged with a
        # one-shot advisory nudge but STANDS. It used to be rejected, which discarded
        # the plan the model had just recorded with nothing guaranteeing it would
        # submit that form again — a model that answered by re-writing its prose
        # document instead then spun until max_steps and delivered nothing.
        backend = ScriptedBackend([
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is the plan."},
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS))
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "refactor the solver in the repo"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="refactor the solver in the repo", messages=messages,
                    execution_context=build_execution_context(), max_steps=10, thinking=False, streaming=False,
                    logger=None, cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "Here is the plan.")
        # The one-shot evidence nudge must have been appended...
        self.assertTrue(any(
            m["role"] == "user" and m["content"] == plan_loop_module.PLAN_EVIDENCE_NUDGE
            for m in messages
        ))
        # ...alongside the deliver nudge, not instead of it: the plan was accepted on
        # the spot, so no extra round trip was spent re-recording it.
        self.assertTrue(any(
            m["role"] == "user" and m["content"] == plan_loop_module.PLAN_DELIVER_ANSWER
            for m in messages
        ))
        self.assertEqual(len(backend.calls), 2)

    def test_plan_evidence_gate_skipped_when_enforcement_off(self) -> None:
        # At enforcement level "off" the plan-evidence gate is disabled: the first
        # todo_write (zero evidence) is accepted without a rejection nudge.
        backend = ScriptedBackend([
            {"content": "planning", "tool_calls": [_tool_call("todo_write")]},
            {"content": "Here is the plan."},
        ])

        async def _plan_dispatch(tool_calls, agent, messages, execution_context):
            _record_plan_flags(tool_calls, agent, execution_context)

        async def _finalize(agent, query, answer, execution_context, messages, logger):
            return answer

        agent = types.SimpleNamespace(model="m", tools=[], tool_caps=dict(_CHECKLIST_CAPS), enforcement="off")
        messages = [{"role": "system", "content": "S"}, {"role": "user", "content": "refactor the solver in the repo"}]

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []), \
             patch.object(plan_loop_module, "_dispatch_tool_calls", _plan_dispatch), \
             patch.object(plan_loop_module, "_finalize_answer", _finalize):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="refactor the solver in the repo", messages=messages,
                    execution_context=build_execution_context(), max_steps=10, thinking=False, streaming=False,
                    logger=None, cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "Here is the plan.")
        self.assertFalse(any(
            m["role"] == "user" and "without gathering any evidence" in m["content"]
            for m in messages
        ))
        self.assertEqual(len(backend.calls), 2)


class RunAgentQueryNonInteractiveTests(unittest.TestCase):
    def _query_agent(self, continue_calls):
        class _QueryAgent:
            mode = "agent"
            model = "dummy"
            tools = []
            tool_owner = {}
            tool_caps = {}
            repo_baseline_context = ""
            thinking_budget = -1
            allow_continue_prompt = False
            _cancel_flag = None

            @staticmethod
            def _new_execution_context():
                return MimirAgent._new_execution_context()

            @staticmethod
            def _get_todo_file():
                return ""

            async def _ensure_repo_baseline(self, query):
                return None

            def _seed_execution_context_from_baseline(self, execution_context):
                pass

            @staticmethod
            def _normalize_mode(mode):
                return "agent"

            async def _build_system_content(self, **kwargs):
                return "system"

            async def _run_tool(self, tool, args, execution_context=None, run_auto_validation=True):
                return "{}"

            @staticmethod
            def _normalize_arguments(args):
                return args

            @staticmethod
            def _truncate_text(text, limit=600):
                return text[:limit]

            approvals = types.SimpleNamespace(flush_pending_review=lambda: None)

            def _apply_carry_context(self, execution_context):
                pass

            def _update_carry_context(self, execution_context):
                pass

            def _request_continue(self, summary):
                continue_calls["n"] += 1
                return False

        return _QueryAgent()

    def test_two_step_script_returns_final_answer_without_prompting(self) -> None:
        continue_calls = {"n": 0}
        agent = self._query_agent(continue_calls)

        backend = ScriptedBackend([
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "final answer"},
        ])

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(agent_loop_module, "_dispatch_tool_calls", _noop_async), \
             patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: []), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(
                m.run_agent_query(agent=agent, query="do a thing", max_steps=5)
            )

        self.assertEqual(result, "final answer")
        self.assertEqual(continue_calls["n"], 0)   # non-interactive: never prompts
        self.assertEqual(len(backend.calls), 2)     # tool-call step, then final answer


class DrainSteerTests(unittest.TestCase):
    """Unit coverage for ``_drain_steer`` — the chat-while-busy injection point."""

    def test_injects_queued_messages_as_user_turns(self) -> None:
        agent = types.SimpleNamespace(_poll_steer=lambda: ["focus on the parser", "skip tests"])
        messages: list[dict] = [{"role": "tool", "content": "prev result"}]
        ec: dict = {}
        emitted: list[dict] = []
        with patch.object(agent_loop_module, "emit", lambda ev: emitted.append(ev)):
            agent_loop_module._drain_steer(agent, messages, ec)

        # Both steer messages appended, in order, as user turns.
        self.assertEqual(messages[-2], {"role": "user", "content": "focus on the parser"})
        self.assertEqual(messages[-1], {"role": "user", "content": "skip tests"})
        self.assertTrue(ec["user_steered"])
        # Each injection is surfaced to the front-end for the "delivered" affordance.
        self.assertEqual([e["type"] for e in emitted], ["steer_injected", "steer_injected"])
        self.assertEqual(emitted[0]["text"], "focus on the parser")

    def test_blank_and_missing_poll_are_noops(self) -> None:
        # Whitespace-only messages are skipped.
        agent = types.SimpleNamespace(_poll_steer=lambda: ["  ", ""])
        messages: list[dict] = []
        agent_loop_module._drain_steer(agent, messages, {})
        self.assertEqual(messages, [])

        # No _poll_steer attr at all (CLI / sub-agents / tests) → untouched.
        bare = types.SimpleNamespace()
        msgs2: list[dict] = [{"role": "user", "content": "x"}]
        ec2: dict = {}
        agent_loop_module._drain_steer(bare, msgs2, ec2)
        self.assertEqual(msgs2, [{"role": "user", "content": "x"}])
        self.assertNotIn("user_steered", ec2)


class SteerInjectionInLoopTests(RunAgentQueryNonInteractiveTests):
    """The steer is injected at the next step boundary of the real agent loop."""

    def test_steer_reaches_the_next_model_call(self) -> None:
        continue_calls = {"n": 0}
        agent = self._query_agent(continue_calls)
        # Simulate a steer that arrives WHILE step 1 runs: the first drain (top of
        # step 1) sees nothing; the message is queued and picked up at step 2's drain.
        drains = {"n": 0}

        def _poll():
            drains["n"] += 1
            return ["focus on the parser"] if drains["n"] == 2 else []

        agent._poll_steer = _poll

        backend = ScriptedBackend([
            {"content": "working", "tool_calls": [_tool_call("noop")]},
            {"content": "final answer"},
        ])

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(agent_loop_module, "_dispatch_tool_calls", _noop_async), \
             patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: []), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(
                m.run_agent_query(agent=agent, query="do a thing", max_steps=5)
            )

        self.assertEqual(result, "final answer")
        # Step 1's prompt (calls[0]) predates the steer; step 2's prompt (calls[1])
        # must contain the injected user turn ahead of the model call.
        step1_users = [msg["content"] for msg in backend.calls[0]["messages"] if msg["role"] == "user"]
        step2_users = [msg["content"] for msg in backend.calls[1]["messages"] if msg["role"] == "user"]
        self.assertNotIn("focus on the parser", step1_users)
        self.assertIn("focus on the parser", step2_users)


class DomainRearmInLoopTests(RunAgentQueryNonInteractiveTests):
    """A domain the query never signaled is unlocked once the work reveals the need."""

    def _run(self, tool_result: str, query: str):
        continue_calls = {"n": 0}
        agent = self._query_agent(continue_calls)

        backend = ScriptedBackend([
            {"content": "checking the environment", "tool_calls": [_tool_call("noop")]},
            {"content": "final answer"},
        ])

        async def _dispatch(tool_calls, agent_, messages, execution_context):
            messages.append({"role": "tool", "content": tool_result})

        async def _noop_async(*a, **k):
            return None

        # Stand-in tool list: one always-present tool plus the cluster family, which
        # only appears once the group has been re-armed in the execution context.
        def _fake_tools_for_context(**kwargs):
            ec = kwargs.get("execution_context") or {}
            names = ["read_file"]
            if "slurm_" in (ec.get("rearmed_domains") or set()):
                names.append("slurm_submit")
            return [{"function": {"name": n}} for n in names]

        emitted: list[dict] = []
        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(agent_loop_module, "_dispatch_tool_calls", _dispatch), \
             patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", _fake_tools_for_context), \
             patch.object(m, "emit", lambda ev: emitted.append(ev)), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(m.run_agent_query(agent=agent, query=query, max_steps=5))

        return result, backend, emitted

    def _tool_names(self, call: dict) -> list[str]:
        return [t["function"]["name"] for t in call["tools"]]

    def test_tool_result_unlocks_the_pruned_domain_for_the_next_step(self) -> None:
        result, backend, emitted = self._run(
            tool_result="sbatch: command not found — no allocation on this login node",
            query="run the test suite",
        )

        self.assertEqual(result, "final answer")
        # Step 1 ran without the cluster tools (the query never mentioned them);
        # step 2 sees them, because the tool result showed the work needs them.
        self.assertNotIn("slurm_submit", self._tool_names(backend.calls[0]))
        self.assertIn("slurm_submit", self._tool_names(backend.calls[1]))
        # The cache break is reported rather than silent.
        rearm_events = [e for e in emitted if e["type"] == "tools_rearmed"]
        self.assertEqual(len(rearm_events), 1)
        self.assertEqual(rearm_events[0]["domains"], ["slurm_"])

    def test_unrelated_tool_result_leaves_the_list_frozen(self) -> None:
        # The common case: nothing signals a pruned domain, so no rebuild happens and
        # the prompt prefix stays byte-identical across steps.
        result, backend, emitted = self._run(
            tool_result="3 passed in 0.4s",
            query="run the test suite",
        )

        self.assertEqual(result, "final answer")
        self.assertEqual(self._tool_names(backend.calls[0]), self._tool_names(backend.calls[1]))
        self.assertEqual([e for e in emitted if e["type"] == "tools_rearmed"], [])


class NormalizeModeTests(unittest.TestCase):
    def test_accepts_every_valid_mode_case_insensitively(self) -> None:
        for mode in ("agent", "plan", "ask"):
            self.assertEqual(MimirAgent._normalize_mode(mode), mode)
            self.assertEqual(MimirAgent._normalize_mode(f"  {mode.upper()} "), mode)

    def test_rejects_unknown_mode_and_names_the_valid_ones(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            MimirAgent._normalize_mode("chat")
        for mode in ("agent", "plan", "ask"):
            self.assertIn(mode, str(ctx.exception))


class AskModeInLoopTests(RunAgentQueryNonInteractiveTests):
    """Ask mode runs the ordinary agent loop with a read-only tool surface.

    Unlike plan mode it has no checklist requirement, no evidence gate, and no
    approval prompt — it just explores and answers.
    """

    def _ask_agent(self):
        agent = self._query_agent({"n": 0})
        agent.mode = "ask"
        agent._normalize_mode = staticmethod(lambda mode: "ask")
        # The shared stub's _normalize_arguments is an identity; the guard needs the
        # real one, which parses the JSON-string arguments a backend actually emits.
        agent._normalize_arguments = staticmethod(MimirAgent._normalize_arguments)
        agent.tool_caps = {
            "write_file": ToolCaps(name="write_file", capabilities=frozenset({PLAN_BLOCKED})),
            "bash_run": ToolCaps(name="bash_run", capabilities=frozenset({PLAN_READONLY})),
            "read_file": ToolCaps(name="read_file", capabilities=frozenset()),
        }
        agent.tools = [{"function": {"name": n}} for n in agent.tool_caps]
        return agent

    def _run(self, script):
        agent = self._ask_agent()
        backend = ScriptedBackend(script)
        dispatched: list[list[dict]] = []

        async def _dispatch(tool_calls, agent_, messages, execution_context):
            dispatched.append(list(tool_calls))

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(agent_loop_module, "_dispatch_tool_calls", _dispatch), \
             patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async), \
             patch.object(history_module, "_trim_tool_history", lambda *a, **k: None), \
             patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             patch.object(m, "emit", lambda ev: None), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(m.run_agent_query(agent=agent, query="how does X work?", mode="ask"))
        return result, backend, dispatched

    def test_write_tools_are_never_advertised(self) -> None:
        result, backend, _ = self._run([{"content": "the answer"}])
        self.assertEqual(result, "the answer")
        advertised = {t["function"]["name"] for t in backend.calls[0]["tools"]}
        self.assertEqual(advertised, {"bash_run", "read_file"})

    def test_hallucinated_write_call_is_dropped_before_dispatch(self) -> None:
        result, _, dispatched = self._run([
            {"content": "looking", "tool_calls": [
                _tool_call("write_file", '{"path": "a.py"}', "1"),
                _tool_call("read_file", '{"path": "a.py"}', "2"),
            ]},
            {"content": "the answer"},
        ])
        self.assertEqual(result, "the answer")
        self.assertEqual(
            [tc["function"]["name"] for tc in dispatched[0]], ["read_file"],
        )

    def test_exec_command_is_dropped_but_discovery_survives(self) -> None:
        _, _, dispatched = self._run([
            {"content": "looking", "tool_calls": [
                _tool_call("bash_run", '{"command": "python train.py"}', "1"),
                _tool_call("bash_run", '{"command": "rg foo src"}', "2"),
            ]},
            {"content": "the answer"},
        ])
        self.assertEqual([tc["id"] for tc in dispatched[0]], ["2"])

    def test_no_plan_checklist_or_approval_pressure(self) -> None:
        # A bare no-tool-call turn ends the query immediately: unlike plan mode there
        # is no todo tool to call first and no Accept/Reject prompt afterwards.
        result, backend, dispatched = self._run([{"content": "the answer"}])
        self.assertEqual(result, "the answer")
        self.assertEqual(len(backend.calls), 1)
        self.assertEqual(dispatched, [])


class MidQueryModeSwitchTests(RunAgentQueryNonInteractiveTests):
    """The mode is live: flipping it mid-run lands on the very next step.

    The flip is performed from inside the dispatch stub, which is where a real
    user's click lands — while the current step's tool call is running.
    """

    _CAPS = {
        "write_file": ToolCaps(name="write_file", capabilities=frozenset({PLAN_BLOCKED})),
        "read_file": ToolCaps(name="read_file", capabilities=frozenset()),
    }

    def _agent(self, start_mode="agent"):
        agent = self._query_agent({"n": 0})
        agent.mode = start_mode
        agent._normalize_mode = staticmethod(lambda mode: mode)
        agent._normalize_arguments = staticmethod(MimirAgent._normalize_arguments)
        agent.tool_caps = dict(self._CAPS)
        agent.tools = [{"function": {"name": n}} for n in self._CAPS]
        return agent

    def _run(self, agent, script, switch_to, extra_patches=()):
        backend = ScriptedBackend(script)
        emitted: list[dict] = []

        async def _dispatch(tool_calls, agent_, messages, execution_context):
            # The user flips the mode while this step's tool call is in flight.
            agent.mode = switch_to

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        patches = [
            patch.object(streaming_module, "get_backend", lambda: backend),
            patch.object(finalize_module, "auto_store_memory", new=_noop_async),
            patch.object(agent_loop_module, "_dispatch_tool_calls", _dispatch),
            patch.object(agent_loop_module, "_post_dispatch_inject", _noop_async),
            patch.object(history_module, "_trim_tool_history", lambda *a, **k: None),
            patch.object(history_module, "_maybe_compact_intra_query", lambda *a, **k: None),
            patch.object(m, "_inject_pin", lambda *a, **k: None),
            patch.object(m, "tools_for_context", lambda **k: k["tools"]),
            patch.object(m, "emit", lambda ev: emitted.append(ev)),
            patch.object(m, "needs_incomplete_finalization", lambda ec: False),
            *extra_patches,
        ]
        with contextlib.ExitStack() as stack:
            for p in patches:
                stack.enter_context(p)
            result = asyncio.run(m.run_agent_query(agent=agent, query="do a thing", max_steps=5))
        return result, backend, emitted

    def _names(self, call):
        return {t["function"]["name"] for t in call["tools"]}

    def test_agent_to_ask_revokes_write_tools_on_the_next_step(self) -> None:
        agent = self._agent("agent")
        result, backend, emitted = self._run(
            agent,
            [
                {"content": "editing", "tool_calls": [_tool_call("read_file")]},
                {"content": "the answer"},
            ],
            switch_to="ask",
        )
        self.assertEqual(result, "the answer")
        # Step 1 had the full surface; step 2 is read-only.
        self.assertIn("write_file", self._names(backend.calls[0]))
        self.assertNotIn("write_file", self._names(backend.calls[1]))
        # The switch is reported, and the front-end toggle is synced.
        self.assertIn({"type": "mode", "mode": "ask"}, emitted)
        self.assertTrue(any(
            e["type"] == "status" and "ask mode mid-run" in e.get("text", "")
            for e in emitted
        ))

    def test_switch_rebuilds_the_system_prompt_for_the_new_mode(self) -> None:
        agent = self._agent("agent")
        built: list[str] = []

        async def _build(**kwargs):
            built.append(kwargs["active_mode"])
            return f"system:{kwargs['active_mode']}"

        agent._build_system_content = _build
        _, backend, _ = self._run(
            agent,
            [
                {"content": "working", "tool_calls": [_tool_call("read_file")]},
                {"content": "the answer"},
            ],
            switch_to="ask",
        )
        self.assertEqual(built, ["agent", "ask"])
        self.assertEqual(backend.calls[1]["messages"][0]["content"], "system:ask")

    def test_switch_to_plan_hands_the_run_to_the_plan_loop(self) -> None:
        agent = self._agent("agent")
        handoff: dict = {}

        async def _fake_plan(**kwargs):
            handoff.update(kwargs)
            return "the plan"

        result, backend, emitted = self._run(
            agent,
            [
                {"content": "working", "tool_calls": [_tool_call("read_file")]},
                {"content": "never reached"},
            ],
            switch_to="plan",
            extra_patches=[patch.object(agent_loop_module, "_run_plan_mode", _fake_plan)],
        )
        self.assertEqual(result, "the plan")
        # Handed off before a second agent-loop model call.
        self.assertEqual(len(backend.calls), 1)
        # The conversation so far travels with it, prompt already rebuilt for plan.
        self.assertEqual(handoff["query"], "do a thing")
        self.assertEqual(handoff["messages"][0]["role"], "system")
        self.assertTrue(any(m["role"] == "assistant" for m in handoff["messages"]))
        self.assertIn({"type": "mode", "mode": "plan"}, emitted)

    def test_explicit_mode_override_is_not_a_user_switch(self) -> None:
        # Sub-agents and the runner pass mode= for one query without touching
        # agent.mode; that must not read as the user flipping the mode mid-run.
        agent = self._agent("agent")
        backend = ScriptedBackend([{"content": "the answer"}])
        emitted: list[dict] = []

        async def _noop_async(*a, **k):
            return None

        m = agent_loop_module
        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(finalize_module, "auto_store_memory", new=_noop_async), \
             patch.object(m, "_inject_pin", lambda *a, **k: None), \
             patch.object(m, "tools_for_context", lambda **k: k["tools"]), \
             patch.object(m, "emit", lambda ev: emitted.append(ev)), \
             patch.object(m, "needs_incomplete_finalization", lambda ec: False):
            result = asyncio.run(
                m.run_agent_query(agent=agent, query="q", mode="ask", max_steps=5)
            )

        self.assertEqual(result, "the answer")
        # Ran as ask (write tools stripped) without ever reporting a switch.
        self.assertNotIn("write_file", self._names(backend.calls[0]))
        self.assertEqual([e for e in emitted if e["type"] == "mode"], [])


class PlanLoopModeSwitchTests(unittest.TestCase):
    """Leaving plan mode mid-draft hands the run to the agent loop."""

    def test_switch_out_of_plan_tail_calls_the_agent_loop(self) -> None:
        handoff: dict = {}

        async def _fake_agent_loop(**kwargs):
            handoff.update(kwargs)
            return "executed"

        async def _build(**kwargs):
            return f"system:{kwargs['active_mode']}"

        agent = types.SimpleNamespace(
            model="m", tools=[], tool_caps={}, mode="agent",
            _build_system_content=_build,
        )
        messages = [{"role": "system", "content": "system:plan"},
                    {"role": "user", "content": "plan it"}]
        # The user already flipped to agent: the loop was entered observing "plan".
        ec = {agent_loop_module._OBSERVED_MODE: "plan"}
        backend = ScriptedBackend([{"content": "unused"}])

        with patch.object(streaming_module, "get_backend", lambda: backend), \
             patch.object(agent_loop_module, "_run_agent_loop", _fake_agent_loop), \
             patch.object(agent_loop_module, "emit", lambda ev: None), \
             patch.object(plan_loop_module, "tools_for_plan_mode", lambda tools, caps: []):
            result = asyncio.run(
                plan_loop_module._run_plan_mode(
                    agent=agent, query="q", messages=messages, execution_context=ec,
                    max_steps=10, thinking=False, streaming=False, logger=None,
                    cb={"think_token_callback": None},
                )
            )

        self.assertEqual(result, "executed")
        # Handed off before any plan-mode model call, with the prompt rebuilt.
        self.assertEqual(len(backend.calls), 0)
        self.assertEqual(handoff["active_mode"], "agent")
        self.assertEqual(handoff["system_content"], "system:agent")
        self.assertEqual(messages[0]["content"], "system:agent")


class ForceFitToWindowTests(unittest.TestCase):
    """The deterministic hard-fit backstop that guarantees a prompt fits."""

    # Word-count token counter keeps the arithmetic obvious and stable.
    @staticmethod
    def _tok(text: str) -> int:
        return len(text.split())

    def test_truncates_largest_and_preserves_protected(self) -> None:
        msgs = [
            {"role": "system", "content": "sys " * 50},
            {"role": "user", "content": "first " * 10},
            {"role": "assistant", "content": "",
             "tool_calls": [_tool_call("read", "{}", "c1")]},
            {"role": "tool", "tool_call_id": "c1", "content": "BIG " * 5000},
            {"role": "assistant", "content": "analysis " * 100},
            {"role": "user", "content": "latest " * 20},  # active query (protected)
        ]
        fit = history_module._force_fit_to_window(msgs, 400, self._tok)
        total = sum(self._tok(m.get("content", "")) for m in msgs)
        self.assertTrue(fit)
        self.assertLessEqual(total, 400)
        # System prompt and the most recent user message are never reduced.
        self.assertTrue(msgs[0]["content"].startswith("sys"))
        self.assertTrue(msgs[-1]["content"].startswith("latest"))
        # The oversized tool result was the thing that got cut.
        self.assertIn("truncated", msgs[3]["content"])

    def test_returns_false_when_core_exceeds_target(self) -> None:
        # System + active user message alone are larger than the target, so the
        # list cannot be made to fit — caller/backend must surface a clear error.
        msgs = [
            {"role": "system", "content": "x " * 300},
            {"role": "user", "content": "y " * 300},
        ]
        self.assertFalse(history_module._force_fit_to_window(msgs, 100, self._tok))

    def test_noop_when_already_fits(self) -> None:
        msgs = [
            {"role": "system", "content": "a b c"},
            {"role": "user", "content": "d e"},
        ]
        snapshot = [dict(m) for m in msgs]
        self.assertTrue(history_module._force_fit_to_window(msgs, 1000, self._tok))
        self.assertEqual(msgs, snapshot)  # unchanged


class RepeatedFailingCallGuardTests(unittest.TestCase):
    """The dispatch-level guard for repeated identical FAILING non-write calls.

    Verifies the soft corrective (staged after SOFT_REPEAT_THRESHOLD failures) and the
    hard backstop (the call is no longer executed once HARD_REPEAT_LIMIT failures are
    recorded) — the mid-tool-loop spin that the regular nudges cannot reach.
    """

    def _agent(self, run_tool):
        approvals = types.SimpleNamespace(record_snapshot=lambda p: None, _file_snapshots={})
        return types.SimpleNamespace(
            _normalize_arguments=lambda a: (json.loads(a) if isinstance(a, str) else dict(a)),
            _is_write_tool=lambda n: False,
            tool_caps={"code_check_file": ToolCaps(name="code_check_file")},
            get_tool_file_targets=lambda n, a: [],
            approvals=approvals,
            _run_tool=run_tool,
            _rewrite_tool_for_context=lambda n, a: (n, a),
        )

    def _dispatch(self, agent, ec):
        msgs: list[dict] = []
        tc = [_tool_call("code_check_file", '{"filepath": "a.py"}')]
        asyncio.run(dispatch_module._dispatch_tool_calls(tc, agent, msgs, ec))
        return msgs

    def test_warns_then_blocks_repeated_failure(self) -> None:
        calls = {"n": 0}

        async def run_tool(name, args, execution_context=None, run_auto_validation=True):
            calls["n"] += 1
            return '{"status": "error", "error": "ModuleNotFoundError: No module named x"}'

        agent = self._agent(run_tool)
        ec = build_execution_context()
        key = ("code_check_file", _make_hashable({"filepath": "a.py"}))

        # Failure 1: counted, no corrective yet.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_fails[key], 1)
        self.assertNotIn("_repeat_alert", ec)

        # Failure 2: crosses SOFT_REPEAT_THRESHOLD → corrective staged once.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_fails[key], 2)
        self.assertEqual(ec["_repeat_alert"][0], "code_check_file")
        ec.pop("_repeat_alert")

        # Failure 3: reaches HARD_REPEAT_LIMIT; still executed, not re-warned.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_fails[key], dispatch_module.HARD_REPEAT_LIMIT)
        self.assertNotIn("_repeat_alert", ec)
        executed_before = calls["n"]

        # 4th attempt: hard-blocked — the tool is NOT executed, a synthetic error is returned.
        msgs = self._dispatch(agent, ec)
        self.assertEqual(calls["n"], executed_before)            # no further execution
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "tool")
        self.assertIn("blocked", msgs[0]["content"].lower())

    def test_raised_exception_becomes_tool_error_not_turn_crash(self) -> None:
        """A tool whose MCP call *raises* (e.g. the server subprocess died and the
        session broke) must be converted into a normal tool-error message. The
        exception must NOT bubble out of dispatch and kill the whole turn."""
        async def run_tool(name, args, execution_context=None, run_auto_validation=True):
            raise ConnectionError("Connection closed")

        agent = self._agent(run_tool)
        ec = build_execution_context()

        # Dispatch must complete without raising.
        msgs = self._dispatch(agent, ec)

        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "tool")
        payload = json.loads(msgs[0]["content"])
        self.assertEqual(payload["status"], "error")
        self.assertIn("failed to execute", payload["error"])
        self.assertIn("Connection closed", payload["error"])

    def test_successful_call_with_changing_content_never_blocks(self) -> None:
        """A non-write call that returns DIFFERENT content each time (e.g. reading a file
        that was edited between reads) must never be counted as redundant or blocked."""
        calls = {"n": 0}

        async def run_tool(name, args, execution_context=None, run_auto_validation=True):
            calls["n"] += 1
            return f'{{"status": "ok", "n": {calls["n"]}}}'  # different content every call

        agent = self._agent(run_tool)
        ec = build_execution_context()
        for _ in range(6):
            self._dispatch(agent, ec)
        self.assertEqual(calls["n"], 6)            # every call executed
        self.assertNotIn("_redundant_alert", ec)
        key = ("code_check_file", _make_hashable({"filepath": "a.py"}))
        # Count keeps resetting to 0 because the result hash changes each time.
        self.assertEqual(loop_control(ec).call_results[key][1], 0)

    def test_warns_then_blocks_redundant_successful_call(self) -> None:
        """A non-write call that keeps SUCCEEDING with identical content is warned on the
        first repeat (REDUNDANT_SOFT_THRESHOLD) and hard-blocked on the second
        (REDUNDANT_HARD_LIMIT)."""
        calls = {"n": 0}

        async def run_tool(name, args, execution_context=None, run_auto_validation=True):
            calls["n"] += 1
            return '{"status": "ok", "content": "unchanged"}'  # identical every time

        agent = self._agent(run_tool)
        ec = build_execution_context()
        key = ("code_check_file", _make_hashable({"filepath": "a.py"}))

        # Call 1: first observation, count starts at 0, no corrective.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_results[key][1], 0)
        self.assertNotIn("_redundant_alert", ec)

        # Call 2 (1st repeat): count reaches REDUNDANT_SOFT_THRESHOLD → corrective once.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_results[key][1], dispatch_module.REDUNDANT_SOFT_THRESHOLD)
        self.assertEqual(ec["_redundant_alert"][0], "code_check_file")
        ec.pop("_redundant_alert")

        # Call 3 (2nd repeat): count reaches REDUNDANT_HARD_LIMIT; still executed, not re-warned.
        self._dispatch(agent, ec)
        self.assertEqual(loop_control(ec).call_results[key][1], dispatch_module.REDUNDANT_HARD_LIMIT)
        self.assertNotIn("_redundant_alert", ec)
        executed_before = calls["n"]

        # Call 4: hard-blocked — the tool is NOT executed, a synthetic notice is returned.
        msgs = self._dispatch(agent, ec)
        self.assertEqual(calls["n"], executed_before)            # no further execution
        self.assertEqual(len(msgs), 1)
        self.assertEqual(msgs[0]["role"], "tool")
        self.assertIn("skipped", msgs[0]["content"].lower())

    def test_hard_block_strips_redundant_history_to_first(self) -> None:
        """On the hard block, the intermediate identical exchanges are removed from
        history, leaving only the first occurrence (plus the current block notice)."""

        async def run_tool(name, args, execution_context=None, run_auto_validation=True):
            return '{"status": "ok", "content": "unchanged"}'  # identical every time

        agent = self._agent(run_tool)
        ec = build_execution_context()
        messages: list[dict] = []

        def turn(cid: str) -> None:
            tc = _tool_call("code_check_file", '{"filepath": "a.py"}', call_id=cid)
            # Mirror _process_response: the assistant tool_calls turn is appended first.
            messages.append({"role": "assistant", "content": "", "tool_calls": [tc]})
            asyncio.run(dispatch_module._dispatch_tool_calls([tc], agent, messages, ec))

        turn("c1")   # keeper, count 0
        turn("c2")   # count 1 (soft)
        turn("c3")   # count 2
        turn("c4")   # blocked → strip c2, c3 from history

        tool_ids = [m["tool_call_id"] for m in messages if m.get("role") == "tool"]
        # Only the first occurrence and the current (blocked) notice survive.
        self.assertEqual(tool_ids, ["c1", "c4"])
        # The redundant assistant turns (c2, c3) are gone entirely.
        assistant_call_ids = [
            dispatch_module._tc_id(tc)
            for m in messages if m.get("role") == "assistant"
            for tc in (m.get("tool_calls") or [])
        ]
        self.assertEqual(assistant_call_ids, ["c1", "c4"])
        # Tracking collapses to the surviving keeper.
        key = ("code_check_file", _make_hashable({"filepath": "a.py"}))
        self.assertEqual(loop_control(ec).redundant_call_ids[key], ["c1"])


class WriteDiffEmitTests(unittest.TestCase):
    """The post-execution diff card must reflect the *outcome* of the write.

    A failed edit never writes the file, but the batch baseline snapshot predates
    any earlier successful edit to the same path (record_snapshot keeps only the
    first snapshot). Diffing that stale baseline against the current content would
    surface the earlier change and make the failed call look like it applied — so
    a failed write must emit no diff card.
    """

    def _agent(self, run_tool, snapshots):
        def record_snapshot(path: str) -> None:
            if path in snapshots:
                return
            try:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    snapshots[path] = fh.read()
            except FileNotFoundError:
                snapshots[path] = None

        approvals = types.SimpleNamespace(
            record_snapshot=record_snapshot, _file_snapshots=snapshots
        )
        return types.SimpleNamespace(
            _normalize_arguments=lambda a: (json.loads(a) if isinstance(a, str) else dict(a)),
            _is_write_tool=lambda n: n == "replace_in_file",
            tool_caps={"replace_in_file": ToolCaps(name="replace_in_file")},
            get_tool_file_targets=lambda n, a: [a["path"]],
            approvals=approvals,
            _run_tool=run_tool,
            _rewrite_tool_for_context=lambda n, a: (n, a),
        )

    def _dispatch_capture(self, agent, path: str):
        events: list[dict] = []
        ec = build_execution_context()
        tc = [_tool_call("replace_in_file", json.dumps({"path": path}))]
        with event_sink_module.event_sink(events.append):
            asyncio.run(dispatch_module._dispatch_tool_calls(tc, agent, [], ec))
        return events

    def test_failed_edit_on_prechanged_file_emits_no_diff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wave.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("edited by a prior successful call\n")

            snapshots = {path: "original content\n"}  # baseline predates that edit

            async def run_tool(name, args, execution_context=None, run_auto_validation=True):
                return '{"status": "error", "error": "old_text not found"}'

            agent = self._agent(run_tool, snapshots)
            events = self._dispatch_capture(agent, path)

            self.assertFalse([e for e in events if e.get("type") == "diff"])
            result = [e for e in events if e.get("type") == "tool_result"][-1]
            self.assertFalse(result["ok"])

    def test_successful_edit_emits_diff(self) -> None:
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wave.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("original content\n")

            snapshots: dict[str, str | None] = {}

            async def run_tool(name, args, execution_context=None, run_auto_validation=True):
                with open(args["path"], "w", encoding="utf-8") as fh:
                    fh.write("new content\n")
                return '{"status": "ok", "operation": "replaced"}'

            agent = self._agent(run_tool, snapshots)
            events = self._dispatch_capture(agent, path)

            diffs = [e for e in events if e.get("type") == "diff"]
            self.assertEqual(len(diffs), 1)
            self.assertIn("new content", diffs[0]["patch"])

    def test_slow_post_write_validation_never_fails_the_write(self) -> None:
        """A hung/slow post-write validator must not flip a successful edit to failed.

        The write hits disk first; auto-validation runs afterward under its own
        budget. If validation exceeds that budget the advisory is dropped, but the
        tool row stays ``ok`` and the diff still emits — the regression this fix
        targets (validation timing out inside the write's timeout marked the whole
        edit failed even though the file was written).
        """
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "wave.py")
            with open(path, "w", encoding="utf-8") as fh:
                fh.write("original content\n")

            snapshots: dict[str, str | None] = {}

            async def run_tool(name, args, execution_context=None, run_auto_validation=True):
                with open(args["path"], "w", encoding="utf-8") as fh:
                    fh.write("new content\n")
                # The real executor would append AUTO_VALIDATION here when
                # run_auto_validation is True; the dispatcher passes False and runs
                # it separately (mocked via _auto_validate_written_file below).
                return '{"status": "ok", "operation": "replaced"}'

            async def _hanging_validate(p, ec):
                await asyncio.sleep(10)  # far beyond the patched budget
                return "should never be appended"

            agent = self._agent(run_tool, snapshots)
            # replace_in_file must carry EDIT so the dispatcher runs validation.
            agent.tool_caps = {
                "replace_in_file": ToolCaps(
                    name="replace_in_file", capabilities=frozenset({"edit"})
                )
            }
            agent._normalize_workspace_path = lambda p: p
            agent._auto_validate_written_file = _hanging_validate

            with patch.object(dispatch_module, "_AUTO_VALIDATION_TIMEOUT_SECS", 0.05):
                events = self._dispatch_capture(agent, path)

            result = [e for e in events if e.get("type") == "tool_result"][-1]
            self.assertTrue(result["ok"])                       # write not failed
            self.assertNotIn("should never be appended", result.get("summary", ""))
            diffs = [e for e in events if e.get("type") == "diff"]
            self.assertEqual(len(diffs), 1)
            self.assertIn("new content", diffs[0]["patch"])


if __name__ == "__main__":
    unittest.main()
