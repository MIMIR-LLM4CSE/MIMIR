"""Contracts of the sub-agent spawn server.

The tool is the delegation channel: a broad sweep the orchestrator sends out instead of
paying for it in its own window. What is tested here is what makes that channel usable —
which tools a child may be given, the toolkit an exploration gets by default, the
anti-recursion, and the time budget it declares to its caller — none of which any
other test covered.
"""
import asyncio
import contextlib
import tempfile
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import types
import unittest
from unittest import mock
from pathlib import Path

SERVERS_DIR = Path(__file__).resolve().parents[1] / "servers"
for _p in (SERVERS_DIR / "_shared", SERVERS_DIR / "agent_state"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


spawn = _load("server_spawn_agent", SERVERS_DIR / "agent_state" / "server_spawn_agent.py")


# What each fake server advertises when the child connects it: the tools, and the one
# capability that decides whether the child works or only reads. Connecting a server
# brings its siblings, which is exactly what the pruning has to undo.
_SERVER_TOOLS: dict[str, list[tuple[str, set]]] = {
    "files": [("read_file_lines", set()), ("write_file", {"plan_blocked"})],
    "search": [("grep_files", set())],
    "code_intel": [("find_definition", set())],
    "bash": [("bash_run", {"plan_readonly"})],
    "todo": [("todo_write", {"task_planning"}), ("todo_read", set())],
}

_GRANTABLE = {name: server for server, tools in _SERVER_TOOLS.items() for name, _ in tools}
# What the client marks as changing the tree: the caller knows, the spawn server does
# not. A child granted one of these gets a copy of the repository.
_WRITERS = sorted(name for _, tools in _SERVER_TOOLS.items() for name, caps in tools
                  if caps & {"plan_blocked", "plan_readonly"})


class _FakeAgent:
    """A MimirAgent stand-in recording what the spawn server does to it."""

    def __init__(self, answer: str = "done"):
        self.connected: list[str] = []
        self.model = "parent/model"
        self.server_env: dict = {}
        self.tools: list[dict] = []
        self.tool_owner: dict = {}
        self.tool_caps: dict = {}
        self.mode = "agent"
        self.thinking_depth = 99
        self.run_kwargs: dict = {}
        self._carry_context = {
            "read_files": {"/w/a.py", "/w/b.py"},
            "last_query_written_files": set(),
        }
        self._answer = answer

        self.approvals = types.SimpleNamespace(
            approval_mode="manual", unattended=False, mode_blocked=[],
            _allowed_paths=set())

    def set_mode(self, mode): self.mode = mode
    def set_approval_mode(self, mode): self.approvals.approval_mode = mode
    def set_thinking_depth(self, depth): self.thinking_depth = depth
    def seed_classification_from_caps(self): pass

    async def connect_server(self, name, script):
        from mimir.client.context.capabilities import ToolCaps
        self.connected.append(name)
        for tool, caps in _SERVER_TOOLS.get(name, []):
            self.tools.append({"type": "function", "function": {"name": tool}})
            self.tool_owner[tool] = name
            self.tool_caps[tool] = ToolCaps(name=tool, capabilities=frozenset(caps))

    async def cleanup(self):
        pass

    def tool_names(self) -> set:
        return {t["function"]["name"] for t in self.tools}

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._answer


def _run_child(tools: list[str] | None = None, answer: str = "done",
               agent: _FakeAgent | None = None, on_event=None,
               approval_mode: str = "manual", level: str = "parallel"):
    """Drive _run_sub_agent against a fake agent and the real server catalog."""
    agent = agent or _FakeAgent(answer)
    with _patched_agent(agent):
        result = asyncio.run(spawn._run_sub_agent(
            "find X", "", list(tools or []), dict(_GRANTABLE), 5, approval_mode,
            on_event=on_event, level=level))
    return agent, result


@contextlib.contextmanager
def _patched_agent(agent: _FakeAgent):
    import mimir.client.agent_core as agent_core
    original = agent_core.MimirAgent
    agent_core.MimirAgent = lambda **_kw: agent
    # The tool points stdout at stderr on first use (its stdout is the JSON-RPC pipe);
    # under a test runner that would swallow the rest of the session's output.
    saved_stdout, spawn._stdout_silenced = sys.stdout, False
    try:
        yield agent
    finally:
        agent_core.MimirAgent = original
        sys.stdout = saved_stdout
        spawn._stdout_silenced = False


class ToolGrantTests(unittest.TestCase):
    """What the caller grants is what the child gets — no more, and nothing implied."""

    def test_naming_no_tool_explores_in_a_readonly_mode(self):
        """Read-only is the child's MODE, not a hand-kept list of servers.

        `files` is in an explorer's server set and carries the write tools; what stops
        it writing is the mode's capability filter and its dual-use call gate.
        """
        from mimir.client.config.models import READONLY_MODES
        agent, _ = _run_child()
        self.assertIn(agent.mode, READONLY_MODES)
        self.assertIn(agent.run_kwargs["mode"], READONLY_MODES)

    def test_a_writing_tool_puts_the_child_in_agent_mode(self):
        agent, _ = _run_child(["read_file_lines", "write_file"])
        self.assertEqual(agent.mode, "agent")
        self.assertEqual(agent.run_kwargs["mode"], "agent")

    def test_a_granted_shell_may_actually_run(self):
        """A child given the shell to build with, then held in a read-only mode, could
        not run the build it was delegated."""
        agent, _ = _run_child(["bash_run"])
        self.assertEqual(agent.mode, "agent")

    def test_reading_tools_alone_stay_read_only(self):
        from mimir.client.config.models import READONLY_MODES
        agent, _ = _run_child(["read_file_lines", "find_definition"])
        self.assertIn(agent.mode, READONLY_MODES)

    def test_only_the_granted_tools_survive_their_server(self):
        """Connecting a server brings its siblings; the grant has to be exact."""
        agent, _ = _run_child(["read_file_lines"])
        self.assertEqual(agent.tool_names(), {"read_file_lines"})
        self.assertEqual(set(agent.tool_owner), {"read_file_lines"})
        self.assertEqual(set(agent.tool_caps), {"read_file_lines"})

    def test_only_the_servers_the_grant_needs_are_started(self):
        agent, _ = _run_child(["read_file_lines"])
        self.assertEqual(agent.connected, ["files"])

    def test_an_exploration_gets_symbol_navigation_and_a_shell(self):
        """The set once said "code", which matches no server: an explorer with no
        symbol navigation and no grep is an explorer that reads whole files."""
        agent, _ = _run_child()
        self.assertIn("code_intel", agent.connected)
        self.assertIn("bash", agent.connected)
        self.assertIn("search", agent.connected)

    def test_no_child_ever_connects_the_spawn_server(self):
        """A child that can spawn its own children recurses without a budget."""
        for tools in (None, ["read_file_lines", "write_file"]):
            with self.subTest(tools=tools):
                agent, _ = _run_child(tools)
                self.assertNotIn("agent", agent.connected)

    def test_an_exploration_is_briefed_to_return_a_conclusion(self):
        agent, _ = _run_child()
        self.assertIn("CONCLUSION", agent.run_kwargs["query"])
        self.assertIn("find X", agent.run_kwargs["query"])

    def test_a_working_child_is_briefed_to_hand_over_before_its_time_runs_out(self):
        agent, _ = _run_child(["write_file"])
        query = agent.run_kwargs["query"]
        self.assertNotIn("CONCLUSION", query)
        self.assertIn("HANDOFF", query)

    def test_child_does_not_reason_out_loud(self):
        agent, _ = _run_child()
        self.assertEqual(agent.thinking_depth, 0)


class PayloadTests(unittest.TestCase):
    def test_files_read_comes_back_so_the_caller_can_record_the_evidence(self):
        _, result = _run_child()
        self.assertEqual(result["files_read"], ["/w/a.py", "/w/b.py"])

    def test_completed_is_false_when_the_step_budget_ran_out(self):
        _, result = _run_child(answer="Reached the maximum number of steps (5).")
        self.assertFalse(result["completed"])
        self.assertTrue(result["answer"])  # the partial answer is still informative

    def test_completed_is_true_on_a_clean_run(self):
        _, result = _run_child(None, answer="X is defined in a.py:12.")
        self.assertTrue(result["completed"])

    def test_an_empty_answer_is_not_a_completed_run(self):
        # The answer IS the payload: a blank one delegates nothing back, whatever the
        # child touched. Observed in the wild reported as ok/completed, which reads as
        # "ran, found nothing to say" — the parent then dropped delegation for the rest
        # of the run and did the whole sweep serially.
        _, result = _run_child(None, answer="   ")
        self.assertFalse(result["completed"])

    def test_an_empty_answer_reaches_the_caller_as_an_error(self):
        with _patched_agent(_FakeAgent(answer="")):
            out = asyncio.run(spawn.spawn_agent("find X"))
        self.assertEqual(out["status"], "error")
        self.assertFalse(out["completed"])
        # The files it did read still come back — that is where it got to.
        self.assertEqual(out["files_read"], ["/w/a.py", "/w/b.py"])


class ChildVisibilityTests(unittest.TestCase):
    """What the child is doing has to leave this process, and only one way out works.

    This server's stdout IS the JSON-RPC pipe, and the child runs in-process: an
    unbound event sink puts every one of its events through emit()'s print fallback,
    straight into the protocol stream. Binding a sink plugs that leak and is also what
    lets the caller show the child's tool calls.
    """

    def test_the_child_engine_gets_a_sink_instead_of_printing(self):
        seen: list = []
        agent, _ = _run_child(None, on_event=seen.append)
        self.assertIsNotNone(agent.run_kwargs["event_callback"])

    def test_the_tool_is_async_so_a_running_child_frees_the_loop(self):
        """FastMCP awaits a sync tool inline: a child blocking for its whole cap would
        block this server's loop, and a fan-out of calls would run one after another."""
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(spawn.spawn_agent))

    def test_the_stream_itself_does_not_travel(self):
        """Tokens and diffs would cost far more than they show on a delegated run.

        Status used to be dropped with them, on the same reasoning. It does not belong
        with them: it is one line per step boundary, not one per token, and it is the
        only account of the model calls a step makes without calling a tool — see
        StatusForwardingTests."""
        self.assertIsNone(spawn._compact_event({"type": "token", "text": "hello"}))
        self.assertIsNone(spawn._compact_event({"type": "diff", "file": "a.py"}))

    def test_a_forwarded_event_is_clipped_to_a_notification_sized_payload(self):
        out = spawn._compact_event({
            "type": "tool_call", "id": "c1", "name": "grep",
            "label": "x" * 500, "detail": "y" * 500,
        })
        self.assertEqual(out["t"], "tc")
        self.assertLessEqual(len(out["l"]), 120)
        self.assertLessEqual(len(out["d"]), 160)

    def test_a_full_queue_drops_the_event_rather_than_breaking_the_child(self):
        """Raising here would put the child's engine back on the print fallback."""
        import queue as _queue
        q = _queue.Queue(maxsize=1)
        counters: dict = {"dropped": 0}
        sink = spawn._make_child_sink(q, counters)
        sink({"type": "tool_call", "id": "c1", "name": "grep"})
        sink({"type": "tool_call", "id": "c2", "name": "grep"})  # no room left
        self.assertEqual(q.qsize(), 1)
        self.assertEqual(counters["dropped"], 1)


class _RecordingCtx:
    """A caller that listens: records what the tool reports while the child runs."""

    def __init__(self, fail: bool = False):
        self.reports: list[tuple] = []
        self._fail = fail

    async def report_progress(self, progress, total=None, message=None):
        if self._fail:
            raise RuntimeError("broken pipe")
        self.reports.append((progress, message))


class _EmittingAgent(_FakeAgent):
    """A child that calls one tool before answering."""

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        sink = kwargs.get("event_callback")
        if sink:
            sink({"type": "tool_call", "id": "c1", "name": "grep",
                  "label": "Searching: _ROUTER", "detail": "mimir/"})
            sink({"type": "tool_result", "id": "c1", "ok": True,
                  "summary": "3 matches", "duration_ms": 41})
        return self._answer


def _run_tool(agent: _FakeAgent, ctx):
    with _patched_agent(agent):
        return asyncio.run(spawn.spawn_agent("find X", ctx=ctx))


class ProgressForwardingTests(unittest.TestCase):
    def test_the_child_s_tool_calls_reach_the_caller_in_order(self):
        ctx = _RecordingCtx()
        _run_tool(_EmittingAgent("X is in a.py:12."), ctx)
        kinds = [json.loads(m)["t"] for _, m in ctx.reports]
        self.assertEqual(kinds, ["tc", "tr"])
        self.assertEqual(json.loads(ctx.reports[0][1])["n"], "grep")

    def test_the_progress_counter_only_moves_forward(self):
        ctx = _RecordingCtx()
        _run_tool(_EmittingAgent(), ctx)
        counts = [p for p, _ in ctx.reports]
        self.assertEqual(counts, sorted(set(counts)))

    def test_the_answer_is_unchanged_by_any_of_this(self):
        ctx = _RecordingCtx()
        out = _run_tool(_EmittingAgent("X is in a.py:12."), ctx)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["answer"], "X is in a.py:12.")
        self.assertTrue(out["completed"])
        self.assertEqual(out["files_read"], ["/w/a.py", "/w/b.py"])

    def test_a_caller_that_is_not_listening_changes_nothing(self):
        out = _run_tool(_EmittingAgent("X is in a.py:12."), None)
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["answer"], "X is in a.py:12.")

    def test_a_dead_channel_never_costs_the_run_its_answer(self):
        out = _run_tool(_EmittingAgent("X is in a.py:12."), _RecordingCtx(fail=True))
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["answer"], "X is in a.py:12.")



class _MetaCtx(_RecordingCtx):
    """A caller that sends its approval mode in the request _meta, as the client does."""

    def __init__(self, mode):
        super().__init__()
        from mcp.types import RequestParams
        self.request_context = types.SimpleNamespace(
            meta=RequestParams.Meta(**{spawn._APPROVAL_MODE_META: mode}))


class _BlockedAgent(_FakeAgent):
    """A child whose mode refused one action during its run."""

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        self.approvals.mode_blocked.append({"action": "Running: make", "needs": "auto"})
        return "Skipped the build: the approval mode does not allow it."


class ApprovalModeInheritanceTests(unittest.TestCase):
    def test_the_meta_key_matches_the_client_s(self):
        from mimir.client.guardrails.policy.approval import APPROVAL_MODE_META, ApprovalManager
        self.assertEqual(spawn._APPROVAL_MODE_META, APPROVAL_MODE_META)
        self.assertEqual(spawn._APPROVAL_MODES, ApprovalManager.APPROVAL_MODES)

    def test_the_child_runs_in_the_caller_s_mode(self):
        for mode in ("manual", "auto", "auto_all"):
            with self.subTest(mode=mode):
                agent = _FakeAgent()
                _run_tool(agent, _MetaCtx(mode))
                self.assertEqual(agent.approvals.approval_mode, mode)


class UserChannelTests(unittest.TestCase):
    """What a sub-agent cannot decide reaches the person, or it does not run."""

    def test_a_blocking_child_can_reach_the_user(self):
        """Its call is still open, so its approvals and questions ride that session
        rather than being refused on the user's behalf."""
        agent = _FakeAgent()
        _run_tool(agent, _MetaCtx("manual"))
        self.assertFalse(agent.approvals.unattended)
        for hook in ("_request_tool_approval", "_request_path_approval",
                     "_request_user_question"):
            self.assertTrue(callable(getattr(agent, hook, None)), hook)

    def test_a_detached_child_has_nobody_to_ask(self):
        """Its call has returned: there is no session left to raise a card on, so it
        runs unattended and reports what its mode refused."""
        agent = _FakeAgent()
        done = threading.Event()
        seen: dict = {}

        async def _drive(child, *a, **kw):
            # No session to raise a card on, so no channel is handed down; the real
            # driver then leaves the child unattended, as it always did.
            seen["ask_ctx"] = kw.get("ask_ctx")
            seen["ask_loop"] = kw.get("ask_loop")
            done.set()
            return {"answer": "x", "completed": True, "files_read": [],
                    "files_written": [], "blocked_by_mode": [], "error": None}

        with _patched_agent(agent), mock.patch.object(spawn, "_drive_sub_agent", _drive):
            asyncio.run(spawn.spawn_agent("axis", ctx=_GrantCtx(), background=True))
            self.assertTrue(done.wait(5))
        self.assertIsNone(seen["ask_ctx"])
        self.assertIsNone(seen["ask_loop"])

    def test_the_cards_of_two_children_are_asked_one_at_a_time(self):
        """Several children work at once; several cards at once is a pile nobody can
        answer in order."""
        order: list[str] = []

        class _Ctx:
            def __init__(self, name):
                self.name = name
                self.session = self

            async def elicit_form(self, message, requestedSchema):
                order.append(f"{self.name}:in")
                await asyncio.sleep(0.05)
                order.append(f"{self.name}:out")
                return types.SimpleNamespace(action="decline", content=None)

        async def _both():
            await asyncio.gather(
                spawn._ask_the_user(_Ctx("a"), "h", "q", []),
                spawn._ask_the_user(_Ctx("b"), "h", "q", []),
            )

        asyncio.run(_both())
        self.assertIn(order, (["a:in", "a:out", "b:in", "b:out"],
                              ["b:in", "b:out", "a:in", "a:out"]))

    def test_a_card_says_which_sub_agent_is_asking(self):
        asked: dict = {}

        class _Ctx:
            session = None

            def __init__(self):
                self.session = self

            async def elicit_form(self, message, requestedSchema):
                asked["message"] = message
                return types.SimpleNamespace(action="decline", content=None)

        agent = _FakeAgent()
        loop = asyncio.new_event_loop()
        threading.Thread(target=loop.run_forever, daemon=True).start()
        spawn._install_user_channel(agent, _Ctx(), loop, "vectorise the inner loop")
        approved, _ = agent._request_tool_approval("write_file", {"path": "/w/a.c"})
        loop.call_soon_threadsafe(loop.stop)
        self.assertFalse(approved)          # a declined card is not an approval
        self.assertIn("vectorise the inner loop", asked["message"])
        self.assertIn("write_file", asked["message"])

    def test_no_or_unknown_mode_falls_back_to_manual(self):
        self.assertEqual(spawn._caller_approval_mode(None), "manual")
        self.assertEqual(spawn._caller_approval_mode(_MetaCtx("yolo")), "manual")
        agent = _FakeAgent()
        _run_tool(agent, _RecordingCtx())
        self.assertEqual(agent.approvals.approval_mode, "manual")

    def test_what_the_mode_blocked_reaches_the_caller_and_the_run_is_not_complete(self):
        out = _run_tool(_BlockedAgent(), _MetaCtx("manual"))
        self.assertEqual(out["status"], "ok")
        self.assertFalse(out["completed"])
        self.assertEqual(out["blocked_by_mode"],
                         [{"action": "Running: make", "needs": "auto"}])

    def test_a_clean_run_reports_nothing_blocked(self):
        _, result = _run_child(["write_file"], answer="done", approval_mode="auto")
        self.assertEqual(result["blocked_by_mode"], [])
        self.assertTrue(result["completed"])


class _ModelCtx(_RecordingCtx):
    """A caller sending its current model in the request _meta, as the client does."""

    def __init__(self, model):
        super().__init__()
        from mcp.types import RequestParams
        self.request_context = types.SimpleNamespace(
            meta=RequestParams.Meta(**{spawn._CALLER_MODEL_META: model}))


_SERVED = [{"id": "RedHatAI/GLM-5.3-Flash-NVFP4"},
           {"id": "deepseek-ai/DeepSeek-V4.1-Flash"},
           {"id": "Qwen/Qwen3-0.6B"}]


class SubAgentModelTests(unittest.TestCase):
    """The child runs on the caller's model, or on a served one the caller names."""

    def _spawn(self, ctx, **kwargs):
        built: list[str] = []
        agent = _FakeAgent()
        import mimir.client.agent_core as agent_core
        saved = spawn._SERVED_MODELS
        spawn._SERVED_MODELS = _SERVED
        try:
            with _patched_agent(agent):
                agent_core.MimirAgent = lambda **kw: built.append(kw.get("model")) or agent
                out = asyncio.run(spawn.spawn_agent("find X", ctx=ctx, **kwargs))
        finally:
            spawn._SERVED_MODELS = saved
        return out, built

    def test_the_meta_key_matches_the_client_s(self):
        from mimir.client.config.models import CALLER_MODEL_META
        self.assertEqual(spawn._CALLER_MODEL_META, CALLER_MODEL_META)

    def test_without_a_model_the_child_follows_the_caller_s_current_one(self):
        """The env was frozen at spawn: a mid-session switch reaches the child via _meta."""
        out, built = self._spawn(_ModelCtx("deepseek-ai/DeepSeek-V4.1-Flash"))
        self.assertEqual(built, ["deepseek-ai/DeepSeek-V4.1-Flash"])
        self.assertEqual(out["model"], "deepseek-ai/DeepSeek-V4.1-Flash")

    def test_a_served_model_the_caller_names_runs_the_child(self):
        out, built = self._spawn(_ModelCtx("deepseek-ai/DeepSeek-V4.1-Flash"),
                                 model="RedHatAI/GLM-5.3-Flash-NVFP4")
        self.assertEqual(out["status"], "ok")
        self.assertEqual(built, ["RedHatAI/GLM-5.3-Flash-NVFP4"])

    def test_an_unserved_model_is_refused_with_the_served_list(self):
        out, built = self._spawn(_ModelCtx("deepseek-ai/DeepSeek-V4.1-Flash"), model="glm")
        self.assertEqual(out["status"], "error")
        self.assertIn("RedHatAI/GLM-5.3-Flash-NVFP4", out["error"])
        self.assertNotIn("Qwen/Qwen3-0.6B", out["error"])
        self.assertEqual(built, [])

    def test_a_model_the_catalog_marks_not_delegable_is_refused(self):
        out, built = self._spawn(_ModelCtx("deepseek-ai/DeepSeek-V4.1-Flash"),
                                 model="Qwen/Qwen3-0.6B")
        self.assertEqual(out["status"], "error")
        self.assertIn("cannot run a sub-agent", out["error"])
        self.assertEqual(built, [])

    def test_naming_the_caller_s_own_model_needs_no_catalog_check(self):
        """Even when the endpoint lists nothing (Ollama, Anthropic) the own model runs."""
        saved = _SERVED[:]
        _SERVED.clear()
        try:
            out, built = self._spawn(_ModelCtx("m"), model="m")
        finally:
            _SERVED.extend(saved)
        self.assertEqual(built, ["m"])
        self.assertEqual(out["status"], "ok")


class _GrantCtx(_RecordingCtx):
    """A caller sending what it may grant, its mode and its rung, as the client does."""

    def __init__(self, grantable=None, mode="agent", level="parallel"):
        super().__init__()
        from mcp.types import RequestParams
        self.request_context = types.SimpleNamespace(meta=RequestParams.Meta(**{
            spawn._GRANTABLE_TOOLS_META: dict(_GRANTABLE if grantable is None else grantable),
            spawn._GRANTABLE_WRITERS_META: list(_WRITERS),
            spawn._CALLER_MODE_META: mode,
            spawn._SUBAGENT_LEVEL_META: level,
        }))


class GrantRefusalTests(unittest.TestCase):
    """A name the caller cannot grant is refused, and nothing is started."""

    def _spawn(self, ctx, **kwargs):
        agent = _FakeAgent()
        with _patched_agent(agent):
            return asyncio.run(spawn.spawn_agent("find X", ctx=ctx, **kwargs)), agent

    def test_the_meta_keys_match_the_client_s(self):
        from mimir.client.config.models import CALLER_MODE_META, GRANTABLE_TOOLS_META
        self.assertEqual(spawn._GRANTABLE_TOOLS_META, GRANTABLE_TOOLS_META)
        self.assertEqual(spawn._CALLER_MODE_META, CALLER_MODE_META)

    def test_a_granted_tool_runs(self):
        out, agent = self._spawn(_GrantCtx(), tools=["read_file_lines"])
        self.assertEqual(out["status"], "ok")
        self.assertEqual(out["tools"], ["read_file_lines"])
        self.assertEqual(agent.tool_names(), {"read_file_lines"})

    def test_a_tool_the_caller_cannot_grant_is_refused_with_what_it_can(self):
        out, agent = self._spawn(_GrantCtx(), tools=["read_file_lines", "memory_add"])
        self.assertEqual(out["status"], "error")
        self.assertIn("memory_add", out["error"])
        self.assertIn("read_file_lines", out["error"])
        self.assertEqual(agent.connected, [])   # nothing was started

    def test_a_caller_in_a_readonly_mode_has_nothing_writing_to_grant(self):
        """Plan mode is not worked around by delegating: the table the caller sends is
        computed under its own mode, so the writing tools are simply not in it."""
        readonly = {k: v for k, v in _GRANTABLE.items() if k != "write_file"}
        out, _ = self._spawn(_GrantCtx(readonly, mode="plan"), tools=["write_file"])
        self.assertEqual(out["status"], "error")

    def test_without_a_table_only_an_exploration_runs(self):
        """An unknown caller vouches for nothing, so it grants nothing."""
        out, _ = self._spawn(_RecordingCtx(), tools=["read_file_lines"])
        self.assertEqual(out["status"], "error")
        out, agent = self._spawn(_RecordingCtx())
        self.assertEqual(out["status"], "ok")
        self.assertIn("files", agent.connected)


class SubAgentLevelTests(unittest.TestCase):
    """The rung the user set, read off the call rather than guessed."""

    def test_the_meta_keys_and_the_rungs_match_the_client_s(self):
        from mimir.client.config.constants import SUBAGENT_LEVELS
        from mimir.client.config.models import (
            GRANTABLE_WRITERS_META, SUBAGENT_LEVEL_META,
        )
        self.assertEqual(spawn._SUBAGENT_LEVEL_META, SUBAGENT_LEVEL_META)
        self.assertEqual(spawn._GRANTABLE_WRITERS_META, GRANTABLE_WRITERS_META)
        self.assertEqual(spawn._SUBAGENT_LEVELS, SUBAGENT_LEVELS)

    def test_there_is_no_rung_for_writing_in_the_shared_tree(self):
        """Two sub-agents editing one tree overwrite each other in silence, and no
        instruction to a model prevents that. So writing means a copy, or nothing."""
        self.assertEqual(spawn._SUBAGENT_LEVELS, ("explore", "parallel"))

    def test_an_absent_or_unknown_rung_is_the_weakest(self):
        self.assertEqual(spawn._subagent_level(_RecordingCtx()), "explore")
        self.assertEqual(spawn._subagent_level(_GrantCtx(level="everything")), "explore")

    def test_at_explore_a_writing_tool_does_not_buy_agent_mode(self):
        """The caller should not have been able to grant one; if one arrives anyway,
        the rung still decides how the child runs."""
        from mimir.client.config.models import READONLY_MODES
        agent, _ = _run_child(["write_file"], level="explore")
        self.assertIn(agent.mode, READONLY_MODES)

    def test_at_explore_a_reading_child_still_runs_in_the_shared_tree(self):
        """Nothing to isolate: it cannot write, so it costs no branch."""
        agent, _ = _run_child(["read_file_lines"], level="explore")
        self.assertEqual(agent.server_env.get("MCP_FILES_ROOT"), None)


class SubSessionTests(unittest.TestCase):
    """Each child writes under its own session, below the caller's."""

    def _session_of(self, ctx=None):
        agent = _FakeAgent()
        with _patched_agent(agent):
            out = asyncio.run(spawn.spawn_agent("find X", ctx=ctx or _GrantCtx()))
        return out, agent

    def test_the_child_hands_its_own_session_to_the_servers_it_starts(self):
        out, agent = self._session_of()
        self.assertTrue(out["session"])
        self.assertEqual(agent.server_env, {"MIMIR_SESSION_ID": out["session"]})

    def test_a_sub_session_is_stored_under_its_parent(self):
        import state_paths
        with mock.patch.object(state_paths, "active_session_id", return_value="parent-1"):
            out, _ = self._session_of()
        self.assertTrue(out["session"].startswith("parent-1/subagents/sub-"))

    def test_the_child_itself_knows_its_session(self):
        """Not only its servers: the agent resolves its own todo file and advertises its
        scratchpad from this, and several children share one process."""
        out, agent = self._session_of()
        self.assertEqual(agent.session_id, out["session"])

    def test_two_children_never_share_a_session(self):
        first, _ = self._session_of()
        second, _ = self._session_of()
        self.assertNotEqual(first["session"], second["session"])

    def test_the_run_is_recorded_for_the_panel(self):
        import state_paths
        with tempfile.TemporaryDirectory() as tmp:
            with mock.patch.object(state_paths, "state_dir", return_value=tmp), \
                 mock.patch.object(state_paths, "active_session_id", return_value="parent-1"):
                out, _ = self._session_of()
            card = json.loads((Path(tmp) / "sessions" / out["session"] / "subagent.json")
                              .read_text(encoding="utf-8"))
        self.assertEqual(card["task"], "find X")
        self.assertEqual(card["state"], "finished")
        self.assertTrue(card["completed"])


class TimeBudgetTests(unittest.TestCase):
    def test_the_budget_is_clamped_to_what_the_tool_declares(self):
        agent = _FakeAgent()
        seen: dict = {}

        async def _slow(*a, **kw):
            seen.update(kw)
            return {"answer": "done", "completed": True, "files_read": [],
                    "files_written": [], "blocked_by_mode": [], "error": None}

        with _patched_agent(agent), mock.patch.object(spawn, "_drive_sub_agent", _slow):
            asyncio.run(spawn.spawn_agent("find X", ctx=_GrantCtx(), time_budget_secs=99999))
        self.assertEqual(seen["budget"], spawn.SUBAGENT_HARD_CAP_SECS)

    def test_a_child_out_of_time_hands_back_where_it_got_to(self):
        """Its answer is gone with the thread; what it touched is not, and that is what
        lets the caller carry the axis on instead of starting it again."""
        agent = _FakeAgent()
        agent._carry_context["last_query_written_files"] = {"/w/solver.py"}

        async def _never(*a, **kw):
            await asyncio.sleep(5)
            return {}

        with _patched_agent(agent), \
             mock.patch.object(spawn, "_drive_sub_agent", _never), \
             mock.patch.object(spawn, "SUBAGENT_MIN_BUDGET_SECS", 1):
            out = asyncio.run(spawn.spawn_agent(
                "optimise", ctx=_GrantCtx(), time_budget_secs=1))
        self.assertEqual(out["status"], "error")
        self.assertIn("/w/solver.py", out["answer"])
        self.assertEqual(out["files_written"], ["/w/solver.py"])
        self.assertTrue(out["session"])


class BackgroundTests(unittest.TestCase):
    """A detached sub-agent: the caller gets a handle and goes on working."""

    def setUp(self):
        spawn._JOBS.clear()

    @contextlib.contextmanager
    def _launched(self, agent, drive):
        """A detached child outlives its call, so the fakes must outlive it too."""
        with _patched_agent(agent), mock.patch.object(spawn, "_drive_sub_agent", drive):
            yield asyncio.run(spawn.spawn_agent(
                "optimise the inner loop", ctx=_GrantCtx(), background=True))

    @staticmethod
    def _settled(job_key: str) -> dict:
        for _ in range(100):
            state = spawn.subagent_job(job_key=job_key).get("state")
            if state and state != "running":
                return spawn.subagent_job("result", job_key)
            time.sleep(0.05)
        raise AssertionError(f"job {job_key} never settled")

    def test_the_call_returns_a_handle_instead_of_an_answer(self):
        done = threading.Event()

        async def _drive(*a, **kw):
            done.wait(5)
            return {"answer": "3.2s → 1.9s", "completed": True, "files_read": [],
                    "files_written": ["/w/solver.py"], "blocked_by_mode": [], "error": None}

        with self._launched(_FakeAgent(), _drive) as out:
            self.assertEqual(out["status"], "ok")
            job = out["background_job"]
            self.assertEqual(job["kind"], "sub-agent")
            self.assertEqual(spawn.subagent_job(job_key=job["job_key"])["state"], "running")
            done.set()
            settled = self._settled(job["job_key"])
        self.assertEqual(settled["state"], "done")
        self.assertEqual(settled["answer"], "3.2s → 1.9s")
        self.assertTrue(settled["completed"])
        self.assertEqual(settled["files_written"], ["/w/solver.py"])

    def test_its_result_can_be_read_while_it_still_works(self):
        """Progress, not "are we there yet": what it has touched is real before its
        answer is."""
        release = threading.Event()
        started = threading.Event()
        agent = _FakeAgent()
        agent._carry_context["last_query_written_files"] = {"/w/solver.py"}

        async def _drive(*a, **kw):
            # The child publishes itself just before this runs, so waiting here is what
            # makes "read it while it works" a defined moment rather than a race.
            started.set()
            release.wait(5)
            return {"answer": "done", "completed": True, "files_read": [],
                    "files_written": [], "blocked_by_mode": [], "error": None}

        with self._launched(agent, _drive) as out:
            self.assertTrue(started.wait(5))
            try:
                partial = spawn.subagent_job("result", out["background_job"]["job_key"])
                self.assertEqual(partial["state"], "running")
                self.assertFalse(partial["completed"])
                self.assertIn("/w/solver.py", partial["answer"])
            finally:
                release.set()
            self._settled(out["background_job"]["job_key"])

    def test_a_child_that_crashes_settles_as_crashed(self):
        async def _drive(*a, **kw):
            raise RuntimeError("no such file")

        with self._launched(_FakeAgent(), _drive) as out:
            settled = self._settled(out["background_job"]["job_key"])
        self.assertEqual(settled["state"], "crashed")

    def test_an_unknown_handle_is_an_error_not_an_empty_state(self):
        out = spawn.subagent_job(job_key="sub-nope")
        self.assertEqual(out["status"], "error")

    def test_the_client_refuses_polling_but_allows_reading_the_result(self):
        """The wake answers "is it done"; asking it again is the poll the watcher
        exists to replace. Reading what it has is a different question."""
        from mimir.client.query_engine.dispatch import _asks_whether_a_watched_run_is_done
        descriptor = spawn._job_descriptor("sub-ab12cd34")
        agent = types.SimpleNamespace(_watched_background_jobs=lambda: [descriptor])
        status = descriptor["status_op"]
        summary = descriptor["summary_op"]
        self.assertEqual(
            _asks_whether_a_watched_run_is_done(agent, status["tool"], status["args"]),
            "sub-ab12cd34")
        self.assertIsNone(
            _asks_whether_a_watched_run_is_done(agent, summary["tool"], summary["args"]))


class WorktreeTests(unittest.TestCase):
    """An axis working in a copy of its own: the copy survives, an empty branch does not."""

    def setUp(self):
        self.repo = tempfile.mkdtemp(prefix="mimir-repo-")
        self.base = tempfile.mkdtemp(prefix="mimir-wt-")
        self.addCleanup(shutil.rmtree, self.repo, ignore_errors=True)
        self.addCleanup(shutil.rmtree, self.base, ignore_errors=True)
        for args in (["init", "-q", "-b", "main"],
                     ["config", "user.email", "t@t"], ["config", "user.name", "t"]):
            subprocess.run(["git", *args], cwd=self.repo, check=True,
                           capture_output=True)
        (Path(self.repo) / "solver.c").write_text("int main(){}\n", encoding="utf-8")
        subprocess.run(["git", "add", "-A"], cwd=self.repo, check=True, capture_output=True)
        subprocess.run(["git", "commit", "-qm", "base"], cwd=self.repo, check=True,
                       capture_output=True)
        self._env = mock.patch.dict(os.environ, {"MCP_FILES_ROOT": self.repo})
        self._env.start()
        self.addCleanup(self._env.stop)
        self._wt_base = mock.patch.object(spawn, "_WORKTREE_BASE", self.base)
        self._wt_base.start()
        self.addCleanup(self._wt_base.stop)

    def _branches(self) -> list[str]:
        out = subprocess.run(["git", "branch", "--format=%(refname:short)"],
                             cwd=self.repo, capture_output=True, text=True)
        return out.stdout.split()

    def _spawn(self, edit: str | None = "vectorised", fail: bool = False):
        agent = _FakeAgent()

        async def _drive(child, *a, **kw):
            if edit is not None:
                Path(child.workspace_root, "solver.c").write_text(edit, encoding="utf-8")
            if fail:
                raise RuntimeError("the build died")
            return {"answer": "3.2s → 1.9s", "completed": True, "files_read": [],
                    "files_written": [], "blocked_by_mode": [], "error": None}

        with _patched_agent(agent), mock.patch.object(spawn, "_drive_sub_agent", _drive):
            out = asyncio.run(spawn.spawn_agent(
                "vectorise the loop", tools=["write_file"],
                ctx=_GrantCtx(level="parallel")))
        return out, agent

    def test_the_copy_is_its_own_checkout_outside_the_scratchpad(self):
        out, agent = self._spawn()
        path = out["workspace"]["path"]
        self.assertTrue(path.startswith(self.base))
        # The scratchpad is writable without approval; a copy there would take every
        # edit the child makes out of the approval layer.
        from mimir.servers._shared.state_paths import scratch_home
        self.assertFalse(path.startswith(scratch_home()))
        self.assertEqual(agent.server_env["MCP_FILES_ROOT"], path)
        self.assertEqual(agent.server_env["SEARCH_ROOT"], path)
        self.assertTrue(agent.server_env["MIMIR_PROXY_BENCH_DIR"].startswith(self.repo))

    def test_a_finished_axis_is_committed_and_its_copy_kept(self):
        out, _ = self._spawn()
        work = out["workspace"]
        self.assertEqual(work["files_changed"], ["solver.c"])
        self.assertTrue(work["kept"])
        # The build tree is the reason: git never had it, and it is what the axis paid
        # its compile for.
        self.assertTrue(os.path.isdir(work["path"]))
        self.assertIn(work["branch"], self._branches())
        shown = subprocess.run(["git", "show", f"{work['branch']}:solver.c"],
                               cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(shown.stdout, "vectorised")

    def test_the_caller_is_told_the_branch_cannot_be_checked_out_elsewhere(self):
        # Git refuses a branch a worktree holds; a caller that does not know that reads
        # the refusal as the work having gone missing.
        out, _ = self._spawn()
        self.assertIn("cherry-pick", out["workspace"]["note"])

    def test_the_copy_carries_a_repo_so_it_can_be_reclaimed_later(self):
        out, _ = self._spawn()
        self.assertEqual(out["workspace"]["repo"], os.path.realpath(self.repo))

    def test_a_failed_axis_keeps_its_copy_because_nothing_else_holds_it(self):
        out, _ = self._spawn(fail=True)
        work = out["workspace"]
        self.assertTrue(work["kept"])
        self.assertTrue(os.path.isdir(work["path"]))
        self.assertEqual(
            Path(work["path"], "solver.c").read_text(encoding="utf-8"), "vectorised")

    def test_an_axis_that_changed_nothing_leaves_no_branch_behind(self):
        # A branch still at the commit it was cut from says nothing HEAD does not, and
        # one was left in the repository by every child that only read.
        out, _ = self._spawn(edit=None)
        work = out["workspace"]
        self.assertEqual(work["files_changed"], [])
        self.assertEqual(work["branch"], "")
        self.assertEqual([b for b in self._branches() if b.startswith("mimir/")], [])
        # The copy still stands: it is detached, not deleted.
        self.assertTrue(os.path.isdir(work["path"]))

    def test_an_axis_that_wrote_code_keeps_its_branch(self):
        out, _ = self._spawn()
        self.assertIn(out["workspace"]["branch"], self._branches())

    def test_the_main_tree_is_untouched_while_the_axis_works(self):
        self._spawn()
        self.assertEqual(
            Path(self.repo, "solver.c").read_text(encoding="utf-8"), "int main(){}\n")

    def test_the_oldest_copy_goes_when_the_next_one_needs_room(self):
        # Kept copies are bounded by a count, and the bound is applied where the disk is
        # about to be wanted — otherwise nothing ever reclaims /tmp.
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state,
                                          "MIMIR_SESSION_ID": "parent-session"}), \
                mock.patch.object(spawn, "SUBAGENT_WORKTREES_KEPT", 1):
            first = self._spawn()[0]["workspace"]["path"]
            second = self._spawn()[0]["workspace"]["path"]
            third = self._spawn()[0]["workspace"]["path"]
        self.assertFalse(os.path.exists(first))
        self.assertTrue(os.path.isdir(second))   # still within the bound when third ran
        self.assertTrue(os.path.isdir(third))

    def _card(self, state_dir, name, card):
        directory = Path(state_dir, "sessions", "parent-session", "subagents", name)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "subagent.json").write_text(json.dumps(card), encoding="utf-8")

    def test_a_copy_still_running_is_never_reclaimed(self):
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        wt, _ = spawn._create_worktree("sub-live")
        self._card(state, "sub-live", {
            "state": "running", "pid": os.getpid(), "started_at": "2000-01-01T00:00:00",
            "workspace": wt,
        })
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            spawn._reclaim_worktrees("parent-session/subagents/sub-new", keep=0)
        self.assertTrue(os.path.isdir(wt["path"]))

    def test_a_card_left_saying_running_by_a_dead_process_is_reclaimed(self):
        # A server killed mid-run never finishes its card. Taking the word at face value
        # would exempt that copy from reclamation for as long as the session lives.
        dead = subprocess.Popen([sys.executable, "-c", ""])
        dead.wait()
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        wt, _ = spawn._create_worktree("sub-orphan")
        self._card(state, "sub-orphan", {
            "state": "running", "pid": dead.pid, "started_at": "2000-01-01T00:00:00",
            "workspace": wt,
        })
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            spawn._reclaim_worktrees("parent-session/subagents/sub-new", keep=0)
        self.assertFalse(os.path.exists(wt["path"]))

    def test_reclaiming_a_copy_commits_what_it_had_not_committed(self):
        # The copy most worth reclaiming is the one a failed run left, and that is
        # exactly the one whose work exists nowhere else. --force would take it silently.
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        wt, _ = spawn._create_worktree("sub-halfway")
        Path(wt["path"], "solver.c").write_text("half done", encoding="utf-8")
        self._card(state, "sub-halfway", {
            "state": "finished", "started_at": "2000-01-01T00:00:00", "workspace": wt,
        })
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            spawn._reclaim_worktrees("parent-session/subagents/sub-new", keep=0)
        self.assertFalse(os.path.exists(wt["path"]))
        shown = subprocess.run(["git", "show", f"{wt['branch']}:solver.c"],
                               cwd=self.repo, capture_output=True, text=True)
        self.assertEqual(shown.stdout, "half done")   # the branch kept what the copy had

    def test_an_empty_branch_goes_even_after_the_repository_moved_on(self):
        # `git branch -d` asks whether the branch is merged into whatever HEAD is now,
        # and answers "no" for an empty branch as soon as the main tree commits again.
        # The recorded base commit is the question that actually holds.
        wt, _ = spawn._create_worktree("sub-quiet")
        Path(self.repo, "other.c").write_text("moved on", encoding="utf-8")
        for args in (["add", "-A"], ["commit", "-qm", "main moved on"]):
            subprocess.run(["git", *args], cwd=self.repo, check=True, capture_output=True)
        report = spawn._finish_worktree(wt, "read the mesh", succeeded=True)
        self.assertEqual(report["branch"], "")
        self.assertNotIn("mimir/sub-quiet", self._branches())

    def test_the_report_does_not_claim_a_branch_went_when_it_did_not(self):
        wt, _ = spawn._create_worktree("sub-stuck")
        with mock.patch.object(spawn, "_drop_branch_if_empty", return_value=False):
            report = spawn._finish_worktree(wt, "read the mesh", succeeded=True)
        self.assertEqual(report["branch"], "mimir/sub-stuck")
        self.assertIn("still where it was cut from", report["note"])


    def test_a_copy_whose_session_was_deleted_is_swept(self):
        # The cards are stored with the session: deleting the conversation takes the
        # registry and used to leave the copy on disk with nothing able to reach it.
        dead = subprocess.Popen([sys.executable, "-c", ""])
        dead.wait()
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            wt, _ = spawn._create_worktree("sub-gone", "deleted-session/subagents/sub-gone")
            spawn._write_marker(spawn._marker_path(self.repo, "sub-gone"),
                                {**wt, "session": "deleted-session/subagents/sub-gone",
                                 "pid": dead.pid})
            spawn._sweep_orphan_copies(self.repo)
        self.assertFalse(os.path.exists(wt["path"]))

    def test_a_copy_whose_session_still_has_its_card_is_left_to_that_rule(self):
        dead = subprocess.Popen([sys.executable, "-c", ""])
        dead.wait()
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        session = "parent-session/subagents/sub-held"
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            wt, _ = spawn._create_worktree("sub-held", session)
            spawn._write_marker(spawn._marker_path(self.repo, "sub-held"),
                                {**wt, "session": session, "pid": dead.pid})
            Path(state, "sessions", session).mkdir(parents=True)
            spawn._sweep_orphan_copies(self.repo)
        self.assertTrue(os.path.isdir(wt["path"]))

    def test_a_copy_whose_owner_is_still_alive_is_never_swept(self):
        # A copy is made before its card is written; a sweep in that window must not
        # take a copy a child is about to work in.
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            wt, _ = spawn._create_worktree("sub-young", "no-card-yet/subagents/sub-young")
            spawn._sweep_orphan_copies(self.repo)
        self.assertTrue(os.path.isdir(wt["path"]))

    def test_a_directory_git_never_registered_is_not_a_copy_of_ours(self):
        state = tempfile.mkdtemp(prefix="mimir-state-")
        self.addCleanup(shutil.rmtree, state, ignore_errors=True)
        junk = Path(self.base, os.path.basename(self.repo), "leftover")
        junk.mkdir(parents=True)
        (junk / "stale.o").write_text("", encoding="utf-8")
        with mock.patch.dict(os.environ, {"MIMIR_STATE_DIR": state}):
            spawn._sweep_orphan_copies(self.repo)
        self.assertFalse(junk.exists())

    def test_the_marker_sits_beside_the_copy_and_never_inside_it(self):
        # Inside, `git add -A` would commit it into the axis.
        wt, _ = spawn._create_worktree("sub-marked", "s/subagents/sub-marked")
        self.assertTrue(os.path.exists(spawn._marker_path(self.repo, "sub-marked")))
        self.assertFalse(os.path.exists(os.path.join(wt["path"], "sub-marked.mimir.json")))
        status = subprocess.run(["git", "status", "--porcelain"], cwd=wt["path"],
                                capture_output=True, text=True)
        self.assertEqual(status.stdout.strip(), "")

    def test_outside_a_git_repository_the_copy_is_refused(self):
        plain = tempfile.mkdtemp(prefix="mimir-plain-")
        self.addCleanup(shutil.rmtree, plain, ignore_errors=True)
        agent = _FakeAgent()
        with mock.patch.dict(os.environ, {"MCP_FILES_ROOT": plain}), \
             _patched_agent(agent):
            out = asyncio.run(spawn.spawn_agent(
                "axis", tools=["write_file"], ctx=_GrantCtx(level="parallel")))
        self.assertEqual(out["status"], "error")
        self.assertIn("git repository", out["error"])
        self.assertEqual(agent.connected, [])

    def test_a_reading_child_creates_no_branch_at_all(self):
        """The copy is the price of writing, not of delegating."""
        agent = _FakeAgent()
        with _patched_agent(agent):
            asyncio.run(spawn.spawn_agent("axis", tools=["read_file_lines"],
                                          ctx=_GrantCtx(level="parallel")))
        self.assertEqual(self._branches(), ["main"])


class ReportedPathTests(unittest.TestCase):
    """What a child says it touched, said the way the repository names it."""

    def test_a_path_inside_the_copy_comes_back_relative_to_the_repo(self):
        wt = {"path": "/tmp/wt/child"}
        self.assertEqual(
            spawn._repo_relative(["/tmp/wt/child/src/solver.c"], wt), ["src/solver.c"])

    def test_a_path_outside_the_copy_is_left_alone(self):
        wt = {"path": "/tmp/wt/child"}
        self.assertEqual(spawn._repo_relative(["/tmp/notes.txt"], wt), ["/tmp/notes.txt"])

    def test_a_child_without_a_copy_reports_what_it_read(self):
        self.assertEqual(spawn._repo_relative(["/repo/src/a.c"], None), ["/repo/src/a.c"])

    def test_a_handoff_past_the_budget_uses_the_same_names(self):
        # The evidence of a child that overran is read off the agent, not its result —
        # the other half of the report, and it must not speak a different language.
        agent = types.SimpleNamespace(_carry_context={
            "read_files": ["/tmp/wt/child/src/solver.c"],
            "last_query_written_files": ["/tmp/wt/child/src/solver.c"],
        })
        out = spawn._partial_handoff({"agent": agent, "worktree": {"path": "/tmp/wt/child"}})
        self.assertEqual(out["files_read"], ["src/solver.c"])
        self.assertEqual(out["files_written"], ["src/solver.c"])

class AbandonedChildTests(unittest.TestCase):
    """What becomes of a child whose caller stopped waiting.

    It runs on a thread of its own that nothing can kill, so the only lever is the
    cooperative cancel flag — and the only honest record is one that says the answer
    was never delivered, however cleanly the thread went on to finish.
    """

    def test_abandoning_sets_the_flag_the_child_s_loop_reads(self):
        flag = threading.Event()
        child = {"agent": types.SimpleNamespace(_cancel_flag=flag)}
        spawn._abandon_child(child)
        self.assertTrue(flag.is_set())
        self.assertTrue(child["abandoned"])

    def test_a_child_too_early_to_have_an_agent_is_still_marked(self):
        # The thread publishes its agent a moment after it starts; a budget that
        # expires inside that window must not raise in the caller's timeout branch.
        child: dict = {}
        spawn._abandon_child(child)
        self.assertTrue(child["abandoned"])

    def test_an_abandoned_run_is_not_recorded_as_finished(self):
        self.assertEqual(
            spawn._final_state({"abandoned": True}, {"completed": True}), "abandoned")

    def test_a_run_nobody_abandoned_keeps_its_own_verdict(self):
        self.assertEqual(spawn._final_state({}, {"completed": True}), "finished")
        self.assertEqual(spawn._final_state({}, None), "failed")
        self.assertEqual(spawn._final_state(None, {"completed": False}), "finished")


class StatusForwardingTests(unittest.TestCase):
    """The step-level lines that make a silent gap readable.

    A step is not one model call: an empty turn is retried, a nudge is answered, a
    checklist refresh re-asks. None is a tool call, and each can run for the child's
    whole per-step ceiling — so dropped, they left minutes of log with nothing in them.
    """

    def test_a_status_line_is_forwarded_as_its_own_row(self):
        out = spawn._compact_event(
            {"type": "status", "text": "  ↻ Empty turn from the model — retrying (1/3)."})
        self.assertEqual(out["t"], "st")
        self.assertIn("Empty turn", out["s"])

    def test_a_blank_status_is_not_a_row(self):
        # The loop emits blank status as a spacer for a terminal; here it would read
        # as an event whose text went missing.
        self.assertIsNone(spawn._compact_event({"type": "status", "text": "   "}))
        self.assertIsNone(spawn._compact_event({"type": "status"}))

    def test_a_long_status_is_clipped_like_every_other_field(self):
        out = spawn._compact_event({"type": "status", "text": "x" * 500})
        self.assertLessEqual(len(out["s"]), 160)

    def test_tokens_and_thinking_still_do_not_travel(self):
        for kind in ("token", "thinking", "diff"):
            self.assertIsNone(spawn._compact_event({"type": kind, "text": "..."}))


class BudgetTimeoutAdviceTests(unittest.TestCase):
    def test_the_timeout_line_refuses_an_identical_respawn(self):
        """The caller reads this line and acts on it. Told only to use a fresh
        sub-agent, it respawned the same task verbatim and burned a second budget on
        it — so the line has to say that repeating it unchanged is the wrong move."""
        src = Path(spawn.__file__).read_text()
        _, _, tail = src.partition("ran out of its")
        advice = tail[:600]
        self.assertIn("Do not spawn the same task again", advice)
        self.assertIn("time_budget_secs", advice)


class DeclaredBudgetTests(unittest.TestCase):
    def _descriptor(self):
        from mimir.tests import _golden_caps as golden
        return golden.build_declared_registry()["spawn_agent"]

    def test_declared_wall_exceeds_the_tool_s_own_cap(self):
        """Whichever fires first owns the outcome. The inner cap hands back the child's
        partial answer; the dispatcher's timeout hands back a failure line."""
        self.assertGreater(self._descriptor().timeout_secs, spawn.SUBAGENT_HARD_CAP_SECS)

    def test_the_dispatcher_reads_that_wall_and_not_the_global_default(self):
        from mimir.client.context.capabilities import timeout_for
        from mimir.client.config.constants import TOOL_CALL_TIMEOUT_SECS
        registry = {"spawn_agent": self._descriptor()}
        self.assertGreater(timeout_for("spawn_agent", registry), TOOL_CALL_TIMEOUT_SECS)
        self.assertEqual(timeout_for("other_tool", registry), TOOL_CALL_TIMEOUT_SECS)

    def test_the_declared_budget_covers_the_largest_the_tool_accepts(self):
        """The caller picks the wall per call; the dispatcher's own timeout must stay
        above the largest one, or a long axis is killed from outside with nothing to
        show."""
        self.assertGreater(spawn.SUBAGENT_HARD_CAP_SECS, spawn.SUBAGENT_DEFAULT_BUDGET_SECS)
        from mimir.client.config.constants import TOOL_CALL_TIMEOUT_MAX_SECS
        self.assertLessEqual(self._descriptor().timeout_secs, TOOL_CALL_TIMEOUT_MAX_SECS)

    def test_delegation_is_no_longer_dual_use_by_argument(self):
        """Its read-only-ness is not an argument any more: in a read-only mode the
        caller simply has nothing writing in the table it may grant."""
        self.assertIsNone(self._descriptor().readonly_when)


if __name__ == "__main__":
    unittest.main()
