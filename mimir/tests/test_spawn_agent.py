"""Contracts of the sub-agent spawn server.

The tool is the delegation channel: a broad sweep the orchestrator sends out instead of
paying for it in its own window. What is tested here is what makes that channel usable —
the role split, the toolkit an explorer actually gets, the anti-recursion, and the time
budget it declares to its caller — none of which any other test covered.
"""
import asyncio
import contextlib
import importlib.util
import json
import sys
import types
import unittest
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


class _FakeAgent:
    """A MimirAgent stand-in recording what the spawn server does to it."""

    def __init__(self, answer: str = "done"):
        self.connected: list[str] = []
        self.mode = "agent"
        self.thinking_depth = 99
        self.run_kwargs: dict = {}
        self._carry_context = {
            "read_files": {"/w/a.py", "/w/b.py"},
            "last_query_written_files": set(),
        }
        self._answer = answer

    def set_mode(self, mode): self.mode = mode
    def set_thinking_depth(self, depth): self.thinking_depth = depth
    def seed_classification_from_caps(self): pass

    async def connect_server(self, name, script):
        self.connected.append(name)

    async def run(self, **kwargs):
        self.run_kwargs = kwargs
        return self._answer


def _run_child(role: str, answer: str = "done", agent: _FakeAgent | None = None,
               on_event=None):
    """Drive _run_sub_agent against a fake agent and the real server catalog."""
    agent = agent or _FakeAgent(answer)
    with _patched_agent(agent):
        result = asyncio.run(spawn._run_sub_agent("find X", "", role, 5, on_event=on_event))
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


class RoleToolkitTests(unittest.TestCase):
    def test_explorer_runs_in_a_readonly_mode(self):
        """Read-only is the child's MODE, not a hand-kept list of servers.

        `files` is in an explorer's server set and carries the write tools; what stops
        it writing is the mode's capability filter and its dual-use call gate.
        """
        from mimir.client.config.models import READONLY_MODES
        agent, _ = _run_child(spawn.ROLE_EXPLORE)
        self.assertIn(agent.mode, READONLY_MODES)
        self.assertIn(agent.run_kwargs["mode"], READONLY_MODES)

    def test_task_role_runs_in_agent_mode(self):
        agent, _ = _run_child(spawn.ROLE_TASK)
        self.assertEqual(agent.mode, "agent")
        self.assertEqual(agent.run_kwargs["mode"], "agent")

    def test_explorer_gets_symbol_navigation_and_a_shell(self):
        """The set once said "code", which matches no server: an explorer with no
        symbol navigation and no grep is an explorer that reads whole files."""
        agent, _ = _run_child(spawn.ROLE_EXPLORE)
        self.assertIn("code_intel", agent.connected)
        self.assertIn("bash", agent.connected)
        self.assertIn("search", agent.connected)

    def test_no_role_ever_connects_the_spawn_server(self):
        """A child that can spawn its own children recurses without a budget."""
        for role in (spawn.ROLE_EXPLORE, spawn.ROLE_TASK):
            with self.subTest(role=role):
                agent, _ = _run_child(role)
                self.assertNotIn("agent", agent.connected)

    def test_unknown_role_is_refused_rather_than_guessed(self):
        out = asyncio.run(spawn.spawn_agent("find X", role="readonly"))
        self.assertEqual(out["status"], "error")
        self.assertIn(spawn.ROLE_EXPLORE, out["error"])

    def test_explorer_is_briefed_to_return_a_conclusion(self):
        agent, _ = _run_child(spawn.ROLE_EXPLORE)
        self.assertIn("CONCLUSION", agent.run_kwargs["query"])
        self.assertIn("find X", agent.run_kwargs["query"])

    def test_task_role_gets_no_exploration_brief(self):
        agent, _ = _run_child(spawn.ROLE_TASK)
        self.assertNotIn("CONCLUSION", agent.run_kwargs["query"])

    def test_child_does_not_reason_out_loud(self):
        agent, _ = _run_child(spawn.ROLE_EXPLORE)
        self.assertEqual(agent.thinking_depth, 0)


class PayloadTests(unittest.TestCase):
    def test_files_read_comes_back_so_the_caller_can_record_the_evidence(self):
        _, result = _run_child(spawn.ROLE_EXPLORE)
        self.assertEqual(result["files_read"], ["/w/a.py", "/w/b.py"])

    def test_completed_is_false_when_the_step_budget_ran_out(self):
        _, result = _run_child(
            spawn.ROLE_EXPLORE, answer="Reached the maximum number of steps (5).")
        self.assertFalse(result["completed"])
        self.assertTrue(result["answer"])  # the partial answer is still informative

    def test_completed_is_true_on_a_clean_run(self):
        _, result = _run_child(spawn.ROLE_EXPLORE, answer="X is defined in a.py:12.")
        self.assertTrue(result["completed"])

    def test_an_empty_answer_is_not_a_completed_run(self):
        # The answer IS the payload: a blank one delegates nothing back, whatever the
        # child touched. Observed in the wild reported as ok/completed, which reads as
        # "ran, found nothing to say" — the parent then dropped delegation for the rest
        # of the run and did the whole sweep serially.
        _, result = _run_child(spawn.ROLE_EXPLORE, answer="   ")
        self.assertFalse(result["completed"])

    def test_an_empty_answer_reaches_the_caller_as_an_error(self):
        with _patched_agent(_FakeAgent(answer="")):
            out = asyncio.run(spawn.spawn_agent("find X", role=spawn.ROLE_EXPLORE))
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
        agent, _ = _run_child(spawn.ROLE_EXPLORE, on_event=seen.append)
        self.assertIsNotNone(agent.run_kwargs["event_callback"])

    def test_the_tool_is_async_so_a_running_child_frees_the_loop(self):
        """FastMCP awaits a sync tool inline: a child blocking for its whole cap would
        block this server's loop, and a fan-out of calls would run one after another."""
        import inspect
        self.assertTrue(inspect.iscoroutinefunction(spawn.spawn_agent))

    def test_only_tool_activity_travels(self):
        """Tokens and diffs would cost far more than they show on a delegated run."""
        self.assertIsNone(spawn._compact_event({"type": "token", "text": "hello"}))
        self.assertIsNone(spawn._compact_event({"type": "diff", "file": "a.py"}))
        self.assertIsNone(spawn._compact_event({"type": "status", "text": "thinking"}))

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
        return asyncio.run(spawn.spawn_agent("find X", role=spawn.ROLE_EXPLORE, ctx=ctx))


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

    def test_only_the_exploring_role_survives_a_readonly_mode(self):
        from mimir.client.query_engine.readonly_guard import filter_readonly_tool_calls
        agent = types.SimpleNamespace(
            tool_caps={"spawn_agent": self._descriptor()},
            _normalize_arguments=lambda a: a,
        )
        def _call(args):
            return [{"id": "c1", "function": {"name": "spawn_agent", "arguments": args}}]

        for args, allowed in (
            ({"task": "t", "role": spawn.ROLE_EXPLORE}, True),
            ({"task": "t"}, True),                       # the tool's own default
            ({"task": "t", "role": spawn.ROLE_TASK}, False),
        ):
            with self.subTest(args=args):
                messages: list = []
                kept = filter_readonly_tool_calls(
                    _call(args), agent=agent, messages=messages, mode_label="plan")
                self.assertEqual(bool(kept), allowed)
                self.assertEqual(bool(messages), not allowed)


if __name__ == "__main__":
    unittest.main()
