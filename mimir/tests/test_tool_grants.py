"""What the orchestrator may hand to a sub-agent.

The grant is computed on the client, because only the client knows all three inputs:
the servers the user switched off, the mode the orchestrator is in, and which tools
their own servers declare reserved. The spawn server then trusts that table — so what
is tested here is the gate itself.
"""
from __future__ import annotations

import os
import tempfile
import types

from mimir.client.context.capabilities import ToolCaps, reserved_for_main
from mimir.client.query_engine.toollist import grantable_tools
from mimir.servers._shared.state_paths import active_session_id

# name -> (owning server, declared capabilities)
_TOOLS = {
    "read_file_lines": ("files", set()),
    "write_file": ("files", {"edit", "plan_blocked"}),
    "bash_run": ("bash", {"plan_readonly", "code_exec"}),
    "todo_write": ("todo", {"task_planning"}),
    "todo_set_plan": ("todo", {"task_planning", "main_only"}),
    "ask_user_question": ("interaction", {"main_only"}),
    "memory_add": ("memory", {"main_only"}),
    "memory_search": ("memory", set()),
    "spawn_agent": ("agent", {"delegate"}),
    "sbatch_submit": ("hpc", {"cluster_submit", "plan_blocked"}),
    "string_op": ("strings", set()),
}


def _agent(mode: str = "agent", disabled: set[str] | None = None,
           level: str = "parallel"):
    disabled = disabled or set()
    tools = [{"type": "function", "function": {"name": n}} for n in _TOOLS]
    owner = {n: s for n, (s, _) in _TOOLS.items()}
    caps = {n: ToolCaps(name=n, capabilities=frozenset(c)) for n, (_, c) in _TOOLS.items()}
    agent = types.SimpleNamespace(mode=mode, tool_owner=owner, tool_caps=caps, tools=tools,
                                  subagent_level=level)
    agent.advertised_tools = lambda: [
        t for t in tools if owner.get(t["function"]["name"]) not in disabled
    ]
    return agent


class TestReservedForMain:
    def test_the_three_reasons_are_read_off_the_declarations(self):
        reserved = reserved_for_main(_agent().tool_caps)
        assert "todo_set_plan" in reserved      # main_only
        assert "spawn_agent" in reserved        # delegate: no recursion
        assert "sbatch_submit" in reserved      # cluster_submit: allocation hours

    def test_a_working_checklist_is_not_reserved(self):
        """A sub-agent working one axis keeps a checklist — in its own session, so the
        orchestrator's list is untouched. Reserving every task_planning tool would have
        taken that with the plan."""
        assert "todo_write" not in reserved_for_main(_agent().tool_caps)

    def test_reading_memory_is_not_reserved(self):
        assert "memory_search" not in reserved_for_main(_agent().tool_caps)


class TestGrantableTools:
    def test_each_granted_tool_carries_its_server(self):
        """The child connects servers, not tools."""
        assert grantable_tools(_agent())["read_file_lines"] == "files"

    def test_nothing_reserved_is_ever_offered(self):
        granted = grantable_tools(_agent())
        assert not set(granted) & reserved_for_main(_agent().tool_caps)

    def test_a_readonly_caller_has_nothing_writing_to_give(self):
        """Delegation is not the way around plan mode: the table is computed under the
        caller's own mode, so a writing tool is simply not in it."""
        for mode in ("plan", "ask"):
            granted = grantable_tools(_agent(mode=mode))
            assert "write_file" not in granted
            assert "read_file_lines" in granted
            # The dual-use shell stays: its exec use is judged per command, and the
            # child inherits the same read-only mode.
            assert "bash_run" in granted

    def test_a_working_caller_may_give_what_it_uses(self):
        granted = grantable_tools(_agent())
        assert {"write_file", "bash_run", "todo_write"} <= set(granted)

    def test_a_server_the_user_switched_off_is_not_offered(self):
        """The user judged those tools irrelevant; a sub-agent must not reinstate them."""
        granted = grantable_tools(_agent(disabled={"files"}))
        assert "read_file_lines" not in granted and "write_file" not in granted
        assert "bash_run" in granted


class TestSubAgentLevels:
    """The rung the user set decides what may be handed over at all."""

    def test_explore_has_nothing_writing_to_give(self):
        granted = grantable_tools(_agent(level="explore"))
        assert "write_file" not in granted
        assert "read_file_lines" in granted

    def test_parallel_may_give_the_writing_tools(self):
        assert "write_file" in grantable_tools(_agent(level="parallel"))

    def test_an_agent_with_no_rung_is_treated_as_explore(self):
        """The absence of a choice is the cautious answer, not the permissive one."""
        agent = _agent()
        del agent.subagent_level
        assert "write_file" not in grantable_tools(agent)

    def test_an_unknown_rung_falls_back_to_explore(self):
        assert "write_file" not in grantable_tools(_agent(level="everything"))

    def test_the_rung_does_not_widen_a_readonly_mode(self):
        """A rung hands something over; it never takes a mode's word back."""
        assert "write_file" not in grantable_tools(_agent(mode="plan", level="parallel"))


class TestSubSessionEnv:
    def test_the_environment_wins_over_the_shared_pointer(self):
        """How a sub-agent gets a session of its own: the pointer file is shared by
        every process of the workspace, the environment belongs to one server."""
        with tempfile.TemporaryDirectory() as base:
            with open(os.path.join(base, "active_session"), "w", encoding="utf-8") as fh:
                fh.write("parent-1")
            assert active_session_id(base) == "parent-1"
            os.environ["MIMIR_SESSION_ID"] = "parent-1/subagents/sub-ab12cd34"
            try:
                assert active_session_id(base) == "parent-1/subagents/sub-ab12cd34"
            finally:
                del os.environ["MIMIR_SESSION_ID"]

    def test_a_named_session_wins_over_the_environment_too(self):
        """Several sub-agents live in one process, so the child's own scratchpad cannot
        be resolved from the environment either — it is named outright."""
        from mimir.servers._shared.state_paths import scratch_dir

        with tempfile.TemporaryDirectory() as base:
            with open(os.path.join(base, "active_session"), "w", encoding="utf-8") as fh:
                fh.write("parent-1")
            mine = scratch_dir(base, "parent-1/subagents/sub-ab12cd34")
            assert mine.endswith(os.path.join("parent-1", "subagents", "sub-ab12cd34"))
            assert scratch_dir(base) != mine

    def test_a_sub_session_does_not_show_up_in_the_session_list(self):
        """It writes a directory, never a <id>.json at the root of sessions/ — which is
        what the front end enumerates."""
        from mimir.client.ui.ws import session_store

        with tempfile.TemporaryDirectory() as base:
            sessions = os.path.join(base, "sessions")
            os.makedirs(os.path.join(sessions, "parent-1", "subagents", "sub-ab12cd34"))
            with open(os.path.join(sessions, "parent-1.json"), "w", encoding="utf-8") as fh:
                fh.write('{"id": "parent-1", "title": "t", "created_at": "", "updated_at": ""}')
            store = session_store.SessionStore()
            original = session_store._sessions_dir
            session_store._sessions_dir = lambda: sessions
            try:
                assert [m.id for m in store.list_sessions()] == ["parent-1"]
            finally:
                session_store._sessions_dir = original
