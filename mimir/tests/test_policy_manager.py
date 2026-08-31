import os
import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

import mimir.client.guardrails.policy.engine as policy_manager_module
import mimir.client.guardrails.policy.gates as gates
from mimir.client.query_engine import toollist
from mimir.tests._golden_caps import build_declared_registry

# Per-agent registry, built once from the server declarations (what connect_server
# produces at runtime), so policy decisions match production classification.
_DECLARED_REGISTRY = build_declared_registry()


class _FakeAgent:
    def __init__(self) -> None:
        self.tool_owner = {"read_file_lines": "search", "replace_in_file": "search"}
        self.tool_caps = dict(_DECLARED_REGISTRY)
        self.approvals = SimpleNamespace(is_sensitive=lambda tool, args: False)
        self.denied_calls: list[tuple[str, dict, dict | None]] = []

    @staticmethod
    def _json_error_payload(message: str, hint: str = "", **extra) -> str:
        return f"error:{message}|{hint}|{extra.get('tool', '')}"

    @staticmethod
    def _normalize_tool_arguments(tool_name: str, arguments: dict) -> dict:
        normalized = dict(arguments)
        normalized["normalized"] = True
        return normalized

    @staticmethod
    def _rewrite_tool_for_context(tool_name: str, arguments: dict) -> tuple[str, dict]:
        rewritten = dict(arguments)
        rewritten["rewritten"] = True
        return tool_name, rewritten

    def _request_tool_approval(self, tool_name: str, arguments: dict) -> tuple[bool, str]:
        return False, "denied by user"

    @staticmethod
    def approval_scope(tool_name: str, arguments: dict) -> str:
        return f"fake:{tool_name}"

    def _record_denied_tool_call(
        self, tool_name: str, arguments: dict, execution_context: dict | None, note: str = "",
    ) -> None:
        self.denied_calls.append((tool_name, arguments, execution_context, note))

    @staticmethod
    def _denied_tool_result(
        tool_name: str, arguments: dict, note: str = "", execution_context: dict | None = None,
    ) -> str:
        return f"denied:{tool_name}:{arguments.get('path', '')}:{note}"

    @staticmethod
    def _is_write_tool(tool_name: str) -> bool:
        return tool_name in {
            "write_file",
            "append_file",
            "replace_in_file",
            "delete_file",
        }

    @staticmethod
    def _normalize_workspace_path(path: str | None) -> str:
        return path or ""

    @staticmethod
    def _is_code_filepath(path: str) -> bool:
        # Simple check for common code extensions
        code_extensions = {".py", ".js", ".ts", ".cpp", ".c", ".h", ".java", ".go", ".rs"}
        return os.path.splitext(path)[1].lower() in code_extensions


class PolicyManagerTests(unittest.TestCase):
    def test_blocked_tools_for_context_returns_empty_without_context(self) -> None:
        blocked = toollist.blocked_tools_for_context(
            "please refactor this function",
            None,
        )
        self.assertEqual(blocked, set())

    def test_blocked_tools_for_context_blocks_create_like_tools_for_edit_intent(self) -> None:
        blocked = toollist.blocked_tools_for_context(
            "please refactor this function",
            {"searched": True},
            _DECLARED_REGISTRY,
        )
        self.assertEqual(blocked, {"write_file", "append_file"})

    def test_tools_for_context_filters_ollama_tool_list(self) -> None:
        tools = [
            {"function": {"name": "write_file"}},
            {"function": {"name": "append_file"}},
            {"function": {"name": "replace_in_file"}},
            {"function": {"name": "read_file_lines"}},
        ]

        filtered = toollist.tools_for_context(
            query="refactor this function",
            execution_context={"searched": True},
            tools=tools,
            tool_caps=_DECLARED_REGISTRY,
        )

        self.assertEqual(
            [tool["function"]["name"] for tool in filtered],
            ["replace_in_file", "read_file_lines"],
        )

    def test_tools_for_context_keeps_all_tools_when_not_blocked(self) -> None:
        tools = [
            {"function": {"name": "write_file"}},
            {"function": {"name": "read_file_lines"}},
        ]

        filtered = toollist.tools_for_context(
            query="show me status",
            execution_context={"searched": True},
            tools=tools,
        )

        self.assertEqual(filtered, tools)

    def test_tools_for_context_visibility_is_independent_of_discovery_state(self) -> None:
        """Edit tools are no longer *hidden* before discovery.

        The per-query tool list is byte-stable regardless of discovery state, so the
        model-prompt prefix stays cacheable across steps (Workstream C). The guard
        against premature writes/edits now lives in the call-time policy
        (check_write_policy), not in tool visibility — so the filtered list is the
        same before and after discovery.
        """
        tools = [
            {"function": {"name": "replace_in_file"}},
            {"function": {"name": "read_file_lines"}},
            {"function": {"name": "list_directory"}},
        ]

        before = toollist.tools_for_context(
            query="update the helper function",
            execution_context={},
            tools=tools,
            tool_caps=_DECLARED_REGISTRY,
        )
        after = toollist.tools_for_context(
            query="update the helper function",
            execution_context={"searched": True, "read_files": {"x.py"}},
            tools=tools,
            tool_caps=_DECLARED_REGISTRY,
        )

        self.assertEqual(before, after)
        self.assertIn("replace_in_file", [t["function"]["name"] for t in before])

    def test_external_fetch_is_not_gated_on_local_discovery(self) -> None:
        """Reaching outside the workspace does not require having looked inside it.

        There was a gate here holding every EXTERNAL_FETCH call until the model had
        searched or read locally. It was removed rather than repaired. It contradicted
        the prompt, which tells the model to skip the survey for work that builds
        something new outside the repository and lets a literature task conclude
        straight from discovery; it ignored two of the five discovery signals, so a
        model that delegated its exploration — which the sub-agent section asks for —
        counted as having done nothing; and a structural snapshot pre-filled the one
        field it did read, so on a repo-touching query it never fired at all. It bit
        only on the bibliography turn it was least entitled to block.
        """
        agent = _FakeAgent()
        agent.tool_owner["github_get_file"] = "github"

        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent,
            tool_name="github_get_file",
            arguments={"owner": "o", "repo": "r", "path": "x"},
            execution_context={},
        )
        self.assertIsNone(result.violation)

    def test_cluster_submit_stays_held_until_something_is_validated(self) -> None:
        """The hold is a precondition, not a counter — retrying alone must not clear it.

        It used to be one-shot: warn once, set a flag, let the next call through. Against
        a model that simply calls again (the normal reaction to an error) that cost one
        round trip and constrained nothing, on the most expensive action in the system —
        a submission that burns real allocation hours and cannot be taken back. The
        condition is a fact about the session, so it holds until the fact changes.
        """
        agent = _FakeAgent()
        agent.tool_owner["salloc_submit"] = "hpc"
        args = {"partition": "compute", "confirm": True}
        # Something was edited and never checked: the state the hold is about.
        ctx: dict = {"dirty_written_files": {"job.py"}}

        def _submit():
            return policy_manager_module.evaluate_tool_preconditions(
                agent=agent, tool_name="salloc_submit", arguments=args, execution_context=ctx,
            ).violation

        first = _submit()
        self.assertIsNotNone(first)
        self.assertIn("held", first.lower())

        second = _submit()
        self.assertIsNotNone(second, "a bare retry must not clear the hold")
        self.assertIn("held", second.lower())

        # The stated exit — one local check that passes — is the only thing that lifts it.
        ctx["validated_files"] = {"job.py"}
        self.assertIsNone(_submit())

    def test_cluster_submit_not_held_when_the_session_wrote_nothing(self) -> None:
        """The stated exit has to be reachable, or the hold is a wall.

        ``validated_files`` is credited only by a checker run against a file the model
        edited. A session whose whole job is to launch something that already exists
        ("resubmit this job on 64 nodes") edits nothing, so it can never satisfy the
        condition however many times it tries. Nothing changed ⇒ nothing ought to have
        been checked; the user's approval prompt, which these irreversible tools always
        raise, is the protection that applies there.
        """
        agent = _FakeAgent()
        agent.tool_owner["salloc_submit"] = "hpc"
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent,
            tool_name="salloc_submit",
            arguments={"partition": "compute", "confirm": True},
            execution_context={},
        )
        self.assertIsNone(result.violation)

    def test_cluster_submit_allowed_when_validated_locally(self) -> None:
        """With local-validation evidence (validated_files), the submit is not held."""
        agent = _FakeAgent()
        agent.tool_owner["proxy_slurm"] = "proxy"
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent,
            tool_name="proxy_slurm",
            arguments={"op": "run", "proxy_name": "p", "confirm": True},
            execution_context={"validated_files": {"solver.py"}},
        )
        self.assertIsNone(result.violation)

    def test_blocked_tools_for_context_has_no_external_fetch_literals(self) -> None:
        """The former hardcoded github-name block is gone: a fresh context no longer
        hides EXTERNAL_FETCH tools (they are gated at call time instead)."""
        blocked = toollist.blocked_tools_for_context(
            "look up the upstream repo",
            {},  # no discovery yet
            _DECLARED_REGISTRY,
        )
        self.assertEqual(blocked & _DECLARED_REGISTRY.keys() & {
            "github_get_file", "github_search_repositories", "github_list_issues",
        }, set())

    def test_unknown_tool_returns_json_error_payload(self) -> None:
        agent = _FakeAgent()
        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent,
            tool_name="missing_tool",
            arguments={"path": "x.py"},
            execution_context=None,
        )

        self.assertEqual(result.tool_name, "missing_tool")
        self.assertIn("Unknown tool 'missing_tool'", result.violation or "")

    def _wire_real_rewrite(self, agent: "_FakeAgent") -> None:
        """Swap the fake's stub rewrite for the production rewrite_tool_for_context
        so the read_file→read_file_lines heal is exercised end to end."""
        from mimir.client.tool_execution.normalizer import rewrite_tool_for_context

        agent._rewrite_tool_for_context = lambda tool_name, arguments: rewrite_tool_for_context(
            tool_name,
            arguments,
            tool_owner=agent.tool_owner,
            is_code_filepath=agent._is_code_filepath,
            normalize_workspace_path_fn=agent._normalize_workspace_path,
        )

    def test_read_file_alias_rewritten_before_registry_check(self) -> None:
        # Regression: `read_file` is intentionally unregistered — it exists only to
        # be healed into the registered `read_file_lines`. The registry check must
        # run AFTER the rewrite; otherwise the alias is rejected ("blocked ·
        # registry") before the heal can happen.
        agent = _FakeAgent()
        self._wire_real_rewrite(agent)

        with patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file",
                arguments={"path": "README.md"},
                execution_context={},
            )

        self.assertIsNone(result.violation)
        self.assertEqual(result.tool_name, "read_file_lines")

    def test_genuinely_unknown_tool_still_blocks_after_reorder(self) -> None:
        # The reorder must not swallow real unknown tools: a name the rewrite leaves
        # untouched still reports an "Unknown tool" registry violation.
        agent = _FakeAgent()
        self._wire_real_rewrite(agent)

        result = policy_manager_module.evaluate_tool_preconditions(
            agent=agent,
            tool_name="frobnicate",
            arguments={"path": "x.py"},
            execution_context={},
        )

        self.assertEqual(result.tool_name, "frobnicate")
        self.assertIn("Unknown tool 'frobnicate'", result.violation or "")

    def test_state_violation_short_circuits_other_checks(self) -> None:
        agent = _FakeAgent()
        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value="state_violation") as state_guard, \
             patch.object(policy_manager_module, "check_write_policy", return_value=None) as write_guard:
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertEqual(result.violation, "state_violation")
        state_guard.assert_called_once()
        write_guard.assert_not_called()

    def test_approval_denial_records_call_and_returns_denied_result(self) -> None:
        agent = _FakeAgent()
        agent.approvals = SimpleNamespace(is_sensitive=lambda tool, args: True)

        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        # The refusal note reaches both the ledger and the tool result — it used to be
        # computed by the front-end and then dropped on the floor.
        self.assertEqual(result.violation, "denied:read_file_lines:x.py:denied by user")
        self.assertEqual(len(agent.denied_calls), 1)
        self.assertEqual(agent.denied_calls[0][3], "denied by user")

    def test_write_policy_violation_short_circuits_approval(self) -> None:
        agent = _FakeAgent()
        approval_probe = {"called": False}
        agent.approvals = SimpleNamespace(is_sensitive=lambda tool, args: approval_probe.__setitem__("called", True))

        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value="write_violation"):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertEqual(result.violation, "write_violation")
        self.assertFalse(approval_probe["called"])

    def test_sensitive_tool_approved_returns_no_violation(self) -> None:
        agent = _FakeAgent()
        agent.approvals = SimpleNamespace(is_sensitive=lambda tool, args: True)
        agent._request_tool_approval = lambda tool_name, arguments: (True, "approved")

        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertIsNone(result.violation)
        self.assertEqual(agent.denied_calls, [])

    def test_scope_at_the_end_of_the_ladder_is_refused_without_asking_again(self) -> None:
        # Being shown the same card a fourth time after saying no three times is the
        # friction the ladder exists to remove: the gate refuses on the user's behalf.
        agent = _FakeAgent()
        agent.approvals = SimpleNamespace(is_sensitive=lambda tool, args: True)
        asked = {"count": 0}

        def _never_called(tool_name, arguments):
            asked["count"] += 1
            return False, "denied by user"

        agent._request_tool_approval = _never_called
        scope = agent.approval_scope("read_file_lines", {})
        context = {"denial_history": [
            {"tool": "read_file_lines", "scope": scope, "kind": "denied", "reason": "denied by user"}
            for _ in range(3)
        ]}

        with patch.object(policy_manager_module, "ensure_execution_context", return_value=context), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context=context,
            )

        self.assertEqual(asked["count"], 0)
        self.assertIn("already refused; not asked again", result.violation)

    def test_an_unrelated_scope_is_still_put_to_the_user(self) -> None:
        # The ladder escalates a goal, not the session: a different action starts fresh.
        agent = _FakeAgent()
        agent.approvals = SimpleNamespace(is_sensitive=lambda tool, args: True)
        agent._request_tool_approval = lambda tool_name, arguments: (True, "approved once")
        context = {"denial_history": [
            {"tool": "replace_in_file", "scope": "fake:replace_in_file",
             "kind": "denied", "reason": "denied by user"}
            for _ in range(3)
        ]}

        with patch.object(policy_manager_module, "ensure_execution_context", return_value=context), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context=context,
            )

        self.assertIsNone(result.violation)

    def test_returns_normalized_inputs_when_allowed(self) -> None:
        agent = _FakeAgent()
        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=None), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertIsNone(result.violation)
        self.assertEqual(result.tool_name, "read_file_lines")
        self.assertTrue(result.arguments.get("normalized"))
        self.assertTrue(result.arguments.get("rewritten"))
        # The engine seeds its own bookkeeping keys on top of the ensured context,
        # so assert the supplied value is preserved rather than exact equality.
        self.assertEqual(result.execution_context.get("ctx"), True)

    def test_json_state_violation_is_enriched_with_policy_metadata(self) -> None:
        agent = _FakeAgent()
        json_violation = json.dumps({
            "status": "error",
            "error": "blocked",
            "hint": "do discovery first",
        })
        execution_context = {
            "workflow_state": "discover",
            "searched": False,
            "inspected_dirs": set(),
            "read_files": set(),
            "search_tool_calls": 0,
        }

        with patch.object(policy_manager_module, "ensure_execution_context", return_value=execution_context), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value=json_violation), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertIsNotNone(result.violation)
        payload = json.loads(result.violation)
        self.assertEqual(payload["policy_stage"], "state_guard")
        self.assertEqual(payload["state"], "discover")
        self.assertEqual(payload["tool"], "read_file_lines")
        self.assertEqual(payload["suggested_next_tool_class"], "discovery")

    def test_non_json_violation_stays_compatible(self) -> None:
        agent = _FakeAgent()
        with patch.object(policy_manager_module, "ensure_execution_context", return_value={"ctx": True}), \
             patch.object(policy_manager_module, "check_state_machine_guard", return_value="state_violation"), \
             patch.object(policy_manager_module, "check_write_policy", return_value=None):
            result = policy_manager_module.evaluate_tool_preconditions(
                agent=agent,
                tool_name="read_file_lines",
                arguments={"path": "x.py"},
                execution_context={},
            )

        self.assertEqual(result.violation, "state_violation")

class MissingEvidenceTests(unittest.TestCase):
    def test_missing_evidence_suggests_read_when_write_blocked(self) -> None:
        from mimir.client.guardrails import policy as policy_module

        # Context simulating a write blocked due to missing read
        context = {
            "read_files": set(),
            "checked_paths": set(),
            "existing_paths": set(),
            "last_replace_file": "src/foo.py",
            "last_replace_old_text": "old_val",
        }
        missing = policy_module.engine._missing_evidence(context)
        self.assertIn("Read at least one concrete file before proposing code changes.", missing)

    def test_missing_evidence_suggests_search_when_discover_and_empty(self) -> None:
        from mimir.client.guardrails import policy as policy_module

        context = {
            "workflow_state": "discover",
            "searched": False,
            "read_files": set(),
        }
        missing = policy_module.engine._missing_evidence(context)
        self.assertIn("Run a targeted local search first.", missing)

    def test_missing_evidence_silent_once_model_has_evidence(self) -> None:
        # Coherent with the discovery gates: once the model has 2+ real signals,
        # the discover-branch stops nagging (was previously per-field).
        from mimir.client.guardrails import policy as policy_module

        context = {
            "workflow_state": "discover",
            "searched": True,
            "read_files": {"src/foo.py"},
        }
        missing = policy_module.engine._missing_evidence(context)
        self.assertNotIn("Run a targeted local search first.", missing)
        self.assertNotIn("Read at least one concrete file before proposing code changes.", missing)

    def test_missing_evidence_empty_when_context_sufficient(self) -> None:
        from mimir.client.guardrails import policy as policy_module

        context = {
            "workflow_state": "edit",
            "read_files": {"src/foo.py"},
        }
        missing = policy_module.engine._missing_evidence(context)
        self.assertEqual(missing, [])


class ValidationNudgeTests(unittest.TestCase):
    def test_validation_nudge_message_includes_line_hint(self) -> None:
        from mimir.client.guardrails.nudges import messages as state_machine

        agent = _FakeAgent()

        context = {
            "pending_validation": set(),
            "workflow_state": "validate",
            "steps_since_last_edit": 2,
            "validation_fail_count_by_file": {"src/foo.py": 3},
            "builtin_check_findings": {"src/foo.py": "line 7: invalid syntax"},
            "last_replace_file": "src/foo.py",
            "last_replace_old_text": "def foo():",
        }
        with patch.object(state_machine, "_validation_nudge_message", return_value=""):
            nudge = state_machine.validation_nudge_message(agent, context)
        self.assertIn("Read the local failing region around the last replacement anchor first", nudge)

    def test_validation_nudge_message_without_line_hint_falls_back(self) -> None:
        from mimir.client.guardrails.nudges import messages as state_machine

        agent = _FakeAgent()

        context = {
            "pending_validation": set(),
            "workflow_state": "validate",
            "steps_since_last_edit": 2,
            "validation_fail_count_by_file": {"src/bar.py": 4},
            "builtin_check_findings": {"src/bar.py": "line 7: invalid syntax"},
            "last_replace_file": "",
            "last_replace_old_text": "",
        }
        with patch.object(state_machine, "_validation_nudge_message", return_value=""):
            nudge = state_machine.validation_nudge_message(agent, context)
        self.assertIn("Prefer a smaller, localized repair", nudge)


if __name__ == "__main__":
    unittest.main()


class PlanShapeGateTests(unittest.TestCase):
    """A plan axis is a change to make, never a step of the exploration.

    PHASE 2 of the plan-mode prompt has always said so; nothing checked it. Observed in
    the wild: a plan whose first axis was "Audit Existing Bindings". The audit then
    reported nothing missing, every axis after it was vacuous, and the run was padded
    with cosmetic edits rather than re-decided.
    """

    def setUp(self) -> None:
        self.agent = _FakeAgent()
        self.agent.tool_caps = dict(_DECLARED_REGISTRY)

    def _check(self, text: str):
        return gates._check_plan_shape(
            self.agent, "todo_set_plan", {"text": text, "title": "t"}, {})

    _FUNTIDES = (
        "## Overview\nBindings exist for gradient, model and solver.\n"
        "## Approach\n"
        "### 1. Audit Existing Bindings\nList every public class.\n"
        "### 2. Extend Bindings for Missing Components\nAdd the declarations.\n"
        "## Validation\n- Run the examples.\n"
    )

    def test_an_exploration_axis_is_refused_and_named(self) -> None:
        violation = self._check(self._FUNTIDES)
        self.assertIsNotNone(violation)
        # The refusal has to be trivially clearable, so it quotes the offending axis.
        self.assertIn("Audit Existing Bindings", violation)
        self.assertNotIn("Extend Bindings", violation)

    def test_a_plan_of_changes_passes(self) -> None:
        self.assertIsNone(self._check(
            "## Approach\n### Extend the solver bindings\nAdd the six methods.\n"
            "### Wire them into the DG module\n## Validation\n- pytest\n"))

    def test_a_gerund_and_a_numbered_list_are_the_same_axis(self) -> None:
        violation = self._check(
            "# Approach\n1. Auditing the mesh module\n2. Rewrite the dispatch table\n")
        self.assertIsNotNone(violation)
        self.assertIn("Auditing the mesh module", violation)

    def test_a_sub_step_of_an_axis_is_not_an_axis(self) -> None:
        # Observed in the wild: a plan refused three times over the numbered steps
        # *inside* its axes ("Add a conditional block that: 1. Check the flag"), which
        # are how a change is spelled out. The model cleared the gate by deleting them,
        # so the guard bought a vaguer plan than the one it turned down. Only top-level
        # items are axes; the indentation is the distinction.
        self.assertIsNone(self._check(
            "## Approach\n"
            "### Extend the operator with the forward skip\n"
            "- **What**: add a conditional block that:\n"
            "  1. Check the new flag\n"
            "  2. Compute the seabed depth\n"
            "- **Where**: `operator_tw_init_time`\n"
            "## Validation\n- pytest\n"))

    def test_the_prescribed_validation_section_never_fires(self) -> None:
        # "## Validation" is structure the tool's own docstring asks for, and the axes
        # under Approach are the only thing read.
        self.assertIsNone(self._check(
            "## Approach\n### Add the missing methods\n"
            "## Validation\n### Review the diff\n### Verify the build\n"))

    def test_an_unverified_assumption_in_prose_is_not_an_axis(self) -> None:
        # PLAN_EXPLORE_BUDGET_SPENT explicitly asks for this sentence. Only axis titles
        # are read, so the two instructions cannot collide.
        self.assertIsNone(self._check(
            "## Approach\n### Extend the solver bindings\n"
            "I could not verify whether orders 4-9 are reachable; reviewing the "
            "builders would settle it. Identifying that gap is left open.\n"))

    def test_the_checklist_tool_is_never_inspected(self) -> None:
        # todo_write carries plan_steps, not plan_document: "validate the solver" is a
        # legitimate implementation step there.
        self.assertIsNone(gates._check_plan_shape(
            self.agent, "todo_write", {"steps": ["Audit the bindings"]}, {}))

    def test_a_non_planning_tool_is_never_inspected(self) -> None:
        self.assertIsNone(gates._check_plan_shape(
            self.agent, "read_file_lines", {"text": self._FUNTIDES}, {}))

    def test_an_empty_plan_is_left_to_the_other_gates(self) -> None:
        self.assertIsNone(self._check("   "))
