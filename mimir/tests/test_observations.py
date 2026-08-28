"""Regression guard for the load-bearing observer dispatch in record_tool_observation.

The ``_observe_*`` handlers run in a fixed order (see observations.py) and each one
decides for itself whether the call is its business. That ordering is the real
fragility of the observation layer — these tests pin it so a future reorder/insert
can't silently break it.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import json
import os
import types
import unittest
from contextlib import ExitStack
from unittest.mock import patch

import mimir.client.guardrails.observations as runtime
from mimir.client.context.execution_context import build_execution_context, unsettled_runs


# The authoritative dispatch order, mirroring record_tool_observation's body.
EXPECTED_ORDER = [
    "_observe_edit_outcome",
    "_observe_todo_flags",
    "_observe_replacement_tracking",
    "_observe_validation_tool",
    "_observe_missing_module",
    "_observe_env_probe",
    "_observe_env_mutation",
    "_observe_denial_clearing",
    "_observe_edit_loop_clear",
    "_observe_delete",
    "_observe_search_flags",
    "_observe_discover_transition",
    "_observe_candidates",
    "_observe_dir_inspect",
    "_observe_existence_check",
    "_observe_read",
    "_observe_delegated_exploration",
    "_observe_declared_edit_set",
    "_observe_bash_validation",
    "_observe_command",
    "_observe_action_op",
    "_observe_tool_run",
    "_observe_verdict_tool",
]


def _stub_agent():
    return types.SimpleNamespace(
        _parse_tool_payload=lambda result: {},
        _normalize_workspace_path=lambda p: p or "",
    )


def _record_order() -> list[str]:
    """Patch every _observe_* with a recorder, run the dispatcher, return call order."""
    order: list[str] = []

    def _make_recorder(name: str):
        def _rec(*_a, **_k):
            order.append(name)
        return _rec

    with ExitStack() as stack:
        for name in EXPECTED_ORDER:
            stack.enter_context(patch.object(runtime, name, _make_recorder(name)))
        runtime.record_tool_observation(
            _stub_agent(), "any_tool", {}, "{}", build_execution_context(),
        )
    return order


class ObserverDispatchOrderTests(unittest.TestCase):
    def test_dispatch_runs_all_handlers_in_order(self) -> None:
        # No handler claims the call: every one runs, in the declared order.
        self.assertEqual(_record_order(), EXPECTED_ORDER)


class ActionOpCountTests(unittest.TestCase):
    """action_op_count increments once per successful PLAN_BLOCKED (write/exec) call.

    The counter keys off the PLAN_BLOCKED capability, never a tool name, so it is
    server-agnostic. The stub registry below uses synthetic names to make that
    generality explicit: one tool that carries the flag, one that does not.
    """

    def _agent(self):
        from mimir.client.context.capabilities import PLAN_BLOCKED, ToolCaps
        return types.SimpleNamespace(
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: p or "",
            _is_code_filepath=lambda p: False,
            tool_caps={
                "action_tool": ToolCaps(name="action_tool", capabilities=frozenset({PLAN_BLOCKED})),
                "readonly_tool": ToolCaps(name="readonly_tool"),
            },
        )

    def test_plan_blocked_success_increments(self) -> None:
        agent = self._agent()
        ec = build_execution_context()
        runtime.record_tool_observation(agent, "action_tool", {}, '{"status": "ok"}', ec)
        runtime.record_tool_observation(agent, "action_tool", {}, '{"status": "ok"}', ec)
        self.assertEqual(ec["action_op_count"], 2)

    def test_failed_call_and_reads_do_not_increment(self) -> None:
        agent = self._agent()
        ec = build_execution_context()
        runtime.record_tool_observation(agent, "action_tool", {}, '{"status": "error"}', ec)
        runtime.record_tool_observation(agent, "readonly_tool", {}, '{"status": "ok"}', ec)
        self.assertEqual(ec["action_op_count"], 0)


class DelegatedExplorationTests(unittest.TestCase):
    """A sub-agent's reading is the caller's evidence — but not its pre-edit warrant."""

    def _agent(self):
        from mimir.client.context.capabilities import DELEGATE, ToolCaps
        return types.SimpleNamespace(
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: p or "",
            _is_code_filepath=lambda p: False,
            tool_caps={
                "delegating_tool": ToolCaps(
                    name="delegating_tool", capabilities=frozenset({DELEGATE})),
                "plain_tool": ToolCaps(name="plain_tool"),
            },
        )

    _OK = '{"status": "ok", "files_read": ["/w/a.py", "/w/b.py"]}'

    def test_reported_reads_become_discovery_evidence(self) -> None:
        ec = build_execution_context()
        runtime.record_tool_observation(self._agent(), "delegating_tool", {}, self._OK, ec)
        self.assertEqual(ec["delegated_read_files"], {"/w/a.py", "/w/b.py"})
        self.assertEqual(ec["existing_paths"], {"/w/a.py", "/w/b.py"})

    def test_they_never_land_in_read_files(self) -> None:
        """`read_files` also answers "does THIS agent hold the lines it is about to
        edit". Only a read of its own settles that."""
        ec = build_execution_context()
        runtime.record_tool_observation(self._agent(), "delegating_tool", {}, self._OK, ec)
        self.assertEqual(ec["read_files"], set())

    def test_a_failed_delegation_credits_nothing(self) -> None:
        ec = build_execution_context()
        runtime.record_tool_observation(
            self._agent(), "delegating_tool", {},
            '{"status": "error", "files_read": ["/w/a.py"]}', ec)
        self.assertEqual(ec["delegated_read_files"], set())

    def test_a_tool_without_the_capability_is_ignored(self) -> None:
        ec = build_execution_context()
        runtime.record_tool_observation(self._agent(), "plain_tool", {}, self._OK, ec)
        self.assertEqual(ec["delegated_read_files"], set())

    def test_a_delegated_sweep_can_carry_the_plan_evidence_bar(self) -> None:
        """Otherwise plan mode punishes the fan-out it asks for: the explore phase
        would never end and the turn budget would drain."""
        from mimir.client.context.execution_context import plan_evidence_ready
        ec = build_execution_context()
        agent = self._agent()
        runtime.record_tool_observation(agent, "delegating_tool", {}, self._OK, ec)
        self.assertFalse(plan_evidence_ready(ec))  # one signal kind is not breadth
        ec["searched"] = True
        self.assertTrue(plan_evidence_ready(ec))


class EditFailureStreakTests(unittest.TestCase):
    """Per-file edit-failure escalation must count failures regardless of the patch.

    edit_loop_state stays signature-based (guards only the identical-patch spin in
    check_write_policy). edit_fail_streak_by_file counts every consecutive failure on
    a file, so a model that keeps trying *different* wrong anchors still gets the
    error_recovery escalation. The file is NOT evicted from the read sets: a wrong
    anchor is not missing content, and the failure already carries the current text.
    """

    def _agent(self, extra_caps=None):
        from mimir.client.context.capabilities import EDIT, ToolCaps
        caps = {
            "replace_in_file": ToolCaps(
                name="replace_in_file",
                capabilities=frozenset({EDIT}),
                arg_roles={"edit_sig": ("old_text", "new_text")},
            ),
        }
        if extra_caps:
            caps.update(extra_caps)
        return types.SimpleNamespace(
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: p or "",
            _is_write_tool=lambda n: n == "replace_in_file",
            _is_code_filepath=lambda p: str(p).endswith(".py"),
            tool_caps=caps,
        )

    def _fail(self, agent, ec, old, new, path="mod.py"):
        runtime.record_tool_observation(
            agent, "replace_in_file",
            {"path": path, "old_text": old, "new_text": new},
            '{"status": "error", "error": "Target text was not found."}', ec,
        )

    def test_varied_anchor_failures_escalate_without_evicting_the_file(self) -> None:
        from mimir.client.guardrails.nudges import engine as nudge_logic
        agent = self._agent()
        ec = build_execution_context()
        ec["read_files"].add("mod.py")

        self._fail(agent, ec, "anchor A", "A2")   # failure 1
        self.assertEqual(ec["edit_fail_streak_by_file"]["mod.py"], 1)
        self.assertIn("mod.py", ec["read_files"])

        self._fail(agent, ec, "anchor B", "B2")   # failure 2 — DIFFERENT anchor
        self.assertEqual(ec["edit_fail_streak_by_file"]["mod.py"], 2)
        # The escalation is the nudge, not a forged "you have not read this".
        self.assertIn("mod.py", ec["read_files"])
        self.assertEqual(nudge_logic._first_failing_edit_path(ec), "mod.py")

    def test_success_resets_streak(self) -> None:
        agent = self._agent()
        ec = build_execution_context()
        self._fail(agent, ec, "anchor A", "A2")
        self.assertEqual(ec["edit_fail_streak_by_file"].get("mod.py"), 1)
        runtime.record_tool_observation(
            agent, "replace_in_file",
            {"path": "mod.py", "old_text": "anchor A", "new_text": "A2"},
            '{"status": "ok"}', ec,
        )
        self.assertNotIn("mod.py", ec["edit_fail_streak_by_file"])

    def test_identical_hard_block_count_still_patch_specific(self) -> None:
        # edit_loop_state (consumed by check_write_policy) must keep resetting on a new
        # patch, so a corrected different attempt is never pre-blocked — while the
        # per-file streak keeps climbing across the different patches.
        agent = self._agent()
        ec = build_execution_context()
        self._fail(agent, ec, "anchor A", "A2")
        self._fail(agent, ec, "anchor A", "A2")   # identical patch
        self.assertEqual(ec["edit_loop_state"]["mod.py"][1], 2)
        self._fail(agent, ec, "anchor B", "B2")   # different patch → identical count resets
        self.assertEqual(ec["edit_loop_state"]["mod.py"][1], 1)
        self.assertEqual(ec["edit_fail_streak_by_file"]["mod.py"], 3)

    def test_reread_resets_streak(self) -> None:
        from mimir.client.context.capabilities import READ, ToolCaps
        agent = self._agent(extra_caps={
            "read_file": ToolCaps(name="read_file", capabilities=frozenset({READ})),
        })
        ec = build_execution_context()
        self._fail(agent, ec, "anchor A", "A2")
        self.assertEqual(ec["edit_fail_streak_by_file"]["mod.py"], 1)
        runtime.record_tool_observation(
            agent, "read_file", {"path": "mod.py"}, '{"status": "ok", "content": "x"}', ec,
        )
        self.assertNotIn("mod.py", ec["edit_fail_streak_by_file"])

    def test_resolving_streak_rearms_error_recovery_budget(self) -> None:
        # The error_recovery reminder is capped per query; once the streak that spent
        # the budget is resolved and no file is failing, the budget is re-armed so a
        # later, distinct spate of failures earns fresh reminders.
        agent = self._agent()
        ec = build_execution_context()
        ec["nudge_counts"]["error_recovery"] = 2  # budget spent
        self._fail(agent, ec, "anchor A", "A2")
        runtime.record_tool_observation(
            agent, "replace_in_file",
            {"path": "mod.py", "old_text": "anchor A", "new_text": "A2"},
            '{"status": "ok"}', ec,
        )
        self.assertEqual(ec["nudge_counts"]["error_recovery"], 0)

    def test_budget_stays_spent_while_another_file_still_failing(self) -> None:
        # Clearing one file's streak must NOT re-arm the budget while a different file
        # is still stuck — otherwise the stuck file would get unlimited reminders.
        agent = self._agent()
        ec = build_execution_context()
        ec["nudge_counts"]["error_recovery"] = 2
        self._fail(agent, ec, "anchor A", "A2", path="a.py")
        self._fail(agent, ec, "anchor B", "B2", path="b.py")
        runtime.record_tool_observation(
            agent, "replace_in_file",
            {"path": "a.py", "old_text": "anchor A", "new_text": "A2"},
            '{"status": "ok"}', ec,
        )
        self.assertEqual(ec["nudge_counts"]["error_recovery"], 2)  # b.py still failing
        self.assertIn("b.py", ec["edit_fail_streak_by_file"])


class BashCommandObservationTests(unittest.TestCase):
    """A shell-command tool feeds the blackboard by classified capability.

    ``_observe_command`` classifies the ``command`` string (registry-driven off the
    ``command_prefix`` scope kind) and credits the same execution_context fields the
    dedicated file/search tools would — so a bash ``cat``/``grep``/``sed -i`` is no
    longer invisible to the policy/nudge layer.
    """

    def _agent(self):
        from mimir.client.context.capabilities import ToolCaps
        return types.SimpleNamespace(
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: p or "",
            _is_code_filepath=lambda p: str(p).endswith(".py"),
            tool_caps={
                "bash_run": ToolCaps(
                    name="bash_run",
                    scope={"kind": "command_prefix", "args": ["command"]},
                ),
            },
        )

    def _run(self, command, status="ok", stdout="out"):
        agent = self._agent()
        ec = build_execution_context()
        payload = json.dumps({"status": status, "stdout": stdout})
        runtime.record_tool_observation(agent, "bash_run", {"command": command}, payload, ec)
        return ec

    def test_read_command_populates_read_files(self):
        ec = self._run("cat a.py")
        self.assertIn("a.py", ec["read_files"])
        self.assertIn("a.py", ec["checked_paths"])
        self.assertIn("a.py", ec["existing_paths"])

    def test_search_command_sets_searched(self):
        ec = self._run("grep foo bar.py")
        self.assertTrue(ec["searched"])
        self.assertEqual(ec["search_tool_calls"], 1)

    def test_inspect_command_populates_inspected_dirs(self):
        ec = self._run("ls src")
        self.assertIn("src", ec["inspected_dirs"])

    def test_inplace_write_marks_file_dirty(self):
        ec = self._run("sed -i s/a/b/ mod.py")
        self.assertIn("mod.py", ec["dirty_written_files"])
        self.assertIn("mod.py", ec["planned_edit_targets"])
        self.assertEqual(ec["action_op_count"], 1)

    def test_exec_command_increments_action_op(self):
        ec = self._run("python x.py")
        self.assertEqual(ec["action_op_count"], 1)

    def test_pipe_reads_and_searches(self):
        ec = self._run("cat a.py | grep foo")
        self.assertIn("a.py", ec["read_files"])
        self.assertTrue(ec["searched"])

    def test_module_load_records_env_mutation(self):
        ec = self._run("module load cuda")
        self.assertTrue(any(r["installed"] == ["cuda"] for r in ec.get("env_mutations", [])))

    def test_module_avail_sets_env_probed(self):
        ec = self._run("module avail")
        self.assertTrue(ec.get("env_probed"))

    def test_failed_call_credits_nothing(self):
        ec = self._run("cat a.py", status="error")
        self.assertNotIn("a.py", ec["read_files"])

    def test_opaque_command_is_noop(self):
        ec = self._run("cat $(which python)")  # substitution → unclassifiable
        self.assertEqual(len(ec["read_files"]), 0)

    def test_read_search_leaves_discover_state(self):
        ec = self._run("grep foo bar.py")
        self.assertEqual(ec["workflow_state"], "edit")


class BashValidationObservationTests(unittest.TestCase):
    """Two axes from one bash command, and they never mix.

    A **checker** (`py_compile`, `ruff`, `mypy`, a compiler) validates the files it
    names: its output is a list of problems, and an empty one is the finding. An
    **execution** (`pytest`, `python solver.py`, `./solver`) validates nothing at all —
    it is recorded as a run, which owes a reading of what it printed. `cd` in the chain
    rebases relative operands so the resolved path matches the (root-relative) dirty
    path exactly, which also means two same-named files in different directories are
    never confused.
    """

    def setUp(self):
        # These cases are about the check/run split, never about a missing toolchain, so
        # whether this host happens to have `make` or `gfortran` must not decide them.
        # The missing-command imputation has its own class below.
        patcher = patch.object(runtime, "_any_command_on_path", lambda cmds: True)
        patcher.start()
        self.addCleanup(patcher.stop)

    def _agent(self):
        from mimir.client.context.capabilities import ToolCaps
        reg = {"bash_run": ToolCaps(
            name="bash_run", scope={"kind": "command_prefix", "args": ["command"]},
        )}
        return types.SimpleNamespace(
            tool_caps=reg,
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: os.path.normpath(p) if p else "",
            _is_code_filepath=lambda p: str(p).endswith(".py"),
        )

    def _ctx(self, dirty):
        ec = build_execution_context()
        ec["dirty_written_files"] = set(dirty)
        ec["code_mutation_started"] = True
        return ec

    def _run(self, agent, command, ec, status="ok", stdout="x"):
        payload = {"status": status, "stdout": stdout}
        runtime.record_tool_observation(
            agent, "bash_run", {"command": command}, json.dumps(payload), ec,
        )

    def _judge(self, ec, verdict="pass", reason="the printed residual is below tolerance"):
        """What the model says the run's output showed — exit 0 never speaks for it."""
        from mimir.client.guardrails.verdict import apply_verdict
        return bool(apply_verdict(verdict, reason, "", ec))

    # ── executions: a run, never a validation ──────────────────────────────

    def test_a_green_execution_validates_nothing(self):
        # The wave2d case: the suite ends cleanly and that says the program reached its
        # end. Whether solver.py parses is a checker's answer, and nobody ran one.
        agent, ec = self._agent(), self._ctx({"pkg/mod.py"})
        self._run(agent, "pytest -q pkg/mod.py", ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertTrue(ec["runs"]["pytest -q pkg/mod.py"]["completed"])

    def test_a_passing_verdict_validates_nothing_either(self):
        agent, ec = self._agent(), self._ctx({"pkg/mod.py"})
        self._run(agent, "pytest -q pkg/mod.py", ec)
        self._judge(ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(ec["runs"]["pytest -q pkg/mod.py"]["verdict"], "pass")

    def test_a_failing_verdict_charges_the_run_not_the_file(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec)
        self._judge(ec, "fail", "0.00% amplitude reduction against a >90% criterion")
        self.assertEqual(ec["validation_fail_count_by_file"], {})
        self.assertEqual(ec["runs"]["python foo.py"]["failures"], 1)
        self.assertEqual(
            ec["runs"]["python foo.py"]["attempts"],
            ["0.00% amplitude reduction against a >90% criterion"],
        )
        self.assertEqual(ec["workflow_state"], "edit")

    def test_unknown_verdict_leaves_the_run_outstanding(self):
        # "I cannot tell" is a state somebody has to be told about at the end: the run
        # keeps its place in the ledger, carrying the verdict that did not close it.
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec)
        self._judge(ec, "unknown", "no reference solution available for this regime")
        self.assertEqual(ec["runs"]["python foo.py"]["verdict"], "unknown")

    def test_a_red_execution_owes_no_verdict(self):
        # Its non-zero exit is the finding; it goes straight onto the repair ladder.
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec, status="error")
        self.assertFalse(ec["runs"]["python foo.py"]["completed"])
        self.assertEqual(ec["runs"]["python foo.py"]["failures"], 1)
        self.assertEqual(ec["workflow_state"], "edit")

    def test_a_run_that_declares_its_own_failure_is_treated_as_red(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec, stdout="reflection is high\ncheck=fail")
        run = ec["runs"]["python foo.py"]
        self.assertFalse(run["completed"])
        self.assertEqual(run["attempts"], [runtime._SELF_DECLARED_FAILURE])

    def test_a_bare_suite_validates_nothing(self):
        # `pytest` exercises the tree, but exercising is not checking: it tells us the
        # programs ran, which is exactly what the verdict axis is for.
        agent, ec = self._agent(), self._ctx({"a.py", "pkg/b.py"})
        self._run(agent, "pytest -q", ec)
        self._judge(ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(list(ec["runs"]), ["pytest -q"])

    # ── environment setup: preparation, never evidence ─────────────────────

    def test_sourcing_an_env_script_is_not_a_run(self):
        # The Diva case: `source set_env_cmake.sh` was recorded as a run owing a
        # verdict, purely because `source` was absent from the validator table. It
        # executes, but it exercises nothing of the project.
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "source set_env_cmake.sh", ec)
        self.assertEqual(ec["runs"], {})
        self.assertEqual(ec["validated_files"], set())

    def test_env_setup_around_a_chdir_and_a_probe_is_not_a_run(self):
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "cd sub && source env.sh && echo done", ec)
        self.assertEqual(ec["runs"], {})

    # ── builds: judged by the machine, like the compilers they drive ───────

    def test_a_green_build_is_settled_by_its_exit_code(self):
        # A build reports diagnostics, not results, so there is no output for the
        # model to read — it must never surface as "ran but never judged".
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "make -f makefile.cmake", ec)
        self.assertEqual(ec["runs"]["make -f makefile.cmake"]["verdict"], "pass")
        self.assertEqual(unsettled_runs(ec), {})

    def test_a_build_that_only_prepares_its_environment_first_still_counts(self):
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "source env.sh && make", ec, status="error")
        run = ec["runs"]["source env.sh && make"]
        self.assertFalse(run["completed"])
        self.assertEqual(run["effect"], "build")

    def test_a_run_in_the_chain_outranks_the_build_before_it(self):
        # `make && ./solver` produced output somebody must judge; the build's own exit
        # code stopped being the last word the moment something ran.
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "make && ./solver", ec)
        run = ec["runs"]["make && ./solver"]
        self.assertEqual(run["effect"], "run")
        self.assertEqual(run["verdict"], "")

    def test_a_green_build_credits_no_file(self):
        # Which source files a build compiled is not recorded anywhere, so crediting
        # one would be a guess — and the check axis is the one that must stay honest.
        agent, ec = self._agent(), self._ctx({"solver.f90"})
        self._run(agent, "make", ec)
        self.assertEqual(ec["validated_files"], set())

    def test_an_inline_snippet_is_a_run(self):
        # Unclassifiable (parentheses defeat the tokenizer), so it names nothing — but
        # it ran, and the base prompt asks for exactly this idiom.
        agent, ec = self._agent(), self._ctx({"a.py", "b.py"})
        self._run(agent, "python -c 'print(check(a))'", ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertIn("python -c 'print(check(a))'", ec["runs"])

    def test_a_scratchpad_probe_is_a_run_like_any_other(self):
        agent, ec = self._agent(), self._ctx({"solver.py", "mesh.py"})
        self._run(agent, "python /tmp/mimir-scratch/probe.py", ec)
        self._judge(ec)
        self.assertEqual(ec["validated_files"], set())
        self.assertEqual(
            ec["runs"]["python /tmp/mimir-scratch/probe.py"]["verdict"], "pass",
        )

    def test_a_module_exercised_through_an_entry_point_still_owes_a_verdict(self):
        # The command names main.py, which was never edited; mesh.py was. The run is
        # recorded and judged all the same — which file it exercised is not asked.
        agent, ec = self._agent(), self._ctx({"mesh.py"})
        self._run(agent, "python main.py", ec)
        self.assertIn("python main.py", ec["runs"])
        self._judge(ec)
        self.assertEqual(ec["runs"]["python main.py"]["verdict"], "pass")

    def test_a_run_that_cannot_be_fixed_releases_the_workflow(self):
        from mimir.client.config.constants import VALIDATION_RETRY_BUDGET
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec)
        for _ in range(VALIDATION_RETRY_BUDGET):
            self._run(agent, "python foo.py", ec, status="error")
        self.assertEqual(ec["workflow_state"], "conclude")

    # ── checkers: a file, never a run ──────────────────────────────────────

    def test_py_compile_marks_validated(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python -m py_compile foo.py", ec)
        self.assertIn("foo.py", ec["validated_files"])
        self.assertFalse(ec["runs"])

    def test_cd_rebases_relative_operand(self):
        agent, ec = self._agent(), self._ctx({"pkg/mod.py"})
        self._run(agent, "cd pkg && ruff check mod.py", ec)
        self.assertIn("pkg/mod.py", ec["validated_files"])

    def test_same_name_other_dir_is_not_credited(self):
        agent, ec = self._agent(), self._ctx({"a/mod.py"})
        self._run(agent, "cd b && ruff check mod.py", ec)
        self.assertFalse(ec["validated_files"])

    def test_conclude_when_last_pending_file_validated(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec)
        self.assertEqual(ec["workflow_state"], "conclude")

    def test_failed_check_increments_fail_count_and_returns_to_edit(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec, status="error")
        self.assertEqual(ec["validation_fail_count_by_file"].get("foo.py"), 1)
        self.assertNotIn("foo.py", ec["validated_files"])
        self.assertEqual(ec["workflow_state"], "edit")

    def test_exhausted_retry_budget_escapes_to_conclude(self):
        from mimir.client.config.constants import VALIDATION_RETRY_BUDGET
        agent, ec = self._agent(), self._ctx({"foo.py"})
        for _ in range(VALIDATION_RETRY_BUDGET):
            self._run(agent, "ruff check foo.py", ec, status="error")
        self.assertGreaterEqual(
            ec["validation_fail_count_by_file"]["foo.py"], VALIDATION_RETRY_BUDGET,
        )
        self.assertEqual(ec["workflow_state"], "conclude")

    def test_test_file_recorded_in_tests_run_even_on_failure(self):
        agent, ec = self._agent(), self._ctx({"test_foo.py"})
        self._run(agent, "pytest -q test_foo.py", ec, status="error")
        self.assertIn("test_foo.py", ec["tests_run"])

    def test_multiple_named_files_all_validated(self):
        agent, ec = self._agent(), self._ctx({"a.py", "b.py", "c.py"})
        self._run(agent, "ruff check a.py b.py c.py", ec)
        self.assertEqual(ec["validated_files"], {"a.py", "b.py", "c.py"})

    def test_whole_project_checker_clears_all_pending(self):
        agent, ec = self._agent(), self._ctx({"src/a.py", "src/b.py"})
        self._run(agent, "ruff check .", ec)
        self.assertEqual(ec["validated_files"], {"src/a.py", "src/b.py"})

    def test_c_source_compile_marks_it_validated(self):
        # Language-agnostic per-file: naming a C file in a compile command validates it.
        agent, ec = self._agent(), self._ctx({"solver.c"})
        self._run(agent, "gcc solver.c -O2 -o solver.out", ec)
        self.assertIn("solver.c", ec["validated_files"])

    def test_a_build_is_credited_as_such_and_owes_no_verdict(self):
        # A build sits on its own tier: it needs a toolchain, so nothing demands it,
        # and its exit code is the finding — nothing executed, nothing to judge.
        agent, ec = self._agent(), self._ctx({"solver.c"})
        self._run(agent, "gcc -c solver.c -o solver.o", ec)
        self.assertEqual(ec["validation_tier_by_file"]["solver.c"], "compiled")
        self.assertFalse(ec["runs"])

    def test_a_syntax_only_compile_is_the_cheap_check(self):
        # The mandatory floor for a compiled language: it parses the translation unit
        # and emits nothing, so it asks no more of the environment than a linter does.
        agent, ec = self._agent(), self._ctx({"model.f90"})
        self._run(agent, "gfortran -fsyntax-only model.f90", ec)
        self.assertEqual(ec["validation_tier_by_file"]["model.f90"], "syntax")

    def test_running_the_built_binary_is_a_run(self):
        agent, ec = self._agent(), self._ctx({"solver.c"})
        self._run(agent, "gcc -c solver.c -o solver.o", ec)
        self._run(agent, "./solver.out", ec)
        self.assertIn("./solver.out", ec["runs"])

    def test_fortran_compile_failure_counts(self):
        agent, ec = self._agent(), self._ctx({"model.f90"})
        self._run(agent, "gfortran model.f90 -o model.out", ec, status="error")
        self.assertEqual(ec["validation_fail_count_by_file"].get("model.f90"), 1)

    def test_a_chain_checks_and_runs_at_once(self):
        # `py_compile a.py && python a.py`: the checker validates the file, the run is
        # registered on its own. One command, two answers, neither standing in for the
        # other.
        agent, ec = self._agent(), self._ctx({"a.py"})
        self._run(agent, "python -m py_compile a.py && python a.py", ec)
        self.assertIn("a.py", ec["validated_files"])
        self.assertIn("python -m py_compile a.py && python a.py", ec["runs"])


class PluginValidatorObservationTests(unittest.TestCase):
    """A dedicated VALIDATE tool (e.g. from an extension pack) marks its file validated.

    The VALIDATE capability stays declarable even though the first-party stack validates
    via bash — a plugin validator makes the same conclude-gate contribution.
    """

    def _agent(self):
        from mimir.client.context.capabilities import ToolCaps, VALIDATE
        reg = {"pack_validate": ToolCaps(
            name="pack_validate",
            capabilities=frozenset({VALIDATE}),
            arg_roles={"path": ("filepath",)},
        )}
        return types.SimpleNamespace(
            tool_caps=reg,
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: os.path.normpath(p) if p else "",
        )

    def _ctx(self, dirty):
        ec = build_execution_context()
        ec["dirty_written_files"] = set(dirty)
        ec["code_mutation_started"] = True
        return ec

    def test_successful_validate_tool_marks_file_validated(self):
        agent, ec = self._agent(), self._ctx({"pkg/mod.py"})
        runtime.record_tool_observation(
            agent, "pack_validate", {"filepath": "pkg/mod.py"}, '{"status": "ok"}', ec,
        )
        self.assertIn("pkg/mod.py", ec["validated_files"])
        self.assertEqual(ec["workflow_state"], "conclude")

    def test_failing_validate_tool_leaves_file_pending(self):
        agent, ec = self._agent(), self._ctx({"pkg/mod.py"})
        runtime.record_tool_observation(
            agent, "pack_validate", {"filepath": "pkg/mod.py"}, '{"status": "error"}', ec,
        )
        self.assertNotIn("pkg/mod.py", ec["validated_files"])


class ExecutionToolObservationTests(unittest.TestCase):
    """An execution tool that is not bash owes a verdict just the same.

    Split by *surface*, not by purpose: a shell tool's calls differ in kind call by call
    (`cat` reads, `python` executes), so only the command text can decide and
    `_observe_bash_validation` owns it. A tool like `proxy_exec` has no such variation —
    the call *is* the execution — so the declared capability decides. The two must stay
    mutually exclusive, or a bash run gets registered twice.
    """

    def _agent(self, caps, **kwargs):
        from mimir.client.context.capabilities import ToolCaps
        reg = {"runner": ToolCaps(name="runner", capabilities=frozenset(caps), **kwargs)}
        return types.SimpleNamespace(
            tool_caps=reg,
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: os.path.normpath(p) if p else "",
            _is_code_filepath=lambda p: str(p).endswith(".py"),
        )

    def _ctx(self, dirty=()):
        ec = build_execution_context()
        ec["dirty_written_files"] = set(dirty)
        ec["code_mutation_started"] = bool(dirty)
        return ec

    def _call(self, agent, ec, **args):
        runtime.record_tool_observation(
            agent, "runner", args, '{"status": "ok"}', ec,
        )

    def test_a_code_exec_tool_leaves_its_run_awaiting_a_verdict(self):
        from mimir.client.context.capabilities import CODE_EXEC
        agent, ec = self._agent({CODE_EXEC}), self._ctx({"solver.py"})
        self._call(agent, ec)
        self.assertTrue(ec["runs"]["runner"]["completed"])
        self.assertEqual(ec["runs"]["runner"]["verdict"], "")
        self.assertFalse(ec["validated_files"])

    def test_a_declared_path_arg_validates_nothing(self):
        # Naming the file it ran does not make the call a check of that file.
        from mimir.client.context.capabilities import CODE_EXEC
        agent = self._agent({CODE_EXEC}, arg_roles={"path": ("target",)})
        ec = self._ctx({"a.py", "b.py"})
        self._call(agent, ec, target="b.py")
        self.assertEqual(list(ec["runs"]), ["runner"])
        self.assertFalse(ec["validated_files"])

    def test_with_nothing_written_the_run_is_recorded_all_the_same(self):
        # An analysis-only session: no file was touched, but the answer rests entirely
        # on that output.
        from mimir.client.context.capabilities import CODE_EXEC
        agent, ec = self._agent({CODE_EXEC}), self._ctx()
        self._call(agent, ec)
        self.assertIn("runner", ec["runs"])

    def test_a_writer_owes_nothing(self):
        from mimir.client.context.capabilities import EDIT
        agent, ec = self._agent({EDIT}), self._ctx({"solver.py"})
        self._call(agent, ec, path="solver.py")
        self.assertFalse(ec["runs"])

    def test_a_shell_tool_is_left_to_the_bash_observer(self):
        # Mutual exclusion: otherwise the same run is registered twice, once per handler.
        from mimir.client.context.capabilities import CODE_EXEC
        agent = self._agent(
            {CODE_EXEC}, scope={"args": ["command"], "kind": "command_prefix"},
        )
        ec = self._ctx({"solver.py"})
        self._call(agent, ec, command="python solver.py")
        self.assertEqual(list(ec["runs"]), ["python solver.py"])

    def test_a_failed_call_owes_nothing(self):
        from mimir.client.context.capabilities import CODE_EXEC
        agent, ec = self._agent({CODE_EXEC}), self._ctx({"solver.py"})
        runtime.record_tool_observation(
            agent, "runner", {}, '{"status": "error"}', ec,
        )
        self.assertFalse(ec["runs"])



class ValidationTierTests(unittest.TestCase):
    """How much a green check actually proved.

    ``validated_files`` answers "was this file checked?"; ``validation_tier_by_file``
    answers "with what?". Both are about checkers only. Running the code is not on this
    ladder at all — a vacuous test and a rigorous one produce the same exit code, so
    "it ran" is recorded as a run and read by the model, never as file evidence.
    """

    _agent = BashValidationObservationTests._agent
    _ctx = BashValidationObservationTests._ctx
    _run = BashValidationObservationTests._run
    _judge = BashValidationObservationTests._judge

    def tier(self, ec, path="foo.py"):
        return ec["validation_tier_by_file"].get(path)

    def test_py_compile_is_syntax_tier(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python -m py_compile foo.py", ec)
        self.assertEqual(self.tier(ec), "syntax")

    def test_linters_are_static_tier(self):
        for cmd in ("ruff check foo.py", "python -m mypy foo.py"):
            agent, ec = self._agent(), self._ctx({"foo.py"})
            self._run(agent, cmd, ec)
            self.assertEqual(self.tier(ec), "static", cmd)

    def test_an_execution_earns_no_tier_however_green(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "pytest -q foo.py", ec,
                  stdout="1 passed in 0.3s\nTest passed: solution is finite and stable.")
        self._judge(ec)
        self.assertIsNone(self.tier(ec))

    def test_a_printed_invariant_earns_nothing(self):
        # A number in stdout is a string: unfalsifiable from out here, so it buys no
        # tier. The proxy seals references server-side precisely because of this.
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "pytest -q foo.py", ec,
                  stdout="l2_rel=3.2e-4\nconvergence_order=3.98\n1 passed")
        self._judge(ec)
        self.assertIsNone(self.tier(ec))

    def test_failing_check_is_not_credited(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec, status="error")
        self.assertIsNone(self.tier(ec))
        self.assertNotIn("foo.py", ec["validated_files"])

    def test_declared_failing_verdict_overrides_a_green_exit(self):
        # The wave2d boundary test: the script decided its own criteria were unmet,
        # printed so, and returned 0 — which the ledger would otherwise read as a pass.
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec, stdout="reflection is high\ncheck=fail")
        self.assertFalse(ec["runs"]["python foo.py"]["completed"])

    def test_prose_about_a_failed_check_is_not_a_verdict(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "pytest -q foo.py", ec, stdout="the check failed to converge early on")
        self.assertTrue(ec["runs"]["pytest -q foo.py"]["completed"])

    def test_passing_verdict_never_rescues_a_red_exit(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "pytest -q foo.py", ec, status="error", stdout="check=pass")
        self.assertFalse(ec["runs"]["pytest -q foo.py"]["completed"])

    def test_tier_never_downgrades(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec)
        self.assertEqual(self.tier(ec), "static")
        # A weaker check afterwards does not un-prove what the stronger one established.
        ec["dirty_written_files"].add("foo.py")
        self._run(agent, "python -m py_compile foo.py", ec)
        self.assertEqual(self.tier(ec), "static")

    def test_strongest_check_in_a_chain_wins(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python -m py_compile foo.py && ruff check foo.py", ec)
        self.assertEqual(self.tier(ec), "static")

    def test_reedit_retracts_the_tier(self):
        # Evidence is about a specific revision of the file.
        from mimir.client.guardrails.observations import _record_code_edit
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec)
        self.assertEqual(self.tier(ec), "static")
        _record_code_edit(ec, "foo.py")
        self.assertIsNone(self.tier(ec))
        self.assertNotIn("foo.py", ec["validated_files"])

    def test_reedit_does_not_retract_a_run(self):
        # A run is a past event and its verdict is a statement about what that event
        # showed; re-editing does not undo either, it only makes them out of date.
        from mimir.client.guardrails.observations import _record_code_edit
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "python foo.py", ec)
        self._judge(ec)
        _record_code_edit(ec, "foo.py")
        self.assertEqual(ec["runs"]["python foo.py"]["verdict"], "pass")

    def test_failed_validation_retracts_the_tier(self):
        agent, ec = self._agent(), self._ctx({"foo.py"})
        self._run(agent, "ruff check foo.py", ec)
        self.assertEqual(self.tier(ec), "static")
        ec["dirty_written_files"].add("foo.py")
        self._run(agent, "ruff check foo.py", ec, status="error")
        self.assertIsNone(self.tier(ec))

    def test_whole_project_checker_stamps_every_pending_file(self):
        agent, ec = self._agent(), self._ctx({"a.py", "b.py"})
        self._run(agent, "ruff check .", ec)
        self.assertEqual(ec["validated_files"], {"a.py", "b.py"})
        for p in ("a.py", "b.py"):
            self.assertEqual(self.tier(ec, p), "static")

    def test_extension_pack_validator_defaults_to_static(self):
        # A VALIDATE-capability tool checked something but tells us no more than that.
        from mimir.client.guardrails.observations import _mark_file_validated
        ec = self._ctx({"foo.py"})
        _mark_file_validated(ec, "foo.py")
        self.assertEqual(self.tier(ec), "static")


class UncheckableFileTests(unittest.TestCase):
    """The check is required where it is possible, and reported where it is not.

    A `.cu` edited on a node with no CUDA toolkit has no checker to run. Demanding one
    anyway turns the conclude gate into a dead end, which a model escapes by inventing
    a check — the one outcome worse than an unchecked file.
    """

    _ctx = BashValidationObservationTests._ctx

    def _edit(self, path, *, available):
        from mimir.client.guardrails import observations
        ec = self._ctx(set())
        with patch.object(
            observations, "_any_command_on_path", lambda cmds: available
        ):
            observations._record_code_edit(ec, path)
        return ec

    def test_a_file_with_no_checker_here_does_not_block_the_conclusion(self):
        from mimir.client.guardrails.workflow import pending_validation_paths
        ec = self._edit("kernel.cu", available=False)
        self.assertIn("kernel.cu", ec["unverifiable_files"])
        self.assertEqual(pending_validation_paths(ec), [])

    def test_the_same_file_owes_a_check_where_the_toolchain_exists(self):
        from mimir.client.guardrails.workflow import pending_validation_paths
        ec = self._edit("kernel.cu", available=True)
        self.assertEqual(ec["unverifiable_files"], set())
        self.assertEqual(pending_validation_paths(ec), ["kernel.cu"])

    def test_a_language_with_no_declared_checker_is_unverifiable_too(self):
        # The default used to be the opposite ("the model may know a checker we do
        # not"), and it wedged every language outside the table: a `.rs` edit stayed
        # pending for a check no runnable command could ever credit, so the run could
        # not conclude. Nothing here can check it — same situation, same ending.
        ec = self._edit("engine.rs", available=True)
        self.assertIn("engine.rs", ec["unverifiable_files"])

    def test_every_declared_checker_extension_is_recognised_as_source(self):
        # The Fortran gap in one line: `.f03` was in the checker table and missing
        # from SOURCE_FILE_EXTENSIONS, so the edit was never recorded at all and no
        # check was ever asked for.
        from mimir.client.guardrails.observations import _CHECKER_COMMANDS_BY_EXTENSION
        from mimir.client.context.signals import SOURCE_FILE_EXTENSIONS
        self.assertEqual(
            set(_CHECKER_COMMANDS_BY_EXTENSION) - set(SOURCE_FILE_EXTENSIONS), set(),
        )

    def test_the_ledger_says_why_it_was_not_checked(self):
        from mimir.client.query_engine.verification import build_ledger
        ec = self._edit("kernel.cu", available=False)
        rows = "\n".join(build_ledger(ec)["rows"])
        self.assertIn("no checker here", rows)


class MissingCommandImputationTests(unittest.TestCase):
    """A command that is not installed said nothing about the patch.

    The one imputation the machine settles alone, so PATH is patched rather than trusted:
    whether this host happens to have `make` must not decide whether these pass.
    """

    def _agent(self):
        from mimir.client.context.capabilities import ToolCaps
        reg = {"bash_run": ToolCaps(
            name="bash_run", scope={"kind": "command_prefix", "args": ["command"]},
        )}
        return types.SimpleNamespace(
            tool_caps=reg,
            _parse_tool_payload=lambda result: json.loads(result),
            _normalize_workspace_path=lambda p: os.path.normpath(p) if p else "",
            _is_code_filepath=lambda p: str(p).endswith(".py"),
        )

    def _red(self, command, dirty={"solver.c"}, absent=()):
        """Run *command* red, with *absent* commands taken off PATH."""
        ec = build_execution_context()
        ec["dirty_written_files"] = set(dirty)
        ec["code_mutation_started"] = True
        real = runtime._any_command_on_path
        runtime._any_command_on_path = lambda cmds: not any(c in absent for c in cmds)
        try:
            runtime.record_tool_observation(
                self._agent(), "bash_run", {"command": command},
                json.dumps({"status": "error", "stdout": "", "stderr": "boom"}), ec,
            )
        finally:
            runtime._any_command_on_path = real
        return ec

    def test_a_command_that_is_not_installed_is_not_a_defect_in_the_patch(self):
        ec = self._red("make", absent=("make",))
        run = ec["runs"]["make"]
        self.assertEqual(run["blocked"], "make is not installed here")
        self.assertEqual(run["failures"], 0)
        # Steers nothing: the wall is not a reason to go back and edit.
        self.assertNotEqual(ec["workflow_state"], "edit")

    def test_a_red_exit_from_an_installed_command_still_drives_the_repair_ladder(self):
        # The default does not move. Every other wall needs the model to claim it.
        ec = self._red("make")
        self.assertEqual(ec["runs"]["make"]["blocked"], "")
        self.assertEqual(ec["runs"]["make"]["failures"], 1)
        self.assertEqual(ec["workflow_state"], "edit")

    def test_a_stdlib_module_head_is_never_read_as_a_missing_command(self):
        # `_exec_head` reports the *module* for `python -m X`, and `py_compile` has no
        # command of its own — reading `head` here would mark the REQUIRED check axis
        # as uninstallable on every box.
        ec = self._red("python -m py_compile foo.py", dirty={"foo.py"},
                       absent=("py_compile",))
        self.assertEqual(ec["runs"].get("python -m py_compile foo.py", {}).get("blocked", ""), "")

    def test_a_missing_local_executable_is_a_missed_build(self):
        # `./solver` absent is a build that did not happen — attributable, not a wall.
        ec = self._red("./solver", absent=("./solver",))
        self.assertEqual(ec["runs"]["./solver"]["blocked"], "")
        self.assertEqual(ec["runs"]["./solver"]["failures"], 1)

    def test_an_env_setup_segment_is_not_tested_against_path(self):
        # `source` is a shell builtin: without the effect filter every chain that sets
        # up an environment would read as blocked.
        ec = self._red("source env.sh && make", absent=("source",))
        self.assertEqual(ec["runs"]["source env.sh && make"]["blocked"], "")

    def test_a_blocked_run_closes_the_advisory_axis(self):
        ec = self._red("make", absent=("make",))
        self.assertTrue(ec["exercise_advice_closed"])
        self.assertEqual(ec["exercise_blocked_reason"], "make is not installed here")


if __name__ == "__main__":
    unittest.main()
