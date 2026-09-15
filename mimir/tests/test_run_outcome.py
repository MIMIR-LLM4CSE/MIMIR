"""The machine floor under a stated verdict, for servers that run code themselves.

A shell run is floored by its exit code. A tool like ``proxy_eval`` answers ``ok`` for
the *call* even when the run it launched came back red, so the ledger held no machine
reading at all and a stated ``pass`` was uncontradicted — the hole these tests close.

Covers the ``run_outcome`` descriptor round-trip, the one-way floor it feeds, the
per-run ledger key, and the ``measured`` validation tier.

Run:
    python -m pytest mimir/tests/test_run_outcome.py
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mimir.client.context.capabilities import (  # noqa: E402
    CODE_EXEC, PLAN_BLOCKED, PLAN_READONLY, ToolCaps, infer_tool_caps, run_outcome_spec,
)
from mimir.client.context.execution_context import (  # noqa: E402
    build_execution_context, validation_tier,
)
from mimir.client.context import failed_runs, unsettled_runs  # noqa: E402
from mimir.client.guardrails import observations as O  # noqa: E402
from mimir.client.guardrails import workflow  # noqa: E402
from mimir.client.guardrails.policy.bash_classify import opaque_command_executes  # noqa: E402
from mimir.client.guardrails.verdict import apply_verdict  # noqa: E402
from mimir.servers._shared.capabilities import build_descriptor  # noqa: E402
from mimir.servers.proxy._lib import store  # noqa: E402

# The shape server_proxy declares, restated here so a change to it fails a test rather
# than silently redefining what these assertions mean.
OUTCOME = {
    "id":            "run_dir",
    "crashed_when":  {"state": ["crashed"]},
    "failed_when":   {"feasible": [False]},
    "measured_when": {"state": ["done"]},
}
ROWS_OUTCOME = {**OUTCOME,
                "rows": {"field": "rows", "id": "run_dir",
                         "failed_when_present": ["error"]}}


def _agent(**caps) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        tool_caps=caps, _normalize_workspace_path=lambda p: p or "")


def _runner_agent():
    """A launcher and a reporter: two tools speaking about the same runs."""
    return _agent(
        runner=ToolCaps("runner", frozenset({PLAN_BLOCKED, CODE_EXEC}), run_outcome=OUTCOME),
        reporter=ToolCaps("reporter", frozenset(), run_outcome=OUTCOME),
        batch=ToolCaps("batch", frozenset({PLAN_BLOCKED, CODE_EXEC}), run_outcome=ROWS_OUTCOME),
        plain=ToolCaps("plain", frozenset({PLAN_BLOCKED, CODE_EXEC})),
    )


def _observe(agent, ec, tool: str, payload: dict, status: str = "ok") -> None:
    O._observe_tool_run(agent, tool, {}, status, ec, "call", payload)
    O._observe_run_outcome(agent, tool, payload, ec, "call")


class DescriptorRoundTripTests(unittest.TestCase):
    def test_spec_survives_server_to_client(self) -> None:
        desc = build_descriptor(caps=["executes"], run_outcome=ROWS_OUTCOME)
        tool = types.SimpleNamespace(name="batch", meta={"mimir": desc}, inputSchema={})
        caps = infer_tool_caps(tool)
        self.assertEqual(caps.run_outcome, ROWS_OUTCOME)
        self.assertIsNotNone(run_outcome_spec("batch", {"batch": caps}))

    def test_incomplete_specs_are_dropped(self) -> None:
        # An identifier with nothing to judge on, and a judgement with nothing to
        # attach it to, are both meaningless — neither reaches the client.
        self.assertNotIn("run_outcome", build_descriptor(run_outcome={"id": "run_dir"}))
        self.assertNotIn("run_outcome", build_descriptor(
            run_outcome={"crashed_when": {"state": ["crashed"]}}))

    def test_no_positive_verdict_form_exists(self) -> None:
        """A server may credit evidence, never grant itself a passing verdict."""
        desc = build_descriptor(run_outcome={
            "id": "run_dir", "crashed_when": {"state": ["crashed"]},
            "passed_when": {"state": ["done"]},
        })
        self.assertNotIn("passed_when", desc["run_outcome"])


class MachineFloorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.agent = _runner_agent()
        self.ec = build_execution_context()

    def test_crash_did_not_complete(self) -> None:
        _observe(self.agent, self.ec, "runner", {"run_dir": "/r/a"})
        _observe(self.agent, self.ec, "reporter", {"run_dir": "/r/a", "state": "crashed"})
        self.assertIn("/r/a", failed_runs(self.ec))
        self.assertNotIn("/r/a", unsettled_runs(self.ec))

    def test_machine_failure_cannot_be_overwritten_by_pass(self) -> None:
        """The whole point: the model may lower credit, never raise it past the machine."""
        _observe(self.agent, self.ec, "runner", {"run_dir": "/r/b"})
        _observe(self.agent, self.ec, "reporter",
                 {"run_dir": "/r/b", "state": "done", "feasible": False})
        self.assertEqual(self.ec["runs"]["/r/b"]["verdict"], "fail")
        self.assertEqual(apply_verdict("pass", "looks good", "", self.ec), [])
        self.assertEqual(self.ec["runs"]["/r/b"]["verdict"], "fail")
        self.assertIn("/r/b", failed_runs(self.ec))

    def test_a_measured_run_that_did_not_improve_costs_nothing(self) -> None:
        """A ratchet 'reject' is the ordinary outcome of an experiment, not a failure."""
        _observe(self.agent, self.ec, "runner", {"run_dir": "/r/c"})
        _observe(self.agent, self.ec, "reporter",
                 {"run_dir": "/r/c", "state": "done", "feasible": True, "verdict": "reject"})
        self.assertEqual(failed_runs(self.ec), {})
        self.assertIn("/r/c", unsettled_runs(self.ec))

    def test_launch_and_report_settle_the_same_entry(self) -> None:
        """Keyed on the run, not the tool: otherwise the floor lands on a second row."""
        _observe(self.agent, self.ec, "runner", {"run_dir": "/r/d"})
        self.assertEqual(list(self.ec["runs"]), ["/r/d"])
        _observe(self.agent, self.ec, "reporter", {"run_dir": "/r/d", "state": "crashed"})
        self.assertEqual(list(self.ec["runs"]), ["/r/d"])

    def test_iterations_do_not_collapse(self) -> None:
        for tag in ("r1", "r2", "r3"):
            _observe(self.agent, self.ec, "runner", {"run_dir": f"/runs/{tag}"})
        self.assertEqual(list(self.ec["runs"]), ["/runs/r1", "/runs/r2", "/runs/r3"])

    def test_undeclared_tool_keeps_the_old_key(self) -> None:
        """Non-regression: a tool declaring no outcome still records against its name."""
        _observe(self.agent, self.ec, "plain", {"run_dir": "/runs/x"})
        _observe(self.agent, self.ec, "plain", {"run_dir": "/runs/y"})
        self.assertEqual(list(self.ec["runs"]), ["plain"])

    def test_a_failing_row_fails_its_own_run(self) -> None:
        """A suite answers ok while individual cases crashed; each row is its own run."""
        _observe(self.agent, self.ec, "batch", {
            "run_dir": "/suite/top", "state": "done", "feasible": True,
            "rows": [{"run_dir": "/suite/case0"},
                     {"run_dir": "/suite/case1", "error": "solver exited with code 2"}],
        })
        self.assertNotIn("/suite/case0", failed_runs(self.ec))
        self.assertIn("/suite/case1", failed_runs(self.ec))
        self.assertEqual(self.ec["runs"]["/suite/case1"]["verdict"], "fail")


class MeasuredTierTests(unittest.TestCase):
    """A run measures the file only where attribution is recorded, not guessed."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self._saved_root = store._CACHE_DIR
        store._CACHE_DIR = self._tmp.name
        self.source = os.path.join(self._tmp.name, "solver.py")
        with open(self.source, "w") as fh:
            fh.write("x = 1\n")
        os.makedirs(store._opt_session_runs_dir("p"), exist_ok=True)
        store._write_json_atomic(
            store._opt_config_file("p"),
            {"proxy_name": "p", "proxy_source_path": self.source})
        store._write_active_session("p")
        self.agent = _runner_agent()
        self.agent.tool_caps["bash_run"] = ToolCaps(
            "bash_run", frozenset({PLAN_READONLY, CODE_EXEC}),
            scope={"kind": "command_prefix", "args": ["command"]})
        self.ec = build_execution_context()
        self.ec["dirty_written_files"].add(self.source)

    def tearDown(self) -> None:
        store._CACHE_DIR = self._saved_root
        self._tmp.cleanup()

    def _report(self, **fields) -> None:
        _observe(self.agent, self.ec, "reporter", {"run_dir": "/r/x", **fields})

    def test_finished_run_measures_the_session_source(self) -> None:
        self._report(state="done", feasible=True)
        self.assertEqual(validation_tier(self.ec, self.source), "measured")
        self.assertIn(self.source, self.ec["validated_files"])

    def test_a_run_still_going_credits_nothing(self) -> None:
        # A launched or diverted run reports status ok with whatever it had printed
        # so far. Judged as a result, a whole-project check would credit every pending
        # file at the instant the suite was started, before it had proved anything.
        O._observe_bash_validation(
            self.agent, "bash_run", {"command": f"ruff check {self.source}"},
            "ok",
            {"stdout": "", "background_job": {"job_key": "20260101T120000Z-ab12"}},
            self.ec, "call")
        self.assertIsNone(validation_tier(self.ec, self.source))

    def test_a_static_check_alone_does_not_reach_measured(self) -> None:
        O._observe_bash_validation(
            self.agent, "bash_run", {"command": f"ruff check {self.source}"},
            "ok", {"stdout": ""}, self.ec, "call")
        self.assertEqual(validation_tier(self.ec, self.source), "static")

    def test_crashed_and_infeasible_runs_credit_nothing(self) -> None:
        self._report(state="crashed")
        self.assertIsNone(validation_tier(self.ec, self.source))
        self._report(state="done", feasible=False)
        self.assertIsNone(validation_tier(self.ec, self.source))

    def test_an_unedited_source_is_not_credited(self) -> None:
        self.ec["dirty_written_files"].discard(self.source)
        self._report(state="done", feasible=True)
        self.assertIsNone(validation_tier(self.ec, self.source))

    def test_report_names_an_unmeasured_source_and_raises_risk(self) -> None:
        O._observe_bash_validation(
            self.agent, "bash_run", {"command": f"ruff check {self.source}"},
            "ok", {"stdout": ""}, self.ec, "call")
        lines = workflow.unmeasured_proxy_source_lines(self.ec)
        self.assertEqual(len(lines), 1)
        self.assertIn("never measured", lines[0])
        summary = workflow.finalize_incomplete_answer("done", self.ec)
        self.assertIn("Checked but never measured", summary)
        self.assertIn("Residual risk: medium", summary)

    def test_a_measured_source_is_not_reported(self) -> None:
        self._report(state="done", feasible=True)
        self.assertEqual(workflow.unmeasured_proxy_source_lines(self.ec), [])


if __name__ == "__main__":
    unittest.main()


# The command that produced the report this class exists to prevent: a heredoc the
# server refused outright, whose C++ body carries two `//` comments. Kept verbatim,
# comments included, because both are load-bearing.
REFUSED_HEREDOC = """cat > /tmp/ws/team_size_test.cpp << 'EOF'
#include <Kokkos_Core.hpp>
int main() {
  // Mimic the elastic Q1 scratch: fields (2*8*4=64B) + gemm (576B) = 640B
  size_t bytes = 640;
  TeamPolicy p(1000000, Kokkos::AUTO);
  // Fixed small team
  TeamPolicy p2(1000000, 32);
  return 0;
}
EOF
echo "written"
"""

# What server_bash answers for it. `refused` is the whole point: it is how a rejection
# decided before any shell opened is told apart from a command that ran and came back red.
REFUSAL_PAYLOAD = {
    "refused": True,
    "status": "error",
    "error": "Redirection operator '<<' is not supported.",
    "hint": "A heredoc body is not a command this can validate.",
}


class RefusedCallTests(unittest.TestCase):
    """A call the server refused never ran, so it owes and charges nothing.

    Both halves of one observed failure. A refused heredoc was entered in the run
    ledger as red: it charged the repair budget, steered the workflow back to `edit`,
    and printed its whole body into the user's chat under `Run failing, unresolved`.
    It got there because `opaque_command_executes` read `//` — the start of a C++
    comment in the heredoc body — as a program invoked by path.
    """

    def setUp(self) -> None:
        self.agent = _agent(bash_run=ToolCaps(
            "bash_run", frozenset({PLAN_READONLY, CODE_EXEC}),
            scope={"kind": "command_prefix", "args": ["command"]}))
        self.ec = build_execution_context()

    def _observe_bash(self, command: str, status: str, payload: dict) -> None:
        O._observe_bash_validation(
            self.agent, "bash_run", {"command": command}, status, payload, self.ec, "call")

    def test_a_comment_in_a_heredoc_is_not_a_run(self) -> None:
        """`//` is path-like and names no program; only the second test is the barrier."""
        self.assertFalse(opaque_command_executes(REFUSED_HEREDOC))

    def test_a_refused_heredoc_opens_no_run(self) -> None:
        self._observe_bash(REFUSED_HEREDOC, "error", REFUSAL_PAYLOAD)
        self.assertEqual(self.ec["runs"], {})
        self.assertEqual(failed_runs(self.ec), {})

    def test_a_refused_call_does_not_reach_the_completion_report(self) -> None:
        """The symptom itself: the heredoc body, printed to the user as an open issue."""
        self._observe_bash(REFUSED_HEREDOC, "error", REFUSAL_PAYLOAD)
        summary = workflow.finalize_incomplete_answer("done", self.ec)
        self.assertNotIn("Kokkos", summary)
        self.assertNotIn("failing, unresolved", summary)

    def test_a_refused_call_charges_no_validation_budget(self) -> None:
        """The other half: a rejection must not count against the file it named."""
        self.ec["dirty_written_files"].add("/tmp/ws/solver.py")
        self._observe_bash("ruff check /tmp/ws/solver.py", "error", REFUSAL_PAYLOAD)
        self.assertEqual(self.ec.get("validation_fail_count_by_file", {}), {})

    def test_a_real_red_exit_is_still_a_failing_run(self) -> None:
        """Non-regression: same error status, no `refused`, so it is a real finding."""
        self._observe_bash(
            "python solver.py", "error",
            {"status": "error", "returncode": 1, "cwd": "/tmp/ws", "stderr": "boom"})
        self.assertIn("python solver.py", failed_runs(self.ec))

    def test_a_timeout_is_still_a_failing_run(self) -> None:
        """It reached a shell and hit a wall there, which is a finding like any other."""
        self._observe_bash(
            "python solver.py", "error",
            {"status": "error", "error": "Command timed out after 300s.", "cwd": "/tmp/ws"})
        self.assertIn("python solver.py", failed_runs(self.ec))
