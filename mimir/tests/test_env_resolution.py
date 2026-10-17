"""Behavioral tests for the environment-resolution cascade (Tiers 1 & 4).

Covers the two reality signals the cascade rests on and the verification-layer
nudge they drive:

* ``env.missing_module`` — a failing import-check or a failing run whose output
  names a missing module flips ``unresolved_modules``; a successful call never does.
* ``env.env_probe`` — calling any ENV_DISCOVERY tool flips ``env_probed``.
* the ``env_resolution`` nudge fires once while a module is unresolved and no env has
  been probed, and stops once the agent probes the environments.

Pure-Python + stubs (no live model/servers): runs on x86 and ARM.
"""
from __future__ import annotations

import types
import unittest

import mimir.client.guardrails.observations as runtime
from mimir.client.context import capabilities as caps
from mimir.client.context.execution_context import build_execution_context
from mimir.client.guardrails.nudges.engine import (
    maybe_append_nudge,
    maybe_inject_env_resolution,
)
from mimir.tests._golden_caps import build_declared_registry
from mimir.client.guardrails.observations import Call  # noqa: E402

_REGISTRY = build_declared_registry()


def _agent():
    return types.SimpleNamespace(
        tool_caps=_REGISTRY,
        enforcement="strict",
        normalize_workspace_path=lambda p: p or "",
    )


def _env_discovery_tool() -> str:
    names = caps.names_with_cap(caps.ENV_DISCOVERY, _REGISTRY)
    assert names, "no ENV_DISCOVERY tool in the declared registry"
    return sorted(names)[0]


class MissingModuleObserverTests(unittest.TestCase):
    def test_unresolved_import_text_flags_unresolved(self) -> None:
        # A failing check whose output names an unresolved import is flagged from the
        # result text (the dedicated imports validator was removed; bash `python`/
        # `pytest` failures carry the same ModuleNotFoundError / unresolved-import text).
        ctx = build_execution_context()
        payload = {"stderr": "unresolved import: scipy"}
        runtime.env.missing_module(Call(
            agent=_agent(), tool_name="code_run", payload=payload, status="error",
            execution_context=ctx,
        ))
        self.assertIn("scipy", ctx.unresolved_modules)

    def test_module_not_found_in_failing_run_flags_unresolved(self) -> None:
        ctx = build_execution_context()
        payload = {"stderr": "ModuleNotFoundError: No module named 'numpy'"}
        runtime.env.missing_module(Call(
            agent=_agent(), tool_name="code_run", payload=payload, status="error",
            execution_context=ctx,
        ))
        self.assertIn("numpy", ctx.unresolved_modules)

    def test_success_never_flags(self) -> None:
        ctx = build_execution_context()
        payload = {"stdout": "ModuleNotFoundError mentioned in a file we read"}
        runtime.env.missing_module(Call(
            agent=_agent(), tool_name="code_run", payload=payload, status="ok",
            execution_context=ctx,
        ))
        self.assertFalse(ctx.unresolved_modules)

    def test_unrelated_failure_does_not_flag(self) -> None:
        ctx = build_execution_context()
        payload = {"stderr": "SyntaxError: invalid syntax"}
        runtime.env.missing_module(Call(
            agent=_agent(), tool_name="code_run", payload=payload, status="error",
            execution_context=ctx,
        ))
        self.assertFalse(ctx.unresolved_modules)


class EnvProbeObserverTests(unittest.TestCase):
    def test_env_discovery_tool_flips_flag(self) -> None:
        ctx = build_execution_context()
        runtime.env.env_probe(Call(
            agent=_agent(), tool_name=_env_discovery_tool(), execution_context=ctx,
        ))
        self.assertTrue(ctx.env_probed)

    def test_other_tool_does_not_flip_flag(self) -> None:
        ctx = build_execution_context()
        runtime.env.env_probe(Call(agent=_agent(), tool_name="code_run", execution_context=ctx))
        self.assertFalse(ctx.env_probed)


class EnvMutationObserverTests(unittest.TestCase):
    def _mutate_tool(self) -> str:
        names = caps.names_with_cap(caps.ENV_MUTATE, _REGISTRY)
        assert names, "no ENV_MUTATE tool in the declared registry"
        return sorted(names)[0]

    def test_successful_install_records_cleanup_obligation(self) -> None:
        ctx = build_execution_context()
        payload = {"installed": ["numpy"], "python": "/envs/foo/bin/python"}
        runtime.env.env_mutation(Call(
            agent=_agent(), tool_name=self._mutate_tool(), payload=payload, status="ok",
            execution_context=ctx,
        ))
        self.assertEqual(len(ctx.env_mutations), 1)
        self.assertEqual(ctx.env_mutations[0]["installed"], ["numpy"])

    def test_failed_mutation_records_nothing(self) -> None:
        ctx = build_execution_context()
        runtime.env.env_mutation(Call(
            agent=_agent(), tool_name=self._mutate_tool(), payload={}, status="error",
            execution_context=ctx,
        ))
        self.assertFalse(ctx.env_mutations)

    def test_non_mutating_tool_records_nothing(self) -> None:
        ctx = build_execution_context()
        runtime.env.env_mutation(Call(
            agent=_agent(), tool_name="code_run", payload={"installed": ["x"]}, status="ok",
            execution_context=ctx,
        ))
        self.assertFalse(ctx.env_mutations)

    def test_duplicate_mutation_not_double_recorded(self) -> None:
        ctx = build_execution_context()
        payload = {"installed": ["numpy"], "python": "/envs/foo/bin/python"}
        tool = self._mutate_tool()
        runtime.env.env_mutation(Call(
            agent=_agent(), tool_name=tool, payload=payload, status="ok", execution_context=ctx,
        ))
        runtime.env.env_mutation(Call(
            agent=_agent(), tool_name=tool, payload=payload, status="ok", execution_context=ctx,
        ))
        self.assertEqual(len(ctx.env_mutations), 1)


class EnvCleanupNudgeTests(unittest.TestCase):
    def _fire(self, ctx, enforcement="strict") -> list[dict]:
        # The env-cleanup nudge survives into "light" (it guards a real, non-self-
        # correcting side effect); it is cut only at "off". light/off exercised below.
        agent = types.SimpleNamespace(
            tool_caps=_REGISTRY, enforcement=enforcement,
            normalize_workspace_path=lambda p: p or "",
        )
        messages: list[dict] = []
        maybe_append_nudge(
            agent=agent, query="check this file", active_mode="agent",
            execution_context=ctx, messages=messages,
        )
        return messages

    def test_fires_at_conclude_with_mutations(self) -> None:
        # The env-cleanup nudge is placed ahead of the other strict guidance nudges,
        # so it wins when its condition holds.
        ctx = build_execution_context()
        ctx.env_mutations = [{"installed": ["numpy"], "python": "/envs/foo/bin/python"}]
        ctx.workflow_state = "conclude"
        messages = self._fire(ctx)
        self.assertEqual(ctx.nudge_counts.get("env_cleanup"), 1)
        self.assertIn("undo", messages[0]["content"].lower())

    def test_not_fired_before_conclude(self) -> None:
        ctx = build_execution_context()
        ctx.env_mutations = [{"installed": ["numpy"]}]
        ctx.workflow_state = "edit"
        self._fire(ctx)
        self.assertIsNone(ctx.nudge_counts.get("env_cleanup"))

    def test_not_fired_without_mutations(self) -> None:
        ctx = build_execution_context()
        ctx.workflow_state = "conclude"
        self._fire(ctx)
        self.assertIsNone(ctx.nudge_counts.get("env_cleanup"))

    def test_fires_at_light(self) -> None:
        # env_cleanup is part of the deliberate "light" subset (agent mode).
        ctx = build_execution_context()
        ctx.env_mutations = [{"installed": ["numpy"]}]
        ctx.workflow_state = "conclude"
        self._fire(ctx, enforcement="light")
        self.assertEqual(ctx.nudge_counts.get("env_cleanup"), 1)

    def test_suppressed_at_off(self) -> None:
        # off cuts the whole guidance layer.
        ctx = build_execution_context()
        ctx.env_mutations = [{"installed": ["numpy"]}]
        ctx.workflow_state = "conclude"
        self._fire(ctx, enforcement="off")
        self.assertIsNone(ctx.nudge_counts.get("env_cleanup"))


class EnvResolutionNudgeTests(unittest.TestCase):
    def _fire(self, ctx, enforcement="strict") -> list[dict]:
        # The env-resolution nudge is strict-only guidance (discovery-flavoured), so
        # the default agent is "strict"; light/off are exercised explicitly below.
        agent = types.SimpleNamespace(
            tool_caps=_REGISTRY, enforcement=enforcement,
            normalize_workspace_path=lambda p: p or "",
        )
        messages: list[dict] = []
        maybe_append_nudge(
            agent=agent, query="check this file", active_mode="agent",
            execution_context=ctx, messages=messages,
        )
        return messages

    def test_fires_when_unresolved_and_not_probed(self) -> None:
        ctx = build_execution_context()
        ctx.unresolved_modules = {"torch"}
        ctx.env_probed = False
        messages = self._fire(ctx)
        self.assertEqual(len(messages), 1)
        self.assertIn("environment", messages[0]["content"].lower())
        self.assertEqual(ctx.nudge_counts.get("env_resolution"), 1)

    def test_suppressed_once_env_probed(self) -> None:
        ctx = build_execution_context()
        ctx.unresolved_modules = {"torch"}
        ctx.env_probed = True
        self._fire(ctx)
        self.assertIsNone(ctx.nudge_counts.get("env_resolution"))

    def test_fires_at_most_once(self) -> None:
        ctx = build_execution_context()
        ctx.unresolved_modules = {"torch"}
        ctx.env_probed = False
        self._fire(ctx)
        self.assertEqual(ctx.nudge_counts.get("env_resolution"), 1)
        # Second pass: counter already at 1 -> no further env-resolution nudge.
        self._fire(ctx)
        self.assertEqual(ctx.nudge_counts.get("env_resolution"), 1)

    def test_suppressed_when_enforcement_not_strict(self) -> None:
        # light/off trust the model to resolve the import failure itself.
        for level in ("light", "off"):
            ctx = build_execution_context()
            ctx.unresolved_modules = {"torch"}
            ctx.env_probed = False
            self._fire(ctx, enforcement=level)
            self.assertIsNone(ctx.nudge_counts.get("env_resolution"), level)


class MidLoopEnvResolutionTests(unittest.TestCase):
    """The cascade arrives at the failing call, not a step ceiling later.

    Same gate and same single budget as the end-of-turn table row — only the moment
    differs, which is the whole point: by the time the loop ends, the model has spent
    its steps retrying against the interpreter that could never resolve the module.
    """

    def _agent(self, enforcement="strict"):
        return types.SimpleNamespace(
            tool_caps=_REGISTRY, enforcement=enforcement,
            normalize_workspace_path=lambda p: p or "",
        )

    def test_fires_immediately_after_a_failing_call(self) -> None:
        ctx = build_execution_context()
        ctx.unresolved_modules = {"torch"}
        messages: list[dict] = []
        fired = maybe_inject_env_resolution(
            agent=self._agent(), active_mode="agent",
            execution_context=ctx, messages=messages,
        )
        self.assertTrue(fired)
        self.assertEqual(len(messages), 1)
        self.assertIn("environment", messages[0]["content"].lower())

    def test_spends_the_budget_the_table_row_shares(self) -> None:
        ctx = build_execution_context()
        ctx.unresolved_modules = {"torch"}
        agent = self._agent()
        maybe_inject_env_resolution(
            agent=agent, active_mode="agent", execution_context=ctx, messages=[],
        )
        self.assertEqual(ctx.nudge_counts.get("env_resolution"), 1)
        # The end-of-turn row must now stay silent rather than repeat the cascade.
        end_of_turn: list[dict] = []
        maybe_append_nudge(
            agent=agent, query="check this file", active_mode="agent",
            execution_context=ctx, messages=end_of_turn,
        )
        self.assertEqual(ctx.nudge_counts.get("env_resolution"), 1)

    def test_silent_without_an_unresolved_module(self) -> None:
        ctx = build_execution_context()
        self.assertFalse(maybe_inject_env_resolution(
            agent=self._agent(), active_mode="agent",
            execution_context=ctx, messages=[],
        ))

    def test_respects_enforcement(self) -> None:
        for level in ("light", "off"):
            ctx = build_execution_context()
            ctx.unresolved_modules = {"torch"}
            self.assertFalse(maybe_inject_env_resolution(
                agent=self._agent(level), active_mode="agent",
                execution_context=ctx, messages=[],
            ), level)


class ResolvedEnvironmentRearmsExerciseTests(unittest.TestCase):
    """A successful execution retracts the unresolved-module flag.

    The flag is read by the exercise-feasibility gate, so leaving it set after the
    model found the right interpreter silenced the run/verdict advice for the rest of
    the query — the one run proving the environment resolved was what used to bury it.
    """

    def _exec_tool(self) -> str:
        names = caps.names_with_cap(caps.CODE_EXEC, _REGISTRY)
        assert names, "no CODE_EXEC tool in the declared registry"
        return sorted(names)[0]

    def test_successful_execution_clears_the_flag(self) -> None:
        ctx = build_execution_context()
        agent = _agent()
        tool = self._exec_tool()
        runtime.env.missing_module(Call(
            agent=agent, tool_name=tool, payload={"stderr": "No module named 'torch'"},
            status="error", execution_context=ctx,
        ))
        self.assertIn("torch", ctx.unresolved_modules)
        runtime.env.missing_module(Call(
            agent=agent, tool_name=tool, payload={"stdout": "ok"}, status="ok",
            execution_context=ctx,
        ))
        self.assertFalse(ctx.unresolved_modules)

    def test_non_execution_success_leaves_it_alone(self) -> None:
        # Reading a file that merely happens to succeed says nothing about the
        # interpreter the check failed under.
        ctx = build_execution_context()
        agent = _agent()
        runtime.env.missing_module(Call(
            agent=agent, tool_name=self._exec_tool(),
            payload={"stderr": "No module named 'torch'"}, status="error", execution_context=ctx,
        ))
        runtime.env.missing_module(Call(
            agent=agent, tool_name="read_file", payload={"stdout": "ok"}, status="ok",
            execution_context=ctx,
        ))
        self.assertIn("torch", ctx.unresolved_modules)


if __name__ == "__main__":
    unittest.main()
