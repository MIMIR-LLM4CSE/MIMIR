"""Pluggable policy checks — the extension seam for application-specific constraints.

The built-in precondition chain in :mod:`policy.engine` (registry → external_fetch →
cluster_submit → state_guard → write_policy → approval) is the fixed *core*. This module
lets an application register **additional** checks on top — e.g. an API-contract or
cybersecurity guard — without editing core code. Custom checks can only *add*
constraints: the core runs first and independently, and a custom check that returns
``None`` never relaxes anything.

A pack (a Python module under the plugins dir; see :mod:`client.extensions`) declares a
:class:`PolicyCheck` and registers it at import time::

    from mimir.client.extensions import PolicyCheck, register_policy_check
    from mimir.client.context.capabilities import has_cap, EXTERNAL_FETCH

    def _no_secret_exfil(agent, tool, args, ec):
        if has_cap(tool, EXTERNAL_FETCH, agent.tool_caps) and _looks_secret(args):
            return '{"status": "error", "error": "Outbound call carries a secret"}'
        return None

    register_policy_check(PolicyCheck(name="no_secret_exfil", check=_no_secret_exfil))

Application policies are **locked** (mandatory) — there is deliberately no toggle for
them, unlike nudges. Reference *capabilities* (``has_cap``/``names_with_cap``), never
literal tool names, so a pack stays robust across tool renames.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

# A check receives (agent, tool_name, arguments, execution_context) and returns a
# violation payload string (ideally the agent's JSON-error shape) to BLOCK the call,
# or None to allow it to proceed to the next check.
PolicyCheckFn = Callable[[Any, str, dict, "dict | None"], "str | None"]

# Where a custom check runs relative to the built-in stages.
#   "pre_mutation" — right after the (post-rewrite) registry check, before any built-in
#                    reach/state/write gate. Use to veto a call outright.
#   "pre_approval" — after the built-in write_policy gate, just before approval. Use for
#                    constraints that should see the fully-resolved tool + arguments.
VALID_POLICY_STAGES = ("pre_mutation", "pre_approval")


@dataclass(frozen=True)
class PolicyCheck:
    name: str            # becomes ``policy_stage`` in the violation payload
    check: PolicyCheckFn
    stage: str = "pre_approval"
    order: int = 100     # lower runs first within a stage

    def __post_init__(self) -> None:
        if self.stage not in VALID_POLICY_STAGES:
            raise ValueError(
                f"PolicyCheck.stage must be one of {VALID_POLICY_STAGES}, got {self.stage!r}"
            )


class _PolicyRegistry:
    """Process-global registry of application policy checks (keyed by name)."""

    def __init__(self) -> None:
        self._checks: dict[str, PolicyCheck] = {}

    def register(self, check: PolicyCheck) -> None:
        # Idempotent by name: re-registration replaces, so re-importing a pack (agent
        # re-init, multiple agents in one process) never duplicates a check.
        self._checks[check.name] = check

    def active_checks(self) -> list[PolicyCheck]:
        """All registered checks, sorted by (stage, order, name) for deterministic runs."""
        stage_rank = {name: i for i, name in enumerate(VALID_POLICY_STAGES)}
        return sorted(
            self._checks.values(),
            key=lambda c: (stage_rank.get(c.stage, 99), c.order, c.name),
        )

    def names(self) -> list[str]:
        return sorted(self._checks)

    def clear(self) -> None:
        """Drop all registered checks (used by tests to isolate the global registry)."""
        self._checks.clear()


# The single process-global instance.
PolicyRegistry = _PolicyRegistry()


def register_policy_check(check: PolicyCheck) -> None:
    """Public entry point a pack calls at import time to add a policy check."""
    PolicyRegistry.register(check)
