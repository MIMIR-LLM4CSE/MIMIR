"""Application policy checks registered on top of the core pipeline; they can only block.

A ``PolicyCheck`` runs at ``pre_mutation`` or ``pre_approval`` (see ``policy.engine``).
Register at import time with ``register_policy_check``; read capabilities, never tool
names. No toggle: application policies are mandatory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from ...core.extension_set import POLICY_STAGES, registration_target

# A check receives (agent, tool_name, arguments, execution_context) and returns a
# violation payload string (ideally the agent's JSON-error shape) to BLOCK the call,
# or None to allow it to proceed to the next check.
PolicyCheckFn = Callable[[Any, str, dict, "dict | None"], "str | None"]

# Where a custom check runs relative to the built-in stages.
#   "pre_mutation" — right after the (post-rewrite) registry check, before any built-in
#                    reach/state/write gate. Use to veto a call outright.
#   "pre_approval" — after the built-in write_policy gate, just before approval. Use for
#                    constraints that should see the fully-resolved tool + arguments.
VALID_POLICY_STAGES = POLICY_STAGES


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


def register_policy_check(check: PolicyCheck) -> None:
    """Add a policy check to the extension set being loaded (see ``core.extension_set``)."""
    registration_target().add_policy_check(check)
