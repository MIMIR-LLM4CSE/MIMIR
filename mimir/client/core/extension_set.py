"""The extension packs one agent runs with: policy checks, nudge rules, post-tool hooks.

A pack registers its descriptors while it is imported; ``register_*`` add them to the
set being loaded (:func:`registering_into`). Registrations made outside any load (an
application embedding MIMIR registers in code) go to :data:`AMBIENT`, which every set
reads through as its parent. Agents of one workspace share that workspace's set
(``extensions.load_plugins``), and nothing leaks between workspaces.
"""
from __future__ import annotations

import contextlib
from contextvars import ContextVar
from typing import Any, TYPE_CHECKING, Iterator

if TYPE_CHECKING:
    from ..guardrails.nudges.plugins import NudgeRule
    from ..guardrails.policy.plugins import PolicyCheck
    from ..tool_execution.plugins import PostToolRule

# The stages a policy check may run at, in pipeline order.
POLICY_STAGES = ("pre_mutation", "pre_approval")


class ExtensionSet:
    def __init__(self, parent: ExtensionSet | None = None) -> None:
        self.parent = parent
        # Registered by name: re-registering replaces, so importing a pack twice never
        # duplicates a rule.
        self._checks: dict[str, PolicyCheck] = {}
        self._nudges: dict[str, NudgeRule] = {}
        self._post_tool: dict[str, PostToolRule] = {}
        # The pack files imported into this set.
        self.packs: list[str] = []

    def add_policy_check(self, check: PolicyCheck) -> None:
        self._checks[check.name] = check

    def add_nudge(self, rule: NudgeRule) -> None:
        self._nudges[rule.name] = rule

    def add_post_tool(self, rule: PostToolRule) -> None:
        self._post_tool[rule.name] = rule

    def _chain(self, attr: str) -> dict[str, Any]:
        """This set's entries over its parent's: a pack overrides a same-named one."""
        merged = self.parent._chain(attr) if self.parent is not None else {}
        merged.update(getattr(self, attr))
        return merged

    def policy_checks(self) -> list[PolicyCheck]:
        """Sorted by (stage, order, name), so runs are deterministic."""
        rank = {stage: i for i, stage in enumerate(POLICY_STAGES)}
        return sorted(self._chain("_checks").values(),
                      key=lambda c: (rank.get(c.stage, 99), c.order, c.name))

    def nudge_rules(self, layer: str | None = None) -> list[NudgeRule]:
        """Sorted by (priority, name), optionally only those of *layer*."""
        rules = [r for r in self._chain("_nudges").values() if layer is None or r.layer == layer]
        return sorted(rules, key=lambda r: (r.priority, r.name))

    def post_tool_rules(self) -> list[PostToolRule]:
        return sorted(self._chain("_post_tool").values(), key=lambda r: (r.order, r.name))

    def policy_names(self) -> list[str]:
        return sorted(self._chain("_checks"))

    def nudge_names(self) -> list[str]:
        return sorted(self._chain("_nudges"))

    def post_tool_names(self) -> list[str]:
        return sorted(self._chain("_post_tool"))


AMBIENT = ExtensionSet()

_target: ContextVar[ExtensionSet] = ContextVar("extension_target", default=AMBIENT)


def registration_target() -> ExtensionSet:
    """Where a ``register_*`` call made now lands."""
    return _target.get()


@contextlib.contextmanager
def registering_into(extensions: ExtensionSet) -> Iterator[ExtensionSet]:
    """Send every ``register_*`` call made inside the block to *extensions*."""
    token = _target.set(extensions)
    try:
        yield extensions
    finally:
        _target.reset(token)


def extensions_of(agent: object) -> ExtensionSet:
    """The agent's extension set; an agent without one runs with the ambient set."""
    return getattr(agent, "extensions", None) or AMBIENT
