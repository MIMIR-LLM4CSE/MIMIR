"""Drift guard: every declared capability must have a live client consumer.

The dual of :func:`mimir.client.context.capabilities.unannotated_live_tools`,
which flags the *tool* side of capability drift (a connected tool that declares
no capability). This guards the *vocabulary* side: a capability constant that
lingers in the vocab after its last consumer was removed — exactly what happened
to ``VALIDATION_BYPASS`` once its only reader (a never-firing policy branch) was
deleted. A capability only earns its place if some client control-flow reacts to
it; servers merely *declare* caps, so consumption is measured in ``client/``.

Static AST scan (no runtime / model / mcp deps) so it runs on the x86 build host,
same approach as ``_golden_caps.build_declared_registry``.

Scope / caveat: this catches *declared-but-never-referenced* drift — the common,
cheap case (you removed the last consumer but left the constant). It deliberately
does NOT try to catch a consumer that exists but is itself unreachable (a branch
gated on a context field nothing populates). That reachability class needs
behavioural / effect tests, not a static reference scan.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

from mimir.client.context import capabilities as caps

CLIENT_DIR = pathlib.Path(caps.__file__).resolve().parents[1]
CAP_DEF_FILE = pathlib.Path(caps.__file__).resolve()


def capability_constants() -> set[str]:
    """The vocabulary: uppercase names in ``__all__`` bound to a capability string."""
    return {
        name for name in caps.__all__
        if name.isupper() and isinstance(getattr(caps, name), str)
    }


def referenced_identifiers(*, exclude: set[pathlib.Path]) -> set[str]:
    """Every identifier *used* (Name load / attribute access) in client source.

    Import aliases are intentionally not counted — importing a constant without
    using it is an unused import, not a consumer.
    """
    used: set[str] = set()
    for path in CLIENT_DIR.rglob("*.py"):
        if path.resolve() in exclude:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)  # e.g. ``caps.VALIDATE``
    return used


def capabilities_without_consumer() -> list[str]:
    """Declared capabilities that no client module outside the vocab file references."""
    declared = capability_constants()
    referenced = referenced_identifiers(exclude={CAP_DEF_FILE})
    return sorted(declared - referenced)


class CapabilityConsumerTests(unittest.TestCase):
    def test_every_capability_has_a_client_consumer(self) -> None:
        orphans = capabilities_without_consumer()
        self.assertEqual(
            orphans, [],
            "These capabilities are declared in the vocabulary but no client code "
            f"consumes them (drop them or wire a consumer): {orphans}",
        )

    def test_detector_is_not_vacuous(self) -> None:
        """A synthetic, never-referenced capability name must be reported.

        Guards the guard: proves the scan would actually catch a real orphan rather
        than passing because, say, the reference set was computed wrong.
        """
        declared = capability_constants() | {"ZZ_NONEXISTENT_CAP"}
        referenced = referenced_identifiers(exclude={CAP_DEF_FILE})
        self.assertIn("ZZ_NONEXISTENT_CAP", declared - referenced)


if __name__ == "__main__":
    unittest.main()
