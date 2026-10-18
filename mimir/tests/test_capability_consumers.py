"""Drift guard: every declared capability has a live consumer in ``client/`` or the
shipped example packs (some capabilities are pack-facing).

A static AST scan: it catches a constant nothing references, not an unreachable consumer.
"""
from __future__ import annotations

import ast
import pathlib
import unittest

from mimir.client.tools import capabilities as caps

CLIENT_DIR = pathlib.Path(caps.__file__).resolve().parents[1]
CAP_DEF_FILE = pathlib.Path(caps.__file__).resolve()
# Version-controlled reference packs. They are never auto-loaded, but they are the
# canonical demonstration of the pack-facing vocabulary, so a capability they key off
# is consumed, not orphaned.
EXAMPLE_PACKS_DIR = CLIENT_DIR.parent / "examples" / "plugins"


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
    sources = list(CLIENT_DIR.rglob("*.py")) + list(EXAMPLE_PACKS_DIR.rglob("*.py"))
    for path in sources:
        if path.resolve() in exclude:
            continue
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id)
            elif isinstance(node, ast.Attribute):
                used.add(node.attr)  # e.g. ``caps.VALIDATE``
    return used


def _vocab_helper_bodies(used_by_client: set[str]) -> set[str]:
    """Identifiers used inside vocab-file helpers that client code calls.

    A capability consumed through such a helper (``CONTENT_WRITE`` via ``is_write``) counts;
    one referenced only by a helper nothing calls is still an orphan.
    """
    tree = ast.parse(CAP_DEF_FILE.read_text(), filename=str(CAP_DEF_FILE))
    used: set[str] = set()
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        # A private helper is followed when a public one the client uses calls it; the
        # two that matter (`_caps_from_meta`, `_derive_reversibility`) are reached from
        # `infer_tool_caps`, which the server manager calls for every connected tool.
        reachable = node.name in used_by_client or any(
            isinstance(n, ast.Name) and n.id == node.name
            for fn in tree.body
            if isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef))
            and fn.name in used_by_client
            for n in ast.walk(fn)
        )
        if not reachable:
            continue
        for inner in ast.walk(node):
            if isinstance(inner, ast.Name):
                used.add(inner.id)
            elif isinstance(inner, ast.Attribute):
                used.add(inner.attr)
    return used


def capabilities_without_consumer() -> list[str]:
    """Declared capabilities that no client module outside the vocab file references."""
    declared = capability_constants()
    referenced = referenced_identifiers(exclude={CAP_DEF_FILE})
    referenced |= _vocab_helper_bodies(referenced)
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
