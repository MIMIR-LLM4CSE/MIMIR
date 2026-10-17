"""Structural rules of the client, checked against the code itself."""
from __future__ import annotations

import inspect
import tempfile
import unittest

from mimir.client.agent_core import MimirAgent
from mimir.client.core.ports import AgentPort
from mimir.tests.support.state import pin_state_dir


def _port_members() -> tuple[set[str], set[str]]:
    """AgentPort's declared attributes, and its methods."""
    attributes = set(AgentPort.__annotations__)
    methods = {name for name, value in vars(AgentPort).items()
               if not name.startswith("__") and callable(value)}
    return attributes, methods


class AgentPortConformanceTests(unittest.TestCase):
    """MimirAgent is what every layer is handed as an AgentPort: it must be one."""

    def setUp(self) -> None:
        state = tempfile.TemporaryDirectory()
        self.addCleanup(state.cleanup)
        pin_state_dir(self, state.name)

    def test_every_declared_attribute_is_set_by_the_constructor(self) -> None:
        agent = MimirAgent()
        attributes, _ = _port_members()
        missing = sorted(name for name in attributes if not hasattr(agent, name))
        self.assertEqual(missing, [])

    def test_every_declared_method_exists_with_a_compatible_signature(self) -> None:
        _, methods = _port_members()
        for name in sorted(methods):
            with self.subTest(method=name):
                self.assertTrue(hasattr(MimirAgent, name), name)
                declared = inspect.signature(getattr(AgentPort, name))
                actual = inspect.signature(getattr(MimirAgent, name))
                required = [p.name for p in declared.parameters.values()
                            if p.name != "self" and p.default is inspect.Parameter.empty
                            and p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)]
                for param in required:
                    self.assertIn(param, actual.parameters, f"{name}({param})")


if __name__ == "__main__":
    unittest.main()
