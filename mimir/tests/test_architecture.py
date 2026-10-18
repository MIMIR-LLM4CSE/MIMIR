"""Structural rules of the code, checked against the code itself (see ARCHITECTURE.md)."""
from __future__ import annotations

import ast
import inspect
import io
import re
import tempfile
import textwrap
import tokenize
import unittest
from pathlib import Path

from mimir.client.agent_core import MimirAgent
from mimir.client.core.ports import AgentPort
from mimir.tests.support.state import pin_state_dir

PACKAGE = Path(__file__).resolve().parents[1]
CLIENT = PACKAGE / "client"
SERVERS = PACKAGE / "servers"

# The client's units, bottom to top. A unit imports, at load time, only units of a
# lower rank; a function may import upwards lazily (the back-edges ARCHITECTURE.md lists).
RANKS = {
    "config": 0, "protocol": 0, "event_sink": 0, "human_pause": 0,
    "context": 1,
    "core": 2,
    "guardrails": 3,
    "tool_execution": 4,
    "extensions": 5, "integration": 5,
    "prompt": 6,
    "query_engine": 7,
    "agent_core": 8,
    "session": 9,
    "ui": 10,
}

# Servers that may import the client: the sub-agent server builds a MimirAgent, and
# these read the client's settings or vocabulary.
SERVERS_USING_THE_CLIENT = {"agent_state", "interaction"}

# How history reads in a comment; the code says what it does now (see CHANGELOG.md).
HISTORICAL = re.compile(
    r"this used to|it used to|which used to|used to be\b|split out of|moved here|"
    r"formerly|historically|the former |legacy", re.IGNORECASE)


def _python_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.py") if "__pycache__" not in p.parts)


def _module_name(path: Path) -> str:
    rel = path.relative_to(PACKAGE.parent).with_suffix("")
    parts = list(rel.parts)
    if parts[-1] == "__init__":
        parts.pop()
    return ".".join(parts)


def _resolve(path: Path, node: ast.ImportFrom) -> str:
    """The absolute module an ``ImportFrom`` names."""
    if node.level == 0:
        return node.module or ""
    package = _module_name(path).split(".")
    if path.name != "__init__.py":
        package = package[:-1]
    base = package[:len(package) - (node.level - 1)]
    return ".".join(base + ([node.module] if node.module else []))


def _client_unit(module: str) -> str | None:
    parts = module.split(".")
    if parts[:2] != ["mimir", "client"] or len(parts) < 3:
        return None
    return parts[2]


def _imports(path: Path, *, top_level_only: bool):
    tree = ast.parse(path.read_text(), filename=str(path))
    nodes = tree.body if top_level_only else list(ast.walk(tree))
    for node in nodes:
        if isinstance(node, ast.ImportFrom):
            target = _resolve(path, node)
            for alias in node.names:
                yield node, target, alias.name
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name, None


class LayerTests(unittest.TestCase):
    def test_every_client_unit_has_a_rank(self) -> None:
        units = {_client_unit(_module_name(p)) for p in _python_files(CLIENT)}
        units.discard(None)
        units.discard("__init__")
        self.assertEqual(sorted(units - set(RANKS)), [])

    def test_client_units_import_only_downwards_at_load_time(self) -> None:
        upward = []
        for path in _python_files(CLIENT):
            source_unit = _client_unit(_module_name(path))
            if source_unit not in RANKS:
                continue
            for _node, target, name in _imports(path, top_level_only=True):
                candidates = [target] + ([f"{target}.{name}"] if name else [])
                for module in candidates:
                    unit = _client_unit(module)
                    if unit in RANKS and unit != source_unit and RANKS[unit] > RANKS[source_unit]:
                        upward.append(f"{path.relative_to(PACKAGE)}: {source_unit} -> {unit}")
        self.assertEqual(upward, [])

    def test_the_shared_server_code_does_not_import_the_client(self) -> None:
        offenders = [str(p.relative_to(PACKAGE)) for p in _python_files(SERVERS / "_shared")
                     for _n, target, _a in _imports(p, top_level_only=False)
                     if target.startswith("mimir.client")]
        self.assertEqual(offenders, [])

    def test_only_the_sub_agent_side_of_the_servers_imports_the_client(self) -> None:
        offenders = set()
        for path in _python_files(SERVERS):
            domain = path.relative_to(SERVERS).parts[0]
            for _n, target, _a in _imports(path, top_level_only=False):
                if target.startswith("mimir.client") and domain not in SERVERS_USING_THE_CLIENT:
                    offenders.add(str(path.relative_to(PACKAGE)))
        self.assertEqual(sorted(offenders), [])


def _module_aliases(path: Path, tree: ast.Module) -> set[str]:
    """The names this file binds to modules of the package (``from . import store``)."""
    aliases = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            target = PACKAGE.parent / _resolve(path, node).replace(".", "/")
            for alias in node.names:
                if ((target / f"{alias.name}.py").exists()
                        or (target / alias.name / "__init__.py").exists()):
                    aliases.add(alias.asname or alias.name)
    return aliases


class PrivateNameTests(unittest.TestCase):
    """A name another module uses is public: no ``from x import _name``, no
    ``module._name``, in the client and the servers alike."""

    def test_no_private_name_crosses_a_module(self) -> None:
        offenders = []
        for path in _python_files(CLIENT) + _python_files(SERVERS):
            tree = ast.parse(path.read_text())
            for _n, target, name in _imports(path, top_level_only=False):
                if name and name.startswith("_") and not name.startswith("__"):
                    offenders.append(f"{path.relative_to(PACKAGE)}: {target}.{name}")
            aliases = _module_aliases(path, tree)
            for node in ast.walk(tree):
                if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                        and node.value.id in aliases and node.attr.startswith("_")
                        and not node.attr.startswith("__")):
                    offenders.append(f"{path.relative_to(PACKAGE)}: "
                                     f"{node.value.id}.{node.attr}")
        self.assertEqual(offenders, [])


class CommentTests(unittest.TestCase):
    def test_no_comment_or_docstring_narrates_history(self) -> None:
        found = []
        for path in _python_files(CLIENT) + _python_files(SERVERS):
            text = path.read_text()
            pieces = [tok.string for tok in tokenize.generate_tokens(io.StringIO(text).readline)
                      if tok.type == tokenize.COMMENT]
            for node in ast.walk(ast.parse(text)):
                if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                     ast.AsyncFunctionDef)):
                    doc = ast.get_docstring(node)
                    if doc:
                        pieces.append(doc)
            for piece in pieces:
                match = HISTORICAL.search(piece)
                if match:
                    found.append(f"{path.relative_to(PACKAGE)}: {match.group(0)!r}")
        self.assertEqual(found, [])


class TestHygieneTests(unittest.TestCase):
    """Tests go through public construction and behaviour, never the source text."""

    def _test_files(self) -> list[Path]:
        here = Path(__file__).resolve()
        return [p for p in sorted((PACKAGE / "tests").glob("test_*.py")) if p != here]

    def test_no_test_reads_the_source(self) -> None:
        offenders = [p.name for p in self._test_files() if "getsource" in p.read_text()]
        self.assertEqual(offenders, [])

    def test_no_test_builds_an_object_around_its_constructor(self) -> None:
        offenders = [p.name for p in self._test_files() if "__new__(" in p.read_text()]
        self.assertEqual(offenders, [])


class ComponentTests(unittest.TestCase):
    """A WebSocket component owns its state: it reads no attribute of a host it does not
    define itself (or inherit from a class of the package). What it reads of another
    object it is handed at construction."""

    def _classes(self) -> dict[str, tuple[str, list[str], set[str], set[str]]]:
        classes = {}
        for path in _python_files(CLIENT / "ui" / "ws"):
            for node in ast.walk(ast.parse(path.read_text())):
                if not isinstance(node, ast.ClassDef):
                    continue
                defined: set[str] = set()
                read: set[str] = set()
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        defined.add(item.name)
                    elif isinstance(item, ast.Assign):
                        defined |= {t.id for t in item.targets if isinstance(t, ast.Name)}
                    elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
                        defined.add(item.target.id)
                for n in ast.walk(node):
                    if (isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name)
                            and n.value.id == "self"):
                        (defined if isinstance(n.ctx, ast.Store) else read).add(n.attr)
                bases = [ast.unparse(b) for b in node.bases]
                classes[node.name] = (path.name, bases, defined, read)
        return classes

    def test_no_class_reads_its_host(self) -> None:
        classes = self._classes()

        def defined(name: str) -> set[str]:
            _, bases, own, _ = classes[name]
            return own.union(*(defined(b) for b in bases if b in classes))

        offenders = []
        for name, (file, bases, _, read) in sorted(classes.items()):
            # A class built on something outside the package inherits what it reads.
            if any(b not in classes for b in bases):
                continue
            missing = sorted(read - defined(name))
            if missing:
                offenders.append(f"{file}:{name} reads {missing}")
        self.assertEqual(offenders, [])


class JobPollingTests(unittest.TestCase):
    """Only the dispatcher refuses a status poll on a run a watcher holds.

    The watchers (the WebSocket one, the in-turn wait, a sub-agent's own) poll through
    ``run_tool``, never through the dispatcher: gated by those guards, they would
    refuse their own probes and no wake would ever come.
    """

    GUARDS = ("_asks_whether_a_watched_run_is_done", "_waits_by_blocking_the_turn")

    def test_the_guards_are_named_only_in_the_dispatcher(self) -> None:
        offenders = []
        for path in _python_files(CLIENT) + _python_files(SERVERS):
            if (CLIENT / "query_engine" / "dispatch") in path.parents:
                continue
            names = {n.id for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.Attribute)}
            offenders += [f"{path.relative_to(PACKAGE)}: {g}" for g in self.GUARDS if g in names]
        self.assertEqual(offenders, [])

    def test_every_watcher_polls_through_run_tool(self) -> None:
        from mimir.client.query_engine.jobs import background
        from mimir.client.ui.ws.worker.jobs import JobWatcher
        from mimir.servers.agent_state.child_jobs import ChildWatcher
        for probe in (background.await_background_job, JobWatcher._poll_to_completion,
                      ChildWatcher._watch):
            with self.subTest(probe=probe.__qualname__):
                tree = ast.parse(textwrap.dedent(inspect.getsource(probe)))
                called = {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
                called |= {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
                self.assertIn("run_tool", called)
                self.assertNotIn("dispatch_tool_calls", called)


def _port_members() -> tuple[set[str], set[str]]:
    """AgentPort's declared attributes, and its methods."""
    attributes = set(AgentPort.__annotations__)
    methods = {name for name, value in vars(AgentPort).items()
               if not name.startswith("__") and callable(value)}
    return attributes, methods


class AgentPortConformanceTests(unittest.TestCase):
    """MimirAgent is what every layer is handed as an AgentPort: it must be one.

    mypy proves it statically (``agent_core``); these check it at run time."""

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
