"""Structural rules of the code, checked against the code itself (see docs/ARCHITECTURE.md)."""
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
SHARED = PACKAGE / "shared"
RUNNER = PACKAGE / "runner"
EVALS = PACKAGE / "evals"

# The client's units, bottom to top. A unit imports, at load time, only units of a
# lower rank; only type annotations (``TYPE_CHECKING``) look upwards.
RANKS = {
    "config": 0, "protocol": 0,
    "tools": 1, "llm": 1,
    "context": 2,
    "core": 3,
    "guardrails": 4,
    "tool_execution": 5,
    "extensions": 6, "integration": 6,
    "prompt": 7,
    "query_engine": 8,
    "agent_core": 9,
    "session": 10,
    "ui": 11,
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


def _mimir_code() -> list[Path]:
    """Every module of MIMIR's own: the client, the servers, shared, the runner and the
    evaluations — not a benchmark's task fixtures, which are the agent's workspaces."""
    fixtures = EVALS / "scientific" / "tasks"
    return [p for root in (CLIENT, SERVERS, SHARED, RUNNER, EVALS) for p in _python_files(root)
            if fixtures not in p.parents]


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

    def test_nothing_the_agent_runs_on_imports_what_measures_it(self) -> None:
        """The runner and the evaluations drive the agent; the agent never knows them."""
        offenders = [str(p.relative_to(PACKAGE))
                     for p in _python_files(CLIENT) + _python_files(SERVERS) + _python_files(SHARED)
                     for _n, target, _a in _imports(p, top_level_only=False)
                     if target.startswith(("mimir.runner", "mimir.evals"))]
        self.assertEqual(offenders, [])

    def test_the_shared_code_imports_neither_the_client_nor_the_servers(self) -> None:
        offenders = [str(p.relative_to(PACKAGE)) for p in _python_files(SHARED)
                     for _n, target, _a in _imports(p, top_level_only=False)
                     if target.startswith(("mimir.client", "mimir.servers"))]
        self.assertEqual(offenders, [])

    def test_the_client_imports_no_server(self) -> None:
        """What the client reads of a server's files is a contract in ``shared``."""
        offenders = {str(p.relative_to(PACKAGE)) for p in _python_files(CLIENT)
                     for _n, target, _a in _imports(p, top_level_only=False)
                     if target.startswith("mimir.servers")}
        self.assertEqual(sorted(offenders), [])

    def test_the_execution_context_is_written_only_by_its_owners(self) -> None:
        """``execution_context.WRITERS`` says who writes each record; the code agrees."""
        from mimir.client.context.execution_context import WRITERS
        offenders = sorted(
            f"{path.relative_to(PACKAGE)}: {record}"
            for path in _python_files(CLIENT) if (CLIENT / "context") not in path.parents
            for record in _context_writes(path)
            if _writer_unit(path) not in WRITERS[record])
        self.assertEqual(offenders, [])

    def test_every_record_has_its_read_only_view(self) -> None:
        """``context_view`` restates each record's fields, no more and no fewer."""
        import dataclasses
        from mimir.client.context import context_view as views
        from mimir.client.context import execution_context as ec
        pairs = {ec.LoopControlState: views.LoopControlView, ec.EditFailures: views.EditFailuresView,
                 ec.Discovery: views.DiscoveryView, ec.Edits: views.EditsView,
                 ec.Checks: views.ChecksView, ec.Execution: views.ExecutionView,
                 ec.Environment: views.EnvironmentView, ec.Denials: views.DenialsView,
                 ec.PlanState: views.PlanStateView, ec.NudgeState: views.NudgeStateView,
                 ec.Bookkeeping: views.BookkeepingView, ec.ExecutionContext: views.ContextView}
        for record, view in pairs.items():
            with self.subTest(record=record.__name__):
                members = {n for n, v in vars(view).items() if isinstance(v, property)}
                self.assertEqual(members, {f.name for f in dataclasses.fields(record)})

    def test_the_guardrails_name_no_server_domain(self) -> None:
        """What a server forbids reaches them as a seal or a capability, never by its name.

        ``workspace`` and ``external`` are also the client's own words (its workspace root,
        the ``EXTERNAL_FETCH`` capability), so they are not read as domains.
        """
        domains = {p.name for p in SERVERS.iterdir()
                   if p.is_dir() and not p.name.startswith("_")} - {"workspace", "external"}
        offenders = set()
        for path in _python_files(CLIENT / "guardrails"):
            tree = ast.parse(path.read_text())
            words: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Constant) and isinstance(node.value, str):
                    words |= set(re.findall(r"[a-z]+", node.value.lower()))
                elif isinstance(node, ast.Name):
                    words |= set(node.id.lower().split("_"))
                elif isinstance(node, ast.Attribute):
                    words |= set(node.attr.lower().split("_"))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    words |= set(node.name.lower().split("_"))
            offenders |= {f"{path.relative_to(PACKAGE)}: {d}" for d in words & domains}
        self.assertEqual(sorted(offenders), [])

    def test_only_the_sub_agent_side_of_the_servers_imports_the_client(self) -> None:
        offenders = set()
        for path in _python_files(SERVERS):
            domain = path.relative_to(SERVERS).parts[0]
            for _n, target, _a in _imports(path, top_level_only=False):
                if target.startswith("mimir.client") and domain not in SERVERS_USING_THE_CLIENT:
                    offenders.add(str(path.relative_to(PACKAGE)))
        self.assertEqual(sorted(offenders), [])


_MUTATORS = frozenset({"add", "update", "append", "pop", "clear", "discard", "extend",
                       "setdefault", "remove", "insert", "popitem", "difference_update",
                       "intersection_update"})
# How the code names an execution context it was handed.
_CONTEXT_NAMES = frozenset({"execution_context", "ec", "ctx", "context"})


def _context_writes(path: Path) -> set[str]:
    """The records (and top-level fields) of an execution context this file writes.

    A write is an assignment, an augmented assignment, an item store or delete, or a
    mutating method call on ``<context>.<record>.<field>`` (or ``<context>.<field>``).
    """
    from mimir.client.context.execution_context import WRITERS
    tree = ast.parse(path.read_text())
    parent = {child: node for node in ast.walk(tree) for child in ast.iter_child_nodes(node)}

    def written(node: ast.Attribute) -> bool:
        up = parent.get(node)
        return (isinstance(node.ctx, (ast.Store, ast.Del))
                or (isinstance(up, ast.AugAssign) and up.target is node)
                or (isinstance(up, ast.Subscript) and up.value is node
                    and isinstance(up.ctx, (ast.Store, ast.Del)))
                or (isinstance(up, ast.Attribute) and up.attr in _MUTATORS
                    and isinstance(parent.get(up), ast.Call) and parent[up].func is up))

    out = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Attribute) and node.attr in WRITERS):
            continue
        base = node.value
        if not ((isinstance(base, ast.Name) and base.id in _CONTEXT_NAMES)
                or (isinstance(base, ast.Attribute) and base.attr in _CONTEXT_NAMES)):
            continue
        field_access = parent.get(node)
        if isinstance(field_access, ast.Attribute) and field_access.value is node:
            if written(field_access):
                out.add(node.attr)
        elif written(node):
            out.add(node.attr)
    return out


def _writer_unit(path: Path) -> str:
    parts = path.relative_to(CLIENT).parts
    if parts[0] == "guardrails" and len(parts) > 2:
        return f"guardrails/{parts[1]}"
    return parts[0].removesuffix(".py")


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
    ``module._name``, in every module of MIMIR's own (``_mimir_code``)."""

    def test_no_private_name_crosses_a_module(self) -> None:
        offenders = []
        for path in _mimir_code():
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
        for path in _mimir_code():
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


class ProseTests(unittest.TestCase):
    """A docstring says what the code is in a few lines; the reasoning lives in the
    .md docs. An MCP tool's docstring is exempt: it is what the model reads."""

    FUNCTION_MAX, MODULE_MAX = 6, 8

    def test_docstrings_stay_short(self) -> None:
        offenders = []
        for path in _python_files(PACKAGE):
            tree = ast.parse(path.read_text())
            if ast.get_docstring(tree, clean=False) is not None:
                doc = tree.body[0]
                if doc.end_lineno - doc.lineno + 1 > self.MODULE_MAX:
                    offenders.append(f"{path.relative_to(PACKAGE)}: module")
            for node in ast.walk(tree):
                if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    continue
                if ast.get_docstring(node, clean=False) is None:
                    continue
                if any("tool" in ast.unparse(d) for d in node.decorator_list):
                    continue
                doc = node.body[0]
                if doc.end_lineno - doc.lineno + 1 > self.FUNCTION_MAX:
                    offenders.append(f"{path.relative_to(PACKAGE)}: {node.name}")
        self.assertEqual(offenders, [])


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
        for path in _mimir_code():
            if (CLIENT / "query_engine" / "dispatch") in path.parents:
                continue
            names = {n.id for n in ast.walk(ast.parse(path.read_text()))
                     if isinstance(n, ast.Name)}
            names |= {n.attr for n in ast.walk(ast.parse(path.read_text()))
                      if isinstance(n, ast.Attribute)}
            offenders += [f"{path.relative_to(PACKAGE)}: {g}" for g in self.GUARDS if g in names]
        self.assertEqual(offenders, [])

    def test_every_watcher_is_the_engines_and_polls_through_run_tool(self) -> None:
        """The conversation's, the sub-agent's and the turn that waits a run out alike."""
        from mimir.client.query_engine.jobs import RunWatcher, background
        from mimir.client.ui.ws.worker.jobs import JobWatcher
        from mimir.servers.agent_state.subagent.watcher import ChildWatcher
        watchers = RunWatcher.__subclasses__()
        self.assertTrue({JobWatcher, ChildWatcher} <= set(watchers))
        waits = _names_in(background.await_background_job)
        self.assertIn("follow", waits, "the in-turn wait polls on its own")
        for cls in watchers:
            with self.subTest(watcher=cls.__qualname__):
                called = _names_in(cls.tool_runner)
                self.assertIn("run_tool", called)
                self.assertNotIn("dispatch_tool_calls", called)


def _names_in(function) -> set[str]:
    """Every attribute, name and string a function's source mentions."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    return ({n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
            | {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
            | {n.value for n in ast.walk(tree)
               if isinstance(n, ast.Constant) and isinstance(n.value, str)})


def _port_members(port: type = AgentPort) -> tuple[set[str], set[str]]:
    """A port's declared attributes, and its methods, its bricks' included."""
    attributes: set[str] = set()
    methods: set[str] = set()
    for brick in port.__mro__:
        if brick is object or not getattr(brick, "_is_protocol", False):
            continue
        attributes |= set(vars(brick).get("__annotations__", {}))
        methods |= {name for name, value in vars(brick).items()
                    if not name.startswith("_") and callable(value)}
    return attributes, methods


def _is_agent(node: ast.AST) -> bool:
    """``agent``, ``self.agent``, ``call.agent``, ``self._agent``…: how the client names one."""
    names = ("agent", "_agent")
    return ((isinstance(node, ast.Name) and node.id in names)
            or (isinstance(node, ast.Attribute) and node.attr in names))


class PortsByNeedTests(unittest.TestCase):
    """Each layer is handed the port of its need, never the whole agent."""

    LAYERS = {"guardrails/nudges": "NudgeAgent", "guardrails/observations": "ObserverAgent",
              "guardrails/policy": "PolicyAgent", "tool_execution": "ExecutorAgent",
              "integration": "ConnectorAgent", "query_engine/dispatch": "DispatchAgent",
              "query_engine/jobs": "JobsAgent"}

    def test_a_layer_names_only_its_own_port(self) -> None:
        from mimir.client.core import ports
        others = {name for name in vars(ports) if name.endswith("Agent") or name == "AgentPort"}
        offenders = []
        for layer, port in self.LAYERS.items():
            for path in _python_files(CLIENT / layer):
                named = {n.id for n in ast.walk(ast.parse(path.read_text()))
                         if isinstance(n, ast.Name)} & others
                offenders += [f"{path.relative_to(PACKAGE)}: {n}" for n in named - {port}]
        self.assertEqual(sorted(offenders), [])

    def test_a_person_is_lent_only_through_attach(self) -> None:
        """``agent.human = …`` is read by nobody: the agent asks its host's person."""
        offenders = []
        for path in _python_files(PACKAGE):
            if "tests" in path.relative_to(PACKAGE).parts:
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Attribute) and node.attr == "human"
                        and isinstance(node.ctx, ast.Store) and _is_agent(node.value)):
                    offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_no_layer_reaches_around_its_port(self) -> None:
        """``getattr(agent, "name")`` reads what no port declares, unseen by the checker."""
        offenders = []
        for path in _python_files(CLIENT):
            if path.name == "agent_core.py":
                continue
            for node in ast.walk(ast.parse(path.read_text())):
                if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                        and node.func.id == "getattr" and len(node.args) >= 2
                        and _is_agent(node.args[0])
                        and isinstance(node.args[1], ast.Constant)):
                    offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
        self.assertEqual(offenders, [])

    def test_every_member_of_a_port_is_used(self) -> None:
        """A member nothing reads is not part of what the layers need: it leaves the port."""
        attributes, methods = _port_members(AgentPort)
        used: set[str] = set()
        for path in _python_files(CLIENT):
            if path.name in ("agent_core.py", "ports.py"):
                continue
            used |= {node.attr for node in ast.walk(ast.parse(path.read_text()))
                     if isinstance(node, ast.Attribute) and _is_agent(node.value)}
        self.assertEqual(sorted((attributes | methods) - used), [])

    def test_every_port_is_made_of_bricks_the_agent_port_holds(self) -> None:
        from mimir.client.core import ports
        all_attributes, all_methods = _port_members(AgentPort)
        for port in self.LAYERS.values():
            with self.subTest(port=port):
                attributes, methods = _port_members(getattr(ports, port))
                self.assertLessEqual(attributes, all_attributes)
                self.assertLessEqual(methods, all_methods)


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
