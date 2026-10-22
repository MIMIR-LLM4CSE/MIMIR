"""What the docs cite exists, and the tables that restate the code say what it says.

The changelogs are history and are not read. A path is a repository file or directory;
the names MIMIR writes at run time, and the examples, are listed here once. A table is
compared with the code it restates; the code is right.
"""
from __future__ import annotations

import ast
import os
import re
import subprocess
import unittest
from collections import defaultdict
from functools import cache
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

DOCS = sorted(p for p in [*REPO.glob("*.md"), *REPO.glob("docs/**/*.md")]
              if p.name != "CHANGELOG.md") + [
    REPO / "mimir" / "vscode-extension" / "README.md"]

# Not repository files: what MIMIR writes at run time (state, run directories, the
# proxy store), what the user owns, and the names examples invent.
NOT_IN_THE_REPOSITORY = frozenset({
    # written at run time
    "best.json", "build.json", "current.json", "embeddings.json", "machine.json",
    "meta.json", "metrics.json", "phase.json", "preferences.json", "probe.json",
    "ratchet.json", "registry.json", "subagent.json", "tree_at_launch.json",
    "watcher.json", "todo_list.md", "postrun.py", "MEMORY.md", "PLANS.md", "seals/proxy.json",
    "harnesses/", "plans/", "references/", "scaffolds/", "subagents/",
    # the user's
    ".vscode/settings.json",
    # examples
    "server_weather.py", "solver.py", "mesh.py", "boundary.py",
})

# Names a doc uses for an object rather than a module (``agent.tool_caps``, the setting
# ``mimir.wsUrl``): a module of the same name exists, and the citation is not about it.
INSTANCE_NAMES = frozenset({"agent", "turn", "session", "mimir", "extensions"})

_TICKED = re.compile(r"`([^`\n]+)`")
_PATH = re.compile(r"^[\w.\-/]+(?:\.py|\.ts|\.tsx|\.json|\.md|/)$")
_SYMBOL = re.compile(r"`([A-Za-z_]\w*)\.([A-Za-z_]\w*)(?:\(\))?`")
_EXTENSIONS = frozenset({"py", "json", "jsonl", "md", "log", "ts", "tsx", "sh", "toml", "txt"})


@cache
def _tracked() -> tuple[frozenset[str], frozenset[str], frozenset[str]]:
    """The repository's files, its directories (with a trailing slash), and its file names."""
    files = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True,
                           check=True).stdout.split()
    dirs = {"/".join(f.split("/")[:i]) + "/" for f in files for i in range(1, f.count("/") + 1)}
    return frozenset(files), frozenset(dirs), frozenset(f.rsplit("/", 1)[-1] for f in files)


def _path_exists(cited: str) -> bool:
    """A bare name is any tracked file of that name; a path is a suffix of a tracked one."""
    files, dirs, names = _tracked()
    if cited.endswith("/"):
        return any(d == cited or d.endswith("/" + cited) for d in dirs)
    if "/" in cited:
        return any(f == cited or f.endswith("/" + cited) for f in files)
    return cited in names


@cache
def _modules() -> dict[str, list[str]]:
    """Python module basename (a package by its directory) -> the source files of that name."""
    files, _dirs, _names = _tracked()
    out: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if not f.startswith("mimir/") or not f.endswith(".py") or "/tests/" in f:
            continue
        stem = f.rsplit("/", 1)[-1][:-3]
        out[f.rsplit("/", 2)[-2] if stem == "__init__" else stem].append(f)
    return out


@cache
def _defined(source: str) -> frozenset[str]:
    """Top-level definitions and imports of a module; a package also offers its submodules."""
    tree = ast.parse((REPO / source).read_text(encoding="utf-8"))
    out: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.add(node.name)
        elif isinstance(node, ast.Assign):
            out |= {t.id for t in node.targets if isinstance(t, ast.Name)}
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            out.add(node.target.id)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out |= {(a.asname or a.name).split(".")[0] for a in node.names}
    if source.endswith("__init__.py"):
        package = REPO / os.path.dirname(source)
        out |= {p.stem for p in package.iterdir() if p.suffix == ".py" or p.is_dir()}
    return frozenset(out)


class CitedPathsTests(unittest.TestCase):
    def test_every_cited_path_exists(self) -> None:
        missing = []
        for doc in DOCS:
            for match in _TICKED.finditer(doc.read_text(encoding="utf-8")):
                cited = match.group(1).strip()
                if (not _PATH.match(cited) or cited.startswith(("/", "~", ".mimir/"))
                        or cited in NOT_IN_THE_REPOSITORY):
                    continue
                if not _path_exists(cited):
                    missing.append(f"{doc.name}: {cited}")
        self.assertEqual(missing, [])


_LINK = re.compile(r"\]\(([^)\s#]+)(?:#[^)\s]*)?\)")


class LinkTargetsTests(unittest.TestCase):
    def test_every_relative_link_reaches_a_file(self) -> None:
        """A link is read from the doc's own directory, as GitHub and an editor read it; the
        changelog counts too, since a moved doc breaks its links as well."""
        broken = []
        for doc in [*DOCS, REPO / "CHANGELOG.md"]:
            for match in _LINK.finditer(doc.read_text(encoding="utf-8")):
                target = match.group(1)
                if re.match(r"^[a-z]+:", target):
                    continue
                if not (doc.parent / target).exists():
                    broken.append(f"{doc.relative_to(REPO)}: {target}")
        self.assertEqual(broken, [])


class CitedSymbolsTests(unittest.TestCase):
    def test_every_cited_module_name_is_defined_there(self) -> None:
        """Only a module whose basename is unique, so the citation means one file."""
        missing = []
        for doc in DOCS:
            for match in _SYMBOL.finditer(doc.read_text(encoding="utf-8")):
                module, name = match.groups()
                sources = _modules().get(module, [])
                if module in INSTANCE_NAMES or name in _EXTENSIONS or len(sources) != 1:
                    continue
                if name not in _defined(sources[0]):
                    missing.append(f"{doc.name}: {module}.{name} ({sources[0]})")
        self.assertEqual(missing, [])


def _table(doc: str, header: str) -> list[list[str]]:
    """The body rows of the markdown table under *header*, each a list of stripped cells."""
    text = (REPO / doc).read_text(encoding="utf-8")
    start = text.index(header)
    rows = []
    for line in text[start:].splitlines()[2:]:
        if not line.startswith("|"):
            break
        rows.append([cell.strip() for cell in line.strip().strip("|").split("|")])
    return rows


def _ticked(cell: str) -> str:
    return cell.strip().strip("`")


class TablesTests(unittest.TestCase):
    def test_the_capability_table_is_the_vocabulary(self) -> None:
        from mimir.shared import capabilities
        rows = _table("docs/PLUGINS_DETAILED.md", "| Group | Capability | Meaning |")
        documented = {_ticked(r[1]) for r in rows}
        levels = {name for name in dir(capabilities)
                  if getattr(capabilities, name) in capabilities.REVERSIBILITY_LEVELS}
        flags = {name for name, value in vars(capabilities).items()
                 if name.isupper() and isinstance(value, str) and name not in levels}
        self.assertEqual(sorted(documented ^ flags), [])

    def test_the_reversibility_table_is_the_levels(self) -> None:
        from mimir.shared import capabilities
        rows = _table("docs/PLUGINS_DETAILED.md", "| Level | Declare it when… | Effect |")
        documented = [getattr(capabilities, _ticked(r[0])) for r in rows]
        self.assertEqual(tuple(documented), capabilities.REVERSIBILITY_LEVELS)

    def test_the_nudge_table_is_the_cascade_and_its_tiers(self) -> None:
        """Name, layer, and whether it fires at each level (in agent mode)."""
        from mimir.client.guardrails.nudges.cascade import CORE_NUDGES
        from mimir.client.guardrails.nudges.tiers import GUIDANCE_BY_LEVEL_MODE
        rows = _table("docs/POLICY.md", "| Nudge | Layer | Fires on |")
        documented = {_ticked(r[0]): (r[1], r[3], r[4], r[5]) for r in rows}
        expected = {}
        for nudge in CORE_NUDGES:
            on = [nudge.layer == "verification"
                  or nudge.name in GUIDANCE_BY_LEVEL_MODE.get((level, "agent"), ())
                  for level in ("strict", "light", "off")]
            expected[nudge.name] = (nudge.layer, *("✅" if o else "❌" for o in on))
        self.assertEqual(documented, expected)

    def test_the_constants_table_says_their_values(self) -> None:
        from mimir.client.config import limits
        rows = _table("docs/CLIENT_DETAILED.md", "| Constant | Value | Bounds |")
        wrong = [f"{_ticked(r[0])}: doc {r[1]}, code {getattr(limits, _ticked(r[0]), None)}"
                 for r in rows if str(getattr(limits, _ticked(r[0]), None)) != r[1]]
        self.assertEqual(wrong, [])

    def test_the_trait_table_is_the_traits(self) -> None:
        from mimir.client.context import execution_context
        rows = _table("docs/POLICY.md", "| Trait | Meaning | Derived list |")
        documented = {getattr(execution_context, _ticked(r[0]), None) for r in rows}
        self.assertEqual(documented, set(execution_context.FIELD_TRAITS))

    def test_the_writer_table_names_every_writing_unit(self) -> None:
        """Each record of ``WRITERS`` has a row, and its row names each unit that writes it."""
        from mimir.client.context.execution_context import WRITERS
        query_engine = ("query_engine", "agent_loop", "dispatch", "turn", "plan_loop", "loop_base")
        rows = {_ticked(r[0]): r[1] for r in _table("docs/CLIENT_DETAILED.md",
                                                    "| Record | Written by | Read by |")}
        self.assertEqual(set(rows), set(WRITERS))
        unnamed = [f"{record}: {unit}" for record, units in WRITERS.items() for unit in units
                   if not any(word in rows[record] for word in
                              (query_engine if unit == "query_engine" else (unit.split("/")[-1],)))]
        self.assertEqual(unnamed, [])

    def test_the_layer_diagram_is_the_ranks_top_down(self) -> None:
        """Every unit is drawn or named as a foundation, and drawn units go down in rank."""
        from mimir.tests.test_architecture import RANKS
        text = (REPO / "docs/ARCHITECTURE.md").read_text(encoding="utf-8")
        start = text.index("```mermaid")
        diagram = text[start:text.index("```", start + 3)]
        drawn = [label.split("/")[0].removesuffix(".py")
                 for label in re.findall(r'\["([\w./]+?)(?:<br/>|")', diagram)]
        foundations = set(re.findall(r"`(\w+)`", text[:start].split("foundations", 1)[1]
                                     .split(")", 1)[0]))
        self.assertEqual(set(drawn) | foundations, set(RANKS))
        ranks = [RANKS[u] for u in dict.fromkeys(drawn)]
        self.assertEqual(ranks, sorted(ranks, reverse=True))


if __name__ == "__main__":
    unittest.main()
