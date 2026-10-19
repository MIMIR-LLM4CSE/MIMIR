"""What the docs cite exists: the paths they name and the ``module.name`` they point at.

The changelogs are history and are not read. A path is a repository file or directory;
the names MIMIR writes at run time, and the examples, are listed here once.
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

DOCS = sorted(p for p in REPO.glob("*.md") if p.name != "CHANGELOG.md") + [
    REPO / "mimir" / "vscode-extension" / "README.md"]

# Not repository files: what MIMIR writes at run time (state, run directories, the
# proxy store), what the user owns, and the names examples invent.
NOT_IN_THE_REPOSITORY = frozenset({
    # written at run time
    "best.json", "build.json", "current.json", "embeddings.json", "machine.json",
    "meta.json", "metrics.json", "phase.json", "preferences.json", "probe.json",
    "ratchet.json", "registry.json", "subagent.json", "tree_at_launch.json",
    "watcher.json", "todo_list.md", "postrun.py", "MEMORY.md", "PLANS.md",
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


if __name__ == "__main__":
    unittest.main()
