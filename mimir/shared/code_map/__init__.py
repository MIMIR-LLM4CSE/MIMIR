"""A workspace's code as a map: its Python, C/C++ and Fortran files, the dependencies between
them within a language and across languages, how they group into components and modules,
and what breaks the architecture's rules.

``scan`` reads the disk, or any tree through a ``read`` callable (a snapshot), and returns
a JSON-ready dict. Per-file parses are cached on the file's mtime and size.
"""
from __future__ import annotations

import hashlib
import json
import os
import posixpath
import re
import threading
from collections import Counter, defaultdict
from typing import Callable, Iterable

from . import build_files, layout
from .c_reader import read_c
from .facts import Facts
from .fortran_reader import FIXED_FORM, read_fortran, read_pyf
from .python_reader import read_cython, read_python
from .resolve import BY_NAME, Linker

_C = (".c",)
_CPP = (".cc", ".cpp", ".cxx", ".c++", ".cu", ".h", ".hh", ".hpp", ".hxx", ".h++", ".cuh", ".inl", ".tpp")
_FORTRAN = (".f", ".for", ".ftn", ".f77", ".f90", ".f95", ".f03", ".f08", ".fpp", ".inc", ".fi", ".h90", ".fh")
_CYTHON = (".pyx", ".pxd", ".pxi")
_SKIPPED_DIRS = frozenset({"node_modules", "__pycache__", "venv", "site-packages", "dist",
                           "build", "_build", "builds", "install"})
_BUILD_DIR_RE = re.compile(r"^(build|cmake-build)[-_.]", re.I)
MAX_FILES = 20000
# What a ``CODE_MAP`` tool is called with to answer the whole map rather than its summary.
FULL_ARGS = {"detail": "full"}

_CACHE: dict[tuple[str, str], tuple[float, int, Facts]] = {}
_LOCK = threading.Lock()


def language_of(rel: str) -> str | None:
    """Which reader a file goes to, from its extension; None for a file not read."""
    low = rel.lower()
    if low.endswith(".py"):
        return "python"
    if low.endswith(_CYTHON):
        return "cython"
    if low.endswith(".pyf"):
        return "pyf"
    if low.endswith(_FORTRAN):
        return "fortran"
    if low.endswith(_C):
        return "c"
    if low.endswith(_CPP):
        return "cpp"
    return None


def parse(rel: str, text: str) -> Facts:
    """One file's facts, by the reader its extension names."""
    lang = language_of(rel)
    if lang == "python":
        return read_python(text)
    if lang == "cython":
        return read_cython(text)
    if lang == "pyf":
        return read_pyf(text)
    if lang == "fortran":
        return read_fortran(text, fixed=posixpath.splitext(rel.lower())[1] in FIXED_FORM)
    return read_c(text, is_c=lang == "c")


def _is_build_file(name: str) -> bool:
    low = name.lower()
    return (name == "CMakeLists.txt" or low.endswith(".cmake") or name == "meson.build"
            or name == "setup.py" or name == "pyproject.toml" or low.endswith(".mk")
            or low in ("makefile", "gnumakefile"))


def walk(root: str) -> list[str]:
    """The workspace's source and build files, relative, skipping hidden directories and
    build trees (a directory holding a CMakeCache.txt, or named like one)."""
    out: list[str] = []
    for d, dirs, names in os.walk(root):
        if "CMakeCache.txt" in names and d != root:
            dirs[:] = []
            continue
        dirs[:] = sorted(x for x in dirs if not x.startswith(".") and x not in _SKIPPED_DIRS
                         and not _BUILD_DIR_RE.match(x) and not x.endswith(".egg-info"))
        rel_d = os.path.relpath(d, root).replace(os.sep, "/")
        for n in sorted(names):
            if language_of(n) or _is_build_file(n):
                out.append(n if rel_d == "." else f"{rel_d}/{n}")
                if len(out) >= MAX_FILES:
                    return out
    return out


def _disk_reader(root: str) -> Callable[[str], str | None]:
    def read(rel: str) -> str | None:
        try:
            with open(os.path.join(root, rel), encoding="utf-8", errors="replace") as fh:
                return fh.read()
        except OSError:
            return None
    return read


def _facts(root: str, rel: str, read: Callable[[str], str | None], cached: bool) -> Facts | None:
    st = None
    if cached:
        try:
            st = os.stat(os.path.join(root, rel))
        except OSError:
            return None
        with _LOCK:
            hit = _CACHE.get((root, rel))
        if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
            return hit[2]
    text = read(rel)
    if text is None:
        return None
    facts = parse(rel, text)
    if st is not None:
        with _LOCK:
            _CACHE[(root, rel)] = (st.st_mtime, st.st_size, facts)
    return facts


def _compile_commands(root: str, read: Callable[[str], str | None], rels: list[str]) -> set[str]:
    """The include directories of a ``compile_commands.json`` at the root or one level down."""
    cands = ["compile_commands.json"] + sorted(
        f"{d}/compile_commands.json" for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d))) if os.path.isdir(root) else []
    for cand in cands:
        text = read(cand)
        if not text:
            continue
        try:
            entries = json.loads(text)
        except ValueError:
            continue
        dirs: set[str] = set()
        for e in entries if isinstance(entries, list) else []:
            args = e.get("arguments") or (e.get("command") or "").split()
            base = e.get("directory") or root
            for i, a in enumerate(args):
                path = a[2:] if a.startswith("-I") and len(a) > 2 else (
                    args[i + 1] if a in ("-I", "-isystem") and i + 1 < len(args) else "")
                if path:
                    full = os.path.normpath(os.path.join(base, path))
                    rel = os.path.relpath(full, root).replace(os.sep, "/")
                    if not rel.startswith(".."):
                        dirs.add(rel)
        return dirs
    return set()


def _build(root: str, rels: list[str], read: Callable[[str], str | None]) -> build_files.Build:
    build = build_files.Build()
    for rel in rels:
        name = posixpath.basename(rel)
        if not _is_build_file(name):
            continue
        text = read(rel) or ""
        d = posixpath.dirname(rel)
        if name == "CMakeLists.txt" or name.lower().endswith(".cmake"):
            build_files.read_cmake(d, text, build)
        elif name == "meson.build":
            build_files.read_meson(d, text, build)
        elif name == "setup.py":
            build_files.read_setup_py(d, text, build)
        elif name == "pyproject.toml":
            build_files.read_pyproject(text, build)
        else:
            build_files.read_makefile(d, text, build)
    build.include_dirs |= _compile_commands(root, read, rels)
    return build


def scan(root: str, *, read: Callable[[str], str | None] | None = None,
         files: Iterable[str] | None = None) -> dict:
    """The map of the code under *root*: from the disk, or through *read* over *files*
    (relative paths) for a tree that is not on disk as it was."""
    cached = read is None
    reader = read or _disk_reader(root)
    rels = sorted(files) if files is not None else walk(root)
    facts: dict[str, Facts] = {}
    for rel in rels:
        if language_of(rel):
            f = _facts(root, rel, reader, cached)
            if f is not None:
                facts[rel] = f
    build = _build(root, rels, reader)
    links = Linker(facts, build).run()

    code = sorted(facts)
    src_root = layout.source_root(code)
    comp_of = layout.components(code, src_root)
    mod_of = layout.modules(code, comp_of)
    comps = sorted(set(comp_of.values()))
    declared = layout.declared_links(build, set(comps))
    langs = {f: facts[f].lang for f in code}
    for e in links.edges:
        e["breaks"] = layout.rule_break(e, comp_of, declared, langs, src_root)

    observed: dict[str, set[str]] = defaultdict(set)
    mod_graph: dict[str, set[str]] = defaultdict(set)
    for e in links.edges:
        if e["sure"] == BY_NAME:
            continue
        a, b = comp_of[e["from"]], comp_of[e["to"]]
        if a != b and not e["breaks"]:
            observed[a].add(b)
        ma, mb = mod_of[e["from"]], mod_of[e["to"]]
        if ma != mb and not e["deferred"]:
            mod_graph[ma].add(mb)
    # The build's word where it has one, what the code does everywhere: a Python example
    # that imports an extension module stacks above it though no CMake target says so.
    tier = layout.tiers(comps, {c: declared.get(c, set()) | observed.get(c, set()) for c in comps})
    cycles = layout.strongly_connected(sorted(set(mod_of.values())), mod_graph)

    external: dict[str, Counter] = defaultdict(Counter)
    for path, names in links.external.items():
        external[comp_of[path]].update(names)
    return {
        "source_root": src_root,
        "components": [{"id": c, "name": posixpath.basename(c) if c != "." else "(root)",
                        "tier": tier[c], "declared": sorted(declared.get(c, ())),
                        "tests": layout.is_tests(c),
                        "external": dict(external[c].most_common(12))} for c in comps],
        "modules": [{"id": m, "component": c} for m, c in sorted({mod_of[f]: comp_of[f] for f in code}.items())],
        "files": [{"id": f, "lang": langs[f], "lines": facts[f].lines, "module": mod_of[f],
                   "component": comp_of[f]} for f in code],
        "edges": sorted(links.edges, key=lambda e: (e["from"], e["line"], e["to"])),
        "cycles": cycles,
        "unresolved": dict(links.unresolved),
        "truncated": files is None and len(rels) >= MAX_FILES,
        "fingerprint": hashlib.sha1("\n".join(code).encode() + json.dumps(
            sorted((e["from"], e["to"], e["kind"]) for e in links.edges)).encode()).hexdigest()[:16],
    }


def summary(m: dict, *, limit: int = 12) -> dict:
    """The map in a few hundred tokens: per component its files and tier, the rule breaks,
    the cycles, the bindings by kind, the most coupled modules, what could not be placed."""
    files_in: Counter = Counter(f["component"] for f in m["files"])
    langs: Counter = Counter(f["lang"] for f in m["files"])
    mod_of = {f["id"]: f["module"] for f in m["files"]}
    degree: Counter = Counter()
    for e in m["edges"]:
        if mod_of[e["from"]] != mod_of[e["to"]]:
            degree[mod_of[e["from"]]] += 1
            degree[mod_of[e["to"]]] += 1
    return {
        "files": len(m["files"]), "modules": len(m["modules"]), "languages": dict(langs),
        "components": [{"id": c["id"], "tier": c["tier"], "files": files_in[c["id"]],
                        "declared": c["declared"]} for c in sorted(m["components"], key=lambda c: -c["tier"])],
        "rule_breaks": [{k: e[k] for k in ("from", "line", "to", "breaks")}
                        for e in m["edges"] if e["breaks"]][:limit],
        "module_cycles": m["cycles"][:limit],
        "bindings": dict(Counter(f"{e['via']} ({e['sure']})" for e in m["edges"] if e["kind"] == "binding")),
        "most_coupled_modules": dict(degree.most_common(limit)),
        "unresolved": m["unresolved"],
    }
