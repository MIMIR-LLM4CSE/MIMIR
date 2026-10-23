"""What a workspace's build files declare: its libraries and their links, its include
directories, the Python extension modules it builds and from which sources, and the
shared libraries it produces."""
from __future__ import annotations

import posixpath
import re
from dataclasses import dataclass, field

SOURCE_EXT = (".c", ".cc", ".cpp", ".cxx", ".cu", ".h", ".hh", ".hpp", ".hxx", ".cuh",
              ".f", ".for", ".ftn", ".f77", ".f90", ".f95", ".f03", ".f08", ".fpp",
              ".pyx", ".pxd", ".pyf")


@dataclass
class Target:
    """A library the build declares: its directory, sources and the targets it links."""
    name: str
    dir: str
    sources: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)


@dataclass
class Build:
    """Everything the build files say, merged across the workspace."""
    targets: dict[str, Target] = field(default_factory=dict)
    include_dirs: set[str] = field(default_factory=set)
    pymods: dict[str, list[str]] = field(default_factory=dict)
    aliases: dict[str, str] = field(default_factory=dict)
    wrapped_by: dict[str, str] = field(default_factory=dict)
    texts: list[str] = field(default_factory=list)

    def names_package(self, name: str) -> bool:
        """Whether *name* appears as a word in any build file: a package the build installs."""
        pat = re.compile(rf"\b{re.escape(name)}\b")
        return any(pat.search(t) for t in self.texts)


_CMAKE_COMMENT_RE = re.compile(r"#[^\n]*")
_CMAKE_CALL_RE = re.compile(r"\b(\w+)\s*\(([^()]*(?:\([^()]*\)[^()]*)*)\)", re.S)
_CMAKE_KEYWORDS = frozenset({
    "STATIC", "SHARED", "MODULE", "INTERFACE", "OBJECT", "PUBLIC", "PRIVATE", "EXCLUDE_FROM_ALL",
    "WITH_SOABI", "NO_EXTRA", "THIN_LTO", "OPT_SIZE", "NB_STATIC", "NB_SHARED", "STABLE_ABI",
    "SYSTEM", "BEFORE", "AFTER", "LINK_PUBLIC", "LINK_PRIVATE"})


def _source(base: str, token: str) -> str | None:
    """A source path a build file names, relative to the workspace; None for any other token."""
    token = token.strip("'\"")
    for var in ("${CMAKE_CURRENT_SOURCE_DIR}/", "${CMAKE_CURRENT_LIST_DIR}/", "${PROJECT_SOURCE_DIR}/",
                "${CMAKE_SOURCE_DIR}/"):
        if token.startswith(var):
            token = token[len(var):]
            base = "" if var in ("${PROJECT_SOURCE_DIR}/", "${CMAKE_SOURCE_DIR}/") else base
    if "$" in token or not token.lower().endswith(SOURCE_EXT):
        return None
    return posixpath.normpath(posixpath.join(base, token))


def read_cmake(rel_dir: str, text: str, build: Build) -> None:
    """Add what one CMakeLists.txt (or .cmake file) in *rel_dir* declares to *build*."""
    build.texts.append(text)
    text = _CMAKE_COMMENT_RE.sub("", text)
    variables: dict[str, list[str]] = {}
    for m in _CMAKE_CALL_RE.finditer(text):
        cmd, args = m.group(1).lower(), m.group(2).split()
        if not args:
            continue
        if cmd == "set":
            variables[args[0]] = args[1:]
            continue
        expanded = [x for a in args for x in (
            variables.get(a[2:-1], [a]) if a.startswith("${") and a.endswith("}") else [a])]
        if not expanded:
            continue
        name, rest = expanded[0], expanded[1:]
        if cmd == "add_library" and "ALIAS" in rest and rest.index("ALIAS") + 1 < len(rest):
            build.aliases[name] = rest[rest.index("ALIAS") + 1]
        elif cmd in ("add_library", "add_executable") and "IMPORTED" not in rest:
            target = build.targets.setdefault(name, Target(name, rel_dir))
            target.dir = rel_dir
            target.sources += [s for s in (_source(rel_dir, a) for a in rest) if s]
        elif cmd == "target_sources":
            target = build.targets.setdefault(name, Target(name, rel_dir))
            target.sources += [s for s in (_source(rel_dir, a) for a in rest) if s]
        elif cmd == "target_link_libraries":
            target = build.targets.setdefault(name, Target(name, rel_dir))
            target.links += [a for a in rest if a not in _CMAKE_KEYWORDS]
        elif cmd in ("target_include_directories", "include_directories"):
            for a in (rest if cmd == "target_include_directories" else expanded):
                if a in _CMAKE_KEYWORDS:
                    continue
                d = _dir(rel_dir, a)
                if d is not None:
                    build.include_dirs.add(d)
        elif cmd in ("pybind11_add_module", "nanobind_add_module") or (
                cmd in ("python_add_library", "python3_add_library") and "MODULE" in rest):
            sources = [s for s in (_source(rel_dir, a) for a in rest) if s]
            build.pymods.setdefault(name, []).extend(sources)
            target = build.targets.setdefault(name, Target(name, rel_dir))
            target.sources += sources
    _read_wrapper_commands(rel_dir, text, build)


def _dir(base: str, token: str) -> str | None:
    """An include directory a build file names, relative to the workspace."""
    for var in ("${CMAKE_CURRENT_SOURCE_DIR}", "${CMAKE_CURRENT_LIST_DIR}"):
        token = token.replace(var, ".")
    for var in ("${PROJECT_SOURCE_DIR}", "${CMAKE_SOURCE_DIR}"):
        if token.startswith(var):
            token, base = token[len(var):].lstrip("/") or ".", ""
    if "$" in token or "<" in token or token.startswith("/"):
        return None
    return posixpath.normpath(posixpath.join(base, token))


# f2py and f90wrap, wherever a build calls them: a Makefile, a CMake custom command, Meson.
_WRAPPER_RE = re.compile(r"\b(f2py\d?|numpy\.f2py|f90wrap)\b[^\n]*?-m['\",\s]+(\w+)([^\n]*)")


def _read_wrapper_commands(rel_dir: str, text: str, build: Build) -> None:
    for m in _WRAPPER_RE.finditer(text):
        tokens = re.split(r"[\s,\[\]()]+", m.group(3))
        sources = [s for s in (_source(rel_dir, t) for t in tokens) if s]
        build.pymods.setdefault(m.group(2), []).extend(sources)
        build.wrapped_by[m.group(2)] = "f90wrap" if m.group(1) == "f90wrap" else "f2py"


_MESON_EXT_RE = re.compile(r"\.extension_module\(\s*'(\w+)'\s*,(.*?)\)\s*$", re.S | re.M)


def read_meson(rel_dir: str, text: str, build: Build) -> None:
    """Add a meson.build's Python extension modules and f2py calls to *build*."""
    build.texts.append(text)
    for m in _MESON_EXT_RE.finditer(text):
        sources = [s for s in (_source(rel_dir, q) for q in re.findall(r"'([^']+)'", m.group(2))) if s]
        build.pymods.setdefault(m.group(1), []).extend(sources)
    _read_wrapper_commands(rel_dir, text, build)


_EXTENSION_RE = re.compile(
    r"Extension\(\s*(?:name\s*=\s*)?['\"]([\w.]+)['\"]\s*,\s*(?:sources\s*=\s*)?\[([^\]]*)\]", re.S)


def read_setup_py(rel_dir: str, text: str, build: Build) -> None:
    """Add a setup.py's ``Extension`` modules to *build*, under their last dotted name."""
    build.texts.append(text)
    for m in _EXTENSION_RE.finditer(text):
        sources = [s for s in (_source(rel_dir, q) for q in re.findall(r"['\"]([^'\"]+)['\"]", m.group(2))) if s]
        build.pymods.setdefault(m.group(1).rsplit(".", 1)[-1], []).extend(sources)


def read_makefile(rel_dir: str, text: str, build: Build) -> None:
    """Add the f2py / f90wrap calls of a Makefile or shell script to *build*."""
    build.texts.append(text)
    _read_wrapper_commands(rel_dir, text, build)


def read_pyproject(text: str, build: Build) -> None:
    """Keep a pyproject.toml's text: the package names it installs are read from it."""
    build.texts.append(text)
