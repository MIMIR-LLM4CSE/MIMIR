"""References turned into edges: within each language (includes, imports, uses) and across
languages (bindings), each edge saying how sure it is. A reference that names nothing of
the workspace is counted as external or unresolved, never drawn."""
from __future__ import annotations

import posixpath
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field

from .build_files import Build
from .facts import (BIND_CALL, CALL, CIMPORT, CSYM, EMBED_IMPORT, EXTERN_HEADER, FBIND, FMODULE,
                    FPROC, IMPORT, INCLUDE, INTRINSIC_USE, LIBRARY, LIBRARY_CALL, PYF_WRAPS, PYMOD,
                    SUBMODULE, USE, Facts, Ref)

EXACT, BY_NAME = "exact", "by_name"
_STDLIB = frozenset(getattr(sys, "stdlib_module_names", ()))
_PY_EXT = (".py", ".pyx", ".pxd")


@dataclass
class Links:
    """The edges found, and what could not be placed."""
    edges: list[dict] = field(default_factory=list)
    external: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    unresolved: Counter = field(default_factory=Counter)


def _python_names(files: dict[str, Facts]) -> dict[str, list[str]]:
    """Every dotted module name a Python or Cython file answers to, mapped to its files."""
    packages = {posixpath.dirname(f) for f in files
                if posixpath.basename(f) in ("__init__.py", "__init__.pxd")}
    out: dict[str, list[str]] = defaultdict(list)
    for f in files:
        if not f.endswith(_PY_EXT):
            continue
        d, stem = posixpath.split(f)
        parts = [] if stem.startswith("__init__.") else [stem.rsplit(".", 1)[0]]
        while d in packages:
            d, pkg = posixpath.split(d)
            parts.insert(0, pkg)
        if parts:
            out[".".join(parts)].append(f)
    return out


class Linker:
    """One pass over the files' facts, with the indices the resolution needs."""

    def __init__(self, files: dict[str, Facts], build: Build) -> None:
        self.files, self.build, self.links = files, build, Links()
        self.by_base: dict[str, list[str]] = defaultdict(list)
        for f in files:
            self.by_base[posixpath.basename(f)].append(f)
        self.py = _python_names(files)
        # The package each Python file sits in, for its relative imports.
        self.package: dict[str, str] = {}
        for name, owners in self.py.items():
            for f in owners:
                init = posixpath.basename(f).startswith("__init__.")
                self.package[f] = name if init else name.rpartition(".")[0]
        self.exports: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
        for f, facts in files.items():
            for kind, name in facts.exports:
                self.exports[kind][name].append(f)
        for name, sources in build.pymods.items():
            self.exports[PYMOD][name].extend(s for s in sources if s in files)

    def add(self, src: str, dst: str, ref: Ref, kind: str, via: str = "", sure: str = EXACT) -> None:
        if src != dst:
            self.links.edges.append({"from": src, "to": dst, "line": ref.line, "kind": kind,
                                     "via": via, "sure": sure, "deferred": ref.deferred})

    def external(self, src: str, name: str) -> None:
        if name:
            self.links.external[src][name] += 1

    # ── within a language ────────────────────────────────────────────────────────
    def include(self, src: str, ref: Ref, kind: str = INCLUDE, via: str = "") -> None:
        name = ref.name
        here = posixpath.normpath(posixpath.join(posixpath.dirname(src), name))
        if here in self.files:
            return self.add(src, here, ref, kind, via)
        for d in sorted(self.build.include_dirs):
            cand = posixpath.normpath(posixpath.join(d, name))
            if cand in self.files:
                return self.add(src, cand, ref, kind, via)
        found = [f for f in self.by_base.get(posixpath.basename(name), [])
                 if f == name or f.endswith("/" + name)]
        if len(found) == 1:
            return self.add(src, found[0], ref, kind, via)
        if found:
            self.links.unresolved["ambiguous_include"] += 1
        elif "@" in name or "$" in name:
            self.links.unresolved["generated_include"] += 1
        else:
            self.external(src, name.split("/")[0].split(".")[0])

    def python(self, src: str, ref: Ref) -> None:
        if ref.computed:
            self.links.unresolved["computed_import"] += 1
            return
        base = ref.name
        if ref.level:
            pkg = self.package[src].split(".") if self.package.get(src) else []
            pkg = pkg[:len(pkg) - (ref.level - 1)] if ref.level > 1 else pkg
            base = ".".join(pkg + ([ref.name] if ref.name else []))
        candidates = [f"{base}.{n}" if base else n for n in ref.names] + ([base] if base else [])
        for cand in candidates:
            parts = cand.split(".")
            for cut in range(len(parts), 0, -1):
                owners = self.py.get(".".join(parts[:cut]))
                if owners:
                    pick = sorted(owners, key=lambda f: not f.endswith(".pxd" if ref.kind == CIMPORT else ".py"))
                    return self.add(src, pick[0], ref, IMPORT)
        if self._extension_module(src, ref, candidates):
            return
        if ref.level:
            self.links.unresolved["relative_import"] += 1
        elif base:
            self.external(src, base.split(".")[0])

    def _extension_module(self, src: str, ref: Ref, candidates: list[str]) -> bool:
        """An import of a module the workspace builds as a Python extension.

        Matched by the module's own name, under a package prefix only when the build names
        that package; a standard-library name never matches.
        """
        for cand in candidates:
            parts = cand.split(".")
            owners = self.exports[PYMOD].get(parts[-1])
            if not owners or parts[0] in _STDLIB:
                continue
            if len(parts) > 1 and not self.build.names_package(parts[0]):
                continue
            for dst in sorted(set(owners)):
                via = self.build.wrapped_by.get(parts[-1]) or (
                    "f2py" if dst.endswith(".pyf") or self.files[dst].lang == "fortran" else
                    "cython" if self.files[dst].lang == "cython" else "extension module")
                self.add(src, dst, ref, "binding", via)
            return True
        return False

    def fortran_use(self, src: str, ref: Ref) -> None:
        owners = self.exports[FMODULE].get(ref.name)
        if owners:
            self.add(src, sorted(owners)[0], ref, USE)
            if len(owners) > 1:
                self.links.unresolved["ambiguous_module"] += 1
        else:
            self.external(src, ref.name)

    # ── across languages ─────────────────────────────────────────────────────────
    def symbol(self, src: str, ref: Ref, via: str) -> bool:
        """A call to a C-linkage symbol defined in the workspace, in C or by a bind(c)."""
        owners = sorted(set(self.exports[CSYM].get(ref.name, []) + self.exports[FBIND].get(ref.name, [])))
        owners = [o for o in owners if o != src]
        for dst in owners:
            self.add(src, dst, ref, "binding", via, EXACT if len(owners) == 1 else BY_NAME)
        return bool(owners)

    def c_call(self, src: str, ref: Ref) -> None:
        """A C/C++ call into Fortran: by its bind(c) symbol, else by the compilers' naming."""
        if ref.name in self.exports[FBIND]:
            for dst in sorted(set(self.exports[FBIND][ref.name])):
                self.add(src, dst, ref, "binding", "bind(c)")
            return
        if ref.name in self.exports[CSYM]:
            return
        name = ref.name
        proc = name[:-1].lower() if name.endswith("_") else name.lower() if name.isupper() else ""
        for dst in sorted(set(self.exports[FPROC].get(proc, []))):
            self.add(src, dst, ref, "binding", "fortran naming", BY_NAME)

    def library(self, src: str, ref: Ref) -> None:
        """A library loaded by name: by-name edges to the sources of the target building it;
        a symbol called on it and found in those sources has made the edge exact already."""
        if ref.computed:
            self.links.unresolved["computed_library"] += 1
            return
        target = self.build.targets.get(ref.name)
        if target is None:
            self.external(src, ref.name)
        else:
            for dst in sorted(set(s for s in target.sources if s in self.files)):
                self.add(src, dst, ref, "binding", "library load", BY_NAME)

    def embedded(self, src: str, ref: Ref) -> None:
        parts = ref.name.split(".")
        for cut in range(len(parts), 0, -1):
            owners = self.py.get(".".join(parts[:cut]))
            if owners:
                via = "forpy" if self.files[src].lang == "fortran" else "embedded python"
                return self.add(src, sorted(owners)[0], ref, "binding", via)
        self.external(src, parts[0])

    def run(self) -> Links:
        for src, facts in sorted(self.files.items()):
            for ref in facts.refs:
                if ref.kind == INCLUDE:
                    self.include(src, ref)
                elif ref.kind in (IMPORT, CIMPORT):
                    self.python(src, ref)
                elif ref.kind in (USE, SUBMODULE):
                    if facts.lang == "pyf":
                        owners = self.exports[FMODULE].get(ref.name, [])
                        for dst in sorted(set(owners)):
                            self.add(src, dst, ref, "binding", "f2py")
                    else:
                        self.fortran_use(src, ref)
                elif ref.kind == INTRINSIC_USE:
                    self.external(src, ref.name)
                elif ref.kind == EXTERN_HEADER:
                    self.include(src, ref, "binding", "cython")
                elif ref.kind == EMBED_IMPORT:
                    self.embedded(src, ref)
                elif ref.kind == BIND_CALL:
                    self.symbol(src, ref, "bind(c)")
                elif ref.kind == CALL:
                    self.c_call(src, ref)
                elif ref.kind == LIBRARY_CALL:
                    self.symbol(src, ref, "ctypes")
                elif ref.kind == LIBRARY:
                    self.library(src, ref)
                elif ref.kind == PYF_WRAPS:
                    owners = self.exports[FPROC].get(ref.name, []) + self.exports[FBIND].get(ref.name, [])
                    for dst in sorted(set(owners)):
                        self.add(src, dst, ref, "binding", "f2py")
        self.links.edges = _dedupe(self.links.edges)
        return self.links


def _dedupe(edges: list[dict]) -> list[dict]:
    """One edge per (from, to, kind): its first line, and the surest of its readings."""
    best: dict[tuple[str, str, str], dict] = {}
    for e in edges:
        k = (e["from"], e["to"], e["kind"])
        cur = best.get(k)
        if cur is None:
            best[k] = dict(e)
        else:
            cur["line"] = min(cur["line"], e["line"])
            if e["sure"] == EXACT and cur["sure"] != EXACT:
                cur["sure"], cur["via"] = EXACT, e["via"]
            cur["deferred"] = cur["deferred"] and e["deferred"]
    return list(best.values())
