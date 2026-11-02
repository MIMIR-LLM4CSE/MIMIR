"""A stack frame as the profilers report it, and where it lands: a workspace file, or the
package, library or runtime outside the workspace it belongs to."""
from __future__ import annotations

import os
import posixpath
import re
from typing import Callable, NamedTuple


FORTRAN_EXT = (".f", ".for", ".f77", ".f90", ".f95", ".f03", ".f08", ".fpp", ".cuf")
_GFORTRAN_RE = re.compile(r"^__(?P<mod>\w+?)_MOD_(?P<proc>\w+)$")
_INTEL_RE = re.compile(r"^(?P<mod>[a-z0-9]\w*?)_mp_(?P<proc>\w+?)_$")


def is_fortran(path: str) -> bool:
    return path.lower().endswith(FORTRAN_EXT)


def fortran_name(sym: str, where: str = "") -> str:
    """A Fortran symbol as its source names it: ``module::procedure`` for gfortran's
    ``__mod_MOD_proc`` and Intel's ``mod_mp_proc_``, ``proc`` for an external ``proc_``
    from a Fortran file, ``main program`` for ``MAIN__``; anything else unchanged."""
    m = _GFORTRAN_RE.match(sym) or _INTEL_RE.match(sym)
    if m:
        return f"{m.group('mod')}::{m.group('proc')}"
    if sym == "MAIN__":
        return "main program"
    if is_fortran(where) and re.fullmatch(r"[a-z]\w*_", sym):
        return sym[:-1]
    return sym


class Frame(NamedTuple):
    """One frame: its source file (or library), its function and line when known."""
    where: str
    func: str = ""
    line: int = 0


def resolver(root: str, cwd: str, files: list[str]) -> Callable[[str], str | None]:
    """A function giving a frame's file as a workspace path, or None outside it: absolute
    under *root*, relative to *cwd*, else the one workspace file it is a suffix of."""
    root = os.path.realpath(root)
    known = set(files)
    by_base: dict[str, list[str]] = {}
    for f in files:
        by_base.setdefault(posixpath.basename(f), []).append(f)
    seen: dict[str, str | None] = {}

    def resolve(path: str) -> str | None:
        if not path:
            return None
        if path not in seen:
            seen[path] = _resolve(path)
        return seen[path]

    def _resolve(path: str) -> str | None:
        full = os.path.realpath(path if os.path.isabs(path) else os.path.join(cwd, path))
        if full.startswith(root + os.sep):
            rel = os.path.relpath(full, root).replace(os.sep, "/")
            if rel in known or not known:
                return rel
        clean = path.replace(os.sep, "/").lstrip("./")
        hits = [f for f in by_base.get(posixpath.basename(clean), [])
                if f == clean or clean.endswith("/" + f) or f.endswith("/" + clean)]
        return hits[0] if len(hits) == 1 else None
    return resolve


def outside(path_or_lib: str) -> str:
    """What a frame outside the workspace belongs to: a package, a library, Python itself."""
    p = path_or_lib.replace(os.sep, "/")
    m = re.search(r"(?:site|dist)-packages/([^/]+)", p)
    if m:
        return m.group(1).split(".")[0]
    base = posixpath.basename(p)
    if re.match(r"lib.+\.so", base) or ".so" in base:
        return re.sub(r"^lib|\.so.*$|\.cpython.*$", "", base) or base
    if p.endswith(".py"):
        return "python"
    return base or "unknown"
