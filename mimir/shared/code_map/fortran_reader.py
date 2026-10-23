"""Fortran sources, free and fixed form, read for the modules they define and use, what
they include, the procedures they offer and the C symbols they call; and f2py signature
files for the Python module they define and the procedures they wrap."""
from __future__ import annotations

import re

from .facts import (BIND_CALL, EMBED_IMPORT, FBIND, FMODULE, FPROC, INCLUDE, INTRINSIC_USE,
                    PYF_WRAPS, PYMOD, SUBMODULE, USE, Facts, Ref)

# Modules every compiler or runtime provides: they are dependencies, not the code's own.
INTRINSIC = frozenset({
    "iso_c_binding", "iso_fortran_env", "ieee_arithmetic", "ieee_exceptions",
    "ieee_features", "omp_lib", "omp_lib_kinds", "openacc", "mpi", "mpi_f08", "mpi_f08_types",
})

_MODULE_RE = re.compile(r"^module\s+(\w+)\s*$", re.I)
_SUBMODULE_RE = re.compile(r"^submodule\s*\(\s*(\w+)", re.I)
_USE_RE = re.compile(r"^use\s*(?:,\s*(intrinsic|non_intrinsic)\s*)?(?:::)?\s*(\w+)", re.I)
_INCLUDE_RE = re.compile(r"""^(?:include\s+|#\s*include\s*)['"<]([^'">]+)['">]""", re.I)
_PROC_RE = re.compile(
    r"^(?:(?:pure|impure|elemental|recursive|non_recursive|module)\s+"
    r"|(?:integer|real|logical|complex|character|double\s*precision|type\s*\([^)]*\)"
    r"|class\s*\([^)]*\))(?:\s*\([^)]*\)|\s*\*\s*\d+)?\s+)*(subroutine|function)\s+(\w+)", re.I)
_BIND_RE = re.compile(r"""bind\s*\(\s*c\s*(?:,\s*name\s*=\s*['"]([^'"]*)['"])?\s*\)""", re.I)
_INTERFACE_RE = re.compile(r"^(?:abstract\s+)?interface\b", re.I)
_END_INTERFACE_RE = re.compile(r"^end\s*interface\b", re.I)
_FORPY_RE = re.compile(r"""\bimport_py\s*\(\s*\w+\s*,\s*['"]([\w.]+)['"]""", re.I)
_PYF_MODULE_RE = re.compile(r"^python\s+module\s+(\w+)", re.I)

FIXED_FORM = frozenset({".f", ".for", ".ftn", ".f77", ".fpp"})


def _strip_comment(line: str) -> str:
    """The line up to a ``!`` that is not inside a string."""
    quote = ""
    for i, ch in enumerate(line):
        if quote:
            if ch == quote:
                quote = ""
        elif ch in "'\"":
            quote = ch
        elif ch == "!":
            return line[:i]
    return line


def statements(text: str, *, fixed: bool) -> list[tuple[int, str]]:
    """The file's statements, comments removed and continuations joined, each with the
    line it starts on. Case is kept: a ``bind(c, name=…)`` symbol is case-sensitive."""
    out: list[tuple[int, str]] = []
    pending = ""
    start = 0
    for no, raw in enumerate(text.splitlines(), 1):
        if fixed:
            if raw[:1] in ("c", "C", "*", "!"):
                continue
            if len(raw) > 5 and raw[5] not in (" ", "0") and out and not raw.lstrip().startswith("#"):
                line, prev = out.pop()
                out.append((line, prev + " " + _strip_comment(raw[6:]).strip()))
                continue
            body = _strip_comment(raw[6:] if not raw.lstrip().startswith("#") else raw).strip()
            if body:
                out.append((no, body))
            continue
        body = _strip_comment(raw).strip()
        if pending:
            body = body[1:].lstrip() if body.startswith("&") else body
            pending += " " + body
        else:
            pending, start = body, no
        if pending.endswith("&"):
            pending = pending[:-1].rstrip()
            continue
        for part in pending.split(";"):
            if part.strip():
                out.append((start, part.strip()))
        pending = ""
    return out


def read_fortran(text: str, *, fixed: bool) -> Facts:
    """What a Fortran file defines (modules, procedures, bind(c) symbols) and references."""
    facts = Facts("fortran", text.count("\n") + 1)
    depth = 0
    for line, st in statements(text, fixed=fixed):
        if _INTERFACE_RE.match(st):
            depth += 1
            continue
        if _END_INTERFACE_RE.match(st):
            depth = max(0, depth - 1)
            continue
        m = _MODULE_RE.match(st)
        if m and m.group(1).lower() != "procedure":
            facts.exports.append((FMODULE, m.group(1).lower()))
            continue
        m = _SUBMODULE_RE.match(st)
        if m:
            facts.refs.append(Ref(SUBMODULE, m.group(1).lower(), line))
            continue
        m = _USE_RE.match(st)
        if m:
            name = m.group(2).lower()
            nature = (m.group(1) or "").lower()
            intrinsic = nature == "intrinsic" or (not nature and name in INTRINSIC)
            facts.refs.append(Ref(INTRINSIC_USE if intrinsic else USE, name, line))
            continue
        m = _INCLUDE_RE.match(st)
        if m:
            facts.refs.append(Ref(INCLUDE, m.group(1), line))
            continue
        m = _PROC_RE.match(st)
        if m and not st.lower().startswith("end"):
            name = m.group(2).lower()
            bind = _BIND_RE.search(st)
            symbol = (bind.group(1) or name) if bind else ""
            if depth:
                if bind:
                    facts.refs.append(Ref(BIND_CALL, symbol, line))
            else:
                facts.exports.append((FPROC, name))
                if bind:
                    facts.exports.append((FBIND, symbol))
        facts.refs.extend(Ref(EMBED_IMPORT, f.group(1), line) for f in _FORPY_RE.finditer(st))
    return facts


def read_pyf(text: str) -> Facts:
    """An f2py signature file: the Python module it defines and the procedures it wraps."""
    facts = Facts("pyf", text.count("\n") + 1)
    for line, st in statements(text, fixed=False):
        m = _PYF_MODULE_RE.match(st)
        if m:
            facts.exports.append((PYMOD, m.group(1).lower()))
            continue
        m = _PROC_RE.match(st)
        if m and not st.lower().startswith("end"):
            facts.refs.append(Ref(PYF_WRAPS, m.group(2).lower(), line))
        m = _USE_RE.match(st)
        if m:
            facts.refs.append(Ref(USE, m.group(2).lower(), line))
    return facts
