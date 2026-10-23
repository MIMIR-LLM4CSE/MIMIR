"""C, C++ and CUDA sources read for their includes, the Python modules they define or
embed, the C-linkage functions they offer, and the functions they call."""
from __future__ import annotations

import re

from .facts import CALL, CSYM, EMBED_IMPORT, INCLUDE, PYMOD, Facts, Ref

_COMMENT_RE = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
_INCLUDE_RE = re.compile(r'^[ \t]*#[ \t]*include[ \t]*[<"]([^>"]+)[>"]', re.M)
_PYMOD_RE = re.compile(
    r"\b(?:PYBIND11_MODULE|NB_MODULE|BOOST_PYTHON_MODULE)\s*\(\s*(\w+)"
    r"|\bPyInit_(\w+)\s*\(\s*(?:void)?\s*\)\s*\{"
    r"|\bPyModuleDef\b[^;{]*\{[^}]*?\"(\w+)\"", re.S)
_EMBED_RE = re.compile(
    r"\b(?:PyImport_ImportModule(?:Level)?|PyImport_ImportModuleNoBlock)\s*\(\s*\"([\w.]+)\""
    r"|\b(?:py|nb|pybind11|nanobind)::module_?::import_?\s*\(\s*\"([\w.]+)\"")
_EXTERN_BLOCK_RE = re.compile(r'extern\s+"C"\s*\{')
_EXTERN_ONE_RE = re.compile(r'extern\s+"C"\s+[^{;(]*?\b([A-Za-z_]\w*)\s*\(')
_FUNC_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\([^;{}()]*(?:\([^()]*\)[^;{}()]*)*\)\s*(?:const\s*)?[;{]")
_TOP_DEF_RE = re.compile(
    r"^[A-Za-z_][\w \t\*]*?[ \t\*]([A-Za-z_]\w*)[ \t]*\([^;{}]*\)[ \t\n]*\{", re.M)
_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")
_KEYWORDS = frozenset(
    "if for while switch return sizeof alignof decltype catch static_assert defined "
    "typeid new delete case do else typedef using template operator noexcept".split())


def _blank_comments(text: str) -> str:
    """Comments replaced by blanks of the same lines, so positions keep their line."""
    return _COMMENT_RE.sub(lambda m: re.sub(r"[^\n]", " ", m.group(0)), text)


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _block_end(text: str, start: int) -> int:
    """The index past the brace that closes the one opening at *start*."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                return i + 1
    return len(text)


def read_c(text: str, *, is_c: bool) -> Facts:
    """What a C/C++/CUDA file includes, exports and calls; *is_c* for plain C, where every
    top-level function has C linkage."""
    facts = Facts("c" if is_c else "cpp", text.count("\n") + 1)
    code = _blank_comments(text)
    facts.refs.extend(Ref(INCLUDE, m.group(1), _line(code, m.start()))
                      for m in _INCLUDE_RE.finditer(code))
    for m in _PYMOD_RE.finditer(code):
        facts.exports.append((PYMOD, next(g for g in m.groups() if g)))
    facts.refs.extend(Ref(EMBED_IMPORT, m.group(1) or m.group(2), _line(code, m.start()))
                      for m in _EMBED_RE.finditer(code))
    symbols: set[str] = set()
    for m in _EXTERN_BLOCK_RE.finditer(code):
        body = code[m.end() - 1:_block_end(code, m.end() - 1)]
        symbols.update(f.group(1) for f in _FUNC_RE.finditer(body))
    symbols.update(m.group(1) for m in _EXTERN_ONE_RE.finditer(code))
    if is_c:
        symbols.update(m.group(1) for m in _TOP_DEF_RE.finditer(code))
    facts.exports.extend((CSYM, s) for s in sorted(symbols - _KEYWORDS))
    seen: set[str] = set()
    for m in _CALL_RE.finditer(code):
        name = m.group(1)
        if name not in _KEYWORDS and name not in seen:
            seen.add(name)
            facts.refs.append(Ref(CALL, name, _line(code, m.start())))
    return facts
