"""Python and Cython sources read for what they import and the native libraries they load."""
from __future__ import annotations

import ast
import re

from .facts import (CIMPORT, EXTERN_HEADER, IMPORT, LIBRARY, LIBRARY_CALL, Facts, Ref)

# Calls that load a shared library and return a handle to call into.
_LOADERS = {"CDLL", "PyDLL", "WinDLL", "LoadLibrary", "load_library", "dlopen"}
_DYNAMIC_IMPORTS = {"import_module", "__import__"}


def read_python(text: str) -> Facts:
    """The imports of a ``.py`` file, its library loads and the symbols called on them."""
    facts = Facts("python", text.count("\n") + 1)
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        facts.refs.extend(_regex_imports(text))
        return facts
    handles: set[str] = set()
    for node, deferred in _statements(tree):
        if isinstance(node, ast.Import):
            facts.refs.extend(Ref(IMPORT, a.name, node.lineno, deferred=deferred) for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            facts.refs.append(Ref(IMPORT, node.module or "", node.lineno, level=node.level,
                                  names=tuple(a.name for a in node.names if a.name != "*"),
                                  deferred=deferred))
        elif isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            loaded = _library_of(node.value)
            if loaded is not None:
                facts.refs.append(loaded)
                handles.update(t.id for t in node.targets if isinstance(t, ast.Name))
        elif isinstance(node, ast.Call) and _call_name(node.func) in _DYNAMIC_IMPORTS:
            arg = node.args[0] if node.args else None
            if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
                facts.refs.append(Ref(IMPORT, arg.value, node.lineno))
            else:
                facts.refs.append(Ref(IMPORT, "", node.lineno, computed=True))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name) and node.func.value.id in handles):
            facts.refs.append(Ref(LIBRARY_CALL, node.func.attr, node.lineno))
    return facts


def _statements(tree: ast.AST):
    """Every node with whether it runs only inside a function; ``if TYPE_CHECKING:``
    bodies are left out, as they never run."""
    todo: list[tuple[ast.AST, bool]] = [(tree, False)]
    while todo:
        node, deferred = todo.pop()
        yield node, deferred
        if isinstance(node, ast.If) and "TYPE_CHECKING" in ast.unparse(node.test):
            todo.extend((n, deferred) for n in node.orelse)
            continue
        inner = deferred or isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda))
        todo.extend((n, inner) for n in ast.iter_child_nodes(node))


def _call_name(func: ast.expr) -> str:
    if isinstance(func, ast.Attribute):
        return func.attr
    return func.id if isinstance(func, ast.Name) else ""


def _library_of(call: ast.Call) -> Ref | None:
    """The library a loader call names, or a computed one; None for any other call."""
    if _call_name(call.func) not in _LOADERS:
        return None
    arg = call.args[0] if call.args else None
    if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
        return Ref(LIBRARY, library_stem(arg.value), call.lineno)
    return Ref(LIBRARY, "", call.lineno, computed=True)


def library_stem(name: str) -> str:
    """``./build/libfoo.so.1`` → ``foo``: the name a build target gives the library."""
    base = re.split(r"[\\/]", name)[-1]
    base = re.sub(r"\.(so|dylib|dll|a)(\.[\d.]+)?$", "", base)
    return base[3:] if base.startswith("lib") else base


_IMPORT_RE = re.compile(r"^[ \t]*import[ \t]+([\w.]+)", re.M)
_FROM_RE = re.compile(r"^[ \t]*from[ \t]+(\.*)([\w.]*)[ \t]+(c?import)[ \t]+\(?([\w., \t]+)", re.M)
_CIMPORT_RE = re.compile(r"^[ \t]*cimport[ \t]+([\w.]+)", re.M)
_EXTERN_RE = re.compile(r"^[ \t]*cdef[ \t]+extern[ \t]+from[ \t]+[\"<]([^\">]+)[\">]", re.M)


def _line(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _regex_imports(text: str) -> list[Ref]:
    refs = [Ref(IMPORT, m.group(1), _line(text, m.start())) for m in _IMPORT_RE.finditer(text)]
    for m in _FROM_RE.finditer(text):
        names = tuple(n.strip() for n in m.group(4).split(",") if n.strip() and n.strip() != "*")
        kind = CIMPORT if m.group(3) == "cimport" else IMPORT
        refs.append(Ref(kind, m.group(2), _line(text, m.start()), level=len(m.group(1)), names=names))
    return refs


def read_cython(text: str) -> Facts:
    """The imports, cimports and ``cdef extern`` headers of a ``.pyx``/``.pxd``/``.pxi`` file."""
    facts = Facts("cython", text.count("\n") + 1)
    facts.refs.extend(_regex_imports(text))
    facts.refs.extend(Ref(CIMPORT, m.group(1), _line(text, m.start()))
                      for m in _CIMPORT_RE.finditer(text))
    facts.refs.extend(Ref(EXTERN_HEADER, m.group(1), _line(text, m.start()))
                      for m in _EXTERN_RE.finditer(text))
    return facts
