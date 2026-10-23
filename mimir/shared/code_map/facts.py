"""What one source file states, as a language reader finds it: the names it exports and
the references it makes, each with its line."""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Ref:
    """One reference a file makes: its kind, the name it gives, its line.

    *names* holds a ``from … import`` list; *level* a relative import's dots;
    *computed* a reference whose name is not a literal (an import or load by variable);
    *deferred* one made only when a function runs, not when the file loads.
    """
    kind: str
    name: str
    line: int
    level: int = 0
    names: tuple[str, ...] = ()
    computed: bool = False
    deferred: bool = False


@dataclass
class Facts:
    """A file's language, its length, what it exports ``(kind, name)`` and its references."""
    lang: str
    lines: int
    exports: list[tuple[str, str]] = field(default_factory=list)
    refs: list[Ref] = field(default_factory=list)


# Reference kinds, read by the resolvers.
INCLUDE = "include"            # C/C++/Fortran: a header or an included file, by path
IMPORT = "import"              # Python (and Cython): a module, absolute or relative
CIMPORT = "cimport"            # Cython: a .pxd module
USE = "use"                    # Fortran: a module
INTRINSIC_USE = "intrinsic_use"  # Fortran: a module the compiler provides
SUBMODULE = "submodule"        # Fortran: a submodule's parent module
EXTERN_HEADER = "extern_header"  # Cython: `cdef extern from "x.h"`
EMBED_IMPORT = "embed_import"  # C/C++ or Fortran (forpy) importing a Python module
BIND_CALL = "bind_call"        # Fortran: an `interface … bind(c)` procedure, a C symbol
CALL = "call"                  # C/C++: a function called by name
LIBRARY = "library"            # Python: a shared library loaded (ctypes, cffi)
LIBRARY_CALL = "library_call"  # Python: a symbol called on a loaded library
PYF_WRAPS = "pyf_wraps"        # f2py signature file: a Fortran procedure it wraps

# Export kinds, read by the binding index.
PYMOD = "pymod"                # a Python extension module defined here
FMODULE = "fmodule"            # a Fortran module defined here
FPROC = "fproc"                # a Fortran procedure defined here (no bind(c))
FBIND = "fbind"                # a Fortran procedure's C symbol, from bind(c)
CSYM = "csym"                  # a C-linkage function defined or declared here
