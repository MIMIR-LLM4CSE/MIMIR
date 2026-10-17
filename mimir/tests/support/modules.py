"""Loading a fresh copy of a module of this repository, inside its package.

``spec_for(path)`` is the import spec of the module at *path* under its dotted name,
so ``module_from_spec`` + ``exec_module`` give a copy whose relative imports resolve,
without replacing the copy in ``sys.modules``.
"""
from __future__ import annotations

import importlib.util
import os
from importlib.machinery import ModuleSpec

_PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))


def dotted_name(path: str | os.PathLike) -> str:
    rel = os.path.relpath(os.path.abspath(path), _PACKAGE_PARENT)[:-3]
    return rel.replace(os.sep, ".")


def spec_for(path: str | os.PathLike) -> ModuleSpec:
    spec = importlib.util.find_spec(dotted_name(path))
    if spec is None or spec.loader is None:
        raise ImportError(f"no module at {path}")
    return spec
