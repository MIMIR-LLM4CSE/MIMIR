"""Run a bundled server or helper as a module: ``python _launch.py MODULE [ARGS...]``.

Started by path, so ``sys.path[0]`` is this directory; it is replaced by the directory
holding the ``mimir`` package before *MODULE* runs as ``__main__``. The current
directory, which is the user's workspace, never enters ``sys.path``: a file of the
project cannot shadow a module the server imports, as it could under ``python -m``.
"""
import os
import runpy
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_PACKAGE_PARENT = os.path.dirname(os.path.dirname(_HERE))


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: _launch.py MODULE [ARGS...]")
    if sys.path and os.path.abspath(sys.path[0] or os.curdir) == _HERE:
        sys.path[0] = _PACKAGE_PARENT
    elif _PACKAGE_PARENT not in sys.path:
        sys.path.insert(0, _PACKAGE_PARENT)
    module = sys.argv[1]
    sys.argv = [module, *sys.argv[2:]]
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
