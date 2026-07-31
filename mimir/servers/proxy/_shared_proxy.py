"""Legacy import facade for the proxy helper library.

This module used to hold every proxy helper (~1700 lines); the code now lives
in the ``_lib/`` package (store, procs, metrics, command, report, execute,
ratchet — see ``_lib/__init__.py`` for the map).  Import from ``_lib.*``
directly in new code.

The one thing this file must keep doing is exist under this name and export
``_post_run_finalize``: ``postrun.py`` scripts generated into old run
directories — possibly still queued on Slurm — do
``from _shared_proxy import _post_run_finalize`` after inserting the server
directory into ``sys.path``.
"""

import os as _os
import sys as _sys

_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))

from _lib.execute import _post_run_finalize  # noqa: E402,F401
