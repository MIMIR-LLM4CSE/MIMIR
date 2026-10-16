"""The whole suite runs against a state directory of its own.

``config.paths`` resolves ``STATE_DIR`` once, at import, from ``MIMIR_STATE_DIR`` when it
is set — so it is set here, before any test module imports the client. Without it, a
test that forgot to patch one of the modules that write state (the journal, the active
session pointer, the preferences) wrote into the developer's own ``~/.mimir``, inventing
conversations there that no session ever had.
"""
import atexit
import os
import shutil
import tempfile

_STATE = tempfile.mkdtemp(prefix="mimir-test-state-")
os.environ["MIMIR_STATE_DIR"] = _STATE
atexit.register(shutil.rmtree, _STATE, ignore_errors=True)
