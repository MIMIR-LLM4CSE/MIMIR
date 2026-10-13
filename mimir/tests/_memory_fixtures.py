"""Shared fixture for the memory server: load it, and point BOTH stores at a tmpdir.

The server resolves two stores — the per-workspace one off ``MIMIR_STATE_DIR`` and the
global one off ``MIMIR_GLOBAL_STATE_DIR`` — and `memory_add` reads the global store even
when writing to the workspace (cross-scope duplicate refusal). So a test that repoints
only the workspace paths reads, and through embedding backfill *writes*, the developer's
real ``~/.mimir/global/memory``. Pointing both is what keeps the suite hermetic, and it
belongs in one place rather than in each test file's setUp.
"""

import importlib.util
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _ROOT / "mimir" / "servers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))


def load_memory_server():
    """Import server_memory as a standalone module, the way the server itself runs."""
    spec = importlib.util.spec_from_file_location(
        "server_memory",
        _ROOT / "mimir" / "servers" / "agent_state" / "server_memory.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def point_stores(mem, root) -> tuple[Path, Path]:
    """Repoint both of *mem*'s stores under *root*. Returns (workspace, global) dirs.

    Two sibling subdirectories rather than *root* itself, so a test can assert that an
    operation on one scope left the other's files untouched.
    """
    root = Path(root)
    ws, gl = root / "workspace", root / "global"
    mem.MEMORY_DIR = str(ws)
    mem.INDEX_FILE = str(ws / "MEMORY.md")
    mem.EMBEDDINGS_FILE = str(ws / "embeddings.json")
    mem.GLOBAL_MEMORY_DIR = str(gl)
    mem.GLOBAL_INDEX_FILE = str(gl / "MEMORY.md")
    mem.GLOBAL_EMBEDDINGS_FILE = str(gl / "embeddings.json")
    return ws, gl
