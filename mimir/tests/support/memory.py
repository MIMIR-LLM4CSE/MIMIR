"""The memory server, loaded with both its stores pointed at a tmpdir.

``memory_add`` reads the global store even when writing the workspace one, so pointing
only the workspace store would read and write the developer's ``~/.mimir/global/memory``.
"""

import importlib.util
from pathlib import Path
from mimir.tests.support.modules import spec_for

_ROOT = Path(__file__).resolve().parents[3]
_SHARED = _ROOT / "mimir" / "servers" / "_shared"


def load_memory_server():
    """Import server_memory as a standalone module, the way the server itself runs."""
    spec = spec_for(_ROOT / "mimir" / "servers" / "agent_state" / "server_memory.py")
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
