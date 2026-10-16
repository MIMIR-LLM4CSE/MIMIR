"""Writing files so that a reader never sees half of one."""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def write_json_atomic(path: str, data: Any, *, trailing_newline: bool = False,
                      **dump_kwargs: Any) -> None:
    """Write *data* as JSON to *path*: whole or not at all. Raises on failure.

    Through a temporary file of its own in the same directory, then a rename. The
    temporary name is unique: a fixed ``path + ".tmp"`` is shared by every process
    writing the same file — several agents in one server, a CLI beside it — and two
    of them interleaving there renamed one writer's half-written file into place.
    """
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, **dump_kwargs)
            if trailing_newline:
                fh.write("\n")
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
