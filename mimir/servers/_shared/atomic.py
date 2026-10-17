"""Writing a file so that a reader sees the old content or the new one, never half.

Each write goes through a temporary file of its own in the target's directory and is
renamed into place. The temporary name is unique per call, so concurrent writers of
the same file never share one.
"""
from __future__ import annotations

import json
import os
import tempfile
from typing import Any


def write_text(path: str, text: str, *, newline: str | None = None) -> None:
    """Replace *path* with *text*. Raises on failure, leaving *path* untouched."""
    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=f".{os.path.basename(path)}.",
                               suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline=newline) as fh:
            fh.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def write_json(path: str, data: Any, *, trailing_newline: bool = False,
               **dump_kwargs: Any) -> None:
    """Replace *path* with *data* as JSON; *dump_kwargs* go to ``json.dumps``."""
    text = json.dumps(data, **dump_kwargs)
    write_text(path, text + "\n" if trailing_newline else text)


def replace_symlink(link: str, target: str) -> None:
    """Point *link* at *target*, replacing whatever *link* was, in one rename."""
    directory = os.path.dirname(os.path.abspath(link))
    tmp = os.path.join(directory, f".{os.path.basename(link)}.{os.getpid()}.tmp")
    if os.path.lexists(tmp):
        os.unlink(tmp)
    os.symlink(target, tmp)
    os.replace(tmp, link)
