"""The sessions kept after their state went (a clean, a re-baseline), as the code map drew
them: ``history/<proxy>@<stamp>.json``, self-contained. A key holds an ``@``, a proxy's
name never does (``check_name``), so the two cannot be mistaken.
"""
from __future__ import annotations

import os
import time

from ....shared.atomic import write_json
from .layout import history_dir, read_json


def is_key(name: str) -> bool:
    """Whether *name* names a frozen session rather than a proxy."""
    return "@" in name and os.path.basename(name) == name


def proxy_of(key: str) -> str:
    """The proxy a frozen session was taken from."""
    return key.rsplit("@", 1)[0]


def path(key: str) -> str:
    return os.path.join(history_dir(), key + ".json")


def keys() -> list[str]:
    """Every frozen session, oldest stamp first for each proxy."""
    try:
        names = sorted(os.listdir(history_dir()))
    except OSError:
        return []
    return [n[:-5] for n in names if n.endswith(".json") and "@" in n]


def read(key: str) -> dict:
    """The frozen session *key*, {} when it is not one."""
    return (read_json(path(key), {}) or {}) if is_key(key) else {}


def write(name: str, why: str, session: dict, merge: dict | None = None) -> str:
    """Keep *session* (as the map draws *name*'s) under a new key, before *why* (cleaned,
    rebaselined) takes the state it is rebuilt from; its steps are renamed to the key, as
    the tab picks a session by the proxy its steps name. Raises when it cannot be written."""
    key = f"{name}@{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(history_dir(), exist_ok=True)
    write_json(path(key), {
        **session, "proxy": key, "steps": [{**st, "proxy": key} for st in session.get("steps", [])],
        **({"merge": merge} if merge else {}),
        "frozen": {"proxy": name, "why": why, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}})
    return key
