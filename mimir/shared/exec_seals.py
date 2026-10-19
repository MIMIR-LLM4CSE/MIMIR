"""Files a server seals for a conversation: not to be run directly, only through its tools.

A server writes one seal per owner under the session's state directory
(``seals/<owner>.json``): the paths, the files a run it records measures, and what the
refusal says. The client's execution gate and its evidence read them, never the server.
"""
from __future__ import annotations

import json
import os

from . import state_paths
from .atomic import write_json

SEALS_DIRNAME = "seals"


def _seal_path(owner: str, session_id: str | None) -> str:
    return os.path.join(state_paths.session_state_dir(session_id=session_id),
                        SEALS_DIRNAME, f"{owner}.json")


def seal(owner: str, paths: list[str], *, refusal: str, hint: str,
         measured: list[str] | tuple[str, ...] = (), session_id: str | None = None) -> None:
    """Seal *paths* for the session: running one directly is refused with *refusal*.

    *measured* are the files a run this owner records measures; *hint* says how to run
    them instead, and how to lift the seal.
    """
    write_json(_seal_path(owner, session_id), {
        "owner": owner,
        "paths": [os.path.abspath(p) for p in paths if p],
        "measured": [os.path.abspath(p) for p in measured if p],
        "refusal": refusal,
        "hint": hint,
    }, indent=2)


def unseal(owner: str, session_id: str | None = None) -> None:
    """Drop *owner*'s seal for the session (nothing to drop is fine)."""
    try:
        os.remove(_seal_path(owner, session_id))
    except OSError:
        pass


def _seals(session_id: str | None) -> list[dict]:
    """Every readable seal of the session; one that cannot be read gives no opinion."""
    root = os.path.join(state_paths.session_state_dir(session_id=session_id), SEALS_DIRNAME)
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    out = []
    for name in names:
        if not name.endswith(".json"):
            continue
        try:
            with open(os.path.join(root, name), encoding="utf-8") as fh:
                record = json.load(fh)
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            out.append(record)
    return out


def sealed(session_id: str | None = None) -> dict[str, dict]:
    """Real path -> the seal that covers it, for every path sealed in the session."""
    out: dict[str, dict] = {}
    for record in _seals(session_id):
        for path in record.get("paths") or []:
            if isinstance(path, str) and path:
                out[os.path.realpath(path)] = record
    return out


def measured_sources(session_id: str | None = None) -> set[str]:
    """The files a run recorded by a sealing owner measures, in the session."""
    return {p for record in _seals(session_id) for p in record.get("measured") or []
            if isinstance(p, str) and p}
