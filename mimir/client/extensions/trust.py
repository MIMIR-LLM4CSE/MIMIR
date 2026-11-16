"""Whether a workspace's own plugins and servers may run: accepted once per content.

``.mimir/plugins`` is imported into this process and ``.mimir/servers`` started, so a
cloned repository's would run at startup. VS Code's folder trust answers through
``MIMIR_WORKSPACE_TRUSTED``; a terminal is asked, and the SHA-256 of what it accepts is
kept, so a changed file is asked about again. No tty: only what was accepted loads.
"""
from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
from typing import Callable

from ...shared.atomic import read_json, write_json
from ...shared.state_paths import ensure_private_dir
from ..config.paths import GLOBAL_STATE_DIR

logger = logging.getLogger(__name__)

TRUSTED_ENV = "MIMIR_WORKSPACE_TRUSTED"
_FILENAME = "trusted_extensions.json"

# Decisions of this process, by (path, digest): asked once, whichever loader asks first.
_decided: dict[tuple[str, str], bool] = {}
_lock = threading.Lock()


def _store() -> str:
    return os.path.join(GLOBAL_STATE_DIR, _FILENAME)


def _digest(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def _ask_terminal(paths: list[str]) -> bool:
    if not sys.stdin or not sys.stdin.isatty():
        return False
    listed = "\n".join(f"  {p}" for p in paths)
    print(f"\n⚠ This workspace ships code MIMIR would run:\n{listed}\n"
          "  Plugins are imported into MIMIR itself; servers are started with your rights.",
          flush=True)
    try:
        reply = input("  Load them [y/N]? ").strip().lower()
    except EOFError:
        return False
    return reply in ("y", "yes")


def trusted_only(paths: list[str], ask: Callable[[list[str]], bool] | None = None) -> list[str]:
    """The *paths* that may run: all of them in a trusted folder, else those accepted.

    *ask* is put the files not yet accepted, once, and answers for all of them.
    """
    if os.environ.get(TRUSTED_ENV) == "1" or not paths:
        return list(paths)
    ask = ask or _ask_terminal
    with _lock:
        accepted: dict[str, str] = read_json(_store(), {}) or {}
        digests = {os.path.realpath(p): _digest(p) for p in paths}
        pending = [p for p, d in digests.items()
                   if d and accepted.get(p) != d and (p, d) not in _decided]
        if pending:
            verdict = ask(pending)
            for p in pending:
                _decided[(p, digests[p])] = verdict
            if verdict:
                accepted.update({p: digests[p] for p in pending})
                ensure_private_dir(GLOBAL_STATE_DIR)
                write_json(_store(), accepted)
            else:
                logger.warning("Not loading workspace code nobody accepted: %s",
                               ", ".join(pending))
        allowed = {p for p, d in digests.items()
                   if d and (accepted.get(p) == d or _decided.get((p, d)))}
    return [p for p in paths if os.path.realpath(p) in allowed]
