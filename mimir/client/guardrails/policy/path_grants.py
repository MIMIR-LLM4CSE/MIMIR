"""Paths outside the workspace the user let this conversation reach.

Two sets: every path granted (allow-once and always), which is mirrored to a sidecar
file the servers read on every call — that is what lets the server's sandbox honour
the grant — and the paths granted "always", which also skip the question next time.
Per session: consent is given inside a conversation, and several run in one process.
"""
from __future__ import annotations

import os

from ...config import paths
from ....shared.atomic import write_json
from ....shared.state_paths import session_state_dir


def _real(abspath: str) -> str:
    return os.path.realpath(os.path.abspath(abspath))


class PathGrants:
    def __init__(self, session_id: str | None = None) -> None:
        self.session_id = session_id
        self.allowed: set[str] = set()
        self.always: set[str] = set()

    def covers(self, abspath: str) -> bool:
        """True if an "always" grant of this session contains *abspath* (servers treat grants as roots)."""
        real = _real(abspath)
        return any(real == granted or real.startswith(granted + os.sep)
                   for granted in self.always)

    def grant(self, abspath: str, *, always: bool) -> None:
        """Record an approved path and mirror it to the sidecar.

        Both allow-once and always reach the sidecar, so the server honours this access;
        only ``always`` also skips future prompts for the same path this session.
        """
        real = _real(abspath)
        self.allowed.add(real)
        if always:
            self.always.add(real)
        self._flush()

    def reset(self) -> None:
        """Clear this session's grants and truncate its own sidecar.

        Only ever its own: the file is per session, so a conversation starting fresh
        cannot revoke what another one was granted and is still writing under.
        """
        self.allowed.clear()
        self.always.clear()
        self._flush()

    def sidecar_file(self) -> str:
        """This session's allowlist sidecar: ``<STATE_DIR>/sessions/<sid>/approved_paths.json``.

        From ``session_id``: the grant belongs to the conversation that asked. None resolves
        through the active-session pointer, as the single-session ends do.
        """
        return os.path.join(session_state_dir(paths.STATE_DIR, self.session_id),
                            "approved_paths.json")

    def _flush(self) -> None:
        try:
            write_json(self.sidecar_file(), sorted(self.allowed))
        except OSError:
            pass  # best-effort; a failed write simply keeps the server strict
