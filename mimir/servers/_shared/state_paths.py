"""Single source of truth for the agent's STATE directory, server-side.

The client computes the central per-workspace state dir (``~/.mimir/<ws-id>/``)
and hands it to the server subprocesses via the ``MIMIR_STATE_DIR`` env var (see
client/integration/server_manager.py). Servers should resolve every persistent
state path (memory/, todos, plans/, platform_profile.json, …) off ``state_dir()``
so the whole stack agrees on one location.

When ``MIMIR_STATE_DIR`` is unset — standalone runs and the hermetic test suite,
which only set ``MCP_FILES_ROOT`` — we fall back to the legacy in-workspace
``<workspace>/.mimir`` so those callers keep working unchanged.
"""

import os


def state_dir() -> str:
    """Return the agent's state directory.

    ``MIMIR_STATE_DIR`` if set (the central per-workspace dir), else the legacy
    ``<MCP_FILES_ROOT or cwd>/.mimir`` fallback.
    """
    env = os.environ.get("MIMIR_STATE_DIR")
    if env:
        return os.path.abspath(env)
    root = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())
    return os.path.join(root, ".mimir")


def active_session_id(base: str | None = None) -> str:
    """The session the client is currently driving, or "" outside a session.

    Reads the ``active_session`` sidecar the client rewrites on every session
    switch. The servers' environment is frozen at spawn, so a file on the shared
    state dir is the only live client→server channel (same mechanism the approved
    -paths allowlist uses). Best-effort: any read error means "no session".
    """
    try:
        with open(os.path.join(base or state_dir(), "active_session"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def scratch_dir(base: str | None = None) -> str:
    """The agent's scratchpad: a writable directory *outside* the workspace.

    Somewhere to put throwaway probe scripts, intermediate data, and working files
    that are not deliverables. Without it the only writable place is the workspace
    itself, so every temporary file becomes indistinguishable from produced work —
    it lands in the user's tree and in the change ledger.

    Per session (``sessions/<sid>/scratch/``) so parallel sessions cannot collide,
    falling back to a shared ``scratch/`` outside any session (CLI runs, tests).
    Never auto-deleted: the contents are often exactly what the user wants to
    inspect after a run.

    Not created here — callers that write will create it; callers that only need
    the path for a sandbox check must not have that check materialise directories.

    *base* overrides the state dir, for the client process: it resolves the same
    location from its own ``STATE_DIR`` constant because ``MIMIR_STATE_DIR`` is
    placed only in the *server* subprocesses' environment, never its own.
    """
    root = base or state_dir()
    sid = active_session_id(root)
    if sid:
        return os.path.join(root, "sessions", sid, "scratch")
    return os.path.join(root, "scratch")


def standing_roots(base: str | None = None) -> list[str]:
    """Absolute paths writable without user approval, granted by the system.

    Deliberately separate from ``approved_roots()``: that file is the record of
    decisions the *user* made, and folding a system grant into it would both
    misreport consent and let a stale sidecar revoke the scratchpad.

    Both the session-scoped scratchpad and the session-less fallback are granted,
    so a path stays writable across a session switch mid-run.
    """
    root = base or state_dir()
    return [scratch_dir(root), os.path.join(root, "scratch")]
