"""Single source of truth for the agent's STATE directory, server-side.

The client computes the central per-workspace state dir (``~/.mimir/<ws-id>/``)
and hands it to the server subprocesses via the ``MIMIR_STATE_DIR`` env var (see
client/integration/server_manager.py). Servers should resolve every persistent
state path (memory/, todos, plans/, …) off ``state_dir()``
so the whole stack agrees on one location.

When ``MIMIR_STATE_DIR`` is unset — standalone runs and the hermetic test suite,
which only set ``MCP_FILES_ROOT`` — we fall back to the legacy in-workspace
``<workspace>/.mimir`` so those callers keep working unchanged.
"""

import hashlib
import os
import stat


def workspace_id(root: str) -> str:
    """Readable, collision-free id for a workspace root: ``<basename>-<sha1[:8]>``.

    Lives here rather than in the client's constants because both ends need it: the
    client to build the per-workspace state dir, this module to name the scratchpad
    under a shared ``/tmp``.
    """
    real = os.path.realpath(root)
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:8]
    return f"{os.path.basename(real) or 'root'}-{digest}"


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


def scratch_home() -> str:
    """Root of the agent's scratchpad, under the system temp dir.

    ``MIMIR_SCRATCH_DIR`` if set — the client resolves this once at startup (after
    validating ownership, see :func:`ensure_scratch_home`) and puts it in both its
    own environment and the servers', so every end agrees on one location without
    re-deriving it.

    Default: ``<TMPDIR or /tmp>/mimir-<uid>-<workspace-id>``. Temp is where throwaway
    work belongs, and the OS reclaims it; the uid and the workspace id keep two users
    (and two checkouts) on a shared machine from colliding.
    """
    env = os.environ.get("MIMIR_SCRATCH_DIR")
    if env:
        return os.path.abspath(env)
    tmp = os.environ.get("TMPDIR") or "/tmp"
    root = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())
    return os.path.join(os.path.abspath(tmp), f"mimir-{os.getuid()}-{workspace_id(root)}")


def scratch_dir(base: str | None = None) -> str:
    """The agent's scratchpad: a writable directory *outside* the workspace.

    Somewhere to put throwaway probe scripts, intermediate data, and working files
    that are not deliverables. Without it the only writable place is the workspace
    itself, so every temporary file becomes indistinguishable from produced work —
    it lands in the user's tree and in the change ledger.

    A per-session subdirectory of :func:`scratch_home` so parallel sessions cannot
    collide, falling back to the home itself outside any session (CLI runs, tests).
    Not auto-deleted by MIMIR: temp-dir policy is the OS's business, and the contents
    are often exactly what the user wants to inspect after a run.

    Not created here — callers that write will create it; callers that only need
    the path for a sandbox check must not have that check materialise directories.

    *base* overrides the *state* dir, which is consulted only for the active-session
    sidecar: the client passes its own ``STATE_DIR`` because ``MIMIR_STATE_DIR`` is
    placed only in the server subprocesses' environment, never its own.
    """
    sid = active_session_id(base or state_dir())
    home = scratch_home()
    return os.path.join(home, sid) if sid else home


def standing_roots(base: str | None = None) -> list[str]:
    """Absolute paths writable without user approval, granted by the system.

    Deliberately separate from ``approved_roots()``: that file is the record of
    decisions the *user* made, and folding a system grant into it would both
    misreport consent and let a stale sidecar revoke the scratchpad.

    The grant is the scratchpad *home*, which covers the session subdirectory and the
    session-less fallback alike — so a session switch mid-run cannot revoke a path
    already being written. Still a list: it is one element of a sandbox's extra roots,
    and callers concatenate it with the user-approved ones.

    *base* is accepted for signature parity with :func:`scratch_dir`; the home does
    not depend on the state dir.
    """
    return [scratch_home()]


def ensure_scratch_home() -> str:
    """Create the scratchpad home ``0700`` and vet it, or return "" if unsafe.

    The one function here that touches the disk. ``/tmp`` is world-writable, so an
    existing path at our name is not necessarily ours: a symlink or a foreign-owned
    directory means someone else chose where our writes land, and the answer is to
    decline rather than to use it. The client calls this once at startup and falls
    back to a path under its private state dir on "".
    """
    home = scratch_home()
    try:
        info = os.lstat(home)
    except FileNotFoundError:
        try:
            os.makedirs(home, mode=0o700)
        except OSError:
            return ""
        return home
    except OSError:
        return ""

    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return ""
    if info.st_uid != os.getuid():
        return ""
    if info.st_mode & 0o077:
        try:
            os.chmod(home, 0o700)
        except OSError:
            return ""
    return home
