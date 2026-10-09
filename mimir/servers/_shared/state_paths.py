"""Single source of truth for the agent's STATE directory, server-side.

The client computes the central per-workspace state dir (``~/.mimir/<ws-id>/``)
and hands it to the server subprocesses via the ``MIMIR_STATE_DIR`` env var (see
client/integration/server_manager.py). Servers should resolve every persistent
state path (memory/, todos, plans/, …) off ``state_dir()``
so the whole stack agrees on one location.

When ``MIMIR_STATE_DIR`` is unset — standalone runs and the hermetic test suite,
which only set ``MCP_FILES_ROOT`` — we fall back to the legacy in-workspace
``<workspace>/.mimir`` so those callers keep working unchanged.

Three tiers, and which one a path belongs to is the whole question:

* ``global_state_dir()`` — what belongs to the *user*, across every workspace on the
  machine: facts that hold whichever repository they are in. One directory for the
  whole machine, which means a session in another workspace may be writing it at this
  very moment.
* ``state_dir()`` — what belongs to the *workspace*: the memory store, the Python
  environments, the module catalogue. Common property, shared by every conversation,
  and expensive to rebuild.
* ``session_state_dir()`` — what belongs to one *conversation*: its todo list, plans,
  transcript, approved paths, run channels, job directories. Sessions run turns at the
  same time, so nothing here may be readable or truncatable by another one.
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


def global_state_dir(home: str | None = None) -> str:
    """Where state shared by *every* workspace lives: ``<STATE_HOME>/global/``.

    A sibling of the per-workspace directories, not a parent of them — ``STATE_HOME``
    is not a private namespace of workspace dirs: ``install.sh`` writes ``bin``,
    ``python`` and ``pythons`` there too. A named tier keeps the agent's state out of
    the installer's layout, and no workspace can collide with it since
    :func:`workspace_id` always emits ``<basename>-<sha1[:8]>``.

    *home* is the state home the caller already resolved, and the client must pass it:
    ``MIMIR_STATE_DIR`` reaches only the server subprocesses' environment, never the
    client's own (see client/guardrails/policy/gates.py), which is the same reason
    :func:`session_state_dir` and :func:`scratch_dir` take a *base*.

    Resolution order: ``MIMIR_GLOBAL_STATE_DIR`` (explicit, and what the client
    publishes to the servers), then *home*, then ``MIMIR_STATE_HOME``, and finally
    ``<state_dir()>/global``.

    That last fallback is deliberate, and so is what it is *not*. Deriving the home
    from ``dirname(MIMIR_STATE_DIR)`` would resolve to ``/tmp`` for the tests that
    point the state dir at a ``mkdtemp()`` — a world-writable path shared between
    users and between runs. Reaching for ``~`` would have the hermetic test suite,
    which sets only ``MCP_FILES_ROOT``, write into the developer's real global memory.
    Collapsing the tier inside the workspace state dir is instead hermetic, and a
    harmless degradation: with one workspace, global and workspace mean the same thing.
    """
    env = os.environ.get("MIMIR_GLOBAL_STATE_DIR")
    if env:
        return os.path.abspath(env)
    root = home or os.environ.get("MIMIR_STATE_HOME")
    return os.path.join(os.path.abspath(root), "global") if root \
        else os.path.join(state_dir(), "global")


def active_session_id(base: str | None = None) -> str:
    """The session this process is working for, or "" outside a session.

    ``MIMIR_SESSION_ID`` first. Sessions run turns at the same time, each with its own
    agent and therefore its own set of server subprocesses, so the session a server belongs
    to is fixed for its whole life and is stamped into its environment at spawn (see
    client/integration/server_manager.py). That is what makes the answer *correct* under
    concurrency: a file naming "the" session can name only one, and every other live
    session reading it would act for the wrong conversation.

    The ``active_session`` sidecar is the fallback, for the ends that genuinely have a
    single session: the CLI, standalone server runs, and the test suite. It names the
    session the user is *looking at*, which is the right answer only when nothing else is
    running.

    Best-effort: any read error means "no session".
    """
    env = (os.environ.get("MIMIR_SESSION_ID") or "").strip()
    if env:
        return env
    try:
        with open(os.path.join(base or state_dir(), "active_session"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def session_state_dir(base: str | None = None, session_id: str | None = None) -> str:
    """Where everything belonging to one *conversation* lives.

    ``<state_dir>/sessions/<sid>/`` — the todo list, the plans, the transcript, the
    approved-path allowlist, the run channels, the detached job directories. One
    directory per conversation, so N sessions working at once cannot read or truncate
    each other's state, and deleting a conversation takes its working files with it.

    Falls back to ``state_dir()`` itself outside any session (CLI, standalone servers,
    tests), so the single-session ends resolve the same paths they would without it.

    Not created here — callers that write create it, so a read-only path check never
    materialises a directory (the same rule :func:`scratch_dir` follows).
    """
    root = base or state_dir()
    sid = session_id if session_id is not None else active_session_id(root)
    return os.path.join(root, "sessions", sid) if sid else root


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


def scratch_dir(base: str | None = None, session_id: str | None = None) -> str:
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

    *session_id* names the session outright and wins over both. The client process runs
    several sessions in one ``os.environ``, so there the id can come only from the object
    that knows it — the agent — never from the environment or the sidecar.
    """
    sid = session_id if session_id is not None else active_session_id(base or state_dir())
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
