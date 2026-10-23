"""The agent's state directories, server-side: one source of truth.

Three tiers: ``global_state_dir()`` (the user, every workspace), ``state_dir()`` (the
workspace: memory, environments; from ``MIMIR_STATE_DIR``, else ``<workspace>/.mimir``),
``session_state_dir()`` (one conversation: todo, plans, transcript, grants, jobs).
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

    ``MIMIR_STATE_DIR`` if set (the central per-workspace dir), else
    ``<MCP_FILES_ROOT or cwd>/.mimir``.
    """
    env = os.environ.get("MIMIR_STATE_DIR")
    if env:
        return os.path.abspath(env)
    root = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())
    return os.path.join(root, ".mimir")


def global_state_dir(home: str | None = None) -> str:
    """The state shared by every workspace: ``<STATE_HOME>/global/``.

    From ``MIMIR_GLOBAL_STATE_DIR``, then *home* (the client passes it), then
    ``MIMIR_STATE_HOME``, else ``<state_dir()>/global`` (hermetic for tests).
    """
    env = os.environ.get("MIMIR_GLOBAL_STATE_DIR")
    if env:
        return os.path.abspath(env)
    root = home or os.environ.get("MIMIR_STATE_HOME")
    return os.path.join(os.path.abspath(root), "global") if root \
        else os.path.join(state_dir(), "global")


def active_session_id(base: str | None = None) -> str:
    """The session this process works for: ``MIMIR_SESSION_ID``, else the ``active_session`` sidecar, else "".

    The variable is fixed at spawn, which is correct under concurrency; the sidecar is for
    single-session ends (CLI, tests). May contain ``/`` for a sub-session.
    """
    env = (os.environ.get("MIMIR_SESSION_ID") or "").strip()
    if env:
        return env
    try:
        with open(os.path.join(base or state_dir(), "active_session"), encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""


def session_owns(owner: str) -> bool:
    """Whether work started by session *owner* is this session's to stop.

    Its own, and its sub-agents' at any depth (``<current>/subagents/...``); never a
    parent's or a sibling's.
    """
    cur = active_session_id()
    return owner == cur or (bool(cur) and owner.startswith(f"{cur}/subagents/"))


def session_family(session_dir: str) -> list[str]:
    """*session_dir* and the directories of its sub-agents, at any depth."""
    family = [session_dir]
    pending = [session_dir]
    while pending:
        sub = os.path.join(pending.pop(), SUBAGENTS_DIRNAME)
        try:
            children = sorted(os.listdir(sub))
        except OSError:
            continue
        for child in children:
            family.append(os.path.join(sub, child))
            pending.append(os.path.join(sub, child))
    return family


def session_state_dir(base: str | None = None, session_id: str | None = None) -> str:
    """``<state_dir>/sessions/<sid>/``, or ``state_dir()`` outside a session. Not created here."""
    root = base or state_dir()
    sid = session_id if session_id is not None else active_session_id(root)
    return os.path.join(root, "sessions", sid) if sid else root


# The layout of a conversation's directory. Named here only: every end that reads or
# writes one of these goes through the helpers below.
SESSIONS_DIRNAME = "sessions"
SUBAGENTS_DIRNAME = "subagents"
TODO_FILENAME = "todo_list.md"
TODO_DEPS_FILENAME = "todo_deps.json"
JOURNAL_FILENAME = "transcript.jsonl"
ACTIVITY_FILENAME = "activity.jsonl"


def sessions_root(base: str | None = None) -> str:
    """``<state_dir>/sessions``: one directory per conversation."""
    return os.path.join(base or state_dir(), SESSIONS_DIRNAME)


def session_dir(session_id: str, base: str | None = None) -> str:
    """The directory of conversation *session_id*, which must be given."""
    return os.path.join(sessions_root(base), session_id)


def subagents_dir(parent_id: str, base: str | None = None) -> str:
    """Where the sub-agent sessions of conversation *parent_id* live."""
    return os.path.join(session_dir(parent_id, base), SUBAGENTS_DIRNAME)


def session_file(filename: str, session_id: str | None, base: str | None = None) -> str:
    """*filename* in a conversation's directory, or in the state root without one.

    ``None`` and ``""`` both mean "no session" here: the caller names the session
    outright, so the active-session sidecar is never consulted.
    """
    return os.path.join(session_state_dir(base, session_id or ""), filename)


def site_scratch() -> tuple[str, bool]:
    """``$SCRATCH`` when the site sets it, else the local temp dir; and whether it is the former.

    A cluster's scratch is large and seen by the compute nodes, so a file written there
    can be read by a Slurm job; a login node's ``/tmp`` is small, shared, and its own.
    """
    scratch = os.environ.get("SCRATCH", "").strip()
    if scratch:
        return os.path.abspath(os.path.expanduser(scratch)), True
    return os.path.abspath(os.environ.get("TMPDIR") or "/tmp"), False


def scratch_home() -> str:
    """The scratchpad root: ``MIMIR_SCRATCH_DIR``, else ``<site scratch>/mimir-<uid>-<workspace-id>``."""
    env = os.environ.get("MIMIR_SCRATCH_DIR")
    if env:
        return os.path.abspath(os.path.expanduser(env))
    root = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())
    return os.path.join(site_scratch()[0], f"mimir-{os.getuid()}-{workspace_id(root)}")


def scratch_dir(base: str | None = None, session_id: str | None = None) -> str:
    """The agent's scratchpad, outside the workspace: a per-session directory of :func:`scratch_home`.

    Not created here, never cleaned by MIMIR. *base* overrides the state dir (the client
    passes its own); *session_id* wins over everything.
    """
    sid = session_id if session_id is not None else active_session_id(base or state_dir())
    home = scratch_home()
    return os.path.join(home, sid) if sid else home


def standing_roots(base: str | None = None) -> list[str]:
    """Paths writable without approval, granted by the system: the scratchpad home.

    Kept apart from ``approved_roots()``, the record of the user's own grants.
    """
    return [scratch_home()]


def ensure_private_dir(path: str) -> bool:
    """Create *path* ``0700``, or vet the one there: ours, a real directory, private.

    For a directory under a root others can write (``/tmp``, a project's scratch): one
    made first at our name by someone else would hand them what we put in it.
    """
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        try:
            os.makedirs(path, mode=0o700)
        except OSError:
            return False
        return True
    except OSError:
        return False

    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        return False
    if info.st_uid != os.getuid():
        return False
    if info.st_mode & 0o077:
        try:
            os.chmod(path, 0o700)
        except OSError:
            return False
    return True


def ensure_scratch_home() -> str:
    """Create the scratchpad home ``0700`` and vet it; "" when the path is not safely ours."""
    home = scratch_home()
    return home if ensure_private_dir(home) else ""
