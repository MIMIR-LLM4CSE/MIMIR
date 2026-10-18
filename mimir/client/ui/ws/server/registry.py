"""Where this workspace's server is: ``<STATE_DIR>/server.json``, one server per workspace.

Liveness is a pid, its start time, and a port that answers. Advisory: a stale entry
costs a failed connection.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import time

from .....servers._shared.processes import is_running, proc_starttime
from .....servers._shared.atomic import write_json
from ....config import paths

logger = logging.getLogger(__name__)

# Bumped when the shape of the file changes incompatibly. A reader that does not know a
# version refuses the entry rather than guessing at it: the alternative is connecting to
# a server whose protocol it cannot speak.
PROTOCOL = 1

# How long to wait for the port to answer. Long enough for a loopback accept on a loaded
# machine, short enough that probing a dead address does not stall a window's startup.
_PROBE_TIMEOUT = 0.3

_FILENAME = "server.json"
_LOCKNAME = "server.lock"

# Held for the life of the winning process. A module global because that is what its
# lifetime is: released by the kernel when the process goes, which is the only release
# that can be trusted — a crashed server must not keep a workspace locked.
_LOCK_FH = None


def lock_path() -> str:
    return os.path.join(paths.STATE_DIR, _LOCKNAME)


def acquire() -> bool:
    """Claim the right to be this workspace's server with a kernel ``flock``; True when claimed.

    Two servers would share one journal and collide on ``seq``. Taken before binding.
    """
    global _LOCK_FH
    if _LOCK_FH is not None:
        return True
    path = lock_path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        handle = open(path, "a+", encoding="utf-8")
    except Exception:
        # Nowhere to put the lock. Serving unlocked is worse than not serving: it is
        # the exact state that corrupts the journal.
        logger.warning("registry: could not open %s", path, exc_info=True)
        return False
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        return False
    _LOCK_FH = handle
    return True


def release() -> None:
    """Drop the *lock*. Never raises; the kernel does this anyway when we go.

    Not to be confused with :func:`unclaim`, which drops a conversation's request to
    be left running.
    """
    global _LOCK_FH
    handle, _LOCK_FH = _LOCK_FH, None
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        logger.debug("release: a best-effort step failed and was skipped", exc_info=True)
    try:
        handle.close()
    except Exception:
        logger.debug("release: a best-effort step failed and was skipped", exc_info=True)


def registry_path() -> str:
    return os.path.join(paths.STATE_DIR, _FILENAME)


def publish(*, url: str, host: str, port: int, **extra) -> dict | None:
    """Record this process as the workspace's server, atomically, once bound; return the entry.

    Never raises: a failure costs discoverability, not the server.
    """
    pid = os.getpid()
    entry = {
        "protocol": PROTOCOL,
        "pid": pid,
        "pid_starttime": proc_starttime(pid),
        "url": url,
        "host": host,
        "port": int(port),
        "workspace": os.path.realpath(os.environ.get("MCP_FILES_ROOT") or os.getcwd()),
        "state_dir": paths.STATE_DIR,
        "started_at": time.time(),
        **extra,
    }
    path = registry_path()
    try:
        write_json(path, entry, indent=2, default=str)
    except Exception:
        logger.warning("registry: could not publish %s", path, exc_info=True)
        return None
    logger.info("registry: published %s for workspace %s", url, entry["workspace"])
    return entry


def update(**fields) -> dict | None:
    """Merge *fields* into this process's own entry, keeping its address."""
    entry = read()
    if entry is None:
        return None
    if int(entry.get("pid") or 0) != os.getpid():
        logger.info("registry: not updating the entry of pid %s", entry.get("pid"))
        return None
    merged = {**entry, **fields}
    return publish(url=merged.pop("url"), host=merged.pop("host"),
                   port=merged.pop("port"),
                   **{k: v for k, v in merged.items()
                      if k not in ("protocol", "pid", "pid_starttime", "workspace",
                                   "state_dir", "started_at")})


def read() -> dict | None:
    """The recorded entry, or None when there is none or it cannot be understood.

    An entry from a protocol this build does not know is refused rather than read
    partially: connecting to a server whose contract has changed is worse than deciding
    there is none.
    """
    try:
        with open(registry_path(), encoding="utf-8") as fh:
            entry = json.load(fh)
    except FileNotFoundError:
        return None
    except Exception:
        logger.warning("registry: %s could not be read", registry_path(), exc_info=True)
        return None
    if not isinstance(entry, dict):
        return None
    if int(entry.get("protocol") or 0) != PROTOCOL:
        logger.info("registry: ignoring an entry written for protocol %r",
                    entry.get("protocol"))
        return None
    return entry


def port_answers(host: str, port: int, timeout: float = _PROBE_TIMEOUT) -> bool:
    """Whether something accepts a connection at *host*:*port* (both loopbacks for a name)."""
    if not port:
        return False
    try:
        infos = socket.getaddrinfo(host or "localhost", int(port),
                                   type=socket.SOCK_STREAM)
    except Exception:
        logger.debug("port_answers: a best-effort step failed and was skipped", exc_info=True)
        return False
    for family, socktype, proto, _canon, addr in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(timeout)
                sock.connect(addr)
                return True
        except Exception:
            logger.debug("port_answers: a best-effort step failed and was skipped", exc_info=True)
            continue
    return False


def alive(entry: dict | None, *, probe: bool = True) -> bool:
    """Whether the recorded server is still there.

    *probe* off asks only the process-table half, for a caller that has a reason not to
    open a socket (a sweep over many entries, a test). It is the weaker answer: a pid
    can outlive its listener.
    """
    if not entry:
        return False
    try:
        pid = int(entry.get("pid") or 0)
    except (TypeError, ValueError):
        return False
    starttime = entry.get("pid_starttime")
    if not is_running(pid, starttime if isinstance(starttime, int) else None):
        return False
    if not probe:
        return True
    return port_answers(str(entry.get("host") or "localhost"),
                        int(entry.get("port") or 0))


# ── Who asked for this server to survive ──────────────────────────────────────
# Claims are per conversation (``detached_sessions: {session id: autonomy}``); the process
# survives while any stands. ``detached`` is "anybody at all", what the extension reads.


def claims() -> dict[str, str]:
    """This process's live claims, as ``{session id: autonomy}``.

    Empty for an entry that is not ours: a stranger's claims are not ours to read as
    our own, and acting on them would have one server keep itself alive for another's
    conversations.
    """
    entry = read()
    if not entry or int(entry.get("pid") or 0) != os.getpid():
        return {}
    held = entry.get("detached_sessions")
    if not isinstance(held, dict):
        return {}
    return {str(k): str(v) for k, v in held.items() if k}


def claim(session_ids: list[str], autonomy: str, **extra) -> dict[str, str]:
    """Record that *session_ids* asked to be left running. Returns every live claim."""
    merged = {**claims(), **{str(sid): autonomy for sid in session_ids if sid}}
    update(detached_sessions=merged, detached=bool(merged), autonomy=autonomy, **extra)
    return merged


def unclaim(session_ids: list[str] | None = None) -> dict[str, str]:
    """Drop the claims of *session_ids*, or every one of them. Returns what is left.

    Named apart from :func:`release`, which drops the *lock*: one is about who may be
    the server for this workspace, the other about who wants it to outlive a window.
    """
    held = claims()
    if session_ids is None:
        held = {}
    else:
        for sid in session_ids:
            held.pop(str(sid), None)
    update(detached_sessions=held, detached=bool(held))
    return held


def current(*, probe: bool = True) -> dict | None:
    """The entry if a live server is recorded, else None."""
    entry = read()
    return entry if alive(entry, probe=probe) else None


def clear(*, only_if_ours: bool = True) -> None:
    """Remove the entry. Never raises.

    *only_if_ours* guards the ordinary shutdown path: this process should retire its own
    address and leave a stranger's alone, because deleting another server's entry makes
    a perfectly good server undiscoverable.
    """
    entry = read()
    if entry is None:
        return
    if only_if_ours and int(entry.get("pid") or 0) != os.getpid():
        logger.info("registry: leaving the entry of pid %s alone", entry.get("pid"))
        return
    try:
        os.remove(registry_path())
    except FileNotFoundError:
        pass
    except Exception:
        logger.warning("registry: could not remove %s", registry_path(), exc_info=True)
