"""Where this workspace's server is, written down so it can be found again.

A detached server is only useful if something can find it. Today the port is learned by
regexing the child's stdout — which works exactly as long as the extension host is the
parent holding that pipe, and not a moment longer. A server that outlives the window
that started it has to leave its address somewhere on disk.

``<STATE_DIR>/server.json``, and the choice of directory is the whole of "one server per
workspace": ``STATE_DIR`` is already ``<state home>/<basename>-<sha1(realpath)[:8]>``, so
two checkouts that share a basename do not collide and two windows on the same workspace
resolve to the same file. Nothing new had to be invented to scope it.

**Liveness is three questions, not one.** A pid exists; its start time matches the one
recorded, because a recycled pid wears the same number; and the port actually answers,
because a process can be alive with its listener already gone — mid-shutdown, or wedged.
The first two come from :mod:`job_scan`, which holds that contract for detached runs; the
third is this module's own, and it is the one the other two cannot answer.

The entry is advisory, never authoritative. A stale one costs a connection attempt that
fails and is then replaced; treating it as proof a server exists is how a window ends up
waiting on an address nothing is listening at.
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import socket
import time

from ._ws_runtime import _MIMIR_DIR_WS
from .job_scan import _is_running, _proc_starttime

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
    return os.path.join(_MIMIR_DIR_WS, _LOCKNAME)


def acquire() -> bool:
    """Claim the right to be *the* server for this workspace. True when claimed.

    Two servers on one workspace is not a tidiness problem. They share the sessions
    directory, so both append to the same ``transcript.jsonl`` and both derive ``seq``
    from it — the numbering collides, the watermark built on it stops meaning anything,
    and a client attached to one of them sees nothing of the turn running in the other.
    That is a conversation whose tools run and never appear.

    So the claim is a kernel lock rather than a convention: ``flock`` is released when
    the process dies however it dies, which no file written by the process can promise.
    Taken before the socket is bound, because a loser must not occupy a port either.
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
    """Drop the claim. Never raises; the kernel does this anyway when we go."""
    global _LOCK_FH
    handle, _LOCK_FH = _LOCK_FH, None
    if handle is None:
        return
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    except Exception:
        pass
    try:
        handle.close()
    except Exception:
        pass


def registry_path() -> str:
    return os.path.join(_MIMIR_DIR_WS, _FILENAME)


def publish(*, url: str, host: str, port: int, **extra) -> dict | None:
    """Record this process as the server for this workspace. Returns the entry.

    Written atomically — tmp plus rename — so a reader never sees half a file. Called
    once the socket is actually bound, because the port is the point: with ``--port 0``
    the argument was a placeholder and only the bound socket knows the answer.

    Never raises. Failing to publish costs discoverability, not the server: it is still
    serving, and the extension's other paths (an explicit ``mimir.wsUrl``, a fresh
    spawn) still work.
    """
    pid = os.getpid()
    entry = {
        "protocol": PROTOCOL,
        "pid": pid,
        "pid_starttime": _proc_starttime(pid),
        "url": url,
        "host": host,
        "port": int(port),
        "workspace": os.path.realpath(os.environ.get("MCP_FILES_ROOT") or os.getcwd()),
        "state_dir": _MIMIR_DIR_WS,
        "started_at": time.time(),
        **extra,
    }
    path = registry_path()
    tmp = path + ".tmp"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh, indent=2, default=str)
        os.replace(tmp, path)
    except Exception:
        logger.warning("registry: could not publish %s", path, exc_info=True)
        try:
            os.remove(tmp)
        except Exception:
            pass
        return None
    logger.info("registry: published %s for workspace %s", url, entry["workspace"])
    return entry


def update(**fields) -> dict | None:
    """Merge *fields* into the recorded entry, keeping the address it already holds.

    What a detachment needs: the url, host and port were settled when the socket was
    bound and have not changed, so re-deriving them would mean handing them around for
    no reason. Refuses to touch a stranger's entry — an address this process does not
    serve is not ours to annotate.
    """
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
    """Whether something accepts a connection there.

    The question the process table cannot answer. Tried for both loopback spellings
    when the host is a name: a server bound to IPv4 and probed over IPv6 looks dead,
    which is the same confusion that once sent the extension's own connection through
    a proxy.
    """
    if not port:
        return False
    try:
        infos = socket.getaddrinfo(host or "localhost", int(port),
                                   type=socket.SOCK_STREAM)
    except Exception:
        return False
    for family, socktype, proto, _canon, addr in infos:
        try:
            with socket.socket(family, socktype, proto) as sock:
                sock.settimeout(timeout)
                sock.connect(addr)
                return True
        except Exception:
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
    if not _is_running(pid, starttime if isinstance(starttime, int) else None):
        return False
    if not probe:
        return True
    return port_answers(str(entry.get("host") or "localhost"),
                        int(entry.get("port") or 0))


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
