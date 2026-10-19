"""Letting this process go on without the window that started it.

The fds are re-pointed at a log file (``dup2``), since the host's pipe closing would
kill the next write; ``setsid`` leaves its process group, best-effort. Irreversible.
"""

from __future__ import annotations

import logging
import os
import signal
import sys

logger = logging.getLogger(__name__)

_LOG_DIR = "logs"


def log_path(state_dir: str, pid: int | None = None) -> str:
    """Where a detached server's output goes. Named by pid, so two never collide."""
    return os.path.join(state_dir, _LOG_DIR,
                        f"server-{pid if pid is not None else os.getpid()}.log")


def _redirect_output(path: str) -> bool:
    """Point fds 1 and 2 at *path*, appending. True when both moved.

    Appending rather than truncating: a server may detach, be re-attached to, and
    detach again, and each of those is the same run's log.
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
    except Exception:
        logger.warning("detach: could not create the log directory for %s", path,
                       exc_info=True)
        return False
    try:
        # Flush first: whatever is buffered belongs to the pipe era and would otherwise
        # be written to the file, out of order with what the extension already showed.
        for stream in (sys.stdout, sys.stderr):
            try:
                stream.flush()
            except Exception:
                logger.debug("_redirect_output: a best-effort step failed and was skipped", exc_info=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    except Exception:
        logger.warning("detach: could not open %s", path, exc_info=True)
        return False
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    except Exception:
        logger.warning("detach: could not re-point the output fds", exc_info=True)
        return False
    finally:
        try:
            os.close(fd)
        except Exception:
            logger.debug("_redirect_output: a best-effort step failed and was skipped", exc_info=True)
    return True


def _leave_process_group() -> bool:
    """``setsid``, best effort. False when this process is already a group leader."""
    try:
        os.setsid()
        return True
    except OSError as exc:
        # Already a session/group leader — which is the normal state for a server
        # started from a shell, and harmless here: see the module docstring.
        logger.info("detach: setsid declined (%s); the fds are what matter", exc)
        return False


def _ignore_hangup() -> bool:
    """Survive a SIGHUP, in case something still signals the old group."""
    try:
        signal.signal(signal.SIGHUP, signal.SIG_IGN)
        return True
    except (OSError, ValueError, AttributeError):
        return False


def detach(state_dir: str) -> dict:
    """Make this process survivable. Returns what it managed, for the caller to report.

    Order matters only in that the fds come first: they are the half that decides
    whether the process lives, so a failure in either of the others is reported and
    stepped over rather than aborting.
    """
    path = log_path(state_dir)
    redirected = _redirect_output(path)
    result = {
        "log": path if redirected else None,
        "redirected": redirected,
        "setsid": _leave_process_group(),
        "sighup_ignored": _ignore_hangup(),
        "pid": os.getpid(),
        "sid": None,
        "pgid": None,
    }
    try:
        result["sid"] = os.getsid(0)
        result["pgid"] = os.getpgid(0)
    except OSError:
        pass
    if redirected:
        # Into the new log, as its first line: a reader opening this file wants to know
        # what it is and when it began.
        print(f"--- MIMIR server {os.getpid()} detached; output continues here ---",
              flush=True)
    logger.info("detach: redirected=%s setsid=%s sighup_ignored=%s log=%s",
                redirected, result["setsid"], result["sighup_ignored"], path)
    return result
