"""Run a subprocess under a timeout that actually stops the work.

``subprocess.run(..., timeout=...)`` kills the process it launched and nothing else.
Anything that process forked keeps running, reparented to init, with no record of where
it came from. Observed: an agent ran ``find / -name _proxy_runner.py`` through the bash
tool, the 30s timeout fired and returned an orderly error, and the ``find`` was still
scanning the filesystem **one hour and forty minutes later** — three of them at once,
alongside a benchmark whose whole purpose was to measure elapsed time.

Two costs, and the second is the one that matters. The leaked processes consume the
machine; and because they are invisible to the tool that started them, they silently
corrupt every measurement taken afterwards. A ratchet optimising ``time_s`` against a
disk saturated by forgotten ``find`` jobs is optimising noise.

Why the leak happens here in particular: the bash tool runs ``bash -c "<preamble><cmd>"``.
Given a single command bash would ``exec`` it and be the only process, so killing the
child would be enough — but the preamble makes the script compound, so bash forks. The
kill lands on bash, and the fork survives. A tool that runs code it did not write cannot
assume the shape of the process tree it creates.

The fix is to give the child its own process group and signal the group:
``start_new_session=True`` makes it a session leader, so its pgid equals its pid and can
never be the caller's own group — killing it cannot reach MIMIR itself. SIGTERM first so
a well-behaved program can clean up, SIGKILL after a short grace period for one that
will not.
"""

from __future__ import annotations

import os
import signal
import subprocess
from typing import Any

# How long a process group gets to exit on SIGTERM before SIGKILL. Short: this runs
# after a timeout the caller has already waited out.
_GRACE_S = 2.0


def _terminate_group(proc: subprocess.Popen) -> None:
    """Signal the child's whole process group, escalating TERM -> KILL."""
    if not hasattr(os, "killpg"):  # pragma: no cover - not POSIX
        proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except OSError:
        return  # already reaped; nothing to signal
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(pgid, sig)
        except OSError:
            return  # group is gone
        try:
            proc.wait(timeout=_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def run(argv: list[str], *, timeout: float | None = None, **kwargs: Any):
    """Like :func:`subprocess.run`, but a timeout kills the child's whole process group.

    Drop-in for the call sites that matter: it raises ``subprocess.TimeoutExpired`` and
    returns a ``CompletedProcess`` exactly as ``subprocess.run`` does, so existing
    ``except subprocess.TimeoutExpired`` handlers keep working unchanged.

    *start_new_session* is forced on unless the caller set it: it is what makes the
    group killable, and it is also what stops a child inheriting the controlling
    terminal it has no business reading from.
    """
    kwargs.setdefault("start_new_session", True)
    with subprocess.Popen(argv, **kwargs) as proc:
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            _terminate_group(proc)
            # The group is dead, so the pipes are closed and this cannot block. Bounded
            # anyway: draining output must never be a second way to hang on a timeout.
            try:
                stdout, stderr = proc.communicate(timeout=_GRACE_S)
            except subprocess.TimeoutExpired:
                stdout = stderr = None
            raise subprocess.TimeoutExpired(argv, timeout, output=stdout, stderr=stderr)
        except BaseException:
            # Cancelled or interrupted: the same leak, by a different route.
            _terminate_group(proc)
            raise
        return subprocess.CompletedProcess(argv, proc.returncode, stdout, stderr)
