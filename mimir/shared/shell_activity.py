"""When this session's shell commands ran: the evidence that a Slurm job submitted outside
the cluster tools — by a launcher, a script — is this session's own.

One line per command in ``<session state dir>/shell_activity.jsonl``: when it started and
when it ended; a detached run ends with its job. A job submitted while a command of this
session (or of one of its sub-agents) ran, and while none of another session's did, is
this session's.
"""

import json
import os
import time

from .jobs import EXIT_CODE, META
from .processes import is_running
from .state_paths import session_family, session_state_dir, sessions_root

FILE = "shell_activity.jsonl"
# Allowed between a command's window and a submission time: the clocks of this host and
# a scheduler's, and a launcher that returns before its sbatch has been recorded.
SLACK_S = 60.0


def record(started: float, ended: float | None, job_dir: str = "") -> None:
    """Note one command of this session; *ended* None for a detached run (its end is its
    job's, read from *job_dir*). Never raises: the record is evidence, not the work."""
    entry = {"started": started, "ended": ended, "job_dir": job_dir}
    try:
        directory = session_state_dir()
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, FILE), "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


def _job_end(job_dir: str) -> float:
    """When a detached run ended: its exit code's time, now while it runs, else its last trace."""
    try:
        return os.path.getmtime(os.path.join(job_dir, EXIT_CODE))
    except OSError:
        pass
    try:
        with open(os.path.join(job_dir, META), encoding="utf-8") as fh:
            meta = json.load(fh)
        if is_running(meta.get("pid"), meta.get("pid_starttime")):
            return time.time()
    except (OSError, ValueError):
        pass
    try:
        return os.path.getmtime(job_dir)
    except OSError:
        return 0.0


def _windows(session_dir: str) -> list[tuple[float, float]]:
    """The ``(start, end)`` of every command recorded under *session_dir*."""
    out = []
    try:
        with open(os.path.join(session_dir, FILE), encoding="utf-8") as fh:
            lines = fh.readlines()
    except OSError:
        return []
    for line in lines:
        try:
            entry = json.loads(line)
            start = float(entry["started"])
            ended = entry.get("ended")
            end = float(ended) if ended is not None else _job_end(str(entry.get("job_dir") or ""))
        except (ValueError, KeyError, TypeError):
            continue
        out.append((start, max(start, end)))
    return out


def _ran_at(dirs: list[str], moment: float) -> bool:
    return any(start - SLACK_S <= moment <= end + SLACK_S
               for d in dirs for start, end in _windows(d))


def ran_commands() -> bool:
    """Whether this session or one of its sub-agents ran any shell command."""
    return any(_windows(d) for d in session_family(session_state_dir()))


def submitted_by_this_session(moment: float) -> bool:
    """Whether a submission at *moment* (epoch seconds) falls in a command of this session
    and in none of another session's; overlapping sessions are not told apart."""
    own = session_family(session_state_dir())
    if not _ran_at(own, moment):
        return False
    mine = {os.path.realpath(d) for d in own}
    try:
        names = sorted(os.listdir(sessions_root()))
    except OSError:
        names = []
    others = [d for name in names
              for d in session_family(os.path.join(sessions_root(), name))
              if os.path.realpath(d) not in mine]
    return not _ran_at(others, moment)
