"""What detached runs a session left behind, read off the disk.

Background runs already survive everything. ``bash_run`` launches them in their own
session with ``start_new_session=True`` and a trap that writes the exit code, Slurm jobs
belong to the controller, and both leave a descriptor under
``<state>/sessions/<sid>/jobs/<key>/`` or ``.../hpc_jobs/<dir>/``. What does *not* survive
is the promise to report them: a watcher is an ``asyncio.Task`` on the worker's loop, and
``shutdown()`` cancels it. The run carries on, indifferent; only the promise has to be
re-made.

``query_engine/background.py`` can already re-make it, but only when a status tool is
called — so it needs a turn in which somebody asks "where is the job at?". This module is
the half that lets it happen without asking: it inspects the descriptors directly and says
which runs are still going and which ended while nothing was listening.

**Why the liveness check is here rather than imported.** ``_bash_jobs._is_running`` is the
same contract, and this is deliberately its twin rather than a call into it: the MCP
servers resolve their imports against their own directory (``import state_paths``), so
nothing in ``mimir.client`` can import that tree — the same reason
``tool_execution/run_channel.py`` is a twin of ``servers/_shared/run_channel.py``. The
contract that matters, and that must change in both places together: a pid is alive only
if it exists, its start time matches the one recorded (a recycled pid wears the same
number), and it is not a zombie (nothing reaps a detached child, so a finished job lingers
in the process table and would otherwise never reach a terminal state).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from ._ws_runtime import _MIMIR_DIR_WS

logger = logging.getLogger(__name__)


@dataclass
class DetachedJob:
    """One run found on disk, and whether anything still has to happen for it."""

    session_id: str
    job_key: str
    kind: str                     # "shell" | "slurm"
    live: bool                    # still going, as far as this machine can tell
    exit_code: int | None         # what it recorded, when it recorded one
    command: str = ""
    started_at: float | None = None

    @property
    def state(self) -> str:
        """The state a ``job_complete`` would carry for this run.

        ``unknown`` is not a tidier word for failure: it means the run stopped being
        trackable, and claiming an outcome nobody observed is worse than saying so.
        """
        if self.live:
            return "running"
        if self.exit_code is None:
            return "unknown"
        return "done" if self.exit_code == 0 else "crashed"

    def status_op(self) -> dict:
        """The read-only op a watcher polls this run with."""
        if self.kind == "slurm":
            return {"tool": "slurm_status", "args": {"job_id": self.job_key}}
        return {"tool": "bash_job", "args": {"job_key": self.job_key}}

    def descriptor(self) -> dict:
        """What ``_register_bg_job`` needs to put a watcher back on this run."""
        return {"server": "hpc" if self.kind == "slurm" else "bash",
                "job_key": self.job_key,
                "kind": "slurm-job" if self.kind == "slurm" else "shell-command",
                "status_op": self.status_op()}


def _proc_starttime(pid: int) -> int | None:
    """Field 22 of ``/proc/<pid>/stat`` — the clock tick the process started at."""
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        # The comm field is parenthesized and may contain spaces: split after its close.
        return int(data[data.rfind(")") + 2:].split()[19])
    except (OSError, IndexError, ValueError):
        return None


def _is_running(pid: int, expected_starttime: int | None) -> bool:
    """Twin of ``_bash_jobs._is_running`` — see the module docstring."""
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError):
        return False
    except OSError:
        return False
    if expected_starttime is not None:
        actual = _proc_starttime(pid)
        if actual is not None and actual != expected_starttime:
            return False  # pid recycled — a different process wears it now
    try:
        with open(f"/proc/{pid}/stat") as fh:
            data = fh.read()
        return data[data.rfind(")") + 2:].split()[0] != "Z"
    except (OSError, IndexError):
        return True


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def _read_exit_code(job_dir: str) -> int | None:
    path = os.path.join(job_dir, "exit_code")
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        return None


def _sessions_root() -> str:
    return os.path.join(_MIMIR_DIR_WS, "sessions")


def _scan_shell_jobs(session_id: str) -> list[DetachedJob]:
    root = os.path.join(_sessions_root(), session_id, "jobs")
    if not os.path.isdir(root):
        return []
    found: list[DetachedJob] = []
    for name in sorted(os.listdir(root)):
        job_dir = os.path.join(root, name)
        meta = _read_json(os.path.join(job_dir, "meta.json"))
        if meta is None:
            continue
        # A blocking run's scratch buffer, not a handle anybody was handed. Nothing is
        # waiting to hear about it and the call that made it is long gone.
        if meta.get("ephemeral"):
            continue
        code = _read_exit_code(job_dir)
        # The recorded status wins: the trap writes it as the shell leaves, so a run
        # that got there is finished whatever its pid now looks like.
        live = code is None and _is_running(meta.get("pid"), meta.get("pid_starttime"))
        found.append(DetachedJob(
            session_id=session_id,
            job_key=str(meta.get("job_key") or name),
            kind="shell",
            live=live,
            exit_code=code,
            command=str(meta.get("command") or ""),
            started_at=meta.get("started_at"),
        ))
    return found


def _scan_slurm_jobs(session_id: str) -> list[DetachedJob]:
    """Slurm jobs, which are never declared finished by inspection.

    Their state lives in the controller, not in a pid on this machine, so the only
    honest reading here is "there is a job, ask Slurm". They come back as live so a
    watcher is put on them, and the watcher's first poll is what settles it.
    """
    root = os.path.join(_sessions_root(), session_id, "hpc_jobs")
    if not os.path.isdir(root):
        return []
    found: list[DetachedJob] = []
    for name in sorted(os.listdir(root)):
        job_dir = os.path.join(root, name)
        id_path = os.path.join(job_dir, "slurm_job_id")
        try:
            with open(id_path, encoding="utf-8") as fh:
                job_id = fh.read().strip()
        except Exception:
            continue
        if not job_id:
            continue
        found.append(DetachedJob(session_id=session_id, job_key=job_id,
                                 kind="slurm", live=True, exit_code=None))
    return found


# Name of the marker that says a finished run's wake has been handed to a turn. A file
# in the run's own directory, because "has this been delivered?" has to survive the
# process that delivered it: two places look for runs that ended unwatched — a
# connection arriving and a worker being built — and a run delivered twice is a
# conversation woken twice for one build.
#
# **Delivered, not emitted.** The marker is written where the wake enters a turn, never
# where the event is queued. A ``job_complete`` put on the bus with nothing ready to
# consume it is a wake still owed: marking it at that moment makes the only record of
# the debt say it was already paid, and the run is then filtered out of every later scan
# — a conversation that waits for ever on a job that finished hours ago. Which is
# precisely the case this module exists for, so the marker follows the turn.
_REPORTED = "reported"

# Name of the per-session file that says "scanning has happened here before".
#
# Without it the first scan of an existing workspace reads a whole history as a backlog
# of wakes nobody is owed: a detached job's directory is never swept, however old, so
# every build that ever finished is sitting there with no marker on it — and markers
# only started being written when this was introduced. Waking a conversation for a
# two-month-old build is worse than missing one, and unlike a missed wake it is also
# unbounded.
#
# So the first scan establishes a baseline instead of claiming one: everything already
# finished is marked as reported and nothing is emitted. Afterwards a job that ends is
# genuinely one nothing has spoken for.
_BASELINE = ".wake_baseline"


def _job_dir(session_id: str, job_key: str, kind: str = "shell") -> str:
    sub = "hpc_jobs" if kind == "slurm" else "jobs"
    return os.path.join(_sessions_root(), session_id, sub, job_key)


def _baseline_path(session_id: str) -> str:
    # Beside the session's other sidecars, not inside ``jobs/``. That directory holds
    # job handles and is read as a list of them: anything else in it is a job to
    # whoever enumerates it, and a marker there reads as a run with no exit code —
    # which is to say a live one.
    return os.path.join(_sessions_root(), session_id, _BASELINE)


def has_baseline(session_id: str) -> bool:
    """Whether this session has been scanned before."""
    return os.path.exists(_baseline_path(session_id))


def establish_baseline(session_id: str, jobs: list[DetachedJob]) -> None:
    """Mark everything already finished as reported, and record that we have looked.

    Called instead of emitting, the first time a session is scanned. Best-effort like
    the markers themselves: a baseline that cannot be written costs one run of
    duplicate wakes, while refusing to scan because of it costs the feature.
    """
    for job in jobs:
        if not job.live:
            mark_reported(job)
    try:
        path = _baseline_path(session_id)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(str(time.time()))
    except Exception:
        logger.warning("job scan: could not record a baseline for %s", session_id,
                       exc_info=True)


def was_reported(job: DetachedJob) -> bool:
    """Whether this run's ending has already been announced."""
    return os.path.exists(os.path.join(
        _job_dir(job.session_id, job.job_key, job.kind), _REPORTED))


def mark_reported(job: DetachedJob) -> None:
    """Record that this run's ending has been announced. Never raises.

    Best-effort on purpose: a marker that cannot be written costs a duplicate wake,
    while refusing to announce the run because the marker failed costs the wake
    entirely. The first is a nuisance, the second is the bug this whole path exists to
    fix.
    """
    try:
        path = _job_dir(job.session_id, job.job_key, job.kind)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, _REPORTED), "w", encoding="utf-8") as fh:
            fh.write("1")
    except Exception:
        logger.warning("job scan: could not mark %s/%s reported",
                       job.session_id, job.job_key, exc_info=True)


def mark_job_reported(session_id: str | None, job_key: str,
                      server: str | None = None) -> None:
    """Mark a run reported when all that is in hand is its key.

    What a watcher has: it never built a :class:`DetachedJob`, and inventing one only
    to mark it would be the same write with more ceremony.
    """
    if not session_id or not job_key:
        return
    mark_reported(DetachedJob(
        session_id=session_id, job_key=job_key,
        kind="slurm" if server == "hpc" else "shell",
        live=False, exit_code=None))


def mark_wakes_reported(events: list[dict]) -> None:
    """Settle every run whose wake has just been handed to a turn.

    What a consumer has: the ``job_complete`` events it folded into one message, each
    carrying the session, the key and the server its descriptor named. Called once the
    turn is submitted — before that the debt is still outstanding, and a marker written
    early is indistinguishable from one written on time to every scan that follows.

    A steered wake is deliberately *not* settled here: a steer is only known to have
    been read when the loop says so, and until then the job still needs a turn of its
    own. Its marker is written by the flush that eventually gives it one.
    """
    for ev in events or ():
        mark_job_reported(ev.get("session_id"), ev.get("job_key"), ev.get("server"))


def scan_session(session_id: str, *, include_reported: bool = False
                 ) -> list[DetachedJob]:
    """Every detached run this session left on disk.

    Finished runs already announced are left out unless *include_reported* asks for
    them; a run still going is always returned, since putting a watcher back on one is
    deduplicated by the worker and costs nothing when it is already held.
    """
    if not session_id:
        return []
    try:
        found = _scan_shell_jobs(session_id) + _scan_slurm_jobs(session_id)
    except Exception:
        logger.warning("job scan: session %s could not be read", session_id,
                       exc_info=True)
        return []
    if include_reported:
        return found
    return [j for j in found if j.live or not was_reported(j)]


def scan_all_sessions() -> dict[str, list[DetachedJob]]:
    """Every session's detached runs, keyed by session."""
    root = _sessions_root()
    if not os.path.isdir(root):
        return {}
    out: dict[str, list[DetachedJob]] = {}
    try:
        names = sorted(os.listdir(root))
    except Exception:
        return {}
    for name in names:
        if not os.path.isdir(os.path.join(root, name)):
            continue
        jobs = scan_session(name)
        if jobs:
            out[name] = jobs
    return out
