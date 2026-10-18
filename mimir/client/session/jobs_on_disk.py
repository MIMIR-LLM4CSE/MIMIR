"""The detached runs a session left behind, read off the disk.

Descriptors under ``<state>/sessions/<sid>/jobs/`` and ``.../hpc_jobs/`` say which runs
still go and which ended unwatched. Contracts with the servers: ``exit_code`` (the
``bash_jobs`` trap) and ``slurm_state`` (``server_hpc``).
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass

from ...servers._shared.processes import is_running
from ...servers._shared.state_paths import sessions_root, subagents_dir
from ..config import paths

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
    # Where the run's files are. Carried because a job key alone resolves against the
    # *reading* session, and a sub-agent's run lives under the child's session — which
    # a bare key reads as "no such job", i.e. ``unknown``, i.e. terminal. A run read
    # through the wrong session would be announced as lost while it is still going.
    run_dir: str = ""
    # The handle of a run whose own server watches it — a proxy run is a Slurm job of
    # this workspace and that server's state machine both. Re-arming it with the stock
    # poll instead would put a second watcher on it under a different key, which no
    # dedup catches and which wakes the conversation twice for one result.
    watcher: dict | None = None

    @property
    def owner_session(self) -> str:
        """The conversation to wake for this run: the parent's, for a sub-agent's run."""
        return self.session_id.split("/subagents/", 1)[0]

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
        if self.watcher:
            return dict(self.watcher.get("status_op") or {})
        if self.kind == "slurm":
            # The name the HPC server actually registers, and it has to be: a watcher
            # polling a tool nothing answers to reads unreadable answers until it gives
            # the run up as untrackable, settling nothing — so the next scan re-arms it,
            # and the one after that, for as long as the process lives.
            return {"tool": "slurm_job_status", "args": {"job_id": self.job_key}}
        args = {"job_key": self.job_key}
        if self.run_dir:
            args["run_dir"] = self.run_dir
        return {"tool": "bash_job", "args": args}

    def descriptor(self) -> dict:
        """What ``_register_bg_job`` needs to put a watcher back on this run."""
        if self.watcher:
            return dict(self.watcher)
        return {"server": "hpc" if self.kind == "slurm" else "bash",
                "job_key": self.job_key,
                "kind": "slurm-job" if self.kind == "slurm" else "shell-command",
                **({"run_dir": self.run_dir} if self.run_dir else {}),
                "status_op": self.status_op()}


def _read_json(path: str) -> dict | None:
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        logger.debug("_read_json: a best-effort step failed and was skipped", exc_info=True)
        return None


# What ``server_hpc`` writes into a submission's directory once Slurm has let the job go:
# one of done|crashed|unknown. The name is shared with that module by convention, the way
# ``exit_code`` is shared with ``bash_jobs`` (see the module docstring).
_SLURM_STATE = "slurm_state"

# How long an unsettled Slurm submission reads as running: far past any wall-time, since
# a live watcher settles a finished job within a poll. Only an unwatched submission is
# bounded by it; erring short would stop a server that still owes a wake.
_DEFAULT_SLURM_STALE_AFTER = 7 * 24 * 3600.0


def _slurm_stale_after() -> float:
    try:
        return max(3600.0, float(os.environ.get("MIMIR_SLURM_STALE_AFTER", "")
                                 or _DEFAULT_SLURM_STALE_AFTER))
    except (TypeError, ValueError):
        return _DEFAULT_SLURM_STALE_AFTER


def _read_slurm_state(job_dir: str) -> str | None:
    """The terminal state recorded for this submission, or None while it has none."""
    try:
        with open(os.path.join(job_dir, _SLURM_STATE), encoding="utf-8") as fh:
            word = fh.read().strip()
    except OSError:
        return None
    return word if word in ("done", "crashed", "unknown") else None


def _older_than(path: str, age: float) -> bool:
    """Whether *path* was last written more than *age* ago. False when unreadable.

    Unreadable means unknown, and unknown must not age a submission out: that direction
    stops a server over a stat() that failed.
    """
    try:
        return (time.time() - os.path.getmtime(path)) > age
    except OSError:
        return False


def _written_at(path: str) -> float | None:
    """When *path* was last written. The submission time of a Slurm job, since the file
    that carries its id is written once, as the job is submitted."""
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def is_settled(job_dir: str) -> bool:
    """Whether a run's directory records that it is over: an exit code, or a recorded Slurm state."""
    return (os.path.exists(os.path.join(job_dir, "exit_code"))
            or _read_slurm_state(job_dir) is not None)


def _read_exit_code(job_dir: str) -> int | None:
    path = os.path.join(job_dir, "exit_code")
    try:
        with open(path, encoding="utf-8") as fh:
            return int(fh.read().strip())
    except Exception:
        logger.debug("_read_exit_code: a best-effort step failed and was skipped", exc_info=True)
        return None


def _sessions_root() -> str:
    return sessions_root(paths.STATE_DIR)


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
        live = code is None and is_running(meta.get("pid"), meta.get("pid_starttime"))
        found.append(DetachedJob(
            session_id=session_id,
            job_key=str(meta.get("job_key") or name),
            kind="shell",
            live=live,
            exit_code=code,
            command=str(meta.get("command") or ""),
            started_at=meta.get("started_at"),
            run_dir=job_dir,
        ))
    return found


def _scan_slurm_jobs(session_id: str) -> list[DetachedJob]:
    """Slurm jobs, live until a poll has written their state (``slurm_job_status``).

    Nothing local can conclude a Slurm job ended. A submission nothing settles reads as
    over after :data:`_DEFAULT_SLURM_STALE_AFTER`.
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
            logger.debug("_scan_slurm_jobs: a best-effort step failed and was skipped", exc_info=True)
            continue
        if not job_id:
            continue
        # A run whose own server watches it says so beside its id; the key then is
        # that watcher's, so a watcher still live is recognised as already holding it.
        watcher = _read_json(os.path.join(job_dir, "watcher.json"))
        if not isinstance(watcher, dict) or not watcher.get("job_key"):
            watcher = None
        state = _read_slurm_state(job_dir)
        if state is None and _older_than(id_path, _slurm_stale_after()):
            logger.info("job scan: Slurm submission %s of session %s has gone "
                        "unsettled past the horizon; reading it as untrackable",
                        job_id, session_id)
            state = "unknown"
        # ``unknown`` is a terminal state with no outcome, which is exactly what a None
        # exit code means to DetachedJob.state — so the mapping is total and says what
        # was observed rather than guessing an outcome nobody saw.
        found.append(DetachedJob(
            session_id=session_id,
            job_key=str(watcher["job_key"]) if watcher else job_id,
            kind="slurm",
            live=state is None,
            exit_code={"done": 0, "crashed": 1}.get(state or ""),
            started_at=_written_at(id_path),
            watcher=watcher,
        ))
    return found


# The marker that a finished run's wake was handed to a turn, in the run's own
# directory so it survives the process. Written where the wake enters a turn, never
# where it is queued: marked early, a debt would read as paid.
_REPORTED = "reported"

# The per-session marker that this session was scanned before. The first scan marks
# everything already finished as reported and emits nothing: job directories are never
# swept, and waking for a months-old build is worse than missing one.
_BASELINE = ".wake_baseline"


def _job_dir(session_id: str, job_key: str, kind: str = "shell") -> str:
    """The directory a run's markers belong in (a Slurm key is resolved to its submission dir)."""
    if kind == "slurm":
        return _slurm_job_dir(session_id, job_key)
    if kind == "proxy":
        # A proxy run is filed among the Slurm descriptors when it submitted a job, and
        # nowhere but its own tree when it ran locally. Resolved rather than assumed, so
        # a marker written by the live watcher and one written by a late scan of the
        # same run land in the same directory — otherwise the run is announced twice.
        found = _slurm_job_dir(session_id, job_key, invent=False)
        if found:
            return found
    return os.path.join(_sessions_root(), session_id, "jobs", job_key)


def _slurm_hpc_root(session_id: str) -> str:
    return os.path.join(_sessions_root(), session_id, "hpc_jobs")


def _slurm_job_dir(session_id: str, job_id: str, *, invent: bool = True) -> str:
    """Where this session recorded *job_id*, else ``hpc_jobs/<job id>`` (when *invent*).

    The twin of ``server_hpc._find_job_dir``, narrowed to one session.
    """
    root = _slurm_hpc_root(session_id)
    try:
        names = sorted(os.listdir(root))
    except OSError:
        names = []
    for name in names:
        job_dir = os.path.join(root, name)
        try:
            with open(os.path.join(job_dir, "slurm_job_id"), encoding="utf-8") as fh:
                if fh.read().strip() == job_id:
                    return job_dir
        except OSError:
            continue
        # A run watched by its own server is keyed by that watcher, not by the Slurm
        # id, and its marker belongs beside the descriptor all the same.
        watcher = _read_json(os.path.join(job_dir, "watcher.json"))
        if isinstance(watcher, dict) and str(watcher.get("job_key") or "") == job_id:
            return job_dir
    if not invent:
        return ""
    return os.path.join(root, job_id)


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
    """Whether this run's ending has already been announced.

    Both candidate directories are checked for a Slurm job: markers written before its id
    was resolved to a descriptor sit under the id itself, and reading only the resolved
    directory would announce every one of those runs a second time.
    """
    candidates = [_job_dir(job.session_id, job.job_key, job.kind)]
    if job.watcher:
        # Where the live watcher's own completion writes: the same run under the other
        # spelling of its directory.
        candidates.append(os.path.join(_sessions_root(), job.session_id, "jobs",
                                       job.job_key))
    if job.kind == "slurm":
        candidates.append(os.path.join(_slurm_hpc_root(job.session_id), job.job_key))
    return any(os.path.exists(os.path.join(base, _REPORTED))
               for base in dict.fromkeys(candidates))


def mark_reported(job: DetachedJob) -> None:
    """Record that this run's ending has been announced. Never raises.

    Best-effort on purpose: a marker that cannot be written costs a duplicate wake,
    while refusing to announce the run because the marker failed costs the wake
    entirely.
    """
    try:
        path = _job_dir(job.session_id, job.job_key, job.kind)
        os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, _REPORTED), "w", encoding="utf-8") as fh:
            fh.write("1")
    except Exception:
        logger.warning("job scan: could not mark %s/%s reported",
                       job.session_id, job.job_key, exc_info=True)


def mark_job_reported(session_id: str | None, job_key: str | None,
                      server: str | None = None) -> None:
    """Mark a run reported when all that is in hand is its key.

    What a watcher has: it never built a :class:`DetachedJob`, and inventing one only
    to mark it would be the same write with more ceremony.
    """
    if not session_id or not job_key:
        return
    mark_reported(DetachedJob(
        session_id=session_id, job_key=job_key,
        kind={"hpc": "slurm", "proxy": "proxy"}.get(server or "", "shell"),
        live=False, exit_code=None))


def mark_wakes_reported(events: list[dict]) -> None:
    """Mark reported every run whose wake was just handed to a turn.

    Called once the turn is submitted; a steered wake is marked later, by the flush that
    gives it a turn.
    """
    for ev in events or ():
        mark_job_reported(_run_session(ev), ev.get("job_key"), ev.get("server"))


def _run_session(ev: dict) -> str | None:
    # A sub-agent's run is woken in the parent but its files, and markers, are the child's.
    return ev.get("run_session") or ev.get("session_id")


def wake_reported(ev: dict) -> bool:
    """Whether the run this ``job_complete`` announces has already been handed to a turn.

    Checked by the consumers, because pending-only dedup misses an announcement that
    arrives after the first one was delivered — and several emitters announce one run.
    """
    session_id, job_key = _run_session(ev), ev.get("job_key")
    if not session_id or not job_key:
        return False
    try:
        return was_reported(DetachedJob(
            session_id=session_id, job_key=str(job_key),
            kind={"hpc": "slurm", "proxy": "proxy"}.get(ev.get("server") or "", "shell"),
            live=False, exit_code=None))
    except Exception:
        logger.debug("wake_reported: a best-effort step failed and was skipped", exc_info=True)
        return False


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


def subagent_sessions(session_id: str) -> list[str]:
    """The sub-sessions of *session_id* (``<id>/subagents/<child>``), one level only."""
    root = subagents_dir(session_id, paths.STATE_DIR)
    if not os.path.isdir(root):
        return []
    try:
        names = sorted(os.listdir(root))
    except Exception:
        logger.debug("subagent_sessions: a best-effort step failed and was skipped", exc_info=True)
        return []
    return [f"{session_id}/subagents/{name}" for name in names
            if os.path.isdir(os.path.join(root, name))]


def scan_all_sessions() -> dict[str, list[DetachedJob]]:
    """Every session's detached runs, sub-agents' included, keyed by the session holding the files.

    A killed child's runs are found only here. See :attr:`DetachedJob.owner_session`.
    """
    root = _sessions_root()
    if not os.path.isdir(root):
        return {}
    out: dict[str, list[DetachedJob]] = {}
    try:
        names = sorted(os.listdir(root))
    except Exception:
        logger.debug("scan_all_sessions: a best-effort step failed and was skipped", exc_info=True)
        return {}
    for name in names:
        if not os.path.isdir(os.path.join(root, name)):
            continue
        for session_id in (name, *subagent_sessions(name)):
            jobs = scan_session(session_id)
            if jobs:
                out[session_id] = jobs
    return out
