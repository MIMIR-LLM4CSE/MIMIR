"""A batch job's record on disk and its state in the scheduler.

Every submission or adoption leaves a directory under the session's state: the id, the
scheduler that runs it, the log, and the terminal state once a poll settles it.
``normalized_job_state`` reads squeue then sacct and answers in one of three words."""
from __future__ import annotations

import os
import re
import shlex
from .._shared.responses import err
from .._shared.state_paths import sessions_root
from .._shared.slurm_site import (
    HOST_FILE as _HOST_FILE,
    ID_FILE as _ID_FILE,
    LOCAL as _LOCAL_HOST,
    hosts as slurm_hosts,
    job_host as _job_host,
    jobs_dir as _hpc_jobs_dir,
    queued_by_host as slurm_queued_by_host,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
from .._shared import cpu_facts
from . import cli
from .cli import (
    TIMEOUT_READ,
    lines_of,
    validate_comment,
    validate_mem,
    validate_time,
)

def add_unwatched(payload: dict) -> None:
    """Attach the user's running jobs that no descriptor covers, on every reachable scheduler.

    Only on submission and adoption, where the model holds ids; never on the watcher's
    poll. Best-effort.
    """
    by_host = slurm_unwatched_by_host(slurm_queued_by_host(TIMEOUT_READ))
    if not by_host:
        return
    # Flat, and in inventory order: the client's nudge reads a list of ids, and the
    # hint is what says which scheduler to name when there is more than one.
    payload["unwatched"] = [job_id for ids in by_host.values() for job_id in ids]
    payload["unwatched_hint"] = slurm_unwatched_hint(by_host)


def find_job_dir(job_id: str) -> tuple[str, str]:
    """``(job_dir, session)`` where *job_id* was recorded: this session ("") first, then siblings read-only."""
    def _match(base: str) -> str:
        try:
            names = os.listdir(base)
        except OSError:
            return ""
        for name in names:
            try:
                with open(os.path.join(base, name, _ID_FILE)) as fh:
                    if fh.read().strip() == job_id:
                        return os.path.join(base, name)
            except OSError:
                continue
        return ""

    found = _match(_hpc_jobs_dir())
    if found:
        return found, ""
    sessions = sessions_root()
    try:
        others = sorted(os.listdir(sessions))
    except OSError:
        return "", ""
    for sid in others:
        found = _match(os.path.join(sessions, sid, "hpc_jobs"))
        if found:
            return found, sid
    return "", ""


# What a job's own output is called, beside the work it did. ``%j`` is Slurm's, and the
# id is not knowable until the submission answers.
LOG_NAME = "slurm-%j.log"


# Where that resolved path is written down, so a later status answer can name the file
# instead of re-deriving a rule that may have changed since the job was submitted.
LOG_FILE = "log"


def slurm_background_job(job_id: str) -> dict:
    """The handle a client watcher polls this job with.

    One author: the three producers — a submission, an adoption, and a status answer
    re-arming its own job — were each writing this literal, so a new field or a renamed
    status op had three places to be remembered in.
    """
    return {
        "server":    "hpc",
        "job_key":   job_id,
        "kind":      "slurm-batch",
        "status_op": {"tool": "slurm_job_status", "args": {"job_id": job_id}},
    }


# The word a poll writes into a submission's directory once Slurm has stopped having an
# opinion about the job. This is the counterpart of a detached shell job's exit-code
# trap, and it exists for the same reader: whoever asks later whether this workspace
# still has work running has nothing but the descriptors on disk, and cannot ask Slurm.
# Without it a submission's directory says "there is a job" for ever — the one thing it
# records is an id, and an id never stops existing.
_STATE_FILE = "slurm_state"


# The states that mean Slurm is done answering for a job. ``unknown`` belongs here and is
# not a failure: it is what a job that has left both squeue and sacct reads as — the
# scheduler has forgotten it — and it is already what the client's watcher treats as
# terminal. Recording it keeps the directory saying what the watcher concluded, where
# leaving it blank would read as "still running" for ever.
TERMINAL_STATES = ("done", "crashed", "unknown")


def recorded_log(job_dir: str) -> str:
    """Where this submission told its job to write, as the submission wrote it down.

    A submission with no recorded path wrote ``slurm.log`` in its own directory.
    """
    try:
        with open(os.path.join(job_dir, LOG_FILE)) as fh:
            recorded = fh.read().strip()
    except OSError:
        recorded = ""
    if recorded:
        return recorded
    in_dir = os.path.join(job_dir, "slurm.log")
    return in_dir if os.path.exists(in_dir) else ""


def record_terminal_state(job_dir: str, state: str) -> None:
    """Write the terminal state of *job_dir* once, at the first poll that sees it; best-effort.

    Write-once: a later poll may read ``unknown`` past sacct's retention.
    """
    try:
        with open(os.path.join(job_dir, _STATE_FILE), "x") as fh:
            fh.write(state)
    except FileExistsError:
        pass
    except OSError:
        pass


def recorded_terminal_state(job_dir: str) -> str:
    """The terminal state a poll wrote in *job_dir*, or "" (no need to ask Slurm)."""
    try:
        with open(os.path.join(job_dir, _STATE_FILE)) as fh:
            word = fh.read().strip()
    except OSError:
        return ""
    return word if word in TERMINAL_STATES else ""


def record_job_host(job_dir: str, host: str) -> None:
    """Cache which scheduler a job is on, so the next poll asks only it; best-effort."""
    if not job_dir:
        return
    try:
        with open(os.path.join(job_dir, _HOST_FILE), "w") as fh:
            fh.write(host)
    except OSError:
        pass


def locate_job(job_id: str, requested: str = "",
                job_dir: str = "") -> tuple[str, str, str, list[str]]:
    """``(host, state, raw, nodes)`` of *job_id*: its recorded host first, then every scheduler.

    *requested* skips the search; *job_dir* avoids a second walk. ``("", "unknown", "", [])``
    when nothing knows it.
    """
    if requested:
        state, raw, nodes = normalized_job_state(job_id, host=requested)
        return requested, state, raw, nodes
    if not job_dir:
        job_dir, _ = find_job_dir(job_id)
    hint = _job_host(job_dir) if job_dir else None
    order = [hint] if hint is not None else []
    order += [host for host in slurm_hosts() if host != hint]
    for host in order:
        state, raw, nodes = normalized_job_state(job_id, host=host)
        if state != "unknown":
            # Found, so the next poll of this job is one call rather than a sweep. A
            # recorded host is never replaced: it is what slurm_cancel trusts.
            if job_dir and hint is None:
                record_job_host(job_dir, host)
            return host, state, raw, nodes
    return "", "unknown", "", []


_SACCT_CRASH_PREFIXES = (
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "BOOT_FAIL", "DEADLINE", "PREEMPTED",
)


_ACTIVE_STATES = ("RUNNING", "COMPLETING")


_WAITING_STATES = ("PENDING", "CONFIGURING", "REQUEUED")


# A job is over when it says so, wherever it says it. squeue reports the same words
# sacct does and keeps a finished job for a while — a cancelled one can sit there
# saying CANCELLED — so these are read from either answer. Prefixes because Slurm
# qualifies some of them ("CANCELLED by 1234").
_DONE_PREFIXES = ("COMPLETED",)


def _is_crash_state(state: str) -> bool:
    return any(state.startswith(prefix) for prefix in _SACCT_CRASH_PREFIXES)


def _is_done_state(state: str) -> bool:
    return any(state.startswith(prefix) for prefix in _DONE_PREFIXES)


def normalized_job_state(job_id: str, *,
                          host: str = _LOCAL_HOST) -> tuple[str, str, list[str]]:
    """``(state, raw, nodes)`` of a job: running, pending, done, crashed or unknown.

    ``squeue`` for live jobs, ``sacct`` for finished ones; nodes only while running. For an
    array, any live element keeps it live, and the verdict is the worst element's.
    """
    rows = lines_of(cli.run_bash(f"squeue -j {shlex.quote(job_id)} -h -o '%T|%R'",
                            TIMEOUT_READ, host=host))
    queued, where = [], []
    for row in rows:
        state, _, reason_or_node = row.partition("|")
        queued.append(state.strip())
        where.append(reason_or_node.strip())
    raw = ",".join(sorted(set(queued)))
    if any(state in _ACTIVE_STATES for state in queued):
        nodes = sorted({node
                        for state, expr in zip(queued, where)
                        if state in _ACTIVE_STATES
                        for node in cpu_facts.expand_hostlist(expr) if node})
        return "running", raw, nodes
    if queued and all(state in _WAITING_STATES for state in queued):
        return "pending", raw, []
    # Over, by the queue's own answer. Read before sacct is asked, for two reasons: it
    # saves a round trip in the ordinary case, and it is the only thing that settles a
    # job on a site with no accounting at all — where ``sacct`` answers nothing and the
    # fallthrough below reads "in the queue with a state I do not recognise" as still
    # running. For a CANCELLED job that is a watcher polling, for as long as the process
    # lives, something that will never change again.
    if queued and all(_is_done_state(s) or _is_crash_state(s) for s in queued):
        # The worst outcome among the elements, as below: an array where one task was
        # cancelled is not an array that completed.
        if any(_is_crash_state(s) for s in queued):
            return "crashed", raw, []
        return "done", raw, []

    accounted = lines_of(cli.run_bash(f"sacct -j {shlex.quote(job_id)} -n -X -o State",
                                 TIMEOUT_READ, host=host))
    sraw = ",".join(sorted(set(accounted)))
    # Ahead of the "anything not COMPLETED is crashed" fallback below, which is where
    # most of these land anyway. What this earns is the one case the fallback cannot
    # reach: a terminal crash line present *while* squeue still reports a state that is
    # neither active nor waiting, which the clause below would read as running.
    crashed = [state for state in accounted if _is_crash_state(state)]
    if crashed:
        return "crashed", sraw, []
    if accounted and all(_is_done_state(state) for state in accounted):
        return "done", sraw, []
    if queued:                    # in the queue with a non-standard active state
        return "running", raw, []
    if accounted:                 # some other terminal state we don't classify
        return "crashed", sraw, []
    return "unknown", "", []


# A job, or one task of a job array. Anything wider — a user, a partition, a name — is
# what `scancel` would also take, and is not something to approve from one card.
JOB_ID_RE = re.compile(r"\d+(_\d+)?")


def salloc_argv(partition: str, account: str, qos: str, nodes: int, ntasks: int,
                 cpus_per_task: int, mem: str, time: str, gres: str, constraint: str,
                 job_name: str, extra_args: str,
                 comment: str = "") -> tuple[list[str], dict | None]:
    """Build a validated salloc argv, or return the rejection."""
    if not partition.strip():
        return [], err("partition is required.", hint="Use slurm_partitions() to list them.")
    if nodes < 1 or ntasks < 1 or cpus_per_task < 1:
        return [], err("nodes, ntasks, and cpus_per_task must be >= 1")
    if time and not validate_time(time):
        return [], err("Invalid Slurm time format", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not validate_mem(mem):
        return [], err("Invalid mem format", hint="Use values like 8G, 32000M, 1T")
    comment_err = validate_comment(comment)
    if comment_err:
        return [], err(comment_err)

    argv = [
        "salloc",
        f"--partition={partition}",
        f"--nodes={nodes}",
        f"--ntasks={ntasks}",
        f"--cpus-per-task={cpus_per_task}",
        f"--time={time}",
        f"--job-name={job_name}",
    ]
    for flag, value in (("account", account), ("qos", qos), ("mem", mem),
                        ("gres", gres), ("constraint", constraint),
                        ("comment", comment)):
        if value:
            argv.append(f"--{flag}={value}")
    if extra_args:
        # argv never reaches a shell, so metacharacters cannot inject; requiring a
        # leading dash is what stops extra_args smuggling in a *command* to allocate for.
        try:
            extra = shlex.split(extra_args)
        except ValueError as exc:
            return [], err(f"Could not parse extra_args: {exc}")
        if any(not tok.startswith("-") for tok in extra):
            return [], err("extra_args accepts salloc flags only.",
                           hint="Every token must start with '-'; pass resources via the named arguments.")
        argv += extra
    return argv, None
