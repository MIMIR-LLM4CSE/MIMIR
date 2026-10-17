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
    """Name the user's running jobs that no descriptor covers, if any.

    Attached to the answers of the two tools that *record* a job — a submission and an
    adoption — because those are the moments the model is holding job ids and can act
    on it in the same turn. Deliberately not to ``slurm_job_status``: that one is the
    watcher's poll target, called on a timer for the life of every watched run, and a
    per-user queue read on it would be paid for ever.

    Asked of every scheduler this workspace reaches, because work nobody is watching
    is not more acceptable for being on the other cluster — and because the model
    holding these ids has just chosen between those schedulers and can act on either.

    Best effort: a scheduler that will not answer leaves the payload alone rather than
    turning a successful submission into an error.
    """
    by_host = slurm_unwatched_by_host(slurm_queued_by_host(TIMEOUT_READ))
    if not by_host:
        return
    # Flat, and in inventory order: the client's nudge reads a list of ids, and the
    # hint is what says which scheduler to name when there is more than one.
    payload["unwatched"] = [job_id for ids in by_host.values() for job_id in ids]
    payload["unwatched_hint"] = slurm_unwatched_hint(by_host)


def find_job_dir(job_id: str) -> tuple[str, str]:
    """Where *job_id* was recorded, and which session recorded it ("" for this one).

    Looks in this session's directory first, then — read-only — in the sibling sessions'.
    A Slurm job can outlive the conversation that submitted it and still need checking from
    another one, so filing it per session must not make it unfindable. Returns ``("", "")``
    when no session recorded it.
    """
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
    """Settle *job_dir* at the first poll that finds Slurm has let the job go.

    Write-once, and that is the point: the first observation of a terminal state is the
    truest one available. A later poll of the same job reads ``unknown`` once sacct's
    retention window has passed, and letting that overwrite a recorded ``crashed`` would
    turn every outcome into "the scheduler has forgotten it" given enough time.

    Best-effort, like the markers in ``session.jobs_on_disk``: a status answer is worth more than the
    record of it, so a directory that cannot be written costs a settled descriptor, never
    the reply. Written for another conversation's job too — the record belongs to the job,
    not to whoever happened to look, and the reader that needs it most is scanning
    sessions that may have no agent at all.
    """
    try:
        with open(os.path.join(job_dir, _STATE_FILE), "x") as fh:
            fh.write(state)
    except FileExistsError:
        pass
    except OSError:
        pass


def recorded_terminal_state(job_dir: str) -> str:
    """The terminal state a poll wrote in *job_dir*, or "" while it has none.

    The counterpart of :func:`record_terminal_state`, and the same question the client
    scan asks of the same file (``jobs_on_disk.is_settled``). What it buys here is an answer
    about a recorded job without asking Slurm: over ssh a sweep is two round trips per
    scheduler, and "has this one already ended?" is on disk.
    """
    try:
        with open(os.path.join(job_dir, _STATE_FILE)) as fh:
            word = fh.read().strip()
    except OSError:
        return ""
    return word if word in TERMINAL_STATES else ""


def record_job_host(job_dir: str, host: str) -> None:
    """Write down which scheduler a job is on, so the next poll asks only it.

    A cache and not a binding: it is written when a job is submitted and whenever a
    sweep has just found one, and :func:`locate_job` sweeps again whenever it is
    absent or wrong. Best effort — a directory that cannot be written costs one extra
    ssh per poll, never the answer.
    """
    if not job_dir:
        return
    try:
        with open(os.path.join(job_dir, _HOST_FILE), "w") as fh:
            fh.write(host)
    except OSError:
        pass


def locate_job(job_id: str, requested: str = "",
                job_dir: str = "") -> tuple[str, str, str, list[str]]:
    """Find *job_id*: ``(host, state, raw, nodes)``, among the schedulers reached.

    The hint first — the descriptor's own record, which makes the common case one call
    — then every other reachable scheduler until one knows the id. That fallback is
    what keeps a job trackable without being owned: an adoption of somebody's job, a
    descriptor from before the host was written down, a watcher re-armed from disk
    after a window reload (which polls by id alone, having no host to pass), all land
    here and are answered.

    *requested* skips the search when a caller already knows — a sweep refused for one
    job is not worth paying twice. *job_dir* is for the caller that has already found
    the descriptor, so the sessions tree is walked once per call rather than twice.
    ``("", "unknown", "", [])`` when nothing has heard of it.
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
    """Map a Slurm job's state to the shared vocab: running|pending|done|crashed|unknown.

    Active jobs come from ``squeue``; finished jobs have left the queue, so their
    terminal state is read from ``sacct``. Returns ``(state, raw, nodes)``. Missing
    tools or an unknown job yield ``("unknown", "", [])`` — a terminal signal for the
    watcher.

    A job the queue already reports as finished — COMPLETED, FAILED, CANCELLED — is
    settled from that answer alone. Nothing is gained by following it further, and a
    site without accounting has no second answer to settle it with.

    *nodes* is where the job is actually running, read from the same ``squeue`` call
    rather than a second one. Slurm's ``%R`` column is the nodelist for a running job
    and the *reason it is waiting* for a pending one, so it is only read as nodes when
    the job is running — a pending job has no node yet, which is the honest answer and
    not an empty one to be filled in later.

    **One id can be many jobs.** An array submits as ``15306621`` and runs as
    ``15306621_1 … _8``, and both commands answer for every element: ``squeue`` prints
    a line each, ``sacct`` likewise. Read as one line that is a job whose first element
    has finished while seven still run — which is reported as finished, waking its
    conversation on a run that has barely started. So the answer is taken over the
    whole set: anything still going keeps the job live, and a terminal verdict is the
    worst outcome among the elements, since an array where one task failed is not an
    array that completed.
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
