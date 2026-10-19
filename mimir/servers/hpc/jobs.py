"""A batch job's record on disk and its state in the scheduler.

Every submission or adoption leaves a directory under the session's state: the id, the
scheduler that runs it, the log, and the terminal state once a poll settles it.
``normalized_job_state`` reads squeue then sacct and answers in one of three words."""
from __future__ import annotations

import os
import re
import shlex
from ...shared.responses import err
from ...shared.state_paths import sessions_root
from ...shared.jobs import SLURM_JOB_ID
from ...shared.slurm_site import (
    slurm_state,
    HOST_FILE as _HOST_FILE,
    LOCAL as _LOCAL_HOST,
    hosts as slurm_hosts,
    job_host as _job_host,
    jobs_dir as _hpc_jobs_dir,
    queued_by_host as slurm_queued_by_host,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
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
                with open(os.path.join(base, name, SLURM_JOB_ID)) as fh:
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


def normalized_job_state(job_id: str, *,
                          host: str = _LOCAL_HOST) -> tuple[str, str, list[str]]:
    """``(state, raw, nodes)`` of a job on *host*: running, pending, done, crashed or
    unknown (``shared.slurm_site.slurm_state``)."""
    return slurm_state(job_id, lambda script: lines_of(cli.run_bash(script, TIMEOUT_READ,
                                                                    host=host)))


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
