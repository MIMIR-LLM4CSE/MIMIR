"""A batch job's record on disk and its state in the scheduler.

Every submission or adoption leaves a directory under the session's state: the id, the
scheduler that runs it, the log, and the terminal state once a poll settles it.
``normalized_job_state`` reads squeue then sacct and answers in one of three words."""
from __future__ import annotations

import os
import re
import shlex
from ...shared.responses import err, ok
from ...shared import shell_activity
from ...shared.state_paths import sessions_root
from typing import Any

from ...shared import build_progress, slurm_site
from ...shared.jobs import (SLURM_JOB_ID, is_terminal, record_terminal_state, recorded_terminal_state,
                            slurm_descriptor)
from ...shared.slurm_site import (
    slurm_state,
    HOST_FILE as _HOST_FILE,
    LOCAL as _LOCAL_HOST,
    hosts as slurm_hosts,
    job_host as _job_host,
    jobs_dir as _hpc_jobs_dir,
    name_of as _host_name,
    new_job_dir,
    record_host,
    queued_by_host as slurm_queued_by_host,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
from . import cli
from .cli import (
    TIMEOUT_READ,
    TIMEOUT_SUBMIT,
    no_scheduler_hint,
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


def sbatch_on(host: str, script: str, hint: str) -> tuple[str, dict | None]:
    """Submit *script* to the scheduler on *host*: its job id, or the error answer.

    On stdin, not by path: ``sbatch`` reads its script from standard input when given no
    file, so nothing of ours has to exist on the host running it — the one thing a
    scheduler behind ``ssh`` cannot promise.
    """
    res = cli.run_bash("sbatch", TIMEOUT_SUBMIT, stdin=script, host=host)
    if res.get("status") != "ok":
        return "", err(res.get("stderr") or res.get("error", "sbatch failed"), hint=hint)
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return "", err("Could not parse a job id from sbatch output.",
                       hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    return match.group(1), None


def submit_batch(host: str, work_dir: str, command: str, header: list[str], *,
                 partition: str, comment: str) -> dict:
    """Submit *command* under *header* on *host*, and file what came back.

    A directory of its own records the script, the id, the log and the scheduler; the
    answer carries the ``background_job`` the client polls.
    """
    job_dir = new_job_dir()
    script_path = os.path.join(job_dir, "batch_script.sh")
    script = "\n".join(header + ["", command, ""]) + "\n"
    # Kept for a reader — a model or a person asking what was actually submitted —
    # and not for the scheduler, which is handed the text itself below.
    try:
        with open(script_path, "w") as fh:
            fh.write(script)
    except OSError as exc:
        return err(f"Could not write batch script: {exc}")

    job_id, refusal = sbatch_on(host, script, "Check partition, account/qos, and requested "
                                              "resources. " + no_scheduler_hint(host))
    if refusal:
        return refusal
    log_file = os.path.join(work_dir, LOG_NAME.replace("%j", job_id))
    try:
        with open(os.path.join(job_dir, SLURM_JOB_ID), "w") as fh:
            fh.write(job_id)
        with open(os.path.join(job_dir, LOG_FILE), "w") as fh:
            fh.write(log_file)
    except OSError:
        pass
    # Beside the id, so the watcher's first poll goes straight to the right scheduler
    # instead of sweeping. And into the session's inventory: a submission that
    # succeeded is proof this host reaches that scheduler, which is the same evidence
    # a queue read gives, from the call that mattered more.
    record_job_host(job_dir, host)
    if host:
        record_host(host)

    payload = {
        "job_id":       job_id,
        "job_dir":      job_dir,
        "batch_script": script_path,
        "work_dir":     work_dir,
        "log":          log_file,
        "partition":    partition,
        "scheduler":    _host_name(host),
        "comment":      comment or None,
        "note":         f"Slurm job {job_id} submitted to '{partition}' on "
                        f"{_host_name(host)}.",
        "background_job": slurm_descriptor(job_id),
    }
    # The submission is the other moment the model holds job ids: what else of the
    # user's is running unaccounted for can be acted on here, in the same turn.
    add_unwatched(payload)
    return ok(payload)


def launched_from_this_session(job_id: str, requested: str = "") -> str | None:
    """The scheduler of *job_id* when it was submitted while a command of this session ran
    and no other session's did, else None: whose a job submitted by a launcher or a script
    is. The submit time is read on the scheduler's host and in its own time zone.
    """
    # Read on disk first: a session that ran no command submitted nothing this way, and
    # the scheduler is not asked about a job that cannot be its own.
    if not shell_activity.ran_commands():
        return None
    host, state, _raw, _nodes = locate_job(job_id, requested)
    if state == "unknown":
        return None
    queued = cli.run_argv(["squeue", "-h", "-j", job_id, "-o", "%V"], TIMEOUT_READ, host=host)
    stamp = (queued.get("stdout") or "").strip().splitlines()
    if queued.get("status") != "ok" or not stamp:
        return None
    epoch = cli.run_argv(["date", "-d", stamp[0].strip(), "+%s"], TIMEOUT_READ, host=host)
    try:
        moment = float((epoch.get("stdout") or "").strip())
    except ValueError:
        return None
    return host if shell_activity.submitted_by_this_session(moment) else None


# ── the job tools' bodies: status, adoption, cancellation ──────────────────────


def status(job_id: str, scheduler: str = "") -> dict:
    """The normalized status of *job_id*, settling its directory when it has ended (``slurm_job_status``)."""
    if not str(job_id).strip():
        return err("job_id is required.")
    job_id = str(job_id).strip()
    # Where the submission was recorded, so the script and the log stay reachable. A
    # Slurm job outlives its conversation, so the sibling sessions are searched too and the
    # answer says when the job came from another one.
    job_dir, from_session = find_job_dir(job_id)
    host, state, raw, nodes = locate_job(job_id, scheduler, job_dir)
    payload: dict[str, Any] = {"job_id": job_id, "state": state, "raw_state": raw}
    if host or state != "unknown":
        payload["scheduler"] = slurm_site.name_of(host)
    # Which machines it got. Only ever present for a running job: a pending one has no
    # node yet, and Slurm's own column there is the reason it is waiting.
    if nodes:
        payload["nodes"] = nodes
    if job_dir:
        payload["job_dir"] = job_dir
        # Read, not re-derived: the submission wrote down where it told the job to
        # write, and that answer has to survive a change of rule between versions —
        # a job submitted by an older build still has its log where that build put it.
        log = recorded_log(job_dir)
        if log:
            payload["log"] = log
            if state == "running":
                tail = build_progress.last_lines(log)
                if tail:
                    payload["log_tail"] = tail
        if from_session:
            payload["submitted_by_another_session"] = from_session
        # And if Slurm has let go, say so in the directory. See record_terminal_state:
        # this poll is the only thing that will ever know, and the readers that decide
        # whether a workspace still has work running cannot ask the scheduler.
        if is_terminal(state):
            record_terminal_state(job_dir, state)
            # A finished job whose log never appeared says something specific, and it
            # is not about the job: the directory it was told to write to is not one
            # the node that ran it could write. Named here because this is where it
            # becomes knowable, and because the alternative is reading an empty
            # ``tail`` as a job that printed nothing.
            if log and not os.path.exists(log):
                payload["log_missing"] = (
                    f"The job wrote no log at {log}. A job writes its output from the "
                    "node it ran on, so that directory was not one that node could "
                    "write — pass work_dir to sbatch_submit and the log lands beside "
                    "the work, where it is certain to be reachable.")
    # A handle for a job of this conversation still in flight, so asking re-arms its watcher;
    # never for another session's job (its wake would land here).
    if state in ("running", "pending") and not from_session:
        payload["background_jobs"] = [slurm_descriptor(job_id)]
    return ok(payload)


def adopt(job_id: str, note: str = "", scheduler: str = "") -> dict:
    """Record *job_id*, submitted outside MIMIR, as this conversation's, for its watcher (``slurm_watch``)."""
    job_id = str(job_id or "").strip()
    if not job_id:
        return err("job_id is required.",
                   hint="Use the id the submission printed, or slurm_queue() to list "
                        "what is running.")
    if not JOB_ID_RE.fullmatch(job_id):
        return err(f"{job_id!r} is not a Slurm job id.",
                   hint="One job, or one array task: 15306620 or 15306621_3.")

    # The disk first, and the scheduler only on the path that writes something. An id
    # already recorded is the common answer when a model adopts an array task by task,
    # and polling Slurm to tell it "nothing to adopt" buys nothing — over ssh it is two
    # round trips, per scheduler swept, for an answer the directory already has.
    existing, from_session = find_job_dir(job_id)
    if existing:
        settled = recorded_terminal_state(existing)
        payload: dict[str, Any] = {
            "job_id": job_id,
            "job_dir": existing,
            "recorded_by": from_session or "this conversation",
            "note": (f"Job {job_id} is already recorded ({settled}); nothing to adopt."
                     if settled else
                     f"Job {job_id} is already recorded; nothing to adopt. "
                     "slurm_job_status(job_id) for where it is at."),
        }
        if settled:
            payload["state"] = settled
        # Recorded is not watched: hand the descriptor over so a lost watcher is put back (the
        # client dedups). Withheld for a sibling's job, as in slurm_job_status.
        if not settled and not from_session:
            payload["background_jobs"] = [slurm_descriptor(job_id)]
        return ok(payload)

    # Asked before anything is written, because a descriptor for a job no scheduler has
    # heard of is worse than no descriptor: it holds this process open until the
    # staleness horizon for a run that does not exist. And asked of each of them, which
    # is what makes an id enough — the caller holding a number from a launcher's output
    # has no reason to know which cluster printed it.
    host, state, raw, nodes = locate_job(job_id, scheduler)
    if state == "unknown":
        known = ", ".join(slurm_site.name_of(h) for h in slurm_site.hosts()) or "none reachable"
        return err(f"No scheduler this workspace reaches knows job {job_id} ({known}).",
                   hint="Check the id with slurm_queue(). A job that has already left "
                        "both squeue and sacct cannot be watched, and one on a "
                        "scheduler not listed above needs it added — "
                        "slurm_schedulers(add=\"<host>\").")

    try:
        # The host beside the id, so the watcher's polls go straight to it. A cache,
        # not a binding: a poll that finds it wrong sweeps again and rewrites it.
        job_dir = slurm_site.record_job_dir(job_id, **{
            slurm_site.ADOPTED_FILE: f"{note}\n" if note else "\n",
            slurm_site.HOST_FILE: host,
        })
    except OSError as exc:
        return err(f"Could not record the job: {exc}")

    payload = {
        "job_id": job_id,
        "state": state,
        "raw_state": raw,
        "job_dir": job_dir,
        "scheduler": slurm_site.name_of(host),
        "note": note or None,
    }
    # Where it is running, for the same reason the status read carries it: a watcher
    # keeps what its poll learned, and "on which node" is what a badge says.
    if nodes:
        payload["nodes"] = nodes
    if is_terminal(state):
        # Already over. Settled rather than watched: a watcher put on a finished job
        # reports it at its first poll, which is a wake for news the caller is holding.
        record_terminal_state(job_dir, state)
        payload["headline"] = (f"Job {job_id} has already finished ({state}); "
                               "recorded, with nothing left to watch.")
        return ok(payload)
    # What was adopted, not what will happen next: whether anything watches this job is
    # the client's to know, and it says so itself when it registers a watcher. A server
    # that promises a resume it does not perform produces a model that ends its turn on
    # a guarantee nobody is holding — which is what the front-ends without a watcher
    # hook were handing out here.
    payload["headline"] = (f"Slurm job {job_id} ({state}) is recorded as this "
                           "conversation's, and handed over to be watched.")
    # The plural key, like a status answer and unlike a submission: this call launched
    # nothing, it hands over a run already going. Singular says "this call started it",
    # which is what gates it behind BACKGROUNDABLE — a capability an adopter has no
    # business claiming, and whose absence here is why nothing ever read the descriptor.
    payload["background_jobs"] = [slurm_descriptor(job_id)]
    # What is still unaccounted for, named here because this is the moment the model is
    # holding the ids: a launcher prints a server job and a client array, and adopting
    # the first is the usual place to stop.
    add_unwatched(payload)
    return ok(payload)


def cancel(job_id: str, scheduler: str = "") -> dict:
    """Cancel *job_id* when this conversation or its sub-agents submitted it (``slurm_cancel``)."""
    # Same Unix user is not ownership: other conversations and workspaces share it.
    owned = slurm_site.owned_job_hosts(job_id)
    if scheduler:
        scheduler = scheduler.strip()
        owned &= {slurm_site.LOCAL if scheduler == slurm_site.LOCAL_NAME else scheduler}
    if not owned:
        # Submitted outside the cluster tools — by a launcher or a script — yet while a
        # command of this conversation ran, and no other conversation's did.
        named = scheduler.strip() if scheduler else ""
        launched = launched_from_this_session(
            job_id, "" if named in ("", slurm_site.LOCAL_NAME) else named)
        if launched is not None:
            owned = {launched}
    if not owned:
        return err(f"Job {job_id} was not submitted by this conversation or its "
                   "sub-agents — neither through the cluster tools nor while one of "
                   "its own commands ran — so it is not this conversation's to cancel.",
                   hint="Jobs of other conversations or workspaces are left alone; "
                        "ask the user to cancel them where they were started.")
    if len(owned) > 1:
        return err(f"Job {job_id} was submitted here on several schedulers: "
                   + ", ".join(slurm_site.name_of(h) for h in sorted(owned)) + ".",
                   hint="Name the one to cancel on with scheduler=.")
    # Only the scheduler it was submitted to: the same id elsewhere is another job.
    host = next(iter(owned))
    located, located_raw, _nodes = normalized_job_state(job_id, host=host)
    if located == "unknown":
        return err(f"{slurm_site.name_of(host)} no longer knows job {job_id}, "
                   "so there is nothing to cancel.",
                   hint="List what is running with slurm_queue(scheduler=...).")

    q = cli.run_argv(["squeue", "-h", "-j", job_id, "-o", "%u|%T"], TIMEOUT_READ,
                  host=host)
    line = (q.get("stdout") or "").strip().splitlines()
    if q.get("status") != "ok" or not line:
        return err(f"Job {job_id} is not in {slurm_site.name_of(host)}'s queue, so there is "
                   "nothing to cancel.",
                   hint=f"Its last known state: {located_raw or located}.")
    owner, _, queued_state = line[0].partition("|")
    user = cli.current_user()
    if user and owner.strip() != user:
        return err(f"Job {job_id} belongs to '{owner.strip()}', not to you ('{user}').",
                   hint="Only your own jobs can be cancelled from here.")

    res = cli.run_argv(["scancel", job_id], TIMEOUT_SUBMIT, host=host)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "scancel failed"))
    return ok({
        "job_id": job_id,
        "was": queued_state.strip(),
        "scheduler": slurm_site.name_of(host),
        "note": f"Slurm job {job_id} cancelled on {slurm_site.name_of(host)} "
                f"(it was {queued_state.strip().lower()}).",
    })
