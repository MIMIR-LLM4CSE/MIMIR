"""The cluster jobs a shell command started without showing it: a script's sbatch, srun or bsub.

They are found from the ids the command prints, else from what entered the queue during its
run, for a script naming a submitter or a process of the run running one (the host of a
script's ``ssh`` is only known there). Filed like a submission: this session's to cancel. A detached run keeps one handle and follows its jobs
(:func:`follow`); a blocking run has returned, and hands them over (:func:`hand_over`).
"""

import json
import os
import shlex
import re
import time

from ...shared import schedulers, shell_activity, cluster_site
from ...shared.jobs import (EXIT_CODE, SHELL_SLURM_JOBS, WATCHER_FILE, is_terminal, launched,
                            record_terminal_state, recorded_terminal_state, shell_descriptor,
                            slurm_descriptor)
from ...shared.processes import session_argvs
from ...shared.shell_paths import ShellParseError, parse_segments, unwrap_argv

# What each scheduler's submitters print about a job (``shared/schedulers``).
_PRINTED_IDS = [sched.printed_id for sched in schedulers.BY_KIND.values()]
_NAMES_SUBMITTER = re.compile(rb"\b(?:sbatch|srun|salloc|bsub)\b")
# How much of a script is read for a submitter.
_SCRIPT_BYTES = 256 * 1024
# Allowed between a run's start and a submission: the clocks of this host and a scheduler's.
_QUEUE_SLACK_S = 5.0
# Worst first: the word a run ends on when its parts ended differently.
_ENDINGS = ("crashed", "unknown", "done")


def printed_ids(text: str) -> list[str]:
    """The job ids *text* reports a submission of, each once, in order."""
    found = sorted((m.start(), m.group(1)) for pattern in _PRINTED_IDS
                   for m in pattern.finditer(text))
    return list(dict.fromkeys(job_id for _, job_id in found))


def _argvs(command: str) -> list[list[str]]:
    try:
        return [segment.argv for segment in parse_segments(command)]
    except ShellParseError:
        return []


def _names_slurm(path: str) -> bool:
    try:
        with open(path, "rb") as fh:
            return bool(_NAMES_SUBMITTER.search(fh.read(_SCRIPT_BYTES)))
    except OSError:
        return False


def runs_batch_script(command: str, cwd: str) -> bool:
    """Whether *command* executes a file of the workspace that names a submitter."""
    base = cwd
    for argv in _argvs(command):
        inner, _ = unwrap_argv(argv)
        if not inner:
            continue
        if inner[0] == "cd":
            base = os.path.join(base, inner[1]) if len(inner) > 1 else base
            continue
        files = [inner[0]]
        if os.path.basename(inner[0]).startswith("python"):
            files += [word for word in inner[1:] if not word.startswith("-")][:1]
        if any(os.path.isfile(path) and _names_slurm(path)
               for path in (os.path.join(base, name) for name in files)):
            return True
    return False


def _running_slurm(pid: int) -> list[str]:
    """The schedulers the processes of the run of *pid* are using now: ``LOCAL`` for a scheduler
    command here, the host for one over ``ssh``, whatever the script spelled it with."""
    out: dict[str, None] = {}
    for argv in session_argvs(pid):
        host, remote = cluster_site.ssh_target(argv)
        if host:
            if _NAMES_SUBMITTER.search(" ".join(remote).encode()):
                out.setdefault(host, None)
        elif os.path.basename(argv[0]) in cluster_site.SCHEDULER_COMMANDS:
            out.setdefault(cluster_site.LOCAL, None)
    return list(out)


def _schedulers(command: str, reached: list[str]) -> list[str]:
    """Where a job of *command* may be: every known scheduler, the hosts it ssh'es to, and
    the *reached* ones its processes were seen on."""
    known = cluster_site.hosts()
    named = [cluster_site.ssh_target(argv)[0] for argv in _argvs(command)]
    return known + [host for host in dict.fromkeys(named + reached)
                    if host and host not in known]


def _queued_in_run(hosts: list[str], started: float, ended: float) -> list[str]:
    """What entered a queue between *started* and *ended*, while no other session ran a command."""
    out: dict[str, None] = {}
    for host in hosts:
        for job_id, moment in (cluster_site.queued_submissions(host=host) or {}).items():
            if (started - _QUEUE_SLACK_S <= moment <= ended + _QUEUE_SLACK_S
                    and shell_activity.submitted_by_this_session(moment)):
                out.setdefault(schedulers.array_base(job_id), None)
    return list(out)


def _take_in(filed: dict, job_ids: list[str], hosts: list[str], watcher_of) -> None:
    """File the ids of *job_ids* a scheduler has running or queued in *filed*.

    One already recorded, or already over (an old log read back), is not this run's; one
    no scheduler knows yet stays out, to be found again.
    """
    recorded = cluster_site.recorded_jobs()
    for job_id in job_ids:
        host, found = cluster_site.locate_job(
            None, hosts, lambda h: cluster_site.job_state_on(job_id, h) or cluster_site.UNKNOWN_JOB)
        if host is None:
            continue
        if is_terminal(found.state) or cluster_site.covered_by(job_id, recorded, host):
            filed["passed"].append(job_id)
            continue
        cluster_site.record_host(host, cluster_site.kind_of(host))
        files = {cluster_site.HOST_FILE: host}
        if watcher_of:
            files[WATCHER_FILE] = json.dumps(watcher_of())
        filed["jobs"][job_id] = {"host": host, "dir": cluster_site.record_job_dir(job_id, **files)}


def _discover(filed: dict, command: str, cwd: str, text: str, started: float, ended: float,
              watcher_of=None) -> None:
    """Add to *filed* the jobs the output *text* of *command* shows, else the queue does."""
    job_ids = [job_id for job_id in printed_ids(text)
               if job_id not in filed["jobs"] and job_id not in filed["passed"]]
    from_queue = (not job_ids and not filed["jobs"]
                  and bool(filed["reached"] or runs_batch_script(command, cwd)))
    if not job_ids and not from_queue:
        return
    hosts = _schedulers(command, filed["reached"])
    _take_in(filed, _queued_in_run(hosts, started, ended) if from_queue else job_ids,
             hosts, watcher_of)


def hand_over(result: dict, command: str, cwd: str, started: float) -> None:
    """Hand the jobs a finished blocking run started over to be watched, in *result*."""
    if launched(result):
        return  # diverted: it goes on as a detached run, which follows them itself
    filed = _empty()
    text = f"{result.get('stdout') or ''}\n{result.get('stderr') or ''}"
    _discover(filed, command, cwd, text, started, time.time())
    job_ids = list(filed["jobs"])
    if not job_ids:
        return
    result["slurm_jobs"] = job_ids
    result["background_jobs"] = [slurm_descriptor(job_id) for job_id in job_ids]
    result["where"] = (f"This command started batch job(s) {', '.join(job_ids)}: they run on "
                       "the cluster, not here, and are handed over to be watched.")


def _empty() -> dict:
    return {"jobs": {}, "passed": [], "reached": [], "closed": False}


def _load(job_dir: str) -> dict:
    try:
        with open(os.path.join(job_dir, SHELL_SLURM_JOBS), encoding="utf-8") as fh:
            return {**_empty(), **json.load(fh)}
    except (OSError, ValueError, TypeError):
        return _empty()


def _save(job_dir: str, filed: dict) -> None:
    try:
        with open(os.path.join(job_dir, SHELL_SLURM_JOBS), "w", encoding="utf-8") as fh:
            json.dump(filed, fh)
    except OSError:
        pass


def _ended_at(job_dir: str) -> float:
    """When a run's process ended: the time of its exit code, now without one."""
    try:
        return os.path.getmtime(os.path.join(job_dir, EXIT_CODE))
    except OSError:
        return time.time()


def _job_word(job_id: str, job: dict) -> tuple[str | None, schedulers.JobState | None]:
    """A filed job's state, settling it once it has ended; None when its scheduler is silent."""
    settled = recorded_terminal_state(job["dir"])
    if settled:
        return settled, None
    found = cluster_site.job_state_on(job_id, job["host"])
    if found is None:
        return None, None
    record_terminal_state(job["dir"], found.state)
    return found.state, found


def _overall(process: str, words: list[str]) -> str:
    """The run's state from its process's and its jobs': going while any part is."""
    going = [word for word in words if not is_terminal(word)]
    if process == "running" or "running" in going:
        return "running"
    if going:
        return going[0]
    return next(word for word in _ENDINGS if word in [process, *words])


def follow(payload: dict, job_dir: str, meta: dict, text: str) -> None:
    """Make *payload*, a shell job's status, say the batch jobs it started.

    An empty state is a scheduler that did not answer: the watcher tries again, and gives
    up on the run as unknown if it stays silent.
    """
    filed = _load(job_dir)
    if not filed["closed"]:
        before = json.dumps(filed, sort_keys=True)
        job_key = str(meta.get("job_key") or payload.get("job_key") or "")
        over = is_terminal(payload.get("state"))
        pid = meta.get("pid")
        if not over and isinstance(pid, int) and not filed["jobs"] and not meta.get("adopted"):
            filed["reached"] = list(dict.fromkeys(filed["reached"] + _running_slurm(pid)))
        _discover(filed, str(meta.get("command") or ""), str(meta.get("cwd") or ""), text,
                  float(meta.get("started_at") or 0), _ended_at(job_dir) if over else time.time(),
                  lambda: shell_descriptor(job_key, job_dir))
        # What a run printed and queued is known once it has ended and been read.
        filed["closed"] = over
        if json.dumps(filed, sort_keys=True) != before:
            _save(job_dir, filed)
    jobs = filed["jobs"]
    if not jobs:
        return

    words: dict[str, str | None] = {}
    nodes: dict[str, None] = {}
    partitions: dict[str, None] = {}
    for job_id, job in jobs.items():
        words[job_id], found = _job_word(job_id, job)
        if found is not None and not is_terminal(found.state):
            nodes.update(dict.fromkeys(found.nodes))
            partitions.update(dict.fromkeys(found.partitions))
    open_ids = [job_id for job_id, word in words.items() if not is_terminal(word)]
    current = (open_ids or list(jobs))[0]
    payload["slurm_jobs"] = list(jobs)
    payload["job_id"] = current
    payload["scheduler"] = cluster_site.name_of(jobs[current]["host"])
    if nodes:
        payload["nodes"] = list(nodes)
    if partitions:
        payload["partitions"] = list(partitions)
    known = [word for word in words.values() if word is not None]
    payload["state"] = ("" if len(known) < len(words)
                        else _overall(str(payload.get("state") or "unknown"), known))
    payload["where"] = (f"Batch job(s) {', '.join(jobs)} started by this command run on "
                        f"{payload['scheduler']}, not here; slurm_job_status(job_id) reads "
                        "each one's log.")


def cancel(job_dir: str) -> list[str]:
    """Cancel the jobs this run started that are not over; the ids a scheduler took it for."""
    cancelled = []
    for job_id, job in _load(job_dir)["jobs"].items():
        if recorded_terminal_state(job["dir"]):
            continue
        cancel = cluster_site.scheduler_for(job["host"]).cancel_command
        if cluster_site.read_lines(f"{cancel} {shlex.quote(str(job_id))}", host=job["host"]) is not None:
            cancelled.append(job_id)
    return cancelled
