"""What each batch scheduler MIMIR drives is: its commands, its script header, its words.

One object per scheduler (:data:`SLURM`, :data:`LSF`), so a caller that knows a job's
host asks :func:`cluster_site.scheduler_for` for the object and never spells a
command itself. Everything here is pure, except what a caller hands in to run lines.
"""

from __future__ import annotations

import math
import re
import shlex
from typing import Callable, NamedTuple

from . import cpu_facts
from .job_script import sbatch_header


class JobState(NamedTuple):
    """What the scheduler says of a job: its run word, the scheduler's own words, where it is.

    *nodes* only while it runs; *partitions* (queues, under LSF) while it is queued or
    running (a pending Slurm job submitted to several lists them all).
    """

    state: str
    raw: str = ""
    nodes: list[str] = []
    partitions: list[str] = []


UNKNOWN_JOB = JobState("unknown")

RunLines = Callable[[str], list[str]]


def array_base(job_id: str) -> str:
    """The submission an array element belongs to: ``1234`` of ``1234_5`` or ``1234[5]``."""
    return re.split(r"[_\[]", str(job_id or ""), maxsplit=1)[0]


# ── Slurm ─────────────────────────────────────────────────────────────────────

_SLURM_ACTIVE = ("RUNNING", "COMPLETING")
_SLURM_WAITING = ("PENDING", "CONFIGURING", "REQUEUED")
# Prefixes, because Slurm qualifies some words ("CANCELLED by 1234").
_SLURM_CRASH = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                "BOOT_FAIL", "DEADLINE", "PREEMPTED")
_SLURM_DONE = ("COMPLETED",)


def _slurm_crashed(word: str) -> bool:
    return word.startswith(_SLURM_CRASH)


def _slurm_done(word: str) -> bool:
    return word.startswith(_SLURM_DONE)


class Slurm:
    kind = "slurm"
    label = "Slurm"
    commands = frozenset({
        "squeue", "sbatch", "sacct", "sinfo", "scontrol", "salloc", "scancel", "sstat",
        "sacctmgr", "sprio", "sshare", "srun",
    })
    submit_command = "sbatch"
    cancel_command = "scancel"
    # The variable a batch job's processes find their job id in.
    job_id_env = "SLURM_JOB_ID"
    # Ids only, unheadered, and ``$USER`` left for the far side's shell to expand,
    # because the account that matters is the one the scheduler knows.
    queued_ids_command = "squeue -h -u $USER -o '%i'"
    # The same, with each job's submit time turned into epoch seconds where it is known.
    queued_submissions_command = (
        "squeue -h -u $USER -o '%i|%V' | while IFS='|' read -r job at; "
        'do echo "$job|$(date -d "$at" +%s)"; done')
    # What sbatch, srun and salloc print about a job. ``--parsable`` prints a bare
    # number, which is told from any other number only by the queue.
    printed_id = re.compile(r"(?:Submitted batch job|(?:srun|salloc): (?:job|Granted job "
                            r"allocation|Pending job allocation)) (\d+)")

    @staticmethod
    def submitted_id(stdout: str) -> str | None:
        found = re.search(r"(\d+)", stdout or "")
        return found.group(1) if found else None

    @staticmethod
    def header(**kwargs) -> list[str]:
        """``sbatch_header``; ``after_ok`` names the job this one waits for."""
        after = kwargs.pop("after_ok", "")
        if after:
            kwargs["dependency"] = f"afterok:{after}"
        return sbatch_header(**kwargs)

    @staticmethod
    def state(job_id: str, run_lines: RunLines) -> JobState:
        """``squeue``, then ``sacct``. For an array, any live element keeps it live and the
        verdict is the worst element's. An ended word in squeue settles the job without
        sacct, all a site without accounting has. Names keep Slurm's case."""
        queued, where, parts = [], [], []
        for row in run_lines(f"squeue -j {shlex.quote(job_id)} -h -o '%T|%R|%P'"):
            word, _, rest = row.partition("|")
            reason_or_node, _, partition = rest.partition("|")
            queued.append(word.strip().upper())
            where.append(reason_or_node.strip())
            parts.append(partition.strip())
        raw = ",".join(sorted(set(queued)))
        live = [p for word, p in zip(queued, parts)
                if word in _SLURM_ACTIVE or word in _SLURM_WAITING]
        partitions = sorted({name for p in live for name in p.split(",") if name})
        if any(word in _SLURM_ACTIVE for word in queued):
            nodes = sorted({node for word, expr in zip(queued, where) if word in _SLURM_ACTIVE
                            for node in cpu_facts.expand_hostlist(expr) if node})
            return JobState("running", raw, nodes, partitions)
        if queued and all(word in _SLURM_WAITING for word in queued):
            return JobState("pending", raw, [], partitions)
        if queued and all(_slurm_done(w) or _slurm_crashed(w) for w in queued):
            return JobState("crashed" if any(_slurm_crashed(w) for w in queued) else "done", raw)

        accounted = [line.strip().upper()
                     for line in run_lines(f"sacct -j {shlex.quote(job_id)} -n -X -o State")]
        sraw = ",".join(sorted(set(accounted)))
        if any(_slurm_crashed(word) for word in accounted):
            return JobState("crashed", sraw)
        if accounted and all(_slurm_done(word) for word in accounted):
            return JobState("done", sraw)
        if queued:                    # in the queue under a non-standard active word
            return JobState("running", raw, [], partitions)
        if accounted:                 # some other terminal word
            return JobState("crashed", sraw)
        return UNKNOWN_JOB

    @staticmethod
    def owner_command(job_id: str) -> str:
        """Prints ``user|STATE`` per element of a queued job, nothing once it has left."""
        return f"squeue -h -j {shlex.quote(job_id)} -o '%u|%T'"

    @staticmethod
    def submit_time_command(job_id: str) -> str:
        """Prints when a queued job was submitted, in a form ``date -d`` reads."""
        return f"squeue -h -j {shlex.quote(job_id)} -o '%V'"


# ── LSF ───────────────────────────────────────────────────────────────────────

_LSF_ACTIVE = ("RUN", "USUSP", "SSUSP")
_LSF_WAITING = ("PEND", "PSUSP", "WAIT", "PROV")
_LSF_DONE = ("DONE",)
_LSF_CRASH = ("EXIT", "ZOMBI")
# How an empty answer reads: bjobs exits non-zero on it, and that is not a failure.
_LSF_NO_JOB = "No unfinished job found"


def _lsf_quiet(command: str) -> str:
    """*command* (a bjobs listing) exiting 0 with nothing printed when no job matches."""
    return (f"out=$({command} 2>&1); rc=$?; "
            f'case "$out" in *"{_LSF_NO_JOB}"*) exit 0;; esac; '
            '[ "$rc" -eq 0 ] || exit "$rc"; printf \'%s\\n\' "$out"')


def lsf_minutes(wall_time: str) -> int:
    """Slurm's ``[D-]HH:MM:SS`` (or ``MM``, ``MM:SS``, ``D-HH[:MM]``) in whole minutes, rounded up."""
    days, _, clock = (wall_time or "").rpartition("-")
    parts = [int(p) for p in clock.split(":")] if clock else [0]
    if days:
        hours, minutes, seconds = (parts + [0, 0])[:3]
    elif len(parts) == 3:
        hours, minutes, seconds = parts
    elif len(parts) == 2:
        hours, minutes, seconds = 0, parts[0], parts[1]
    else:
        hours, minutes, seconds = 0, parts[0], 0
    total = (int(days or 0) * 24 + hours) * 60 + minutes + math.ceil(seconds / 60)
    return max(total, 1)


def _lsf_mem(mem: str) -> str:
    """Slurm's ``32G`` as LSF's ``32GB``; a bare number is megabytes, as for Slurm."""
    mem = (mem or "").strip().upper()
    if not mem:
        return ""
    if mem[-1].isdigit():
        return f"{mem}MB"
    return f"{mem}B" if not mem.endswith("B") else mem


def _lsf_select(constraint: str) -> str:
    """A Slurm feature expression as an LSF ``select[]``: ``a&b`` is ``a && b``."""
    return re.sub(r"\s*([&|])\s*", lambda m: f" {m.group(1) * 2} ", constraint).strip()


def bsub_header(
    *,
    job_name: str,
    partition: str,
    cpus_per_task: int,
    wall_time: str,
    log_file: str,
    work_dir: str = "",
    mem: str = "",
    gpus: int = 0,
    account: str = "",
    nodes: int | None = None,
    ntasks: int | None = None,
    constraint: str = "",
    nodelist: str = "",
    exclusive: bool = False,
    comment: str = "",
    after_ok: str = "",
) -> list[str]:
    """The ``#!/bin/bash`` and ``#BSUB`` lines of a job script, from the placement
    :func:`sbatch_header` takes (*partition* is the queue); empty options are omitted.
    """
    # -n counts slots, so tasks × CPUs, kept on one host for one node. -M carries its
    # unit, so the site's LSF_UNIT_FOR_LIMITS cannot change its meaning. A job behind
    # after_ok waits for that one to end DONE, and -ti kills it once it cannot.
    for name, value in (("job_name", job_name), ("partition", partition),
                        ("wall_time", wall_time), ("mem", mem), ("account", account),
                        ("comment", comment), ("work_dir", work_dir), ("log_file", log_file)):
        if isinstance(value, str) and ("\n" in value or "\r" in value):
            raise ValueError(f"{name} cannot hold a line break.")
    minutes = lsf_minutes(wall_time)
    lines = [
        "#!/bin/bash",
        f"#BSUB -J {shlex.quote(job_name)}",
        f"#BSUB -q {shlex.quote(partition)}",
        f"#BSUB -n {max(1, (ntasks or 1) * max(1, cpus_per_task))}",
    ]
    if nodes == 1:
        lines.append('#BSUB -R "span[hosts=1]"')
    lines.append(f"#BSUB -W {minutes // 60}:{minutes % 60:02d}")
    if mem:
        lines.append(f"#BSUB -M {_lsf_mem(mem)}")
    if work_dir:
        lines.append(f"#BSUB -cwd {shlex.quote(work_dir)}")
    # Slurm's ``%j`` (the job id) is LSF's ``%J``.
    log_file = log_file.replace("%j", "%J")
    lines += [f"#BSUB -oo {shlex.quote(log_file)}", f"#BSUB -eo {shlex.quote(log_file)}"]
    if gpus > 0:
        lines.append(f'#BSUB -gpu "num={gpus}"')
    if account:
        lines.append(f"#BSUB -P {shlex.quote(account)}")
    if constraint:
        lines.append(f'#BSUB -R "select[{_lsf_select(constraint)}]"')
    if nodelist:
        lines.append(f'#BSUB -m "{" ".join(cpu_facts.expand_hostlist(nodelist))}"')
    if exclusive:
        lines.append("#BSUB -x")
    if comment:
        lines.append('#BSUB -Jd "' + comment.replace('"', "'") + '"')
    if after_ok:
        lines += [f'#BSUB -w "done({after_ok})"', "#BSUB -ti"]
    return lines


def _lsf_ended(text: str) -> str | None:
    """``done``/``crashed`` from ``bhist -l``'s prose, or None when it says neither.

    bhist wraps its lines at 80 columns, mid-word too: the words are matched with every
    space taken out.
    """
    flat = re.sub(r"\s+", "", text)
    if "Donesuccessfully" in flat:
        return "done"
    if "Exited" in flat:
        return "crashed"
    return None


class Lsf:
    kind = "lsf"
    label = "LSF"
    commands = frozenset({
        "bsub", "bjobs", "bkill", "bhist", "bqueues", "bhosts", "lshosts", "lsload",
        "bacct", "bpeek", "bstop", "bresume", "bmod", "blaunch", "lsid",
    })
    submit_command = "bsub"
    cancel_command = "bkill"
    job_id_env = "LSB_JOBID"
    queued_ids_command = _lsf_quiet("bjobs -noheader -u $USER -o jobid")
    queued_submissions_command = (
        _lsf_quiet("bjobs -noheader -u $USER -o \"jobid submit_time delimiter='|'\"")
        + " | while IFS='|' read -r job at; "
        'do echo "$job|$(date -d "$at" +%s)"; done')
    printed_id = re.compile(r"Job <(\d+)> is submitted")

    @staticmethod
    def submitted_id(stdout: str) -> str | None:
        found = Lsf.printed_id.search(stdout or "")
        return found.group(1) if found else None

    @staticmethod
    def header(**kwargs) -> list[str]:
        kwargs.pop("kill_on_invalid_dep", None)
        return bsub_header(**kwargs)

    @staticmethod
    def state(job_id: str, run_lines: RunLines) -> JobState:
        """``bjobs`` (every element of an array), then ``bhist`` for a job bjobs has
        cleaned out. The worst element decides, as for Slurm."""
        rows = run_lines(
            f"bjobs -noheader -o \"stat exec_host queue delimiter='|'\" {shlex.quote(job_id)}")
        words, hosts, queues = [], [], []
        for row in rows:
            if _LSF_NO_JOB in row or "is not found" in row:
                continue
            word, _, rest = row.partition("|")
            exec_host, _, queue = rest.partition("|")
            words.append(word.strip().upper())
            hosts.append(exec_host.strip())
            queues.append(queue.strip())
        raw = ",".join(sorted(set(words)))
        live = [q for w, q in zip(words, queues) if w in _LSF_ACTIVE or w in _LSF_WAITING]
        partitions = sorted({q for q in live if q and q != "-"})
        if any(w in _LSF_ACTIVE for w in words):
            # exec_host reads ``8*c01:8*c02``: slots, a star, then the host.
            nodes = sorted({part.split("*")[-1] for w, expr in zip(words, hosts)
                            if w in _LSF_ACTIVE for part in expr.split(":")
                            if part and part != "-"})
            return JobState("running", raw, nodes, partitions)
        if words and all(w in _LSF_WAITING for w in words):
            return JobState("pending", raw, [], partitions)
        if words and all(w in _LSF_DONE or w in _LSF_CRASH for w in words):
            return JobState("crashed" if any(w in _LSF_CRASH for w in words) else "done", raw)
        if words:                     # UNKWN and the like: still the scheduler's
            return JobState("running", raw, [], partitions)
        ended = _lsf_ended("".join(run_lines(f"bhist -l -n 0 {shlex.quote(job_id)}")))
        return JobState(ended, ended.upper()) if ended else UNKNOWN_JOB

    @staticmethod
    def owner_command(job_id: str) -> str:
        return f"bjobs -noheader -o \"user stat delimiter='|'\" {shlex.quote(job_id)}"

    @staticmethod
    def submit_time_command(job_id: str) -> str:
        return f"bjobs -noheader -o submit_time {shlex.quote(job_id)}"


SLURM = Slurm()
LSF = Lsf()
BY_KIND = {SLURM.kind: SLURM, LSF.kind: LSF}

Scheduler = Slurm | Lsf

# Every command a scheduler answers, whichever it is: a successful remote call naming
# one of these proves a scheduler on the far side.
ALL_COMMANDS = SLURM.commands | LSF.commands


def kind_of_command(name: str) -> str:
    """The scheduler kind whose command *name* is, or ""."""
    for sched in BY_KIND.values():
        if name in sched.commands:
            return sched.kind
    return ""


def allocation(env) -> tuple[Scheduler, str] | None:
    """``(scheduler, job id)`` when *env* is a batch job's, else None."""
    for sched in BY_KIND.values():
        job_id = (env.get(sched.job_id_env) or "").strip()
        if job_id:
            return sched, job_id
    return None
