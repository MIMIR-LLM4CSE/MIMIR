"""Where this machine's Slurm controller is, learned rather than configured.

Slurm's tools are commands on a host, and the host MIMIR runs on is not always the one
that has them: a login node holding the editor's remote session while the scheduler
lives behind ``ssh`` is an ordinary arrangement. On one of those, every typed cluster
tool answers "sbatch is not available", so the work goes out through a hand-written
``ssh host "sbatch …"`` instead — where nothing records a descriptor, nothing is
watched, no wake is owed when the job ends, and the server stops on its idle timer with
the run still going. The tools being local-only was the whole of that failure.

**Learned from the user's own successful call, not from a setting.** An environment
variable is a thing somebody has to know to set, and nobody sets it before the first
night is lost. What is already happening instead is that the model reaches the
scheduler over ``ssh`` constantly — to read the queue, to check a job. So the bash
server tells this module every time such a call *succeeds*, and the hpc server asks it
where to send its own commands. One successful ``ssh svc03 "squeue -u $USER"`` is
enough, and that call is the first thing anybody runs.

Deterministic, which is what lets it be acted on: the signal is the shell validator's
own parse of the command — the same mechanism that refuses a bare ``sbatch`` — plus the
exit status. Nothing here matches patterns against free text.

Filed in the global state dir, because the answer is a property of the machine and not
of a workspace or a conversation: two workspaces on one login node face the same
scheduler.
"""

from __future__ import annotations

import os
import shlex
import subprocess

from state_paths import global_state_dir, state_dir

# The commands that only exist where a scheduler does. A remote call whose head is one
# of these, and that succeeded, proves the far side is the controller — no other host
# would have answered it.
SLURM_COMMANDS = frozenset({
    "squeue", "sbatch", "sacct", "sinfo", "scontrol", "salloc", "scancel", "sstat",
    "sacctmgr", "sprio", "sshare", "srun",
})

# ``ssh`` options that take a separate value, so the host is not mistaken for one of
# them. Only what ssh actually defines — an unknown flag is left to the fallback of
# "the first bare word is the host", which is ssh's own rule.
_SSH_VALUE_FLAGS = frozenset({
    "-b", "-c", "-D", "-E", "-e", "-F", "-I", "-i", "-J", "-L", "-l", "-m", "-O",
    "-o", "-p", "-Q", "-R", "-S", "-W", "-w",
})

_FILENAME = "slurm_controller"

# How :func:`queued_ids` asks: ids only, unheadered, and ``$USER`` left for the far
# side's shell to expand, because the account that matters is the one the scheduler
# knows rather than the one this process runs as.
_QUEUED_IDS_COMMAND = "squeue -h -u $USER -o '%i'"

# ``BatchMode=yes`` is not optional: a poller that can be asked for a password hangs for
# its timeout on every tick instead of answering. No ssh *options* beyond that pair — an
# identity file, a port, a jump host are ``~/.ssh/config``'s business, which is a
# configuration mechanism the user already has and we have no business duplicating.
_SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def argv_for(script: str, *, piped: bool = False) -> list[str]:
    """The argv that runs *script* where Slurm's tools are.

    A login shell in both cases, local and remote, because that is what puts the
    scheduler's own module files on ``PATH`` — the difference between ``squeue`` and
    "command not found" on most sites. Quoted once for the remote shell: ``ssh`` joins
    its arguments and hands the result to a shell on the far side, so a script that
    arrived unquoted would be re-split there.

    *piped* says this call feeds the command its own stdin. ``-n`` closes stdin for
    every other call — an ``ssh`` that reads a server's stdin eats its MCP protocol
    traffic — and must be left off when the stdin *is* the point, as it is for a
    submission handing ``sbatch`` its script.
    """
    host = controller()
    if not host:
        return ["bash", "-lc", script]
    base = _SSH_BASE if piped else [*_SSH_BASE[:1], "-n", *_SSH_BASE[1:]]
    return base + [host, "bash", "-lc", shlex.quote(script)]


def exec_argv(argv: list[str], *, piped: bool = False) -> list[str]:
    """What to exec to run *argv* where Slurm's tools are.

    The twin of :func:`argv_for` for a caller that built an **argv** rather than a
    script — the submitters, which take their resources as arguments and assemble the
    command themselves precisely so that no shell is involved and no argument can
    become a second command. Locally that property is kept exactly: the argv is
    exec'd as it stands.

    Remotely one shell is unavoidable, because ``ssh`` has no argv to hand over — only
    a command line for the far side to parse. So each element is quoted on the way out
    and the property survives in the form that matters: what the remote shell splits is
    exactly the argv built here.
    """
    if not controller():
        return list(argv)
    return argv_for(shlex.join(argv), piped=piped)


def queued_ids(timeout: float = 15.0) -> list[str] | None:
    """The job ids the scheduler says this user has queued, or None if it did not answer.

    Here rather than in either server because both ask it, and for the same reason: the
    gap between this and the descriptors on disk is the work nothing is watching. None
    is "unknown", which must never be read as "nothing" — the whole failure this exists
    for is silence being taken for an empty queue.
    """
    try:
        res = subprocess.run(
            argv_for(_QUEUED_IDS_COMMAND),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def controller_from_ssh(argv: list[str]) -> str:
    """The host *argv* reached a Slurm command on, or "" when it did not.

    ``argv`` is one command as the shell validator parsed it, so this reads tokens and
    never text. The remote command arrives as a single token — ``ssh host "squeue -u
    $USER"`` is three words to a shell — so its own head is taken from one more round
    of tokenization, which is the same answer the far side's shell will reach.
    """
    if not argv or os.path.basename(argv[0]) != "ssh":
        return ""
    host = ""
    rest: list[str] = []
    index = 1
    while index < len(argv):
        token = argv[index]
        if not host and token.startswith("-"):
            index += 2 if token in _SSH_VALUE_FLAGS else 1
            continue
        if not host:
            host = token
            index += 1
            continue
        rest.append(token)
        index += 1
    if not host or not rest:
        return ""
    try:
        words = shlex.split(rest[0])
    except ValueError:
        return ""
    if not words or os.path.basename(words[0]) not in SLURM_COMMANDS:
        return ""
    # A user@host target reaches the same scheduler; the account is ssh's business and
    # is kept, so the recorded target is exactly what worked.
    return host


def record_controller(host: str, *, home: str | None = None) -> None:
    """Remember *host* as this machine's controller. Never raises.

    Written once and then left alone: a site has one scheduler, and rewriting the file
    on every call would turn a read-mostly fact into a write on every queue listing.
    """
    host = (host or "").strip()
    if not host or host == controller(home=home):
        return
    path = os.path.join(global_state_dir(home), _FILENAME)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(host + "\n")
    except OSError:
        pass


def controller(*, home: str | None = None) -> str:
    """This machine's recorded controller, or "" when none has been seen yet."""
    try:
        with open(os.path.join(global_state_dir(home), _FILENAME)) as fh:
            return fh.read().strip()
    except OSError:
        return ""


# ── What this workspace tracks of what the scheduler is running ───────────────
#
# Two halves, and both servers need the same answer. A job is tracked by the descriptor
# its submission wrote — the id file under ``hpc_jobs/`` — so the gap is "what the queue
# says is running" minus "what the disk says we recorded". Owned here rather than in the
# hpc server because the *bash* server is the one that sees the ordinary case: the work
# is submitted by a site launcher and verified with a ``squeue`` a second later, and
# neither call reaches a typed cluster tool. Two copies of this rule would drift, and
# the one that drifted would be the one the night depends on.

# The file a submission's directory is identified by. The hpc server writes it; this is
# the name both sides agree on.
ID_FILE = "slurm_job_id"

_JOBS_SUBDIR = "hpc_jobs"


def recorded_job_ids(*extra_roots: str) -> set[str]:
    """Every job id this workspace holds a descriptor for, in one pass.

    A set rather than a search per id: the callers are listings, and a search only
    returns early on a hit — so the ids that answer "no", which are the interesting
    ones, would each pay a complete walk of every session's submissions.

    *extra_roots* is for a caller whose own directory is not under the sessions tree,
    which is how the hpc server's test override reaches this.
    """
    found: set[str] = set()
    roots = list(extra_roots)
    sessions = os.path.join(state_dir(), "sessions")
    try:
        roots += [os.path.join(sessions, sid, _JOBS_SUBDIR)
                  for sid in sorted(os.listdir(sessions))]
    except OSError:
        pass
    for root in roots:
        try:
            names = os.listdir(root)
        except OSError:
            continue
        for name in names:
            try:
                with open(os.path.join(root, name, ID_FILE)) as fh:
                    recorded = fh.read().strip()
            except OSError:
                continue
            if recorded:
                found.add(recorded)
    return found


def covered_by(job_id: str, recorded: set[str]) -> bool:
    """Whether *recorded* holds a descriptor covering *job_id*.

    An array is the whole reason this is not ``in``. It is adopted or submitted under
    its base id and the queue then lists its tasks — ``15306621_3``, or
    ``15306621_[4-8]`` for the ones still pending — so the id in a row is rarely the id
    on disk. The base is what covers them: a watcher on it polls the whole array.
    """
    return bool(job_id) and (job_id in recorded
                             or job_id.split("_", 1)[0] in recorded)


# What an unwatched job means, said once, by the layer that knows a scheduler exists.
UNWATCHED_HINT = (
    "Nothing is watching these: they were not submitted through the cluster tools, so "
    "no wake is owed when they end and this workspace's server does not stay alive "
    "for them. slurm_watch(job_id) adopts one — an array's base id covers every task.")


def unwatched(queued_ids, *extra_roots: str) -> list[str]:
    """The ids in *queued_ids* that no descriptor covers, by array rather than by task.

    Eight task rows of one array name one descriptor to write, so they collapse to
    their base: listed one by one, a reminder built on this would ask for eight calls
    that all land on "already recorded". Ordered, so the answer reads in queue order.
    """
    recorded = recorded_job_ids(*extra_roots)
    out: dict[str, None] = {}
    for job_id in queued_ids:
        if job_id and not covered_by(job_id, recorded):
            out.setdefault(job_id.split("_", 1)[0], None)
    return list(out)
