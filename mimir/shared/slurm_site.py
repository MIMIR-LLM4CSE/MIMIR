"""Which Slurm schedulers this workspace reaches, learned from successful calls.

The bash server records a host whenever an ``ssh <host> <slurm command>`` succeeds
(from its own parse, never from text); the hpc server sends its commands there.
Several hosts at once, filed per workspace; job directories are per session.
"""

from __future__ import annotations

from typing import Callable

import os
import shlex
import subprocess
import time

from . import cpu_facts
from .jobs import SLURM_JOB_ID
from .state_paths import SUBAGENTS_DIRNAME, session_state_dir, sessions_root, state_dir

# The commands that only exist where a scheduler does. A remote call whose head is one
# of these, and that succeeded, proves the far side has a scheduler — no other host
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

# The local scheduler, named the way :func:`argv_for` takes it: no host, hence no ssh.
# A real name as far as every caller is concerned — it is recorded in descriptors and
# returned by :func:`hosts` — so that "where is this job" has one kind of answer.
LOCAL = ""

# What the model spells to mean LOCAL, since "" at a tool boundary is "unspecified".
LOCAL_NAME = "local"

# The workspace's registry of reachable schedulers, one host per line.
_FILENAME = "slurm_hosts"

# In a job descriptor: the scheduler that job is on. Empty contents mean LOCAL, and an
# absent file means a descriptor written before the host was recorded with it — see
# :func:`job_host`, which answers None there rather than guessing.
HOST_FILE = "slurm_host"

# Whether this host has Slurm, memoised: asked on every call that reaches a scheduler,
# and the answer cannot change under a running process. Named apart from ``LOCAL``,
# which is a host and not a yes/no.
_HAS_SLURM: bool | None = None

# Short, because this runs inside a tool call the user is waiting on, and a login shell
# that takes longer than this to answer "command -v" is not going to be this server's
# fast path anyway.
_LOCAL_PROBE_TIMEOUT = 10

# How :func:`queued_ids` asks: ids only, unheadered, and ``$USER`` left for the far
# side's shell to expand, because the account that matters is the one the scheduler
# knows rather than the one this process runs as.
_QUEUED_IDS_COMMAND = "squeue -h -u $USER -o '%i'"

# ``BatchMode=yes`` is not optional: a poller that can be asked for a password hangs for
# its timeout on every tick instead of answering. No ssh *options* beyond that pair — an
# identity file, a port, a jump host are ``~/.ssh/config``'s business, which is a
# configuration mechanism the user already has and we have no business duplicating.
_SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]


def argv_for(script: str, *, host: str = LOCAL, piped: bool = False) -> list[str]:
    """The argv running *script* on *host* (``LOCAL`` or over ``ssh``) in a login shell.

    Quoted once for the remote shell; ``-n`` closes stdin unless *piped*.
    """
    if not host:
        return ["bash", "-lc", script]
    base = _SSH_BASE if piped else [*_SSH_BASE[:1], "-n", *_SSH_BASE[1:]]
    return base + [host, "bash", "-lc", shlex.quote(script)]


def exec_argv(argv: list[str], *, host: str = LOCAL,
              piped: bool = False) -> list[str]:
    """The argv running *argv* on *host*: as is locally, each element quoted for ``ssh``."""
    if not host:
        return list(argv)
    return argv_for(shlex.join(argv), host=host, piped=piped)


def queued_ids(timeout: float = 15.0, *, host: str = LOCAL) -> list[str] | None:
    """The job ids *host* has queued for this user, or None ("unknown", never "nothing")."""
    try:
        res = subprocess.run(
            argv_for(_QUEUED_IDS_COMMAND, host=host),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if res.returncode != 0:
        return None
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def queued_by_host(timeout: float = 15.0) -> dict[str, list[str] | None]:
    """:func:`queued_ids` for every reachable host, sequentially; None where one did not answer."""
    return {host: queued_ids(timeout, host=host) for host in hosts()}


def slurm_host_from_ssh(argv: list[str]) -> str:
    """The host an ``ssh`` argv reached a Slurm command on, or ""."""
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


def _path() -> str:
    """The file holding this workspace's registry of reachable schedulers.

    One file for the workspace. Every session in it reads and appends to the same
    inventory, which is what lets a scheduler proved by one conversation be a
    destination for the next without an ``ssh`` that has already succeeded once.
    """
    return os.path.join(state_dir(), _FILENAME)


def slurm_is_local() -> bool:
    """Whether Slurm answers on this host through a login shell (probed once; failure means no)."""
    global _HAS_SLURM
    if _HAS_SLURM is None:
        try:
            res = subprocess.run(
                ["bash", "-lc", "command -v sbatch"],
                stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL, timeout=_LOCAL_PROBE_TIMEOUT,
            )
            _HAS_SLURM = res.returncode == 0
        except (OSError, subprocess.SubprocessError):
            _HAS_SLURM = False
    return _HAS_SLURM


def name_of(host: str) -> str:
    """How to spell *host* back to a caller: ``local``, or the hostname itself.

    One spelling in every direction. ``LOCAL`` is the empty string because that is what
    :func:`argv_for` needs in order to run no ``ssh`` at all, and an empty string in an
    answer reads as a missing value rather than as a scheduler.
    """
    return LOCAL_NAME if not host else host


def hosts() -> list[str]:
    """Every reachable scheduler: the local one first (by probe), then the learned ones in order."""
    out = [LOCAL] if slurm_is_local() else []
    try:
        with open(_path()) as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    for line in lines:
        host = line.strip()
        if host and host not in out:
            out.append(host)
    return out


def record_host(host: str) -> None:
    """Add *host* to the workspace's schedulers if new (never raises)."""
    host = (host or "").strip()
    if not host or host in hosts():
        return
    path = _path()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(host + "\n")
    except OSError:
        pass


def forget_host(host: str) -> bool:
    """Drop *host* from the registry; whether it was there (the local one cannot be dropped)."""
    host = (host or "").strip()
    path = _path()
    try:
        with open(path) as fh:
            lines = [line.strip() for line in fh.read().splitlines() if line.strip()]
    except OSError:
        return False
    if host not in lines:
        return False
    try:
        with open(path, "w") as fh:
            fh.write("".join(f"{h}\n" for h in lines if h != host))
    except OSError:
        return False
    return True


def resolve_host(requested: str = "") -> tuple[str, str]:
    """``(host, "")`` for the scheduler to use, or ``("", reason)`` when the caller must choose.

    A named host is taken as given. Unspecified with several schedulers is refused, listing
    them: the choice belongs to whoever knows what the job needs.
    """
    requested = (requested or "").strip()
    known = hosts()
    if requested == LOCAL_NAME:
        if not slurm_is_local():
            return "", ("No Slurm command answers on this host, so there is no local "
                        "scheduler to use. " + _choices(known))
        return LOCAL, ""
    if requested:
        return requested, ""
    if len(known) == 1:
        return known[0], ""
    if not known:
        return "", ("No scheduler answers on this host and none has been reached over "
                    "ssh yet. Read a queue through the shell once — "
                    "ssh <host> \"squeue -u $USER\" — and it is learned, or name one "
                    "with slurm_schedulers(add=\"<host>\").")
    return "", (f"This workspace reaches more than one scheduler ({_listed(known)}), "
                "so where this goes is a choice rather than a default: pass "
                "scheduler=. "
                "It follows what the job needs — the GPUs, a compilation node, the "
                "filesystem its inputs are on — or what the user asked for; ask them "
                "when the task does not settle it.")


def _listed(known: list[str]) -> str:
    """The inventory as a caller should read it back, in one spelling: ``local, a-host``."""
    return ", ".join(name_of(host) for host in known)


def _choices(known: list[str]) -> str:
    """A sentence naming what is left to choose from, for a refusal that can be acted on."""
    if not known:
        return "No scheduler is reachable from this workspace at all."
    return f"Reachable: {_listed(known)}."


# ── What this workspace tracks of what the scheduler is running ───────────────
# Untracked work is what the queue lists minus the descriptors on disk. Here because
# both the bash and the hpc server need the same answer.

# A submission's directory is identified by its SLURM_JOB_ID file (:func:`record_job_dir`
# is the one writer). WATCHER_FILE is written by a submitter that runs its own watcher
# (a proxy run), so a reload re-arms that watcher rather than a second one beside it.

_JOBS_SUBDIR = "hpc_jobs"


def jobs_dir() -> str:
    """Where this session files its submissions (env-overridable for tests)."""
    return os.environ.get("MIMIR_HPC_JOBS_DIR") or os.path.join(
        session_state_dir(), _JOBS_SUBDIR)


def _new_job_dir_name(base: str) -> str:
    """A sortable, collision-free directory name for one submission.

    The UTC stamp alone collides: it resolves to the second, and two submissions in
    that second would share a directory and overwrite each other's script and id.
    One second is a long time when two conversations are working at once.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    while True:
        name = f"{stamp}-{os.urandom(2).hex()}"
        if not os.path.exists(os.path.join(base, name)):
            return name


def new_job_dir() -> str:
    """A new empty directory for a submission whose id is not known yet."""
    base = jobs_dir()
    job_dir = os.path.join(base, _new_job_dir_name(base))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def record_job_dir(job_id: str, **files: str) -> str:
    """File *job_id* (with *files*) as a job of this workspace; return its directory.

    The one writer of the layout: submissions, adoptions and the proxy's runs.
    """
    job_dir = new_job_dir()
    for name, content in {SLURM_JOB_ID: job_id, **files}.items():
        with open(os.path.join(job_dir, name), "w") as fh:
            fh.write(content)
    return job_dir


def recorded_jobs() -> set[tuple[str | None, str]]:
    """Every ``(host, id)`` this workspace has a descriptor for, across sessions.

    Keyed by host since clusters number independently; a descriptor without a host
    (``None``) covers any (:func:`covered_by`).
    """
    roots = [jobs_dir()]
    sessions = sessions_root()
    try:
        roots += [os.path.join(sessions, sid, _JOBS_SUBDIR)
                  for sid in sorted(os.listdir(sessions))]
    except OSError:
        pass
    return {(host, recorded) for _, recorded, host in _descriptors(roots)}


def _descriptors(roots: list[str]):
    """Every readable job descriptor under *roots*, as ``(job_dir, id, host)``."""
    for root in roots:
        try:
            names = sorted(os.listdir(root))
        except OSError:
            continue
        for name in names:
            job_dir = os.path.join(root, name)
            try:
                with open(os.path.join(job_dir, SLURM_JOB_ID)) as fh:
                    recorded = fh.read().strip()
            except OSError:
                continue
            if recorded:
                yield job_dir, recorded, job_host(job_dir)


# Marks a descriptor written by adopting a job MIMIR did not submit.
ADOPTED_FILE = "adopted"


def _owned_jobs_roots() -> list[str]:
    """The job directories of this session and of its sub-agents, at any depth."""
    roots = [jobs_dir()]
    if os.environ.get("MIMIR_HPC_JOBS_DIR"):
        return roots
    pending = [session_state_dir()]
    while pending:
        sub = os.path.join(pending.pop(), SUBAGENTS_DIRNAME)
        try:
            children = sorted(os.listdir(sub))
        except OSError:
            continue
        for child in children:
            child_dir = os.path.join(sub, child)
            roots.append(os.path.join(child_dir, _JOBS_SUBDIR))
            pending.append(child_dir)
    return roots


def owned_jobs() -> dict[str, set[str]]:
    """``{job id: hosts}`` this session and its sub-agents submitted: the gate of every ``scancel``.

    Adopted jobs and descriptors without a host are left out.
    """
    owned: dict[str, set[str]] = {}
    for job_dir, recorded, host in _descriptors(_owned_jobs_roots()):
        if host is None or os.path.exists(os.path.join(job_dir, ADOPTED_FILE)):
            continue
        owned.setdefault(recorded, set()).add(host)
    return owned


def owned_job_hosts(job_id: str, owned: dict[str, set[str]] | None = None) -> set[str]:
    """The schedulers on which this session submitted *job_id*. See :func:`owned_jobs`.

    *owned* is a map already walked for, so a caller asking about several ids walks
    once. A task of an array is its own submission or part of one: ``1234_5`` is owned
    by a descriptor recording either it or ``1234``.
    """
    job_id = str(job_id or "").strip()
    if not job_id:
        return set()
    if owned is None:
        owned = owned_jobs()
    return (owned.get(job_id, set()) | owned.get(job_id.split("_", 1)[0], set()))


def owns_job(job_id: str, host: str,
             owned: dict[str, set[str]] | None = None) -> bool:
    """Whether *job_id* on *host* is this session's (or a sub-agent's) to cancel."""
    return host in owned_job_hosts(job_id, owned)


def job_host(job_dir: str) -> str | None:
    """The scheduler of *job_dir*'s job, or None when unrecorded (which is not "local")."""
    try:
        with open(os.path.join(job_dir, HOST_FILE)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def covered_by(job_id: str, recorded: set[tuple[str | None, str]],
               host: str) -> bool:
    """Whether *recorded* covers *job_id* on *host*, an array's tasks by their base id."""
    if not job_id:
        return False
    ids = {job_id, job_id.split("_", 1)[0]}
    for rec_host, rec_id in recorded:
        if rec_id in ids and (host is None or rec_host is None or rec_host == host):
            return True
    return False


# What an unwatched job means, said once, by the layer that knows a scheduler exists.
# Private, because the sentence a caller wants is :func:`unwatched_hint`'s: with two
# schedulers in play an id alone is ambiguous, and this text does not say which host.
_UNWATCHED_HINT = (
    "Nothing is watching these: they were not submitted through the cluster tools, so "
    "no wake is owed when they end and this workspace's server does not stay alive "
    "for them. slurm_watch(job_id) adopts one — an array's base id covers every task.")


def unwatched(queued_ids, *, host: str) -> list[str]:
    """The ids in *queued_ids* that no descriptor covers, by array rather than by task.

    Eight task rows of one array name one descriptor to write, so they collapse to
    their base: listed one by one, a reminder built on this would ask for eight calls
    that all land on "already recorded". Ordered, so the answer reads in queue order.
    """
    recorded = recorded_jobs()
    out: dict[str, None] = {}
    for job_id in queued_ids:
        if job_id and not covered_by(job_id, recorded, host):
            out.setdefault(job_id.split("_", 1)[0], None)
    return list(out)


def unwatched_by_host(queued: dict[str, list[str] | None]) -> dict[str, list[str]]:
    """Per scheduler, what it is running that no descriptor covers.

    Takes :func:`queued_by_host`'s answer, so a cluster that did not reply contributes
    nothing rather than having its jobs reported as unwatched — None is "unknown", and
    the whole failure this exists for is silence being read as an empty queue.
    """
    out: dict[str, list[str]] = {}
    for host, ids in queued.items():
        if ids is None:
            continue
        missing = unwatched(ids, host=host)
        if missing:
            out[host] = missing
    return out


def unwatched_hint(by_host: dict[str, list[str]]) -> str:
    """:data:`_UNWATCHED_HINT`, naming the scheduler when more than one is in play.

    Said only when it is load-bearing: with one scheduler the adoption needs no host,
    and with two an id alone is ambiguous — the same number exists on both.
    """
    live = {host: ids for host, ids in by_host.items() if ids}
    if len(live) <= 1:
        return _UNWATCHED_HINT
    named = "; ".join(f"{name_of(host)}: {', '.join(ids)}" for host, ids in live.items())
    return (_UNWATCHED_HINT + " These are on more than one scheduler — " + named
            + " — so name it: slurm_watch(job_id, scheduler=...).")


# ── A job's state, read into the run words ────────────────────────────────────────────────────────────────────
_ACTIVE = ("RUNNING", "COMPLETING")
_WAITING = ("PENDING", "CONFIGURING", "REQUEUED")
# Prefixes, because Slurm qualifies some words ("CANCELLED by 1234").
_CRASH_PREFIXES = ("FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
                   "BOOT_FAIL", "DEADLINE", "PREEMPTED")
_DONE_PREFIXES = ("COMPLETED",)


def _crashed(word: str) -> bool:
    return word.startswith(_CRASH_PREFIXES)


def _done(word: str) -> bool:
    return word.startswith(_DONE_PREFIXES)


def slurm_state(job_id: str, run_lines: Callable[[str], list[str]]
                ) -> tuple[str, str, list[str]]:
    """``(state, raw, nodes)`` of a Slurm job; *run_lines* runs a command on its scheduler.

    ``squeue``, then ``sacct``; nodes only while running. For an array, any live element
    keeps it live and the verdict is the worst element's. An ended word in squeue settles
    the job without sacct, all a site without accounting has.
    """
    queued, where = [], []
    for row in run_lines(f"squeue -j {shlex.quote(job_id)} -h -o '%T|%R'"):
        word, _, reason_or_node = row.partition("|")
        queued.append(word.strip())
        where.append(reason_or_node.strip())
    raw = ",".join(sorted(set(queued)))
    if any(word in _ACTIVE for word in queued):
        nodes = sorted({node for word, expr in zip(queued, where) if word in _ACTIVE
                        for node in cpu_facts.expand_hostlist(expr) if node})
        return "running", raw, nodes
    if queued and all(word in _WAITING for word in queued):
        return "pending", raw, []
    if queued and all(_done(w) or _crashed(w) for w in queued):
        return ("crashed" if any(_crashed(w) for w in queued) else "done"), raw, []

    accounted = run_lines(f"sacct -j {shlex.quote(job_id)} -n -X -o State")
    sraw = ",".join(sorted(set(accounted)))
    if any(_crashed(word) for word in accounted):
        return "crashed", sraw, []
    if accounted and all(_done(word) for word in accounted):
        return "done", sraw, []
    if queued:                    # in the queue under a non-standard active word
        return "running", raw, []
    if accounted:                 # some other terminal word
        return "crashed", sraw, []
    return "unknown", "", []


def locate_slurm_job(hint: str | None, hosts: list[str],
                     state_on: Callable[[str], tuple[str, str, list[str]]]
                     ) -> tuple[str | None, tuple[str, str, list[str]]]:
    """``(host, state_on(host))`` for the first scheduler that knows the job, its recorded
    *hint* first; ``(None, unknown)`` when none does."""
    order = ([hint] if hint is not None else []) + [h for h in hosts if h != hint]
    for host in order:
        found = state_on(host)
        if found[0] != "unknown":
            return host, found
    return None, ("unknown", "", [])
