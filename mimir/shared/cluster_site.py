"""Which batch schedulers (Slurm, LSF) this workspace reaches, learned from successful calls.

The bash server records a host whenever an ``ssh <host> <scheduler command>`` succeeds
(from its own parse, never from text), with the kind of scheduler the command proved;
the hpc and proxy servers send their commands there, spelled by that kind
(``shared/schedulers``). Several hosts at once, filed per workspace; job directories
are per session.
"""

from __future__ import annotations

from typing import Callable

import os
import re
import shlex
import subprocess
import time

from . import schedulers
from .jobs import SLURM_JOB_ID
from .schedulers import UNKNOWN_JOB, JobState, Scheduler, array_base
from .shell_paths import ShellParseError, parse_segments, unwrap_argv
from .state_paths import session_family, session_state_dir, sessions_root, state_dir

# The commands that only exist where a scheduler does. A remote call whose head is one
# of these, and that succeeded, proves the far side has a scheduler — no other host
# would have answered it — and which kind.
SCHEDULER_COMMANDS = schedulers.ALL_COMMANDS

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

# The workspace's registry of reachable schedulers, one host per line, then its kind
# when it is not Slurm (``login2 lsf``).
_FILENAME = "slurm_hosts"

# In a job descriptor: the host of the scheduler that job is on. Empty contents mean
# LOCAL, and an absent file means a descriptor written before the host was recorded
# with it — see :func:`job_host`, which answers None there rather than guessing.
HOST_FILE = "slurm_host"

# The kind of scheduler this host answers for (``""`` for none), memoised: asked on
# every call that reaches a scheduler, and the answer cannot change under a running
# process. None until probed.
_LOCAL_KIND: str | None = None

# Short, because this runs inside a tool call the user is waiting on, and a login shell
# that takes longer than this to answer "command -v" is not going to be this server's
# fast path anyway.
_LOCAL_PROBE_TIMEOUT = 10

# ``BatchMode=yes`` is not optional: a poller that can be asked for a password hangs for
# its timeout on every tick instead of answering. No ssh *options* beyond that pair — an
# identity file, a port, a jump host are ``~/.ssh/config``'s business, which is a
# configuration mechanism the user already has and we have no business duplicating.
_SSH_BASE = ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]

# What a host may be: a name, an address or ``user@host``, never an ssh option. A
# leading "-" would make ``ssh`` read the "host" as one (``-oProxyCommand=…`` runs a
# local command), and a scheduler named by the model reaches ssh unasked.
_HOST_SHAPE = re.compile(r"[A-Za-z0-9_\[][A-Za-z0-9_.@:%\[\]-]*")


class InvalidHost(ValueError):
    """A scheduler name that cannot be an ssh destination."""


def valid_host(host: str) -> bool:
    """Whether *host* can only be read by ``ssh`` as a destination."""
    return bool(_HOST_SHAPE.fullmatch(host or ""))


def _invalid(host: str) -> str:
    return (f"{host!r} is not a host name: a scheduler is a name, an address or "
            "user@host, and cannot start with \"-\".")


def argv_for(script: str, *, host: str = LOCAL, piped: bool = False) -> list[str]:
    """The argv running *script* on *host* (``LOCAL`` or over ``ssh``) in a login shell.

    Quoted once for the remote shell; ``-n`` closes stdin unless *piped*.
    """
    if not host:
        return ["bash", "-lc", script]
    if not valid_host(host):
        raise InvalidHost(_invalid(host))
    base = _SSH_BASE if piped else [*_SSH_BASE[:1], "-n", *_SSH_BASE[1:]]
    return base + ["--", host, "bash", "-lc", shlex.quote(script)]


def exec_argv(argv: list[str], *, host: str = LOCAL,
              piped: bool = False) -> list[str]:
    """The argv running *argv* on *host*: as is locally, each element quoted for ``ssh``."""
    if not host:
        return list(argv)
    return argv_for(shlex.join(argv), host=host, piped=piped)


def read_lines(script: str, timeout: float = 15.0, *, host: str = LOCAL) -> list[str] | None:
    """The non-empty lines *script* prints on *host*, or None when it could not be asked."""
    try:
        res = subprocess.run(
            argv_for(script, host=host),
            stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, text=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError, InvalidHost):
        return None
    if res.returncode != 0:
        return None
    return [line.strip() for line in res.stdout.splitlines() if line.strip()]


def queued_ids(timeout: float = 15.0, *, host: str = LOCAL) -> list[str] | None:
    """The job ids *host* has queued for this user, or None ("unknown", never "nothing")."""
    return read_lines(scheduler_for(host).queued_ids_command, timeout, host=host)


def queued_submissions(timeout: float = 15.0, *, host: str = LOCAL) -> dict[str, float] | None:
    """``{job id: submit time}`` of what *host* has queued for this user, or None.

    The time is read on *host*, in its own time zone, as epoch seconds.
    """
    lines = read_lines(scheduler_for(host).queued_submissions_command, timeout, host=host)
    if lines is None:
        return None
    out: dict[str, float] = {}
    for line in lines:
        job_id, _, moment = line.partition("|")
        try:
            out[job_id.strip()] = float(moment)
        except ValueError:
            continue
    return out


def queued_by_host(timeout: float = 15.0) -> dict[str, list[str] | None]:
    """:func:`queued_ids` for every reachable host, sequentially; None where one did not answer."""
    return {host: queued_ids(timeout, host=host) for host in hosts()}


def ssh_target(argv: list[str]) -> tuple[str, list[str]]:
    """The host an ``ssh`` argv reaches and the words it runs there; ``("", [])`` for any other."""
    if not argv or os.path.basename(argv[0]) != "ssh":
        return "", []
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
    return host, rest


def scheduler_from_ssh(argv: list[str]) -> tuple[str, str]:
    """``(host, kind)`` of the scheduler an ``ssh`` argv reached, or ``("", "")``."""
    host, rest = ssh_target(argv)
    kind = scheduler_named(" ".join(rest)) if host and rest else ""
    # A user@host target reaches the same scheduler; the account is ssh's business and
    # is kept, so the recorded target is exactly what worked.
    return (host, kind) if kind else ("", "")


def scheduler_named(remote: str) -> str:
    """The kind of scheduler the command line *remote* runs a command of, anywhere in
    its chain, or "".

    Anywhere, not first: ``cd run && module load x && srun ./solver`` is the ordinary
    shape. A command behind a wrapper counts, and ``srun`` is itself one.
    """
    try:
        segments = parse_segments(remote)
    except ShellParseError:
        return ""
    for segment in segments:
        if not segment.argv:
            continue
        inner, wrappers = unwrap_argv(segment.argv)
        heads = [os.path.basename(w) for w in wrappers]
        if inner:
            heads.append(os.path.basename(inner[0]))
        for head in heads:
            kind = schedulers.kind_of_command(head)
            if kind:
                return kind
    return ""


def _path() -> str:
    """The file holding this workspace's registry of reachable schedulers.

    One file for the workspace. Every session in it reads and appends to the same
    inventory, which is what lets a scheduler proved by one conversation be a
    destination for the next without an ``ssh`` that has already succeeded once.
    """
    return os.path.join(state_dir(), _FILENAME)


def local_kind() -> str:
    """The scheduler answering on this host through a login shell: ``slurm``, ``lsf`` or ""
    (probed once; failure means none). Slurm first, where a site has both."""
    global _LOCAL_KIND
    if _LOCAL_KIND is None:
        script = ('command -v sbatch >/dev/null && echo slurm && exit; '
                  'command -v bsub >/dev/null && echo lsf; true')
        try:
            res = subprocess.run(
                ["bash", "-lc", script],
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL, text=True, timeout=_LOCAL_PROBE_TIMEOUT,
            )
            found = res.stdout.strip().splitlines()
            _LOCAL_KIND = found[-1] if found and found[-1] in schedulers.BY_KIND else ""
        except (OSError, subprocess.SubprocessError):
            _LOCAL_KIND = ""
    return _LOCAL_KIND


def scheduler_is_local() -> bool:
    """Whether a scheduler, of any kind, answers on this host."""
    return bool(local_kind())


def slurm_is_local() -> bool:
    """Whether the scheduler answering on this host is Slurm."""
    return local_kind() == schedulers.SLURM.kind


def name_of(host: str) -> str:
    """How to spell *host* back to a caller: ``local``, or the hostname itself.

    One spelling in every direction. ``LOCAL`` is the empty string because that is what
    :func:`argv_for` needs in order to run no ``ssh`` at all, and an empty string in an
    answer reads as a missing value rather than as a scheduler.
    """
    return LOCAL_NAME if not host else host


def _registry() -> dict[str, str]:
    """``{host: kind}`` of the learned schedulers, in the order they were learned."""
    try:
        with open(_path()) as fh:
            lines = fh.read().splitlines()
    except OSError:
        lines = []
    out: dict[str, str] = {}
    for line in lines:
        words = line.split()
        if not words or not valid_host(words[0]) or words[0] in out:
            continue
        kind = words[1] if len(words) > 1 else schedulers.SLURM.kind
        out[words[0]] = kind if kind in schedulers.BY_KIND else schedulers.SLURM.kind
    return out


def hosts() -> list[str]:
    """Every reachable scheduler: the local one first (by probe), then the learned ones in order."""
    out = [LOCAL] if scheduler_is_local() else []
    return out + [host for host in _registry() if host not in out]


def kind_of(host: str) -> str:
    """The kind of the scheduler on *host*: the local probe's, else the registry's, else Slurm."""
    if not host:
        return local_kind() or schedulers.SLURM.kind
    return _registry().get(host, schedulers.SLURM.kind)


def scheduler_for(host: str) -> Scheduler:
    """How to speak to the scheduler on *host* (``shared/schedulers``)."""
    return schedulers.BY_KIND[kind_of(host)]


def record_host(host: str, kind: str = schedulers.SLURM.kind) -> None:
    """Add *host* to the workspace's schedulers if new (never raises); a Slurm host is
    written bare, the way a line with no kind reads."""
    host = (host or "").strip()
    if not valid_host(host) or host in hosts() or kind not in schedulers.BY_KIND:
        return
    path = _path()
    line = host if kind == schedulers.SLURM.kind else f"{host} {kind}"
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "a") as fh:
            fh.write(line + "\n")
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
    if host not in [line.split()[0] for line in lines]:
        return False
    try:
        with open(path, "w") as fh:
            fh.write("".join(f"{line}\n" for line in lines if line.split()[0] != host))
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
        if not scheduler_is_local():
            return "", ("No Slurm or LSF command answers on this host, so there is no "
                        "local scheduler to use. " + _choices(known))
        return LOCAL, ""
    if requested:
        return (requested, "") if valid_host(requested) else ("", _invalid(requested))
    if len(known) == 1:
        return known[0], ""
    if not known:
        return "", ("No scheduler answers on this host and none has been reached over "
                    "ssh yet. Read a queue through the shell once — "
                    "ssh <host> \"squeue -u $USER\" (or \"bjobs\" under LSF) — and it "
                    "is learned, or name one with slurm_schedulers(add=\"<host>\").")
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
    return roots + [os.path.join(d, _JOBS_SUBDIR)
                    for d in session_family(session_state_dir())[1:]]


def owned_jobs() -> dict[str, set[str]]:
    """``{job id: hosts}`` this session and its sub-agents submitted: the gate of every cancellation.

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
    return (owned.get(job_id, set()) | owned.get(array_base(job_id), set()))


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
    ids = {job_id, array_base(job_id)}
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
            out.setdefault(array_base(job_id), None)
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


# ── A job's state, read into the run words ─────────────────────────────────────

def job_state(job_id: str, run_lines, host: str) -> JobState:
    """The :class:`JobState` of a job on *host*'s scheduler; *run_lines* runs a command there."""
    return scheduler_for(host).state(job_id, run_lines)


def job_state_on(job_id: str, host: str, timeout: float = 15.0) -> JobState | None:
    """:func:`job_state` of *job_id* on *host*, or None when nothing there answered.

    One read failing says nothing: ``squeue -j`` and ``bjobs`` exit non-zero on a job
    they have forgotten.
    """
    answers: list[bool] = []

    def run_lines(script: str) -> list[str]:
        lines = read_lines(script, timeout, host=host)
        answers.append(lines is not None)
        return lines or []

    found = job_state(job_id, run_lines, host)
    return found if any(answers) else None


def locate_job(hint: str | None, hosts: list[str],
               state_on: Callable[[str], JobState]
               ) -> tuple[str | None, JobState]:
    """``(host, state_on(host))`` for the first scheduler that knows the job, its recorded
    *hint* first; ``(None, UNKNOWN_JOB)`` when none does."""
    order = ([hint] if hint is not None else []) + [h for h in hosts if h != hint]
    for host in order:
        found = state_on(host)
        if found.state != "unknown":
            return host, found
    return None, UNKNOWN_JOB
