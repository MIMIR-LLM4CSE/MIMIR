"""Which Slurm schedulers this workspace reaches, learned rather than configured.

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
where to send its own commands. One successful ``ssh <host> "squeue -u $USER"`` is
enough, and that call is the first thing anybody runs.

**A set of schedulers, not one.** A login node reaches its own and whatever ``ssh``
gets to, and one conversation uses several: a reference run on the local controller
while a sweep goes out to another cluster. So learning a host *adds* a destination and
displaces nothing — the jobs already running keep being polled where they are — and
every job carries the host it was submitted to, which is what lets a wake be owed for
all of them at once. Nothing here picks a destination: a caller names one, and
:func:`resolve_host` says what to answer when it did not.

Deterministic, which is what lets it be acted on: the signal is the shell validator's
own parse of the command — the same mechanism that refuses a bare ``sbatch`` — plus the
exit status. Nothing here matches patterns against free text.

Filed per *workspace*, because the clusters in play are a property of the work and
not of the conversation doing it. The sessions in one workspace share a project, the
filesystems its inputs are on, and therefore the schedulers that can run it — so a
host one of them proved is a host the next can submit to.

That is also what makes the two halves of this module agree. :func:`recorded_jobs`
below reads every session's descriptors, so a job submitted in one conversation is
already findable from the next; filed per session instead, the registry would hand
that conversation a job it cannot poll, because the scheduler running it is not in
its inventory. One fact read at two scopes is how a run ends up watched by nobody.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import time

from state_paths import session_state_dir, state_dir

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
    """The argv that runs *script* on *host* — ``LOCAL`` for this machine's own Slurm.

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
    if not host:
        return ["bash", "-lc", script]
    base = _SSH_BASE if piped else [*_SSH_BASE[:1], "-n", *_SSH_BASE[1:]]
    return base + [host, "bash", "-lc", shlex.quote(script)]


def exec_argv(argv: list[str], *, host: str = LOCAL,
              piped: bool = False) -> list[str]:
    """What to exec to run *argv* on *host* — ``LOCAL`` for this machine's own Slurm.

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
    if not host:
        return list(argv)
    return argv_for(shlex.join(argv), host=host, piped=piped)


def queued_ids(timeout: float = 15.0, *, host: str = LOCAL) -> list[str] | None:
    """The job ids *host* says this user has queued, or None if it did not answer.

    Here rather than in either server because both ask it, and for the same reason: the
    gap between this and the descriptors on disk is the work nothing is watching. None
    is "unknown", which must never be read as "nothing" — the whole failure this exists
    for is silence being taken for an empty queue.
    """
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
    """What every reachable scheduler says this user has queued, keyed by host.

    One entry per host in :func:`hosts`, None where that one did not answer — so a
    cluster that is down leaves its own jobs unknown instead of making the whole sweep
    unknown. Sequential on purpose: the list is short, and each remote call is an ssh
    whose failure mode is a timeout nobody is waiting out twice.
    """
    return {host: queued_ids(timeout, host=host) for host in hosts()}


def slurm_host_from_ssh(argv: list[str]) -> str:
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


def _path() -> str:
    """The file holding this workspace's registry of reachable schedulers.

    One file for the workspace. Every session in it reads and appends to the same
    inventory, which is what lets a scheduler proved by one conversation be a
    destination for the next without an ``ssh`` that has already succeeded once.
    """
    return os.path.join(state_dir(), _FILENAME)


def slurm_is_local() -> bool:
    """Whether Slurm's own tools answer on this host. Probed once per process.

    Through ``bash -lc``, not :func:`shutil.which`, because that is how :func:`argv_for`
    runs the local branch: on a site where ``sbatch`` arrives from a module loaded in
    the login shell, the server process's own ``PATH`` does not have it and the answer
    about that ``PATH`` would be about nothing anybody runs. A probe that will not
    answer is read as "not local", which keeps the remote host — the behaviour of a
    machine that has no scheduler, and the safe direction.
    """
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
    """Every scheduler this workspace can reach, the local one first when there is one.

    The local one is a member by probe rather than by record: it is a fact about the
    machine, true before any call has been made, and there is nothing to learn. It
    comes first because it is the one whose filesystems the work is already on. The
    rest follow in the order they were learned, which is the order they were useful in.
    """
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
    """Add *host* to this workspace's reachable schedulers. Never raises.

    Additive, and that is the whole of it: a scheduler learned displaces nothing, so a
    queue read on a second cluster adds a destination while every job already running
    keeps being polled where it is. Appended only when the host is new, so a
    read-mostly fact does not become a write on every queue listing.
    """
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
    """Drop *host* from the registry, answering whether it was in it.

    For a destination that was a mistake or is gone: a hostname learned from a typo
    that happened to answer, a cluster the user no longer has. The local scheduler is
    not droppable — it is a probe rather than a record, so it would be back on the next
    call, and saying so is better than appearing to work.
    """
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
    """The scheduler to run on, or ``""`` and the reason the caller must choose.

    *requested* is what a tool was handed: :data:`LOCAL_NAME` for this machine's own
    Slurm, a hostname for one behind ``ssh``, or ``""`` for "not specified".

    A name is taken as given rather than checked against the registry. The registry is
    a convenience, not an allowlist — a host the user names is one they can already
    ``ssh`` to — and it joins the registry when a command actually succeeds there.

    Unspecified is where the judgement is. With one scheduler there is no choice and it
    is used. With several there is one, and it belongs to whoever knows what the job
    needs: the GPUs, a compilation node, the filesystem its inputs are on, or simply
    what the user asked for. So this refuses and names the candidates rather than
    taking the local one because it happens to be first — a default here would make
    that decision invisible, which is the failure that put a production run's tools on
    a cluster that had never heard of it.
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
#
# Two halves, and both servers need the same answer. A job is tracked by the descriptor
# its submission wrote — the id file under ``hpc_jobs/`` — so the gap is "what the queue
# says is running" minus "what the disk says we recorded". Owned here rather than in the
# hpc server because the *bash* server is the one that sees the ordinary case: the work
# is submitted by a site launcher and verified with a ``squeue`` a second later, and
# neither call reaches a typed cluster tool. Two copies of this rule would drift, and
# the one that drifted would be the one the night depends on.

# The file a submission's directory is identified by, and the name every reader of a
# descriptor agrees on. :func:`record_job_dir` below is the one writer.
ID_FILE = "slurm_job_id"

# The descriptor a client watcher is to be re-armed from, when the job is not watched
# by the stock poll. Written by a submitter that runs its own watcher — the proxy's
# runs are Slurm jobs of this workspace and its own state machine both — so that a
# reload puts *its* watcher back rather than a second one beside it, and so that
# "is anything watching this id" has one answer. Absent for an ordinary submission,
# whose watcher is slurm_job_status and needs nothing recorded.
WATCHER_FILE = "watcher.json"

_JOBS_SUBDIR = "hpc_jobs"


def jobs_dir() -> str:
    """Where this session files the jobs it submits.

    Per session, unlike the registry above, and for the opposite reason: a submission
    belongs to the conversation that made it, two conversations submitting in the same
    second must not land on one directory, and a conversation's submissions go with it
    when it is deleted. Reading is workspace-wide all the same — see
    :func:`recorded_jobs` — because a job outlives the conversation that started it.

    Env-overridable for tests, which is the one seam that lets a submission be filed
    somewhere a test can see without standing up a session.
    """
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
    """An empty directory for a submission about to be made, created.

    Split from :func:`record_job_dir` for the submitter that needs the directory
    *before* it has an id: the batch script is written there and the id only exists
    once ``sbatch`` has answered. Both halves name the directory the same way, which
    is the whole reason they live together.
    """
    base = jobs_dir()
    job_dir = os.path.join(base, _new_job_dir_name(base))
    os.makedirs(job_dir, exist_ok=True)
    return job_dir


def record_job_dir(job_id: str, **files: str) -> str:
    """File *job_id* as a job of this workspace and return its directory.

    The one writer of the layout every reader here depends on, which is why it lives
    beside them rather than in whichever server happens to submit. Three callers, and
    the third is the point: a submission, an adoption, and the proxy — whose runs are
    Slurm jobs of this workspace as much as they are its own state machine. Filed
    anywhere else, they are jobs that "nothing is watching" to every reader of this
    module, and the remedy it offers — adopt the id — arms a second watcher on a run
    that already has one.
    """
    job_dir = new_job_dir()
    for name, content in {ID_FILE: job_id, **files}.items():
        with open(os.path.join(job_dir, name), "w") as fh:
            fh.write(content)
    return job_dir


def recorded_jobs() -> set[tuple[str | None, str]]:
    """Every job this workspace holds a descriptor for, as ``(host, id)`` pairs.

    A set rather than a search per id: the callers are listings, and a search only
    returns early on a hit — so the ids that answer "no", which are the interesting
    ones, would each pay a complete walk of every session's submissions.

    Keyed by host because two clusters number independently: ``9001`` is a
    production run on one and somebody else's sweep on the other, and an id alone would
    have the descriptor here vouch for the job there — reporting a watched run that
    nothing is watching, which is the one mistake this whole mechanism exists to
    prevent. ``None`` is a descriptor that recorded no host (see :func:`job_host`), and
    :func:`covered_by` lets it vouch for any: an older submission must not be read as
    unwatched work.

    Read from :func:`jobs_dir` and from every session's directory under it. Both,
    because the two name different places outside a session and under the test
    override, and a reader seeing one of them calls this session's own submissions
    unwatched — the one answer this must never give wrongly.

    There is nowhere else to look, and that is a decision rather than a coincidence.
    A submitter that filed its jobs in a tree of its own — the proxy did — is one whose
    runs read as work nobody is watching to everything built on this, so
    :func:`record_job_dir` is the only writer there is.
    """
    found: set[tuple[str | None, str]] = set()
    roots = [jobs_dir()]
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
            job_dir = os.path.join(root, name)
            try:
                with open(os.path.join(job_dir, ID_FILE)) as fh:
                    recorded = fh.read().strip()
            except OSError:
                continue
            if recorded:
                found.add((job_host(job_dir), recorded))
    return found


def job_host(job_dir: str) -> str | None:
    """The scheduler *job_dir*'s job is on: a hostname, ``LOCAL``, or None if unrecorded.

    None is not "local". A descriptor that never recorded a host cannot say where its
    job is, and reading that silence as this machine would poll the wrong scheduler and
    call a running job forgotten. The callers answer it by asking every reachable
    scheduler instead, and write down the one that knew.
    """
    try:
        with open(os.path.join(job_dir, HOST_FILE)) as fh:
            return fh.read().strip()
    except OSError:
        return None


def covered_by(job_id: str, recorded: set[tuple[str | None, str]],
               host: str) -> bool:
    """Whether *recorded* holds a descriptor covering *job_id* on *host*.

    An array is the whole reason this is not ``in``. It is adopted or submitted under
    its base id and the queue then lists its tasks — ``15306621_3``, or
    ``15306621_[4-8]`` for the ones still pending — so the id in a row is rarely the id
    on disk. The base is what covers them: a watcher on it polls the whole array.

    *host* is required, because every caller has just read one scheduler's queue and
    knows which: the descriptor that may vouch is the one filed against that same
    scheduler. The exception is on the *recorded* side — a descriptor that never wrote
    a host down vouches for any, since an older submission read as unwatched work is a
    reminder nobody can act on.
    """
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
