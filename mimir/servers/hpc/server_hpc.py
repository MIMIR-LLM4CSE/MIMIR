"""
MCP HPC Server
==============
HPC helpers for Slurm scheduling and batch job submission.

This server is designed for cluster usage where:
- Slurm tools are available (`sinfo`, `squeue`, `scontrol`, `salloc`, `sbatch`)

Environment Modules / Lmod are handled directly through the bash server's
`module` command, not here.

Safety model:
- Query tools are read-only.
- Both submitters take resources as *arguments* and build the command themselves —
  the model never hands over a command string — and launch it as argv. They are
  approval-gated client-side (``CLUSTER_SUBMIT``). The read-only queries do use a
  shell, for ``$USER`` expansion; their filters are quoted. With a remote controller
  one shell is unavoidable, since ``ssh`` takes a command line and not an argv: each
  element is quoted on the way out, so what the far side splits is exactly the argv
  built here (``_shared/slurm_site.exec_argv``).
- ``sbatch_submit`` returns a ``background_job`` descriptor that the client watcher
  polls to completion via ``slurm_job_status``.
- ``slurm_cancel`` takes one job ID, refuses a job the user does not own, and is
  approval-gated (irreversible) but not ``CLUSTER_SUBMIT``: stopping a job spends no
  allocation, so the local-validation hold has nothing to protect there.
"""

import os
import re
import shlex
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE, IRREVERSIBLE
from responses import err, ok
from state_paths import session_state_dir, state_dir
from slurm_site import (
    ID_FILE as _ID_FILE,
    UNWATCHED_HINT,
    controller as slurm_controller,
    covered_by as _covered_by,
    recorded_job_ids as _recorded_job_ids,
    unwatched as slurm_unwatched,
    argv_for as slurm_argv,
    exec_argv as slurm_exec_argv,
    queued_ids as slurm_queued_ids,
)
from slurm_nodes import (
    aggregate_node_types as _aggregate_node_types,
    as_int as _as_int,
    clean_gres as _clean_gres,
    parse_scontrol_nodes as _parse_scontrol_nodes,
)

mcp = FastMCP(
    "HPCServer",
    debug=False,
    log_level="ERROR",
)

_TIMEOUT_READ = 15
_TIMEOUT_ALLOC = 20
_TIMEOUT_SUBMIT = 30
_MAX_OUTPUT = 128 * 1024

# Where a batch submission's own bookkeeping lives — the script, the id, the state a
# poll settled it at — under the submitting session's state dir, like every other
# artefact belonging to a conversation (env-overridable for tests).
#
# Not the job's output: that goes where the job runs, which is the only directory it is
# certain to be able to write. See _sbatch_header.
#
# Per session, resolved per call. A Slurm job outlives its session, sometimes by days,
# which is the one way these differ from detached shell jobs — so a job is never
# *unreachable* from another conversation: _find_job_dir below looks in the sibling
# sessions read-only and says where it came from. Per session buys two things: two
# conversations submitting at the same moment cannot land on each other's directory, and a
# conversation's submissions go with it when it is deleted.
def _hpc_jobs_dir() -> str:
    env = os.environ.get("MIMIR_HPC_JOBS_DIR")
    if env:
        return env
    return os.path.join(session_state_dir(), "hpc_jobs")


def _new_job_dir_name() -> str:
    """A sortable, collision-free directory name for one submission.

    Shares ``_bash_jobs``' key shape — UTC stamp plus four random hex — because the stamp
    alone collides: it resolves to the second, and with ``makedirs(exist_ok=True)`` two
    submissions in that second would share one directory and overwrite each other's
    batch_script.sh and slurm_job_id. One second is a long time when two sessions are
    working at once.
    """
    import time as _time
    stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
    base = _hpc_jobs_dir()
    while True:
        name = f"{stamp}-{os.urandom(2).hex()}"
        if not os.path.exists(os.path.join(base, name)):
            return name


def _no_scheduler_hint() -> str:
    """What to do when no Slurm command answers here. Self-documenting on purpose.

    There is no setting to point at, so the answer has to be a thing the user is about
    to do anyway: reach the scheduler over ``ssh`` once. That call records where it is
    and every tool here follows. Said only when nothing has been recorded yet — once it
    has, a failure means something else and this would be a wrong turn.
    """
    if slurm_controller():
        return (f"The scheduler is reached through {slurm_controller()}; check that "
                "host answers.")
    return ("No Slurm command answered on this host. If the scheduler lives on "
            "another one, read the queue there once through the shell — "
            "ssh <host> \"squeue -u $USER\" — and every cluster tool here will use "
            "that host from then on.")


def _add_unwatched(payload: dict) -> None:
    """Name the user's running jobs that no descriptor covers, if any.

    Attached to the answers of the two tools that *record* a job — a submission and an
    adoption — because those are the moments the model is holding job ids and can act
    on it in the same turn. Deliberately not to ``slurm_job_status``: that one is the
    watcher's poll target, called on a timer for the life of every watched run, and a
    per-user queue read on it would be paid for ever.

    Best effort: a scheduler that will not answer leaves the payload alone rather than
    turning a successful submission into an error.
    """
    ids = slurm_queued_ids(_TIMEOUT_READ)
    if ids is None:
        return
    missing = slurm_unwatched(ids, _hpc_jobs_dir())
    if missing:
        payload["unwatched"] = missing
        payload["unwatched_hint"] = UNWATCHED_HINT


def _find_job_dir(job_id: str) -> tuple[str, str]:
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
    sessions = os.path.join(state_dir(), "sessions")
    try:
        others = sorted(os.listdir(sessions))
    except OSError:
        return "", ""
    for sid in others:
        found = _match(os.path.join(sessions, sid, "hpc_jobs"))
        if found:
            return found, sid
    return "", ""


# What an adopted submission's directory carries instead of a batch script: the job was
# not built here, so there is nothing of ours to keep but the fact that we took it on.
_ADOPTED_FILE = "adopted"

# What a job's own output is called, beside the work it did. ``%j`` is Slurm's, and the
# id is not knowable until the submission answers.
_LOG_NAME = "slurm-%j.log"

# Where that resolved path is written down, so a later status answer can name the file
# instead of re-deriving a rule that may have changed since the job was submitted.
_LOG_FILE = "log"


def _record_job_dir(job_id: str, **files: str) -> str:
    """Make this session's directory for *job_id* and return it.

    The one writer of the descriptor layout every reader depends on. Both tools that
    record a job — the one that submits it and the one that adopts it — come through
    here, so the id file's name and the directory's shape are stated once.
    """
    job_dir = os.path.join(_hpc_jobs_dir(), _new_job_dir_name())
    os.makedirs(job_dir, exist_ok=True)
    for name, content in {_ID_FILE: job_id, **files}.items():
        with open(os.path.join(job_dir, name), "w") as fh:
            fh.write(content)
    return job_dir


def _slurm_background_job(job_id: str) -> dict:
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
_TERMINAL_STATES = ("done", "crashed", "unknown")


def _recorded_log(job_dir: str) -> str:
    """Where this submission told its job to write, as the submission wrote it down.

    Falls back to the name a batch job's log had when it lived in the submission's own
    directory: a job submitted by an earlier build has no record of its path, and the
    rule it was submitted under is the only thing that can find its output.
    """
    try:
        with open(os.path.join(job_dir, _LOG_FILE)) as fh:
            recorded = fh.read().strip()
    except OSError:
        recorded = ""
    if recorded:
        return recorded
    legacy = os.path.join(job_dir, "slurm.log")
    return legacy if os.path.exists(legacy) else ""


def _record_terminal_state(job_dir: str, state: str) -> None:
    """Settle *job_dir* at the first poll that finds Slurm has let the job go.

    Write-once, and that is the point: the first observation of a terminal state is the
    truest one available. A later poll of the same job reads ``unknown`` once sacct's
    retention window has passed, and letting that overwrite a recorded ``crashed`` would
    turn every outcome into "the scheduler has forgotten it" given enough time.

    Best-effort, like the markers in ``job_scan``: a status answer is worth more than the
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


# ── Where the controller is ───────────────────────────────────────────────────
#
# Not a setting: ``_shared/slurm_site`` learns it from the user's own successful
# ``ssh host "squeue …"``, which is the first thing anybody runs on a login node whose
# scheduler lives elsewhere. See that module for why a variable nobody sets before the
# first lost night is the wrong mechanism, and for how a command is run over there.
#
# One seam: every Slurm command this server runs goes through ``slurm_argv``. Nothing
# else in this file knows whether the scheduler is local.
def _slurm_host() -> str:
    """The host whose Slurm tools to use, or "" when they are this machine's."""
    return slurm_controller()


def _run_bash(script: str, timeout: int, stdin: str | None = None) -> dict:
    """Run *script* where Slurm is, optionally feeding it *stdin*.

    Without *stdin* the command gets none at all: the server's own is its MCP protocol
    pipe, and a command that reads it (``cat``, an ``ssh`` without ``-n``) eats the
    client's traffic. With it, the text is handed over on a pipe we opened — which is
    how a batch script reaches ``sbatch`` without any file having to exist on the host
    that runs it.
    """
    try:
        res = subprocess.run(
            slurm_argv(script, piped=stdin is not None),
            input=stdin,
            stdin=None if stdin is not None else subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        stdout = res.stdout[:_MAX_OUTPUT]
        stderr = res.stderr[:_MAX_OUTPUT]
        payload = {
            "status": "ok" if res.returncode == 0 else "error",
            "returncode": res.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
        if len(res.stdout) > _MAX_OUTPUT or len(res.stderr) > _MAX_OUTPUT:
            payload["truncated"] = True
        return payload
    except subprocess.TimeoutExpired:
        return {
            "status": "error",
            "error": f"Command timed out after {timeout}s.",
            "hint": "Narrow the query or increase timeout.",
        }
    except Exception as e:
        return err(str(e))


def _parse_pipe_table(stdout: str, columns: list[str]) -> list[dict]:
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != len(columns):
            continue
        rows.append({columns[i]: parts[i] for i in range(len(columns))})
    return rows


def _validate_time(value: str) -> bool:
    # Accept Slurm-like forms: MM, MM:SS, HH:MM:SS, D-HH, D-HH:MM, D-HH:MM:SS
    return bool(re.fullmatch(r"\d+(-\d{1,2}(:\d{2}(:\d{2})?)?|(:\d{2}){0,2})", value))


def _validate_mem(value: str) -> bool:
    # Examples: 8G, 32000M, 2T
    return bool(re.fullmatch(r"\d+[KMGTP]", value.upper()))


_COMMENT_MAX = 512


def _validate_comment(value: str) -> str | None:
    """Return why *value* cannot be a Slurm --comment, or None when it can.

    Free text, so the only hard rules are a length Slurm will store and a single
    line: a newline in a value that lands in a ``#SBATCH`` directive turns the rest
    of it into directives of the caller's choosing, and sacct renders the field on
    one line regardless.
    """
    if not value:
        return None
    if len(value) > _COMMENT_MAX:
        return f"comment is too long ({len(value)} chars). Keep it under {_COMMENT_MAX}."
    bad = [ch for ch in value if ord(ch) < 0x20 or ord(ch) == 0x7F]
    if bad:
        return ("Invalid comment: control characters are not allowed "
                f"(found {bad[0]!r}). Keep it to a single line of plain text.")
    return None


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_partitions() -> dict:
    """List Slurm partitions with key scheduling attributes."""
    cmd = "sinfo -h -o '%P|%a|%l|%D|%t|%c|%m'"
    result = _run_bash(cmd, _TIMEOUT_READ)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "sinfo failed"),
                    hint=_no_scheduler_hint())
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["partition", "availability", "time_limit", "nodes", "state", "cpus_per_node", "mem_mb_per_node"],
    )
    return ok({"partitions": rows, "count": len(rows)})


def _sinfo_nodes() -> list[dict]:
    """Degraded node list for a cluster whose `scontrol show node` is restricted.

    sinfo is readable everywhere but does not carry the architecture, so callers are
    told the field is unknown rather than being handed a wrong default.
    """
    result = _run_bash("sinfo -N -h -o '%N|%P|%t|%c|%m|%e|%G|%X|%Y|%Z'", _TIMEOUT_READ)
    if result["status"] != "ok":
        return []
    rows = _parse_pipe_table(result.get("stdout", ""), [
        "node", "partition", "state", "cpus", "mem_mb", "mem_free_mb", "gres",
        "sockets", "cores_per_socket", "threads_per_core",
    ])
    # sinfo -N emits one row per (node, partition), so a node in three partitions
    # appears three times; merge on the node name or every count is inflated.
    nodes: list[dict] = []
    by_name: dict[str, dict] = {}
    for r in rows:
        part = r["partition"].rstrip("*")
        if r["node"] in by_name:
            if part not in by_name[r["node"]]["partitions"]:
                by_name[r["node"]]["partitions"].append(part)
            continue
        by_name[r["node"]] = {
            "node": r["node"], "arch": "", "state": r["state"].upper(),
            "partitions": [part],
            "cpus": _as_int(r["cpus"].rstrip("+")), "cpus_allocated": None, "cpus_free": None,
            "cpu_load": "", "sockets": _as_int(r["sockets"]),
            "cores_per_socket": _as_int(r["cores_per_socket"]),
            "threads_per_core": _as_int(r["threads_per_core"]),
            "mem_mb": _as_int(r["mem_mb"]), "mem_free_mb": _as_int(r["mem_free_mb"]),
            "gres": _clean_gres(r["gres"]), "features": "",
        }
        nodes.append(by_name[r["node"]])
    return nodes


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_nodes(partition: str = "", states: str = "", node: str = "", detail: bool = False) -> dict:
    """Inventory the cluster's compute nodes: hardware, GPUs, and what is free right now.

    Read-only and instant — it reads Slurm's own node database, which is what actually
    governs placement, so it allocates nothing. Use it before a submission to pick the
    partition and resources that fit: note that **architecture varies between nodes** on
    a mixed cluster, so a binary built on the login node will not necessarily run on the
    node you submit to.

    Returns node *types* (nodes collapsed onto their hardware signature, with a count
    per state) unless you ask for detail or name a node — a per-node listing of a large
    cluster is mostly noise.

    Args:
        partition: Only nodes in this partition.
        states: Comma-separated state filter, e.g. 'idle,mix,alloc'.
        node: A single node name; implies detail.
        detail: List every matching node individually instead of aggregating.
    """
    degraded = ""
    result = _run_argv(["scontrol", "show", "node", "-o"], _TIMEOUT_READ)
    if result["status"] == "ok":
        nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    else:
        nodes = _sinfo_nodes()
        degraded = "scontrol unavailable: architecture and live CPU occupancy are unknown."
    if not nodes:
        return err(result.get("stderr") or result.get("error", "node query failed"),
                   hint="Ensure Slurm commands are available on this host.")

    if node:
        wanted = {n.strip() for n in node.split(",") if n.strip()}
        nodes = [n for n in nodes if n["node"] in wanted]
    if partition:
        nodes = [n for n in nodes if partition in n["partitions"]]
    if states:
        wanted = {s.strip().lower() for s in states.split(",") if s.strip()}
        nodes = [n for n in nodes if any(part.lower() in wanted for part in (n["state"] or "").split("+"))]

    if not nodes:
        return ok({"nodes": [], "count": 0,
                   "note": "No node matched the filters."})
    if detail or node:
        payload = {"nodes": nodes, "count": len(nodes)}
    else:
        types = _aggregate_node_types(nodes)
        payload = {
            "node_types":   types,
            "type_count":   len(types),
            "nodes_total":  len(nodes),
            "architectures": sorted({n["arch"] for n in nodes if n["arch"]}),
            "note": "Aggregated by hardware signature; pass detail=True or node='<name>' for individual nodes.",
        }
    if degraded:
        payload["degraded"] = degraded
    return ok(payload)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_queue(user_only: bool = True, states: str = "") -> dict:
    """List jobs from Slurm queue.

    Args:
        user_only: If true, show only jobs for the current user.
        states: Optional state filter (e.g. 'R,PD').
    """
    user_flag = " -u $USER" if user_only else ""
    state_flag = f" -t {shlex.quote(states)}" if states else ""
    cmd = f"squeue -h{user_flag}{state_flag} -o '%i|%u|%T|%M|%l|%D|%R|%j'"
    result = _run_bash(cmd, _TIMEOUT_READ)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "squeue failed"),
                   hint=_no_scheduler_hint())
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["job_id", "user", "state", "elapsed", "time_limit", "nodes", "reason_or_node", "name"],
    )
    # Which of these this workspace is actually watching, which is not a thing the
    # scheduler knows: a job is tracked by the descriptor its submission wrote, and a
    # job submitted any other way — a site launcher, a script, a shell on another host
    # — wrote none. Answered per row rather than left to be worked out, because the
    # consequence is invisible from both ends: nothing polls that job, no wake is owed
    # when it ends, and the process stops on its idle TTL with the run still going.
    #
    # Two facts, no guessing: the id the scheduler printed, and whether a directory
    # under any session holds it. That is also what makes it fit to be *recorded* —
    # the unwatched set is handed to the client, where a nudge reads it at the end of
    # the turn, and a nudge may only ever rest on something checkable.
    recorded = _recorded_job_ids(_hpc_jobs_dir())
    ids = [str(row.get("job_id") or "") for row in rows]
    for row, job_id in zip(rows, ids):
        row["watched"] = _covered_by(job_id, recorded)
    payload = {"jobs": rows, "count": len(rows)}
    # Per row above, by array here: eight task rows name one descriptor to write, and
    # listing them one by one would ask for eight calls that all land on "already
    # recorded".
    missing = slurm_unwatched(ids, _hpc_jobs_dir())
    if missing:
        payload["unwatched"] = missing
        payload["unwatched_hint"] = UNWATCHED_HINT
    return ok(payload)


def _run_argv(argv: list[str], timeout: int) -> dict:
    """Run a command as argv — no shell, so no argument can inject a second command.

    With a remote controller one shell is unavoidable: ``ssh`` has no argv to hand over,
    only a command line for the far side to parse. So each element is quoted on the way
    out and the property survives in the form that matters — what the remote shell
    splits is exactly the argv built here, and an argument cannot become a command.
    """
    try:
        res = subprocess.run(
            slurm_exec_argv(argv), stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
        )
        return {
            "status": "ok" if res.returncode == 0 else "error",
            "returncode": res.returncode,
            "stdout": res.stdout[:_MAX_OUTPUT],
            "stderr": res.stderr[:_MAX_OUTPUT],
        }
    except subprocess.TimeoutExpired:
        return {"status": "error", "error": f"Command timed out after {timeout}s."}
    except Exception as e:
        return err(str(e))


def _salloc_argv(partition: str, account: str, qos: str, nodes: int, ntasks: int,
                 cpus_per_task: int, mem: str, time: str, gres: str, constraint: str,
                 job_name: str, extra_args: str,
                 comment: str = "") -> tuple[list[str], dict | None]:
    """Build a validated salloc argv, or return the rejection."""
    if not partition.strip():
        return [], err("partition is required.", hint="Use slurm_partitions() to list them.")
    if nodes < 1 or ntasks < 1 or cpus_per_task < 1:
        return [], err("nodes, ntasks, and cpus_per_task must be >= 1")
    if time and not _validate_time(time):
        return [], err("Invalid Slurm time format", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not _validate_mem(mem):
        return [], err("Invalid mem format", hint="Use values like 8G, 32000M, 1T")
    comment_err = _validate_comment(comment)
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


@mcp.tool(**tool_caps(kind="slurm",
    caps=[PLAN_BLOCKED, CLUSTER_SUBMIT], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="requests Slurm resource allocation",
))
def salloc_submit(
    partition: str,
    account: str = "",
    qos: str = "",
    nodes: int = 1,
    ntasks: int = 1,
    cpus_per_task: int = 1,
    mem: str = "",
    time: str = "01:00:00",
    gres: str = "",
    constraint: str = "",
    comment: str = "",
    job_name: str = "mimir-interactive",
    confirm: bool = False,
    timeout_seconds: int = _TIMEOUT_ALLOC,
    extra_args: str = "",
) -> dict:
    """Request an interactive Slurm allocation (sensitive, synchronous).

    Takes the resources as arguments and builds the salloc command itself, so what is
    validated is what runs. Call with confirm=False first to see the exact command
    without executing it. For a non-blocking run use sbatch_submit instead.

    Args:
        partition: Slurm partition (required).
        account: Slurm account to charge (optional).
        qos: Quality of service (optional).
        nodes: Nodes to allocate.
        ntasks: Tasks to run.
        cpus_per_task: CPU cores per task.
        mem: Memory in Slurm format (e.g. '8G'); empty = scheduler default.
        time: Wall-clock limit HH:MM:SS or D-HH:MM:SS.
        gres: Generic resources, e.g. 'gpu:2'.
        constraint: Node feature constraint.
        comment: Free-text label stored with the allocation and read back by
            `sacct -j <id> -o Comment`. Single line, under 512 characters.
        job_name: Slurm job name.
        confirm: Must be True to execute; False returns the command as a preview.
        timeout_seconds: Max time to wait for the allocation response.
        extra_args: Additional salloc flags; every token must start with '-'.
    """
    argv, error = _salloc_argv(partition, account, qos, nodes, ntasks, cpus_per_task,
                               mem, time, gres, constraint, job_name, extra_args,
                               comment)
    if error:
        return error
    preview = shlex.join(argv)
    if not confirm:
        return err("Execution not confirmed.",
                   hint="Review the command, then call again with confirm=True after user approval.",
                   command=preview)

    # An interactive allocation wants a terminal on the host that has the scheduler,
    # and over ``ssh`` there is none: ``salloc`` would take the allocation, find no
    # shell to run in it, and give it straight back — spending the request and
    # reporting success at having done nothing. Refused rather than attempted, because
    # a silent no-op is worse than a plain no.
    if _slurm_host():
        return err(f"The scheduler here is reached through {_slurm_host()}, and an "
                   "interactive allocation needs a terminal on that host.",
                   hint="Submit the work as a batch job instead — sbatch_submit(...) "
                        "returns a handle this session watches to completion.",
                   command=preview)

    result = _run_argv(argv, max(5, min(timeout_seconds, 120)))
    if result["status"] == "ok":
        return ok({
            "command": preview,
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
            "returncode": result.get("returncode", 0),
        })
    return err(
        result.get("stderr") or result.get("error", "salloc submission failed"),
        hint=(
            "Allocation may be pending/denied. Check partitions, account/qos, and requested resources. "
            "Use slurm_partitions() and slurm_queue() for diagnostics."
        ),
        command=preview,
    )


# ── async batch submission (backgroundable) ──────────────────────────────────

_SACCT_CRASH_PREFIXES = (
    "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL",
    "BOOT_FAIL", "DEADLINE", "PREEMPTED",
)


_ACTIVE_STATES = ("RUNNING", "COMPLETING")
_WAITING_STATES = ("PENDING", "CONFIGURING", "REQUEUED")


def _lines(result: dict) -> list[str]:
    """The non-empty lines of a command's output, upper-cased. [] when it failed."""
    if result.get("status") != "ok":
        return []
    return [line.strip().upper()
            for line in (result.get("stdout") or "").splitlines() if line.strip()]


def _normalized_job_state(job_id: str) -> tuple[str, str]:
    """Map a Slurm job's state to the shared vocab: running|pending|done|crashed|unknown.

    Active jobs come from ``squeue``; finished jobs have left the queue, so their
    terminal state is read from ``sacct``. Returns ``(state, raw)``. Missing tools
    or an unknown job yield ``("unknown", "")`` — a terminal signal for the watcher.

    **One id can be many jobs.** An array submits as ``15306621`` and runs as
    ``15306621_1 … _8``, and both commands answer for every element: ``squeue`` prints
    a line each, ``sacct`` likewise. Read as one line that is a job whose first element
    has finished while seven still run — which is reported as finished, waking its
    conversation on a run that has barely started. So the answer is taken over the
    whole set: anything still going keeps the job live, and a terminal verdict is the
    worst outcome among the elements, since an array where one task failed is not an
    array that completed.
    """
    queued = _lines(_run_bash(f"squeue -j {shlex.quote(job_id)} -h -o '%T'",
                              _TIMEOUT_READ))
    raw = ",".join(sorted(set(queued)))
    if any(state in _ACTIVE_STATES for state in queued):
        return "running", raw
    if queued and all(state in _WAITING_STATES for state in queued):
        return "pending", raw

    accounted = _lines(_run_bash(f"sacct -j {shlex.quote(job_id)} -n -X -o State",
                                 _TIMEOUT_READ))
    sraw = ",".join(sorted(set(accounted)))
    # Ahead of the "anything not COMPLETED is crashed" fallback below, which is where
    # most of these land anyway. What this earns is the one case the fallback cannot
    # reach: a terminal crash line present *while* squeue still reports a state that is
    # neither active nor waiting, which the clause below would read as running.
    crashed = [state for state in accounted
               if any(state.startswith(prefix) for prefix in _SACCT_CRASH_PREFIXES)]
    if crashed:
        return "crashed", sraw
    if accounted and all(state.startswith("COMPLETED") for state in accounted):
        return "done", sraw
    if queued:                    # in the queue with a non-standard active state
        return "running", raw
    if accounted:                 # some other terminal state we don't classify
        return "crashed", sraw
    return "unknown", ""


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_job_status(job_id: str) -> dict:
    """Normalized status of a single Slurm job (poll target for background jobs).

    Returns ``state`` in running|pending|done|crashed|unknown (squeue for active
    jobs, sacct for finished ones) plus the raw Slurm state string.

    A terminal answer also settles the submission's directory, so that the readers which
    cannot ask Slurm stop reading the job as live — see :func:`_record_terminal_state`.

    Args:
        job_id: The Slurm job ID to poll, as returned when the job was submitted.
    """
    if not str(job_id).strip():
        return err("job_id is required.")
    job_id = str(job_id).strip()
    state, raw = _normalized_job_state(job_id)
    payload = {"job_id": job_id, "state": state, "raw_state": raw}
    # Where the submission was recorded, so the script and the log stay reachable. A
    # Slurm job outlives its conversation, so the sibling sessions are searched too and the
    # answer says when the job came from another one.
    job_dir, from_session = _find_job_dir(job_id)
    if job_dir:
        payload["job_dir"] = job_dir
        # Read, not re-derived: the submission wrote down where it told the job to
        # write, and that answer has to survive a change of rule between versions —
        # a job submitted by an older build still has its log where that build put it.
        log = _recorded_log(job_dir)
        if log:
            payload["log"] = log
        if from_session:
            payload["submitted_by_another_session"] = from_session
        # And if Slurm has let go, say so in the directory. See _record_terminal_state:
        # this poll is the only thing that will ever know, and the readers that decide
        # whether a workspace still has work running cannot ask the scheduler.
        if state in _TERMINAL_STATES:
            _record_terminal_state(job_dir, state)
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
    # A handle for a job of *this* conversation that is still in flight, so asking
    # where it is at puts a watcher back on it — the watchers do not survive the agent
    # that made them, and a window reload is enough to lose every one of them.
    #
    # Never for another session's job, and this is the whole reason the descriptor is
    # withheld rather than always attached: a wake is routed to the conversation whose
    # agent registered it, so re-arming here would deliver another conversation's
    # result into this one. Reading a sibling's job stays allowed; adopting it does not.
    if state in ("running", "pending") and not from_session:
        payload["background_jobs"] = [_slurm_background_job(job_id)]
    return ok(payload)


# A job, or one task of a job array. Anything wider — a user, a partition, a name — is
# what `scancel` would also take, and is not something to approve from one card.
_JOB_ID_RE = re.compile(r"\d+(_\d+)?")

@mcp.tool(**tool_caps(kind="slurm"))
def slurm_watch(job_id: str, note: str = "") -> dict:
    """Watch a Slurm job that was submitted outside MIMIR, so its end reaches you.

    The counterpart of sbatch_submit for every job this server did not launch: one
    started by a site launcher, by a script of the user's, or from a shell. Those are
    invisible here — a job is tracked by the descriptor its submission writes, and a
    submission MIMIR never made writes none — so nothing watches them, nothing is owed
    a wake, and this process stops on its idle TTL with the user's run still going.
    This records the descriptor after the fact, which is all the watcher ever needed.

    Call it for every id the submission reported, the server and the client array alike:
    one id is one thing to be told about. An array id covers all of its tasks.

    Args:
        job_id: The Slurm job id to adopt — ``15306620``, or one array task
            ``15306621_3``. An array's base id watches the whole array.
        note: What this job is, for whoever reads the directory later. Optional.
    """
    job_id = str(job_id or "").strip()
    if not job_id:
        return err("job_id is required.",
                   hint="Use the id the submission printed, or slurm_queue() to list "
                        "what is running.")
    if not _JOB_ID_RE.fullmatch(job_id):
        return err(f"{job_id!r} is not a Slurm job id.",
                   hint="One job, or one array task: 15306620 or 15306621_3.")

    # The disk first, and the scheduler only on the path that writes something. An id
    # already recorded is the common answer when a model adopts an array task by task,
    # and polling Slurm to tell it "nothing to adopt" buys nothing — with a remote
    # controller it is two ssh round trips for an answer the directory already has.
    existing, from_session = _find_job_dir(job_id)
    if existing:
        return ok({
            "job_id": job_id,
            "job_dir": existing,
            "already_watched_by": from_session or "this conversation",
            "note": f"Job {job_id} is already recorded; nothing to adopt. "
                    "slurm_job_status(job_id) for where it is at.",
        })

    # Asked before anything is written, because a descriptor for a job the scheduler
    # has never heard of is worse than no descriptor: it holds this process open until
    # the staleness horizon for a run that does not exist.
    state, raw = _normalized_job_state(job_id)
    if state == "unknown":
        return err(f"Slurm knows nothing about job {job_id}.",
                   hint="Check the id with slurm_queue(). A job that has already left "
                        "both squeue and sacct cannot be watched.")

    try:
        job_dir = _record_job_dir(job_id, **{_ADOPTED_FILE: f"{note}\n" if note else "\n"})
    except OSError as exc:
        return err(f"Could not record the job: {exc}")

    payload = {
        "job_id": job_id,
        "state": state,
        "raw_state": raw,
        "job_dir": job_dir,
        "note": note or None,
    }
    if state in _TERMINAL_STATES:
        # Already over. Settled rather than watched: a watcher put on a finished job
        # reports it at its first poll, which is a wake for news the caller is holding.
        _record_terminal_state(job_dir, state)
        payload["headline"] = (f"Job {job_id} has already finished ({state}); "
                               "recorded, with nothing left to watch.")
        return ok(payload)
    payload["headline"] = (f"Watching Slurm job {job_id} ({state}). End your turn — "
                           "you are resumed when it finishes, with this window closed "
                           "or not.")
    payload["background_job"] = _slurm_background_job(job_id)
    # What is still unaccounted for, named here because this is the moment the model is
    # holding the ids: a launcher prints a server job and a client array, and adopting
    # the first is the usual place to stop.
    _add_unwatched(payload)
    return ok(payload)


def _current_user() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


@mcp.tool(**tool_caps(kind="slurm",
    caps=[PLAN_BLOCKED], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="Cancels a Slurm job; whatever it had not written yet is lost.",
    label="Slurm cancel {job_id}",
))
def slurm_cancel(job_id: str, confirm: bool = False) -> dict:
    """Cancel one of your own Slurm jobs, pending or running (sensitive).

    For a job that is wrong or no longer needed: a bad submission, a run superseded by
    a fix, a hung job burning allocation hours. A running job stops where it is, so its
    output is whatever it had written by then. One job per call — or one task of an
    array, as '1234_5'; a job that is not yours, or no longer in the queue, is refused
    with its state.

    A job this session watches ends like any other: you are resumed with its end, so
    there is nothing to poll after cancelling it.

    Args:
        job_id: The Slurm job ID, as sbatch_submit or slurm_queue gave it.
        confirm: Must be True to cancel.
    """
    job_id = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(job_id):
        return err("job_id must be one Slurm job ID, e.g. '1234' or '1234_5'.",
                   hint="List your jobs with slurm_queue().")
    if not confirm:
        return err("Cancellation not confirmed.",
                   hint="Set confirm=True only after user approval.")

    q = _run_argv(["squeue", "-h", "-j", job_id, "-o", "%u|%T"], _TIMEOUT_READ)
    line = (q.get("stdout") or "").strip().splitlines()
    if q.get("status") != "ok" or not line:
        state, raw = _normalized_job_state(job_id)
        return err(f"Job {job_id} is not in the queue, so there is nothing to cancel.",
                   hint=f"Its last known state: {raw or state}.")
    owner, _, queued_state = line[0].partition("|")
    user = _current_user()
    if user and owner.strip() != user:
        return err(f"Job {job_id} belongs to '{owner.strip()}', not to you ('{user}').",
                   hint="Only your own jobs can be cancelled from here.")

    res = _run_argv(["scancel", job_id], _TIMEOUT_SUBMIT)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "scancel failed"))
    return ok({
        "job_id": job_id,
        "was": queued_state.strip(),
        "note": f"Slurm job {job_id} cancelled (it was {queued_state.strip().lower()}).",
    })


def _sbatch_header(job_name: str, partition: str, cpus_per_task: int, gpus: int,
                   mem: str, wall_time: str, account: str, work_dir: str,
                   comment: str = "") -> list[str]:
    # The job runs in *work_dir* and writes its output there, and that pairing is the
    # whole of why the output is reachable: if the job can start at all, that directory
    # is mounted on the node Slurm started it on — otherwise there would have been
    # nothing to chdir into. Pointed anywhere else, the log depends on a second
    # filesystem being shared, which on a site whose home is not mounted on the compute
    # nodes it is not: the job runs, succeeds, and its output goes nowhere.
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --chdir={work_dir}",
        f"#SBATCH --output={os.path.join(work_dir, _LOG_NAME)}",
        f"#SBATCH --error={os.path.join(work_dir, _LOG_NAME)}",
    ]
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    if comment:
        lines.append(f"#SBATCH --comment={shlex.quote(comment)}")
    return lines


@mcp.tool(**tool_caps(kind="slurm",
    caps=[PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="Submits a Slurm batch job that consumes cluster allocation hours.",
    label="Slurm sbatch submit",
))
def sbatch_submit(
    command: str,
    partition: str,
    cpus_per_task: int = 1,
    gpus: int = 0,
    mem: str = "",
    wall_time: str = "01:00:00",
    account: str = "",
    comment: str = "",
    job_name: str = "mimir-batch",
    work_dir: str = "",
    confirm: bool = False,
) -> dict:
    """Submit *command* as a non-blocking Slurm batch job (sensitive, backgroundable).

    Unlike salloc_submit (synchronous, interactive), this returns immediately with a
    ``job_id`` and a ``background_job`` descriptor so the run can be tracked off the
    critical path: when the result says the run is being watched, end your turn and
    you are auto-resumed when the job finishes; otherwise poll slurm_job_status(job_id).

    Args:
        command: The shell command line to run inside the batch job.
        partition: Slurm partition (required).
        cpus_per_task: CPU cores to allocate (default 1).
        gpus: GPUs per node (0 = CPU-only).
        mem: Memory in Slurm format (e.g. '8G'); empty = scheduler default.
        wall_time: Wall-clock limit HH:MM:SS or D-HH:MM:SS (default '01:00:00').
        account: Slurm account to charge (optional).
        comment: Free-text label stored with the job and read back by
            `sacct -j <id> -o Comment` or `scontrol show job <id>` — what this
            job was for, for whoever reads the queue later. Single line, under
            512 characters.
        job_name: Slurm job name (default 'mimir-batch').
        work_dir: Where the job runs and writes its output (default: the workspace).
            Pass the directory the work belongs to — a scratch filesystem, say — and
            the log lands beside it: the job's own directory is the one place its
            output is certain to be writable, since a job that could not reach it
            would not have started.
        confirm: Must be True to submit.
    """
    if not command.strip():
        return err("command is required.")
    if not partition.strip():
        return err("partition is required.")
    if wall_time and not _validate_time(wall_time):
        return err("Invalid Slurm time format.", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not _validate_mem(mem):
        return err("Invalid mem format.", hint="Use values like 8G, 32000M, 1T")
    if cpus_per_task < 1 or gpus < 0:
        return err("cpus_per_task must be >= 1 and gpus >= 0.")
    comment_err = _validate_comment(comment)
    if comment_err:
        return err(comment_err)
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    work_dir = os.path.realpath(work_dir.strip() or os.getcwd())
    if not os.path.isdir(work_dir):
        return err(f"{work_dir} is not a directory this host can see.",
                   hint="The job starts in it, so a path that is not there is a "
                        "submission that fails at launch.")
    # Said now rather than discovered after the queue wait. And said rather than
    # worked around: the tempting fallback is somewhere always writable, which on a
    # cluster means ``/tmp`` — and ``/tmp`` is the compute node's own disk, so the
    # output would land where this host can never read it. A shared directory that is
    # not writable fails loudly; a node-local one fails silently, with an empty run
    # that looks like a job which printed nothing.
    if not os.access(work_dir, os.W_OK):
        return err(f"{work_dir} is not writable, so the job could not write its "
                   "output there.",
                   hint="Pass work_dir as a directory you can write and the compute "
                        "nodes share — the run's own tree, or a scratch filesystem.")

    job_dir = os.path.join(_hpc_jobs_dir(), _new_job_dir_name())
    os.makedirs(job_dir, exist_ok=True)
    script_path = os.path.join(job_dir, "batch_script.sh")
    header = _sbatch_header(job_name, partition, cpus_per_task, gpus, mem,
                            wall_time, account, work_dir, comment)
    script = "\n".join(header + ["", command, ""]) + "\n"
    # Kept for a reader — a model or a person asking what was actually submitted —
    # and not for the scheduler, which is handed the text itself below.
    try:
        with open(script_path, "w") as fh:
            fh.write(script)
    except OSError as exc:
        return err(f"Could not write batch script: {exc}")

    # On stdin, not by path. ``sbatch`` reads its script from standard input when given
    # no file, which means nothing of ours has to exist on the host running it — the
    # one thing a remote controller could not promise, and the reason this used to
    # check a directory over ssh before submitting. The command line is now one word,
    # so there is no path to quote either.
    res = _run_bash("sbatch", _TIMEOUT_SUBMIT, stdin=script)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account/qos, and requested resources. "
                        + _no_scheduler_hint())
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return err("Could not parse a job id from sbatch output.",
                   hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    job_id = match.group(1)
    log_file = os.path.join(work_dir, _LOG_NAME.replace("%j", job_id))
    try:
        with open(os.path.join(job_dir, _ID_FILE), "w") as fh:
            fh.write(job_id)
        with open(os.path.join(job_dir, _LOG_FILE), "w") as fh:
            fh.write(log_file)
    except OSError:
        pass

    payload = {
        "job_id":       job_id,
        "job_dir":      job_dir,
        "batch_script": script_path,
        "work_dir":     work_dir,
        "log":          log_file,
        "partition":    partition,
        "comment":      comment or None,
        "note":         f"Slurm job {job_id} submitted to '{partition}'.",
        "background_job": _slurm_background_job(job_id),
    }
    # The submission is the other moment the model holds job ids: what else of the
    # user's is running unaccounted for can be acted on here, in the same turn.
    _add_unwatched(payload)
    return ok(payload)


if __name__ == "__main__":
    mcp.run()
