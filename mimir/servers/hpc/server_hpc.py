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
  shell, for ``$USER`` expansion; their filters are quoted. For a scheduler behind
  ``ssh`` one shell is unavoidable, since it takes a command line and not an argv: each
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
from state_paths import state_dir
from slurm_site import (
    HOST_FILE as _HOST_FILE,
    ID_FILE as _ID_FILE,
    LOCAL as _LOCAL_HOST,
    LOCAL_NAME as _LOCAL_NAME,
    covered_by as _covered_by,
    hosts as slurm_hosts,
    job_host as _job_host,
    name_of as _host_name,
    jobs_dir as _hpc_jobs_dir,
    new_job_dir as _new_job_dir,
    record_job_dir as _record_job_dir,
    recorded_jobs as _recorded_jobs,
    resolve_host as _resolve_host,
    argv_for as slurm_argv,
    exec_argv as slurm_exec_argv,
    queued_by_host as slurm_queued_by_host,
    record_host as _record_host,
    forget_host as _forget_host,
    unwatched as slurm_unwatched,
    unwatched_by_host as slurm_unwatched_by_host,
    unwatched_hint as slurm_unwatched_hint,
)
from slurm_digest import (
    cached_cluster_digest as _cluster_digest,
    forget_cluster_digest as _forget_cluster_digest,
)
from slurm_nodes import (
    SCONTROL_NODES_ARGV as _SCONTROL_NODES_ARGV,
    SINFO_NODES_FORMAT as _SINFO_NODES_FORMAT,
    aggregate_node_types as _aggregate_node_types,
    parse_pipe_table as _parse_pipe_table,
    parse_scontrol_nodes as _parse_scontrol_nodes,
    parse_sinfo_nodes as _parse_sinfo_nodes,
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
def _no_scheduler_hint(host: str = _LOCAL_HOST) -> str:
    """What to do when a Slurm command did not answer. Self-documenting on purpose.

    Two different failures, and the cure is only ever offered for the one it fits. A
    scheduler that was *named* and did not answer is a host to check — the inventory is
    not the problem. Nothing reachable at all is the case this learning exists for, and
    there the answer has to be a thing the user is about to do anyway: reach a
    scheduler over ``ssh`` once, which adds it.
    """
    known = slurm_hosts()
    if host:
        return (f"That went to {host}; check the host answers — "
                "slurm_schedulers() lists what this workspace reaches.")
    if known:
        return ("No Slurm command answered on this host. Reachable instead: "
                + ", ".join(_host_name(h) for h in known)
                + " — pass one as scheduler=.")
    return ("No Slurm command answered on this host, and no other scheduler has been "
            "reached yet. If one lives elsewhere, read its queue through the shell "
            "once — ssh <host> \"squeue -u $USER\" — and it joins this workspace's "
            "schedulers, or name it with slurm_schedulers(add=\"<host>\").")


def _add_unwatched(payload: dict) -> None:
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
    by_host = slurm_unwatched_by_host(slurm_queued_by_host(_TIMEOUT_READ))
    if not by_host:
        return
    # Flat, and in inventory order: the client's nudge reads a list of ids, and the
    # hint is what says which scheduler to name when there is more than one.
    payload["unwatched"] = [job_id for ids in by_host.values() for job_id in ids]
    payload["unwatched_hint"] = slurm_unwatched_hint(by_host)


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


# ── Which scheduler a call goes to ────────────────────────────────────────────
#
# An inventory, not a setting: ``_shared/slurm_site`` holds the schedulers this
# workspace reaches — this machine's own when it has one, plus every host a successful
# ``ssh host "squeue …"`` has proved — and a caller names which. See that module for why
# a variable nobody sets before the first lost night is the wrong mechanism, and for how
# a command is run over there.
#
# Two regimes here, and they are different questions. A tool that *acts* resolves a
# host from what it was given (``_resolve_host``), and refuses rather than defaulting
# when the inventory leaves a real choice open. A tool that acts on an *existing job*
# does not choose at all: it finds where that job already is (``_locate_job``), because
# a job is not something a scheduler owns — it is something one of them is running.
#
# One seam either way: every Slurm command this server runs goes through ``slurm_argv``
# or ``slurm_exec_argv`` with that host. Nothing else in this file knows about ssh.


def _record_job_host(job_dir: str, host: str) -> None:
    """Write down which scheduler a job is on, so the next poll asks only it.

    A cache and not a binding: it is written when a job is submitted and whenever a
    sweep has just found one, and :func:`_locate_job` sweeps again whenever it is
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


def _chosen_host(scheduler: str) -> tuple[str, dict | None]:
    """The scheduler a call goes to, or the refusal naming the choice left to make.

    The refusal is the useful half: with two schedulers reachable, "which one" is a
    decision about the work — where the inputs are, which has the GPUs — and this says
    so rather than picking one. See ``slurm_site.resolve_host``.
    """
    host, refusal = _resolve_host(scheduler)
    return host, (err(refusal) if refusal else None)


def _locate_job(job_id: str, requested: str = "",
                job_dir: str = "") -> tuple[str, str, str]:
    """Find *job_id* among the schedulers this workspace reaches: ``(host, state, raw)``.

    The hint first — the descriptor's own record, which makes the common case one call
    — then every other reachable scheduler until one knows the id. That fallback is
    what keeps a job trackable without being owned: an adoption of somebody's job, a
    descriptor from before the host was written down, a watcher re-armed from disk
    after a window reload (which polls by id alone, having no host to pass), all land
    here and are answered.

    *requested* skips the search when a caller already knows — a sweep refused for one
    job is not worth paying twice. *job_dir* is for the caller that has already found
    the descriptor, so the sessions tree is walked once per call rather than twice.
    ``("", "unknown", "")`` when nothing has heard of it.
    """
    if requested:
        state, raw = _normalized_job_state(job_id, host=requested)
        return requested, state, raw
    if not job_dir:
        job_dir, _ = _find_job_dir(job_id)
    hint = _job_host(job_dir) if job_dir else None
    order = [hint] if hint is not None else []
    order += [host for host in slurm_hosts() if host != hint]
    for host in order:
        state, raw = _normalized_job_state(job_id, host=host)
        if state != "unknown":
            # Found, so the next poll of this job is one call rather than a sweep.
            if job_dir and hint != host:
                _record_job_host(job_dir, host)
            return host, state, raw
    return "", "unknown", ""


def _run_bash(script: str, timeout: int, stdin: str | None = None, *,
              host: str = _LOCAL_HOST) -> dict:
    """Run *script* on *host*'s scheduler, optionally feeding it *stdin*.

    Without *stdin* the command gets none at all: the server's own is its MCP protocol
    pipe, and a command that reads it (``cat``, an ``ssh`` without ``-n``) eats the
    client's traffic. With it, the text is handed over on a pipe we opened — which is
    how a batch script reaches ``sbatch`` without any file having to exist on the host
    that runs it.
    """
    try:
        res = subprocess.run(
            slurm_argv(script, host=host, piped=stdin is not None),
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


def _hardware_of(host: str) -> dict:
    """What *host*'s machines are, for the listing a scheduler is chosen from.

    A summary and the stable node types, never occupancy. A host that said nothing
    carries neither and says so, because a cluster described as having no hardware
    reads as a cluster not worth submitting to.
    """
    digest = _cluster_digest(host)
    if not digest["source"]:
        return {"hardware": "",
                "hardware_note": "Did not answer; its hardware is unknown, not absent."}
    types = digest["node_types"]
    arches = sorted({t["arch"] for t in types if t["arch"]})
    # The model, not the whole gres: a site with ten GPU shapes makes a line nobody
    # scans, and the counts are on each node type below for whoever needs them.
    gpus = sorted({t["gres"].split(":")[1] for t in types
                   if t["gres"].startswith("gpu:") and t["gres"].count(":") >= 1})
    parts = [f"{len(types)} node type{'s' if len(types) != 1 else ''}"]
    if arches:
        parts.append("/".join(arches))
    if gpus:
        parts.append(", ".join(gpus))
    parts.append(f"{len(digest['partitions'])} partitions")
    summary = {"hardware": " · ".join(parts), "node_types": types}
    if digest["degraded"]:
        summary["degraded"] = digest["degraded"]
    return summary


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_schedulers(add: str = "", forget: str = "") -> dict:
    """The schedulers this workspace can submit to — read this before choosing one.

    A machine reaches more than one: its own Slurm when it has it, plus any host a
    successful ``ssh host "squeue …"`` has proved. Every cluster tool here takes a
    ``scheduler=`` naming one of these, and a job submitted to any of them is watched
    the same way — the wake is owed wherever the work is.

    Which one a job goes to is a decision about the work, not a setting: the partitions
    and GPUs it needs, and above all the filesystem holding its inputs and its
    work_dir, since a job can only run where its files are. Ask the user when the task
    does not settle it.

    Each entry therefore carries what its machines *are* — architectures, GPUs, the
    kinds of node, how many partitions — because that is half the decision and a model
    that must spend a slurm_nodes call per candidate before it can choose will choose
    without looking. What is *free right now* is deliberately not here: that changes
    between two reads and belongs to slurm_nodes(scheduler=...), which this is the
    companion of rather than a substitute for.

    Built once per host for the life of this server, so the first call pays a round
    trip per cluster and the rest are free. A host that does not answer is listed
    without its hardware rather than making the whole listing an error.

    Args:
        add: A host to add, verified before it is recorded — ``sbatch`` must answer
            there. Not needed for a host you have already reached over ssh.
        forget: A host to drop: one that was a typo, or a cluster you no longer have.
            The local scheduler cannot be dropped, being a fact rather than a record.
    """
    added: dict | None = None
    if add.strip():
        host = add.strip()
        probe = _run_bash("command -v sbatch", _TIMEOUT_READ, host=host)
        if probe.get("status") != "ok":
            return err(f"No Slurm on {host}: {probe.get('stderr') or probe.get('error') or 'sbatch did not answer'}",
                       hint="Nothing was recorded. Check the host answers "
                            f"— ssh {host} \"squeue -u $USER\" — and that ssh needs no "
                            "password: a scheduler that can ask for one hangs every "
                            "poll. Keys and ports belong in ~/.ssh/config.")
        _record_host(host)
        # The explicit re-probe: a host named again is one the user may have changed.
        _forget_cluster_digest(host)
        added = {"scheduler": host, "note": f"sbatch answers on {host}; added."}

    dropped: dict | None = None
    if forget.strip():
        host = forget.strip()
        if host == _LOCAL_NAME:
            dropped = {"scheduler": host,
                       "note": "The local scheduler is not a record to drop: it is "
                               "this host answering sbatch, and it would be back on "
                               "the next call."}
        else:
            gone = _forget_host(host)
            _forget_cluster_digest(host)
            dropped = {"scheduler": host,
                       "note": f"{host} dropped." if gone
                               else f"{host} was not in this workspace's schedulers."}

    known = slurm_hosts()
    listed = [{
        "scheduler": _host_name(host),
        "local": not host,
        "reached": "Slurm answers on this host" if not host
                   else f"over ssh to {host}",
        **_hardware_of(host),
    } for host in known]
    payload: dict = {"schedulers": listed, "count": len(listed)}
    if added:
        payload["added"] = added
    if dropped:
        payload["dropped"] = dropped
    if not known:
        payload["note"] = (
            "None. No Slurm command answers here, and no other host has been reached. "
            "Read a queue through the shell once — ssh <host> \"squeue -u $USER\" — "
            "and it is learned, or name it with slurm_schedulers(add=\"<host>\").")
    elif len(known) == 1:
        payload["note"] = (f"One, so scheduler= can be omitted: "
                           f"{_host_name(known[0])}.")
    else:
        payload["note"] = ("More than one, so every submission names its scheduler. "
                           "The choice follows the job's resources — the hardware on "
                           "each entry above — and the filesystem its inputs are on.")
    return ok(payload)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_partitions(scheduler: str = "") -> dict:
    """List Slurm partitions with key scheduling attributes.

    Args:
        scheduler: Which scheduler to ask — 'local' for this machine's own, or
            a hostname. Omitted is fine when this workspace reaches only one;
            slurm_schedulers() lists them.
    """
    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    cmd = "sinfo -h -o '%P|%a|%l|%D|%t|%c|%m'"
    result = _run_bash(cmd, _TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "sinfo failed"),
                    hint=_no_scheduler_hint(host))
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["partition", "availability", "time_limit", "nodes", "state", "cpus_per_node", "mem_mb_per_node"],
    )
    return ok({"partitions": rows, "count": len(rows),
               "scheduler": _host_name(host)})


def _sinfo_nodes(host: str = _LOCAL_HOST) -> list[dict]:
    """Degraded node list for a cluster whose `scontrol show node` is restricted.

    sinfo is readable everywhere but does not carry the architecture, so callers are
    told the field is unknown rather than being handed a wrong default.
    """
    result = _run_bash(f"sinfo -N -h -o '{_SINFO_NODES_FORMAT}'",
                       _TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return []
    return _parse_sinfo_nodes(result.get("stdout", ""))


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_nodes(partition: str = "", states: str = "", node: str = "",
                detail: bool = False, scheduler: str = "") -> dict:
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
        scheduler: Which scheduler to ask — 'local' for this machine's own, or
            a hostname. Omitted is fine when this workspace reaches only one;
            slurm_schedulers() lists them.
    """
    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    degraded = ""
    result = _run_argv(_SCONTROL_NODES_ARGV, _TIMEOUT_READ, host=host)
    if result["status"] == "ok":
        nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    else:
        nodes = _sinfo_nodes(host)
        degraded = "scontrol unavailable: architecture and live CPU occupancy are unknown."
    if not nodes:
        return err(result.get("stderr") or result.get("error", "node query failed"),
                   hint=_no_scheduler_hint(host))

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
    payload["scheduler"] = _host_name(host)
    return ok(payload)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_queue(user_only: bool = True, states: str = "", scheduler: str = "") -> dict:
    """List jobs from Slurm queue.

    One scheduler's queue, because that is what a queue is. What this workspace runs
    elsewhere is in ``unwatched`` when nothing is tracking it, and slurm_schedulers()
    says where else to look.

    Args:
        user_only: If true, show only jobs for the current user.
        states: Optional state filter (e.g. 'R,PD').
        scheduler: Which scheduler to ask — 'local' for this machine's own, or
            a hostname. Omitted is fine when this workspace reaches only one;
            slurm_schedulers() lists them.
    """
    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    user_flag = " -u $USER" if user_only else ""
    state_flag = f" -t {shlex.quote(states)}" if states else ""
    cmd = f"squeue -h{user_flag}{state_flag} -o '%i|%u|%T|%M|%l|%D|%R|%j'"
    result = _run_bash(cmd, _TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "squeue failed"),
                   hint=_no_scheduler_hint(host))
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
    #
    # Asked of *this* scheduler, which is why the descriptors are keyed by host: the
    # same number is a watched production run here and somebody else's sweep there, and
    # an id alone would report the second as covered by the first's descriptor.
    recorded = _recorded_jobs()
    ids = [str(row.get("job_id") or "") for row in rows]
    for row, job_id in zip(rows, ids):
        row["watched"] = _covered_by(job_id, recorded, host)
    payload = {"jobs": rows, "count": len(rows), "scheduler": _host_name(host)}
    # Per row above, by array here: eight task rows name one descriptor to write, and
    # listing them one by one would ask for eight calls that all land on "already
    # recorded".
    missing = slurm_unwatched(ids, host=host)
    if missing:
        payload["unwatched"] = missing
        payload["unwatched_hint"] = slurm_unwatched_hint({host: missing})
    return ok(payload)


def _run_argv(argv: list[str], timeout: int, *, host: str = _LOCAL_HOST) -> dict:
    """Run a command as argv on *host* — no shell, so no argument can inject a second one.

    With a remote controller one shell is unavoidable: ``ssh`` has no argv to hand over,
    only a command line for the far side to parse. So each element is quoted on the way
    out and the property survives in the form that matters — what the remote shell
    splits is exactly the argv built here, and an argument cannot become a command.
    """
    try:
        res = subprocess.run(
            slurm_exec_argv(argv, host=host), stdin=subprocess.DEVNULL,
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
    scheduler: str = "",
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
        scheduler: Which scheduler to allocate on. 'local' only: an interactive
            allocation needs a terminal on the host that has the scheduler, which
            ssh does not give — for a remote one, use sbatch_submit.
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
    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    if host:
        return err(f"{host} is reached over ssh, and an interactive allocation needs a "
                   "terminal on the host that has the scheduler.",
                   hint="Submit the work as a batch job instead — sbatch_submit(..., "
                        f"scheduler='{host}') returns a handle this session watches to "
                        "completion.",
                   command=preview)

    result = _run_argv(argv, max(5, min(timeout_seconds, 120)), host=host)
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


def _normalized_job_state(job_id: str, *, host: str = _LOCAL_HOST) -> tuple[str, str]:
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
                              _TIMEOUT_READ, host=host))
    raw = ",".join(sorted(set(queued)))
    if any(state in _ACTIVE_STATES for state in queued):
        return "running", raw
    if queued and all(state in _WAITING_STATES for state in queued):
        return "pending", raw

    accounted = _lines(_run_bash(f"sacct -j {shlex.quote(job_id)} -n -X -o State",
                                 _TIMEOUT_READ, host=host))
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
def slurm_job_status(job_id: str, scheduler: str = "") -> dict:
    """Normalized status of a single Slurm job (poll target for background jobs).

    Returns ``state`` in running|pending|done|crashed|unknown (squeue for active
    jobs, sacct for finished ones) plus the raw Slurm state string.

    A terminal answer also settles the submission's directory, so that the readers which
    cannot ask Slurm stop reading the job as live — see :func:`_record_terminal_state`.

    The id is enough. Which scheduler is running the job is found — the descriptor's
    own record first, then every other one this workspace reaches — so a poll works
    for a job on any of them, including a watcher re-armed from disk after a reload,
    which has nothing but the id to poll with.

    Args:
        job_id: The Slurm job ID to poll, as returned when the job was submitted.
        scheduler: Only to skip the search, when the caller already knows which one is
            running it. Omitted is the normal case.
    """
    if not str(job_id).strip():
        return err("job_id is required.")
    job_id = str(job_id).strip()
    # Where the submission was recorded, so the script and the log stay reachable. A
    # Slurm job outlives its conversation, so the sibling sessions are searched too and the
    # answer says when the job came from another one.
    job_dir, from_session = _find_job_dir(job_id)
    host, state, raw = _locate_job(job_id, scheduler, job_dir)
    payload = {"job_id": job_id, "state": state, "raw_state": raw}
    if host or state != "unknown":
        payload["scheduler"] = _host_name(host)
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
def slurm_watch(job_id: str, note: str = "", scheduler: str = "") -> dict:
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
        scheduler: Which scheduler is running it, when known. Omitted, every one this
            workspace reaches is asked — so an id alone is enough, and the one that
            answers is written down for the polls that follow.
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
    # and polling Slurm to tell it "nothing to adopt" buys nothing — over ssh it is two
    # round trips, per scheduler swept, for an answer the directory already has.
    existing, from_session = _find_job_dir(job_id)
    if existing:
        return ok({
            "job_id": job_id,
            "job_dir": existing,
            "already_watched_by": from_session or "this conversation",
            "note": f"Job {job_id} is already recorded; nothing to adopt. "
                    "slurm_job_status(job_id) for where it is at.",
        })

    # Asked before anything is written, because a descriptor for a job no scheduler has
    # heard of is worse than no descriptor: it holds this process open until the
    # staleness horizon for a run that does not exist. And asked of each of them, which
    # is what makes an id enough — the caller holding a number from a launcher's output
    # has no reason to know which cluster printed it.
    host, state, raw = _locate_job(job_id, scheduler)
    if state == "unknown":
        known = ", ".join(_host_name(h) for h in slurm_hosts()) or "none reachable"
        return err(f"No scheduler this workspace reaches knows job {job_id} ({known}).",
                   hint="Check the id with slurm_queue(). A job that has already left "
                        "both squeue and sacct cannot be watched, and one on a "
                        "scheduler not listed above needs it added — "
                        "slurm_schedulers(add=\"<host>\").")

    try:
        # The host beside the id, so the watcher's polls go straight to it. A cache,
        # not a binding: a poll that finds it wrong sweeps again and rewrites it.
        job_dir = _record_job_dir(job_id, **{
            _ADOPTED_FILE: f"{note}\n" if note else "\n",
            _HOST_FILE: host,
        })
    except OSError as exc:
        return err(f"Could not record the job: {exc}")

    payload = {
        "job_id": job_id,
        "state": state,
        "raw_state": raw,
        "job_dir": job_dir,
        "scheduler": _host_name(host),
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
def slurm_cancel(job_id: str, confirm: bool = False, scheduler: str = "") -> dict:
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
        scheduler: Only to skip the search for the one running it. Omitted is the
            normal case — a cancellation goes where the job is.
    """
    job_id = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(job_id):
        return err("job_id must be one Slurm job ID, e.g. '1234' or '1234_5'.",
                   hint="List your jobs with slurm_queue().")
    if not confirm:
        return err("Cancellation not confirmed.",
                   hint="Set confirm=True only after user approval.")

    # Where it is, before anything is sent: a scancel aimed at the wrong scheduler
    # either hits nothing or — the id existing on both — somebody else's job.
    host, located, located_raw = _locate_job(job_id, scheduler)
    if located == "unknown":
        known = ", ".join(_host_name(h) for h in slurm_hosts()) or "none reachable"
        return err(f"No scheduler this workspace reaches knows job {job_id} ({known}), "
                   "so there is nothing to cancel.",
                   hint="List what is running with slurm_queue(scheduler=...).")

    q = _run_argv(["squeue", "-h", "-j", job_id, "-o", "%u|%T"], _TIMEOUT_READ,
                  host=host)
    line = (q.get("stdout") or "").strip().splitlines()
    if q.get("status") != "ok" or not line:
        return err(f"Job {job_id} is not in {_host_name(host)}'s queue, so there is "
                   "nothing to cancel.",
                   hint=f"Its last known state: {located_raw or located}.")
    owner, _, queued_state = line[0].partition("|")
    user = _current_user()
    if user and owner.strip() != user:
        return err(f"Job {job_id} belongs to '{owner.strip()}', not to you ('{user}').",
                   hint="Only your own jobs can be cancelled from here.")

    res = _run_argv(["scancel", job_id], _TIMEOUT_SUBMIT, host=host)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "scancel failed"))
    return ok({
        "job_id": job_id,
        "was": queued_state.strip(),
        "scheduler": _host_name(host),
        "note": f"Slurm job {job_id} cancelled on {_host_name(host)} "
                f"(it was {queued_state.strip().lower()}).",
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
    scheduler: str = "",
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
        scheduler: Which scheduler to submit to — 'local' for this machine's own, or a
            hostname. With more than one reachable this is required, and the choice is
            about the work: the partitions and GPUs it needs, and the filesystem
            holding its inputs and work_dir — a job is submitted to the cluster that
            can see its files. slurm_schedulers() lists them; ask the user when the
            task does not settle it.
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

    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal

    work_dir = os.path.realpath(work_dir.strip() or os.getcwd())
    # Checked here, which is where it can be: this host sees the directory or it does
    # not. A submission to a remote scheduler is the case this cannot settle — the far
    # side has its own filesystems — so the check is kept as the local evidence it is,
    # and a path this host cannot see is still refused, since a work_dir nobody here
    # can read is not one this session can collect output from either.
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

    job_dir = _new_job_dir()
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
    # no file, which means nothing of ours has to exist on the host running it — the one
    # thing a scheduler behind ``ssh`` cannot promise. The command line is one word, so
    # there is no path to quote either.
    res = _run_bash("sbatch", _TIMEOUT_SUBMIT, stdin=script, host=host)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account/qos, and requested resources. "
                        + _no_scheduler_hint(host))
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
    # Beside the id, so the watcher's first poll goes straight to the right scheduler
    # instead of sweeping. And into the session's inventory: a submission that
    # succeeded is proof this host reaches that scheduler, which is the same evidence
    # a queue read gives, from the call that mattered more.
    _record_job_host(job_dir, host)
    if host:
        _record_host(host)

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
        "background_job": _slurm_background_job(job_id),
    }
    # The submission is the other moment the model holds job ids: what else of the
    # user's is running unaccounted for can be acted on here, in the same turn.
    _add_unwatched(payload)
    return ok(payload)


if __name__ == "__main__":
    mcp.run()
