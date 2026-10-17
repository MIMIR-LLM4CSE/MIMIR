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

import asyncio
import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import shlex
import socket
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import (tool_caps, PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE,
                          IRREVERSIBLE, PANEL_REPORT)
from responses import err, ok
from state_paths import sessions_root, state_dir
from slurm_site import (
    ADOPTED_FILE as _ADOPTED_FILE,
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
    owned_job_hosts as _owned_job_hosts,
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
    HARDWARE_FIELDS as _HARDWARE_FIELDS,
    SCONTROL_NODES_ARGV as _SCONTROL_NODES_ARGV,
    SINFO_NODES_FORMAT as _SINFO_NODES_FORMAT,
    aggregate_node_types as _aggregate_node_types,
    as_int as _as_int,
    hardware_key as _hardware_key,
    parse_pipe_table as _parse_pipe_table,
    parse_scontrol_nodes as _parse_scontrol_nodes,
    parse_sinfo_nodes as _parse_sinfo_nodes,
)
from slurm_script import node_python_lines, sbatch_header, validate_target
import build_progress
import cpu_facts

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
# certain to be able to write. See _shared/slurm_script.sbatch_header.
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


def _recorded_terminal_state(job_dir: str) -> str:
    """The terminal state a poll wrote in *job_dir*, or "" while it has none.

    The counterpart of :func:`_record_terminal_state`, and the same question the client
    scan asks of the same file (``job_scan.is_settled``). What it buys here is an answer
    about a recorded job without asking Slurm: over ssh a sweep is two round trips per
    scheduler, and "has this one already ended?" is on disk.
    """
    try:
        with open(os.path.join(job_dir, _STATE_FILE)) as fh:
            word = fh.read().strip()
    except OSError:
        return ""
    return word if word in _TERMINAL_STATES else ""


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
                job_dir: str = "") -> tuple[str, str, str, list[str]]:
    """Find *job_id*: ``(host, state, raw, nodes)``, among the schedulers reached.

    The hint first — the descriptor's own record, which makes the common case one call
    — then every other reachable scheduler until one knows the id. That fallback is
    what keeps a job trackable without being owned: an adoption of somebody's job, a
    descriptor from before the host was written down, a watcher re-armed from disk
    after a window reload (which polls by id alone, having no host to pass), all land
    here and are answered.

    *requested* skips the search when a caller already knows — a sweep refused for one
    job is not worth paying twice. *job_dir* is for the caller that has already found
    the descriptor, so the sessions tree is walked once per call rather than twice.
    ``("", "unknown", "", [])`` when nothing has heard of it.
    """
    if requested:
        state, raw, nodes = _normalized_job_state(job_id, host=requested)
        return requested, state, raw, nodes
    if not job_dir:
        job_dir, _ = _find_job_dir(job_id)
    hint = _job_host(job_dir) if job_dir else None
    order = [hint] if hint is not None else []
    order += [host for host in slurm_hosts() if host != hint]
    for host in order:
        state, raw, nodes = _normalized_job_state(job_id, host=host)
        if state != "unknown":
            # Found, so the next poll of this job is one call rather than a sweep. A
            # recorded host is never replaced: it is what slurm_cancel trusts.
            if job_dir and hint is None:
                _record_job_host(job_dir, host)
            return host, state, raw, nodes
    return "", "unknown", "", []


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

# A job is over when it says so, wherever it says it. squeue reports the same words
# sacct does and keeps a finished job for a while — a cancelled one can sit there
# saying CANCELLED — so these are read from either answer. Prefixes because Slurm
# qualifies some of them ("CANCELLED by 1234").
_DONE_PREFIXES = ("COMPLETED",)


def _is_crash_state(state: str) -> bool:
    return any(state.startswith(prefix) for prefix in _SACCT_CRASH_PREFIXES)


def _is_done_state(state: str) -> bool:
    return any(state.startswith(prefix) for prefix in _DONE_PREFIXES)


def _lines(result: dict) -> list[str]:
    """The non-empty lines of a command's output, upper-cased. [] when it failed."""
    if result.get("status") != "ok":
        return []
    return [line.strip().upper()
            for line in (result.get("stdout") or "").splitlines() if line.strip()]


def _normalized_job_state(job_id: str, *,
                          host: str = _LOCAL_HOST) -> tuple[str, str, list[str]]:
    """Map a Slurm job's state to the shared vocab: running|pending|done|crashed|unknown.

    Active jobs come from ``squeue``; finished jobs have left the queue, so their
    terminal state is read from ``sacct``. Returns ``(state, raw, nodes)``. Missing
    tools or an unknown job yield ``("unknown", "", [])`` — a terminal signal for the
    watcher.

    A job the queue already reports as finished — COMPLETED, FAILED, CANCELLED — is
    settled from that answer alone. Nothing is gained by following it further, and a
    site without accounting has no second answer to settle it with.

    *nodes* is where the job is actually running, read from the same ``squeue`` call
    rather than a second one. Slurm's ``%R`` column is the nodelist for a running job
    and the *reason it is waiting* for a pending one, so it is only read as nodes when
    the job is running — a pending job has no node yet, which is the honest answer and
    not an empty one to be filled in later.

    **One id can be many jobs.** An array submits as ``15306621`` and runs as
    ``15306621_1 … _8``, and both commands answer for every element: ``squeue`` prints
    a line each, ``sacct`` likewise. Read as one line that is a job whose first element
    has finished while seven still run — which is reported as finished, waking its
    conversation on a run that has barely started. So the answer is taken over the
    whole set: anything still going keeps the job live, and a terminal verdict is the
    worst outcome among the elements, since an array where one task failed is not an
    array that completed.
    """
    rows = _lines(_run_bash(f"squeue -j {shlex.quote(job_id)} -h -o '%T|%R'",
                            _TIMEOUT_READ, host=host))
    queued, where = [], []
    for row in rows:
        state, _, reason_or_node = row.partition("|")
        queued.append(state.strip())
        where.append(reason_or_node.strip())
    raw = ",".join(sorted(set(queued)))
    if any(state in _ACTIVE_STATES for state in queued):
        nodes = sorted({node
                        for state, expr in zip(queued, where)
                        if state in _ACTIVE_STATES
                        for node in cpu_facts.expand_hostlist(expr) if node})
        return "running", raw, nodes
    if queued and all(state in _WAITING_STATES for state in queued):
        return "pending", raw, []
    # Over, by the queue's own answer. Read before sacct is asked, for two reasons: it
    # saves a round trip in the ordinary case, and it is the only thing that settles a
    # job on a site with no accounting at all — where ``sacct`` answers nothing and the
    # fallthrough below reads "in the queue with a state I do not recognise" as still
    # running. For a CANCELLED job that is a watcher polling, for as long as the process
    # lives, something that will never change again.
    if queued and all(_is_done_state(s) or _is_crash_state(s) for s in queued):
        # The worst outcome among the elements, as below: an array where one task was
        # cancelled is not an array that completed.
        if any(_is_crash_state(s) for s in queued):
            return "crashed", raw, []
        return "done", raw, []

    accounted = _lines(_run_bash(f"sacct -j {shlex.quote(job_id)} -n -X -o State",
                                 _TIMEOUT_READ, host=host))
    sraw = ",".join(sorted(set(accounted)))
    # Ahead of the "anything not COMPLETED is crashed" fallback below, which is where
    # most of these land anyway. What this earns is the one case the fallback cannot
    # reach: a terminal crash line present *while* squeue still reports a state that is
    # neither active nor waiting, which the clause below would read as running.
    crashed = [state for state in accounted if _is_crash_state(state)]
    if crashed:
        return "crashed", sraw, []
    if accounted and all(_is_done_state(state) for state in accounted):
        return "done", sraw, []
    if queued:                    # in the queue with a non-standard active state
        return "running", raw, []
    if accounted:                 # some other terminal state we don't classify
        return "crashed", sraw, []
    return "unknown", "", []


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
    host, state, raw, nodes = _locate_job(job_id, scheduler, job_dir)
    payload = {"job_id": job_id, "state": state, "raw_state": raw}
    if host or state != "unknown":
        payload["scheduler"] = _host_name(host)
    # Which machines it got. Only ever present for a running job: a pending one has no
    # node yet, and Slurm's own column there is the reason it is waiting.
    if nodes:
        payload["nodes"] = nodes
    if job_dir:
        payload["job_dir"] = job_dir
        # Read, not re-derived: the submission wrote down where it told the job to
        # write, and that answer has to survive a change of rule between versions —
        # a job submitted by an older build still has its log where that build put it.
        log = _recorded_log(job_dir)
        if log:
            payload["log"] = log
            if state == "running":
                tail = build_progress.last_lines(log)
                if tail:
                    payload["log_tail"] = tail
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
        settled = _recorded_terminal_state(existing)
        payload = {
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
        # Recorded is not watched. A watcher is held by the agent that registered it and
        # by nothing else, so a window reload leaves this directory saying "there is a
        # job" with nobody polling it — and this branch used to answer "already watched"
        # into exactly that silence. Handing the descriptor over makes the claim true:
        # the client dedups on the job id, so a watcher that is still there costs
        # nothing, and one that is gone is put back. Withheld for a sibling's job for
        # the reason slurm_job_status withholds it — a wake is routed to the
        # conversation whose agent registered it, and adopting another's delivers its
        # result into this one.
        if not settled and not from_session:
            payload["background_jobs"] = [_slurm_background_job(job_id)]
        return ok(payload)

    # Asked before anything is written, because a descriptor for a job no scheduler has
    # heard of is worse than no descriptor: it holds this process open until the
    # staleness horizon for a run that does not exist. And asked of each of them, which
    # is what makes an id enough — the caller holding a number from a launcher's output
    # has no reason to know which cluster printed it.
    host, state, raw, nodes = _locate_job(job_id, scheduler)
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
    # Where it is running, for the same reason the status read carries it: a watcher
    # keeps what its poll learned, and "on which node" is what a badge says.
    if nodes:
        payload["nodes"] = nodes
    if state in _TERMINAL_STATES:
        # Already over. Settled rather than watched: a watcher put on a finished job
        # reports it at its first poll, which is a wake for news the caller is holding.
        _record_terminal_state(job_dir, state)
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
    payload["background_jobs"] = [_slurm_background_job(job_id)]
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
    """Cancel a Slurm job this conversation submitted, pending or running (sensitive).

    For a job that is wrong or no longer needed: a bad submission, a run superseded by
    a fix, a hung job burning allocation hours. A running job stops where it is, so its
    output is whatever it had written by then. One job per call — or one task of an
    array, as '1234_5'. Only jobs submitted by this conversation or by its sub-agents
    can be cancelled: one from another conversation or workspace, one adopted with
    slurm_watch, or one no longer in the queue is refused.

    A job this session watches ends like any other: you are resumed with its end, so
    there is nothing to poll after cancelling it.

    Args:
        job_id: The Slurm job ID, as sbatch_submit or slurm_queue gave it.
        confirm: Must be True to cancel.
        scheduler: Only when the same id was submitted here on two schedulers, to
            say which. Omitted is the normal case — a cancellation goes to the
            scheduler the job was submitted to.
    """
    job_id = str(job_id or "").strip()
    if not _JOB_ID_RE.fullmatch(job_id):
        return err("job_id must be one Slurm job ID, e.g. '1234' or '1234_5'.",
                   hint="List your jobs with slurm_queue().")
    if not confirm:
        return err("Cancellation not confirmed.",
                   hint="Set confirm=True only after user approval.")

    # Same Unix user is not ownership: other conversations and workspaces share it.
    owned = _owned_job_hosts(job_id)
    if scheduler:
        scheduler = scheduler.strip()
        owned &= {_LOCAL_HOST if scheduler == _LOCAL_NAME else scheduler}
    if not owned:
        return err(f"Job {job_id} was not submitted by this conversation or its "
                   "sub-agents, so it is not this conversation's to cancel.",
                   hint="Jobs of other conversations, other workspaces, or adopted "
                        "with slurm_watch are left alone; ask the user to cancel "
                        "them where they were started.")
    if len(owned) > 1:
        return err(f"Job {job_id} was submitted here on several schedulers: "
                   + ", ".join(_host_name(h) for h in sorted(owned)) + ".",
                   hint="Name the one to cancel on with scheduler=.")
    # Only the scheduler it was submitted to: the same id elsewhere is another job.
    host = next(iter(owned))
    located, located_raw, _nodes = _normalized_job_state(job_id, host=host)
    if located == "unknown":
        return err(f"{_host_name(host)} no longer knows job {job_id}, "
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
    nodes: int = 0,
    ntasks: int = 0,
    constraint: str = "",
    nodelist: str = "",
    exclusive: bool = False,
    work_dir: str = "",
    confirm: bool = False,
    scheduler: str = "",
) -> dict:
    """Submit *command* as a non-blocking Slurm batch job (sensitive, backgroundable).

    Unlike salloc_submit (synchronous, interactive), this returns immediately with a
    ``job_id`` and a ``background_job`` descriptor so the run can be tracked off the
    critical path: when the result says the run is being watched, end your turn and
    you are auto-resumed when the job finishes; otherwise poll slurm_job_status(job_id).

    The job runs on a compute node, in the workspace directory, with the environment
    this server started with — not the modules a previous shell command loaded, so
    put any `module load` in *command* itself. A performance measurement belongs on
    the kind of node it is for (``constraint``/``nodelist``) with ``exclusive=True``:
    a neighbour's job on the same node is measured along with the code. The job's
    output lands in ``log`` and is not recorded as a measurement: a timing meant to
    count goes through ``srun`` in the shell or through the proxy's Slurm route.

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
        nodes: Nodes to allocate (0 = scheduler default).
        ntasks: Tasks to run, e.g. MPI ranks (0 = scheduler default).
        constraint: Slurm feature expression selecting the kind of node.
        nodelist: Specific node(s), as a Slurm hostlist.
        exclusive: Reserve whole nodes (for timings).
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
    target_err = validate_target(constraint, nodelist, nodes or None, ntasks or None)
    if target_err:
        return err(target_err)
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
    header = sbatch_header(
        job_name=job_name, partition=partition, cpus_per_task=cpus_per_task,
        wall_time=wall_time, log_file=os.path.join(work_dir, _LOG_NAME),
        work_dir=work_dir, mem=mem, gpus=gpus, account=account,
        nodes=nodes or None, ntasks=ntasks or None, constraint=constraint,
        nodelist=nodelist, exclusive=exclusive, comment=comment,
    )
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


# ── compute-node profiles ─────────────────────────────────────────────────────
#
# Slurm knows a node's core count, memory, GRES and site features, but not its CPU
# model, its vector ISA, what `-march=native` means there, its caches or its OS. Those
# decide how code should be built and tuned for it, and they are only readable ON the
# node. So a short job reads them there, with the very collectors that describe the
# login node (server_platform --profile-json), and the answer is kept per node until
# Slurm's own description of that node changes. No TTL: hardware does not age.

_PLATFORM_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "server_platform.py")
_NODE_PROFILES_DIR = os.environ.get(
    "MIMIR_NODE_PROFILES_DIR", os.path.join(state_dir(), "hpc", "node_profiles"))
_PROBE_MARKER = "probe.json"


def _profile_host_dir(host: str) -> str:
    """Where this scheduler's node profiles are filed.

    Per host, because Slurm's own node names are per cluster: two schedulers each with
    a ``node01`` are two different machines, and a single flat directory had the second
    one overwrite the first. The segment is the scheduler's own name — ``local``, or the
    hostname — reduced to what a directory name can hold.

    Profiles already sitting at the root of the profiles directory are not read for any
    host. A root file records no scheduler, and reading it as the local one would be a
    guess of exactly the kind the per-host keying exists to stop; there is no TTL and a
    probe is a five-minute job, so a node re-profiles on demand instead.
    """
    name = _host_name(host) or _LOCAL_NAME
    return os.path.join(_NODE_PROFILES_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", name))


def _node_profile_path(node: str, host: str) -> str:
    return os.path.join(_profile_host_dir(host), f"{node}.json")


def _read_node_profile(node: str, host: str) -> dict | None:
    return _read_json(_node_profile_path(node, host))

# Fallback when no MIMIR Python runs on the node: raw sections, parsed server-side
# by the same cpu_facts parsers. Everything that needs no Python is still read.
_FALLBACK_SECTIONS = (
    ("uname", "uname -m"),
    ("lscpu", "LC_ALL=C lscpu"),
    ("march", cpu_facts.MARCH_QUERY + " 2>/dev/null"),
    ("gpu", cpu_facts.GPU_QUERY + " 2>/dev/null"),
    ("os", "cat /etc/os-release 2>/dev/null"),
    ("ldd", "ldd --version 2>&1 | head -n 1"),
    ("nproc", "nproc --all"),
)


def _probe_script(work_dir: str, header: list[str], *, make_dir: bool = False) -> str:
    """The probe, as a script. *work_dir* is where it writes, on the node that runs it.

    For a local scheduler that is the submission's own directory, which already exists.
    For one reached over ssh it is a path on the far side that nothing here can create,
    so the script makes it itself — the first thing on the node, before anything it
    would write there.
    """
    q = shlex.quote
    fallback = [f'  echo "=={name}"; {cmd}' for name, cmd in _FALLBACK_SECTIONS]
    return "\n".join(header + [
        "",
        *([f"mkdir -p {q(work_dir)} || exit 1"] if make_dir else []),
        f"cd {q(work_dir)} || exit 1",
        'echo "${SLURMD_NODENAME:-$(hostname -s)}" > node',
        *node_python_lines(),
        f'if [ -n "$_MIMIR_PY" ] && "$_MIMIR_PY" {q(_PLATFORM_SCRIPT)} --profile-json'
        " > profile.json 2> profile.err; then",
        "  echo full > mode",
        "else",
        "  rm -f profile.json",
        "  {",
        *fallback,
        "  } > fallback.txt 2>&1",
        "  echo partial > mode",
        "fi",
        "",
    ])


def _split_sections(text: str, names) -> dict[str, str]:
    """Split output that labels its parts with ``==<name>`` lines.

    The probe script writes its fallback sections this way, and a remote harvest reads
    a probe's whole directory the same way — one round trip instead of one per file.
    Lines before the first marker, and markers nobody asked for, are dropped.
    """
    wanted = set(names)
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("==") and line[2:].strip() in wanted:
            current = line[2:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# What a probe leaves behind, read whether its directory is on this filesystem or on
# the far side of an ssh. ``mode`` is written last by the script, so its presence is
# what says the probe finished — no scheduler has to have caught up.
_PROBE_FILES = ("mode", "node", "profile.json", "fallback.txt")


def _probe_results(work_dir: str, host: str) -> dict[str, str]:
    """The probe's own files, by name. Absent files are simply missing keys."""
    if not host:
        out = {}
        for name in _PROBE_FILES:
            try:
                with open(os.path.join(work_dir, name)) as fh:
                    out[name] = fh.read()
            except OSError:
                continue
        return out
    # One round trip for all of them: a file per call would be four ssh connections
    # per probe, per panel refresh. Missing files contribute an empty section rather
    # than an error, which is the same answer the local branch gives.
    q = shlex.quote
    script = "; ".join(
        [f"cd {q(work_dir)} || exit 0"] +
        [f'echo "=={name}"; cat {q(name)} 2>/dev/null' for name in _PROBE_FILES])
    res = _run_bash(script, _TIMEOUT_READ, host=host)
    if res.get("status") != "ok":
        return {}
    found = _split_sections(res.get("stdout", ""), _PROBE_FILES)
    return {k: v for k, v in found.items() if v.strip()}


def _fallback_profile(text: str) -> dict:
    """A profile from the raw sections — what the node could say without Python."""
    sec = _split_sections(text, dict(_FALLBACK_SECTIONS))
    arch = sec.get("uname", "").strip()
    gpus = cpu_facts.parse_nvidia_csv(sec.get("gpu", ""))
    os_facts = cpu_facts.parse_os_release(sec.get("os", ""))
    glibc = cpu_facts.parse_ldd_version(sec.get("ldd", ""))
    if glibc:
        os_facts["glibc"] = glibc
    cpu = {"arch": arch, "logical_cpus": _as_int(sec.get("nproc", "").strip()),
           **cpu_facts.cpu_facts(arch, sec.get("lscpu", ""))}
    return {
        "os": os_facts,
        "cpu": cpu,
        "march": cpu_facts.parse_march(sec.get("march", "")),
        "gpu": {"available": bool(gpus), "count": len(gpus), "devices": gpus},
        "machine_signature": cpu_facts.machine_signature(arch, cpu, [g["name"] for g in gpus]),
    }


def _slurm_facts(node: dict) -> dict:
    return {k: node.get(k) for k in _HARDWARE_FIELDS}


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def _write_node_profile(node: str, record: dict, host: str = _LOCAL_HOST) -> None:
    host_dir = _profile_host_dir(host)
    os.makedirs(host_dir, exist_ok=True)
    tmp = os.path.join(host_dir, f".{node}.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(host_dir, f"{node}.json"))


def _harvest_probes(nodes_by_name: dict[str, dict],
                    host: str = _LOCAL_HOST) -> list[dict]:
    """Turn finished probe jobs into node profiles; report the ones not finished.

    A probe is finished when its script wrote ``mode`` — its own last act, so the
    answer never depends on squeue/sacct having caught up. Only a probe without it
    asks the scheduler where it stands.

    *host* is the scheduler whose nodes are being harvested, and it decides where each
    profile is filed. A probe's own directory records the host it was submitted to, so
    one left by an earlier call is harvested for that scheduler rather than for
    whichever one this call is asking about.
    """
    pending = []
    try:
        entries = sorted(os.listdir(_hpc_jobs_dir()))
    except OSError:
        return pending
    for entry in entries:
        job_dir = os.path.join(_hpc_jobs_dir(), entry)
        meta = _read_json(os.path.join(job_dir, _PROBE_MARKER))
        if not meta or meta.get("harvested"):
            continue
        # The scheduler this probe was submitted to, which is not necessarily the one
        # being asked about now, and the directory its result is in — on this
        # filesystem for a local scheduler, on the far side for one reached over ssh.
        probe_host = meta.get("host", host)
        work_dir = meta.get("work_dir") or job_dir
        files = _probe_results(work_dir, probe_host)
        if "mode" not in files:
            state, raw, _nodes = _normalized_job_state(str(meta.get("job_id", "")),
                                                       host=probe_host)
            if state in ("running", "pending"):
                pending.append({"job_id": meta.get("job_id"), "state": state,
                                "partition": meta.get("partition", ""),
                                "scheduler": _host_name(probe_host)})
                continue
            meta.update(harvested=True, outcome=f"no result ({raw or state})")
        else:
            mode = files["mode"].strip()
            node = files.get("node", "").strip()
            if not node:
                meta.update(harvested=True, outcome="no node name recorded")
                with open(os.path.join(job_dir, _PROBE_MARKER), "w") as fh:
                    json.dump(meta, fh)
                continue
            if mode == "full":
                try:
                    profile = json.loads(files.get("profile.json") or "")
                except ValueError:
                    profile = {}
                partial = not profile
            else:
                profile = _fallback_profile(files.get("fallback.txt") or "")
                partial = True
            record = {
                "node": node, "probed_at": meta.get("submitted_at", ""),
                "job_id": meta.get("job_id"), "source": "probe job",
                "scheduler": _host_name(probe_host),
                "slurm": _slurm_facts(nodes_by_name.get(node, {})),
                "partial": partial, "profile": profile,
            }
            if partial:
                record["partial_reason"] = (
                    "No MIMIR Python runs on this node (no .venv-<os>-<arch> for it), so "
                    "only what the shell can read was collected: CPU, -march, GPU, OS. "
                    "Toolchains, modules and Python environments are missing."
                    + (" This scheduler is reached over ssh, where MIMIR's own "
                       "collectors are not installed unless the checkout is shared."
                       if probe_host else ""))
            _write_node_profile(node, record, probe_host)
            meta.update(harvested=True, outcome=f"profiled {node}")
        with open(os.path.join(job_dir, _PROBE_MARKER), "w") as fh:
            json.dump(meta, fh)
    return pending


def _local_profile_record(node: dict) -> dict | None:
    """This host's own profile, when this host is an allocated compute node."""
    res = _run_argv([sys.executable, _PLATFORM_SCRIPT, "--profile-json"], 90)
    try:
        profile = json.loads(res.get("stdout") or "")
    except ValueError:
        return None
    return {"node": node.get("node", ""), "probed_at": profile.get("timestamp", ""),
            "source": "this host (inside the allocation)", "slurm": _slurm_facts(node),
            "partial": False, "profile": profile}


def _host_identity() -> dict:
    """What the host MIMIR runs on is, in the terms a node profile is compared on."""
    lscpu = _run_argv(["env", "LC_ALL=C", "lscpu"], _TIMEOUT_READ)
    import platform as _platform
    arch = _platform.machine()
    cpu = cpu_facts.cpu_facts(arch, lscpu.get("stdout", "") if lscpu.get("status") == "ok" else "")
    return {"arch": arch, "cpu": cpu, "os": cpu_facts.local_os()}


def _compare_with_host(profile: dict, host: dict) -> dict:
    """Which of the facts that decide a build differ between this host and the node."""
    cpu = profile.get("cpu") or {}
    node_os, host_os = profile.get("os") or {}, host["os"]
    checks = {
        "arch": (cpu.get("arch"), host["arch"]),
        "cpu_model": (cpu.get("model"), host["cpu"].get("model")),
        "simd": (sorted(k for k, v in (cpu.get("simd") or {}).items() if v),
                 sorted(k for k, v in (host["cpu"].get("simd") or {}).items() if v)),
        "os": ((node_os.get("id"), node_os.get("version")), (host_os.get("id"), host_os.get("version"))),
        "glibc": (node_os.get("glibc"), host_os.get("glibc")),
    }
    differs = [k for k, (a, b) in checks.items() if a and b and a != b]
    return {"same": not differs, "differs": differs}


@mcp.tool(**tool_caps(kind="slurm",
    caps=[BACKGROUNDABLE], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="Submits a short Slurm job (a few seconds on one node) to read the node's hardware.",
    label="Slurm node probe",
))
def slurm_probe_node(partition: str, constraint: str = "", nodelist: str = "",
                     account: str = "", scheduler: str = "",
                     confirm: bool = False) -> dict:
    """Read a compute node's full profile ON the node, with a short batch job.

    Slurm cannot say what a node's CPU model, vector ISA, native -march, caches, GPU
    compute capability, OS/glibc, toolchains or Python environments are — only the node
    can. This submits a job of a few seconds that collects exactly what the host
    profile shows for this machine, on one node of *partition* (optionally narrowed by
    ``constraint`` or ``nodelist``), and returns a background-job handle: end your turn;
    when you are resumed, read the result with the node-profile tool. Each node is
    probed once and kept until Slurm's description of it changes, so check the
    node-profile tool first — a node already profiled needs no job. Unnecessary when
    MIMIR itself runs inside an allocation on the node: its own profile is the node's.

    Which partition: the one the user named. Otherwise the one the code and the job
    call for (GPU code → a GPU partition, and so on); when that is not clear, ask the
    user rather than guess.

    Args:
        partition: Slurm partition whose node to probe (required).
        constraint: Slurm feature expression selecting the kind of node.
        nodelist: A specific node, when one kind of node in a mixed partition matters.
        account: Slurm account to charge (optional).
        scheduler: Which scheduler to submit to; name it when the workspace reaches
            more than one. The profile is filed under that scheduler, because a node
            name means nothing without the cluster it is on.
        confirm: Must be True to submit.
    """
    if not partition.strip():
        return err("partition is required.")
    target_err = validate_target(constraint, nodelist)
    if target_err:
        return err(target_err)
    host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    import time as _time
    stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
    job_dir = os.path.join(_hpc_jobs_dir(), f"{stamp}-probe")
    suffix = 1
    while os.path.exists(job_dir):
        suffix += 1
        job_dir = os.path.join(_hpc_jobs_dir(), f"{stamp}-probe{suffix}")
    os.makedirs(job_dir)
    # Where the job writes, on the node that runs it. A scheduler on this machine
    # writes into the submission's own directory; one reached over ssh has no access
    # to this filesystem, so it writes beside its own state and the harvest reads it
    # back over the same connection. The expansion is the script's, not ours — only
    # the far side knows what $HOME is there.
    if host:
        work_dir = ('"${MIMIR_STATE_HOME:-$HOME/.mimir}"/node_probes/'
                    + os.path.basename(job_dir))
    else:
        work_dir = job_dir
    log_file = os.path.join(work_dir, "slurm.log")
    header = sbatch_header(
        job_name="mimir-probe", partition=partition, cpus_per_task=1,
        wall_time="00:05:00", log_file=log_file, account=account, nodes=1, ntasks=1,
        constraint=constraint, nodelist=nodelist,
    )
    script = _probe_script(work_dir, header, make_dir=bool(host))
    # Written locally whatever the scheduler, for a reader: the submission's directory
    # is where someone goes to see what was actually asked for.
    script_path = os.path.join(job_dir, "probe.sh")
    try:
        with open(script_path, "w") as fh:
            fh.write(script)
    except OSError as exc:
        return err(f"Could not write the probe script: {exc}")

    # On stdin, not by path: a remote scheduler cannot read a file on this machine,
    # and the local one does not need to.
    res = _run_bash("sbatch", _TIMEOUT_SUBMIT, stdin=script, host=host)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account and constraint.")
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return err("Could not parse a job id from sbatch output.",
                   hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    job_id = match.group(1)
    # The host is recorded twice on purpose: in the marker the harvest reads, and in
    # the descriptor's own host file, which is what every other cluster tool resolves a
    # job's scheduler from.
    _record_job_host(job_dir, host)
    with open(os.path.join(job_dir, _PROBE_MARKER), "w") as fh:
        json.dump({"job_id": job_id, "partition": partition, "constraint": constraint,
                   "nodelist": nodelist, "submitted_at": stamp,
                   "host": host, "work_dir": work_dir}, fh)
    return ok({
        "job_id": job_id,
        "job_dir": job_dir,
        "partition": partition,
        "note": f"Probe job {job_id} submitted to '{partition}'. Its result is read "
                "with the node-profile tool once the job ends.",
        "background_job": {
            "server":    "hpc",
            "job_key":   job_id,
            "kind":      "slurm-batch",
            "status_op": {"tool": "slurm_job_status", "args": {"job_id": job_id}},
        },
    })


# Above this many node types, only the facts that decide a build are shown per type;
# a full profile each (toolchains, environments, modules) would drown the answer.
_FULL_PROFILE_TYPES = 3


def _summary(profile: dict) -> dict:
    cpu = profile.get("cpu") or {}
    return {
        "cpu_model": cpu.get("model", ""),
        "simd": sorted(k for k, v in (cpu.get("simd") or {}).items() if v),
        "caches": cpu.get("caches", {}),
        "march": profile.get("march", {}),
        "gpu": [{k: d.get(k) for k in ("name", "compute_cap") if d.get(k)}
                for d in (profile.get("gpu") or {}).get("devices", [])],
        "os": profile.get("os", {}),
    }


@mcp.tool()
def slurm_node_profile(partition: str = "", node: str = "",
                       scheduler: str = "") -> dict:
    """What the compute nodes are, as the machines code will be built for and run on.

    Read-only and instant. For each kind of node (Slurm's hardware signature) in
    *partition* — or just *node* — returns Slurm's facts plus, when a node of that
    kind has been profiled, its full profile read on the node: CPU model, vector ISA,
    caches, the name `-march=native` resolves to there, GPUs with compute capability,
    OS and glibc, toolchains and Python environments. A kind with no profiled node
    says so; probe one with the node-probe tool. When MIMIR runs inside an allocation,
    the node it runs on is profiled on the spot.

    `matches_this_host` says whether a node differs from the host MIMIR runs on in
    what decides a build: arch, CPU model, SIMD, OS, glibc. When it differs:
    - never build with `-march=native` here for that node — pass the node's `march`,
      or build on the node, inside the job;
    - if OS or glibc differ, build on the node: a binary linked here may not start there;
    - measure performance on the node, not here: a speedup measured on this host
      describes this host.

    Which scheduler: name it when the workspace reaches more than one. Node names are
    per cluster — two schedulers each with a `node01` are two different machines — so a
    profile belongs to the scheduler it was read on and is never read for another.

    Which partition: the one the user named; otherwise the one the code and the job
    call for; when that is not clear, ask the user.

    Args:
        partition: Only node kinds in this partition.
        node: A single node name.
        scheduler: Which scheduler's nodes to describe; name it when the workspace
            reaches more than one. A profile belongs to the scheduler it was read on
            and is never read for another.
    """
    sched_host, refusal = _chosen_host(scheduler)
    if refusal:
        return refusal
    result = _run_argv(["scontrol", "show", "node", "-o"], _TIMEOUT_READ,
                       host=sched_host)
    if result.get("status") != "ok":
        return err(result.get("stderr") or result.get("error", "scontrol failed"),
                   hint=f"Needs Slurm's scontrol on {_host_name(sched_host)}.")
    all_nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    by_name = {n["node"]: n for n in all_nodes}
    pending = _harvest_probes(by_name, sched_host)

    ctx = cpu_facts.execution_context(has_slurm=True)
    if ctx["context"] == "in_allocation":
        here = by_name.get(socket.gethostname().split(".")[0])
        if here:
            cached = _read_node_profile(here["node"], sched_host)
            if not cached or cached.get("slurm") != _slurm_facts(here):
                record = _local_profile_record(here)
                if record:
                    _write_node_profile(here["node"], record, sched_host)

    nodes = all_nodes
    if node:
        nodes = [n for n in nodes if n["node"] == node]
    if partition:
        nodes = [n for n in nodes if partition in n["partitions"]]
    if not nodes:
        return ok({"node_types": [], "note": "No node matched.", "pending_probes": pending})

    host = _host_identity()
    members: dict[tuple, list[dict]] = {}
    for n in nodes:
        members.setdefault(_hardware_key(n), []).append(n)
    types = _aggregate_node_types(nodes)
    full = bool(node) or len(types) <= _FULL_PROFILE_TYPES
    out = []
    for t in types:
        group = members.get(_hardware_key(t), [])
        fresh, stale = [], []
        for n in group:
            rec = _read_node_profile(n["node"], sched_host)
            if not rec:
                continue
            (fresh if rec.get("slurm") == _slurm_facts(n) else stale).append(rec)
        entry = {k: t.get(k) for k in ("arch", "cpus", "mem_gb", "gres", "sockets",
                                       "cores_per_socket", "threads_per_core", "features",
                                       "partitions", "nodes_total", "by_state", "example_nodes")}
        if not fresh:
            entry["profiled"] = False
            entry["note"] = ("No node of this kind profiled"
                             + (" since Slurm's description of it changed" if stale else "")
                             + "; probe one to know its CPU, ISA and -march.")
        else:
            rec = fresh[0]
            profile = rec.get("profile") or {}
            entry.update({
                "profiled": True,
                "profiled_on": rec.get("node"),
                "profiled_at": rec.get("probed_at"),
                "source": rec.get("source"),
                "matches_this_host": _compare_with_host(profile, host),
                "profile": profile if full else _summary(profile),
            })
            if rec.get("partial"):
                entry["partial"] = rec.get("partial_reason", True)
            models = {(r.get("profile") or {}).get("cpu", {}).get("model") for r in fresh}
            models.discard(None)
            if len(models) > 1:
                entry["cpu_models_differ"] = sorted(models)
            if len(fresh) < len(group):
                entry["unprofiled_nodes"] = len(group) - len(fresh)
                entry["caveat"] = ("Slurm's signature does not include the CPU model: "
                                   "the unprofiled nodes of this kind are assumed, not "
                                   "known, to be the same machine.")
        out.append(entry)

    payload = {
        "execution_context": ctx,
        "host": {"hostname": socket.gethostname(), "arch": host["arch"],
                 "cpu_model": host["cpu"].get("model", ""), "os": host["os"]},
        "node_types": out,
        "type_count": len(out),
    }
    if not full:
        payload["note"] = ("Profiles summarised across many node kinds; pass partition "
                           "with one kind, or node, for the full profile.")
    if pending:
        payload["pending_probes"] = pending
    return ok(payload)


# ── Panel section ─────────────────────────────────────────────────────────────
# What the scientific-computing drawer shows about the cluster. Deliberately not the
# host MIMIR runs on — that is usually a login node, which is the one machine nothing
# will be measured on. What is worth a line here is where the work actually goes: the
# queue the user is waiting in, the partitions that have room, and the hardware of the
# nodes their code will be built for.

# The panel refreshes on a timer while a turn runs, so every line below is served from
# a cache. Two clocks, because the facts age at different speeds: a queue moves between
# two refreshes, a partition's shape and a node's hardware do not.
_PANEL_QUEUE_TTL_SECS = 15
_PANEL_CLUSTER_TTL_SECS = 300
# A tree holding a host that did not answer is asked again this soon instead.
_PANEL_RETRY_SECS = 30
_PANEL_CACHE: dict = {}


def _panel_cached(key: str, ttl: int, produce):
    """*produce()* at most once every *ttl* seconds, per server process.

    A failure serves the last good answer rather than a blank: a transient sinfo is a
    reason to show what was true a minute ago, not to empty the section.
    """
    entry = _PANEL_CACHE.get(key)
    now = time.time()
    if entry and now - entry["at"] <= ttl:
        return entry["value"]
    try:
        value = produce()
    except Exception:
        return entry["value"] if entry else None
    _PANEL_CACHE[key] = {"at": now, "value": value}
    return value


def _queue_line() -> dict | None:
    """The user's own jobs, counted by state. None when the queue cannot be read."""
    payload = _panel_cached("queue", _PANEL_QUEUE_TTL_SECS,
                            lambda: slurm_queue(user_only=True))
    if not payload or payload.get("status") != "ok":
        return None
    counts: dict[str, int] = {}
    for job in payload.get("jobs") or []:
        counts[str(job.get("state") or "?")] = counts.get(str(job.get("state") or "?"), 0) + 1
    if not counts:
        return {"label": "your jobs", "value": "none queued"}
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "label": "your jobs",
        "value": ", ".join(f"{n} {state.lower()}" for state, n in order),
        "state": "running" if counts.get("RUNNING") else "",
    }


def _partition_lines(limit: int = 4) -> list[dict]:
    """Idle nodes per partition — whether a job submitted now starts now.

    sinfo emits one row per (partition, state), so the rows are folded back into one
    entry per partition here. The busiest are not the interesting ones: the partitions
    with room come first, because that is the question being asked.
    """
    payload = _panel_cached("partitions", _PANEL_CLUSTER_TTL_SECS, slurm_partitions)
    if not payload or payload.get("status") != "ok":
        return []
    totals: dict[str, list[int]] = {}
    for row in payload.get("partitions") or []:
        name = str(row.get("partition") or "").rstrip("*")
        nodes = _as_int(row.get("nodes")) or 0
        if not name:
            continue
        seen = totals.setdefault(name, [0, 0])
        seen[0] += nodes
        if str(row.get("state") or "").startswith("idle"):
            seen[1] += nodes
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1][1], kv[0]))
    # No state on these: the value already says how much room there is, and the one
    # colour the panel has means "something of yours is running".
    return [{"label": name, "value": f"{idle}/{total} nodes idle"}
            for name, (total, idle) in ranked[:limit]]


def _node_kind_label(kind: dict) -> str:
    """A node kind as a person would name it: what you would say to pick one.

    The GPUs first when there are any, because that is what a kind is chosen for; then
    the core count and the memory, which is what a job is sized against. Slurm's own
    hardware signature is what grouped them, and it is unreadable.
    """
    parts = []
    gpu = _gpu_of(kind)
    if gpu:
        parts.append(gpu)
    if kind.get("cpus"):
        parts.append(f"{kind['cpus']} cpus")
    mem_mb = kind.get("mem_mb")
    if isinstance(mem_mb, (int, float)) and mem_mb:
        parts.append(f"{round(mem_mb / 1024)} GB")
    return " · ".join(parts) or str(kind.get("arch") or "node")


def _gpu_of(kind: dict) -> str:
    """``gpu:a100:4`` as ``4× a100``; empty when the kind has no GPU."""
    gres = str(kind.get("gres") or "")
    if not gres.startswith("gpu:"):
        return ""
    bits = gres.split(":")
    model = bits[1] if len(bits) > 2 else ""
    count = bits[-1] if bits[-1].isdigit() else ""
    return " ".join(x for x in (f"{count}×" if count else "", model) if x).strip() or gres


def _node_names_label(kind: dict) -> str:
    """The kind's nodes by name, ``+N`` for those beyond the sample."""
    names = list(kind.get("example_nodes") or [])
    if not names:
        return _node_kind_label(kind)
    more = (kind.get("nodes_total") or len(names)) - len(names)
    return ", ".join(names) + (f" +{more}" if more > 0 else "")


def _kind_spec_lines(kind: dict, gpu_mem: str = "") -> list[dict]:
    """What Slurm says a node of this kind is, one fact per line.

    *gpu_mem* comes from a probe: Slurm's gres names GPUs but never their memory.
    """
    lines = [{"label": "arch", "value": kind.get("arch") or "?"}]
    if kind.get("cpus"):
        value = str(kind["cpus"])
        topo = [kind.get(f) for f in ("sockets", "cores_per_socket", "threads_per_core")]
        if all(topo):
            value += f" ({topo[0]} sockets × {topo[1]} cores × {topo[2]} threads)"
        lines.append({"label": "cpus", "value": value})
    gpu = _gpu_of(kind)
    if gpu:
        lines.append({"label": "gpu", "value": gpu})
    elif kind.get("gres"):
        lines.append({"label": "gres", "value": str(kind["gres"])})
    mem_mb = kind.get("mem_mb")
    if isinstance(mem_mb, (int, float)) and mem_mb:
        lines.append({"label": "cpu mem", "value": f"{round(mem_mb / 1024)} GB"})
    if gpu:
        lines.append({"label": "gpu mem",
                      "value": gpu_mem or "unknown until a node is profiled"})
    if kind.get("partitions"):
        lines.append({"label": "partitions", "value": ", ".join(kind["partitions"])})
    if kind.get("features"):
        lines.append({"label": "features", "value": kind["features"]})
    if kind.get("nodes_total"):
        lines.append({"label": "nodes", "value": str(kind["nodes_total"])})
    return lines


def _gpu_mem_label(raw: str) -> str:
    """nvidia-smi's ``81920 MiB`` as ``80 GB per GPU``; anything else as it came."""
    m = re.fullmatch(r"\s*(\d+)\s*MiB\s*", raw)
    return f"{round(int(m.group(1)) / 1024)} GB per GPU" if m else raw


def _kind_id(host: str, kind: dict) -> str:
    """A stable id for one node kind of one scheduler.

    Stable across refreshes because the panel's open/closed state is keyed on it: a
    tree that re-identified its rows every poll would collapse under the reader. From
    the hardware fields themselves, so the same machines get the same id whatever
    order the scheduler listed them in.
    """
    payload = json.dumps([kind.get(f) for f in _HARDWARE_FIELDS], sort_keys=True,
                         default=str)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"host:{_host_name(host)}/kind:{digest}"


def _profiled_kind_lines(kind: dict, host: str) -> tuple[list[dict], str]:
    """What a profiled node of this kind actually is, and where that was read.

    Slurm reports a node's core count and memory but never its CPU model, vector ISA
    or the name ``-march=native`` resolves to — only the node can say those, which is
    what a probe is for. Read off the profile cache: nothing here runs a command, the
    same discipline the rest of this panel keeps.
    """
    for node in kind.get("example_nodes") or []:
        record = _read_node_profile(node, host)
        if not record or not record.get("profile"):
            continue
        profile = record["profile"]
        cpu = profile.get("cpu") or {}
        lines = []
        if cpu.get("model"):
            lines.append({"label": "cpu", "value": cpu["model"]})
        simd = sorted(k for k, v in (cpu.get("simd") or {}).items() if v)
        if simd:
            lines.append({"label": "vector isa", "value": ", ".join(simd[-3:])})
        march = (profile.get("march") or {}).get("native")
        if march:
            lines.append({"label": "-march=native", "value": march})
        node_os = profile.get("os") or {}
        if node_os.get("id"):
            value = f"{node_os['id']} {node_os.get('version', '')}".strip()
            if node_os.get("glibc"):
                value += f" · glibc {node_os['glibc']}"
            lines.append({"label": "os", "value": value})
        devices = (profile.get("gpu") or {}).get("devices") or []
        if devices and devices[0].get("memory"):
            lines.append({"label": "gpu mem",
                          "value": _gpu_mem_label(str(devices[0]["memory"]))})
        if devices and devices[0].get("compute_cap"):
            lines.append({"label": "compute capability",
                          "value": str(devices[0]["compute_cap"])})
        detail = f"Profiled on {record.get('node', node)}"
        if record.get("probed_at"):
            detail += f", {record['probed_at']}"
        if record.get("partial"):
            detail += ". " + str(record.get("partial_reason") or "")
        return lines, detail.strip()
    return [], ""


def _host_tree() -> list[dict]:
    """Every scheduler this workspace reaches, and what each is made of.

    One group per host, one subgroup per kind of node. Built from the per-host digest,
    which is occupancy-free and memoised for the life of the server — so opening the
    drawer does not reach across ``ssh`` to every cluster on every refresh, and what
    is *free right now* stays ``slurm_nodes``'s answer rather than a number that was
    true once.

    A host that did not answer is listed with that said, never as a host with no
    machines: an unreachable cluster and an empty one are not the same thing, and only
    one of them is worth waiting for.
    """
    hosts = slurm_hosts()
    # Fill the memo for every host at once. Each miss is an ssh away and waits on a
    # scheduler rather than on this process, so one after another the drawer cost the
    # sum of the clusters it reaches; together it costs the slowest of them. Reads
    # below still go through _cluster_digest, which now finds them already there.
    if len(hosts) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            for done in [pool.submit(_cluster_digest, h) for h in hosts]:
                with contextlib.suppress(Exception):
                    done.result()
    groups = []
    for host in hosts:
        name = _host_name(host)
        digest = _cluster_digest(host)
        group = {"id": f"host:{name}", "label": name, "host": name,
                 "local": not host,
                 "lines": [{"label": "reached",
                            "value": "Slurm answers on this host" if not host
                                     else f"over ssh to {host}"}]}
        if not digest["source"]:
            group["state"] = "warn"
            group["value"] = "did not answer"
            group["lines"].append(
                {"label": "hardware",
                 "value": "unknown, not absent", "state": "warn"})
            groups.append(group)
            continue
        hardware = _hardware_of(host)
        group["value"] = hardware.get("hardware", "")
        partitions = [str(p.get("partition") or "") for p in digest["partitions"]]
        if partitions:
            group["lines"].append({"label": "partitions",
                                   "value": ", ".join(partitions[:6])})
        if digest["degraded"]:
            group["lines"].append({"label": "reading", "value": digest["degraded"],
                                   "state": "warn"})
        kinds = []
        for kind in digest["node_types"]:
            profiled, detail = _profiled_kind_lines(kind, host)
            # GPU memory sits with the other memory line, not among the probe's lines.
            gpu_mem = next((l["value"] for l in profiled if l["label"] == "gpu mem"), "")
            profiled = [l for l in profiled if l["label"] != "gpu mem"]
            entry = {
                "id": _kind_id(host, kind),
                "label": _node_names_label(kind),
                "lines": _kind_spec_lines(kind, gpu_mem) + profiled,
                "nodes": list(kind.get("example_nodes") or []),
            }
            if detail:
                entry["detail"] = detail
            if not profiled:
                # No badge for it. A kind nothing has been probed on is the ordinary
                # case — most workspaces never probe any — so "not profiled" on every
                # row was a warning colour carrying no warning, and it sat next to the
                # one thing on the row worth reading. What is offered instead is the
                # request that would get the answer, inside the row once it is opened.
                # The server writes the words because it is the end that knows the
                # scheduler and the partition.
                part = partitions[0] if partitions else ""
                for candidate in partitions:
                    if candidate in (kind.get("partitions") or []):
                        part = candidate
                        break
                entry["action"] = {
                    "kind": "probe",
                    "label": "Profile a node",
                    "prompt": (
                        f"Probe one node of the '{_node_kind_label(kind)}' kind on "
                        f"scheduler '{name}', partition '{part}', so its CPU model, "
                        f"vector ISA and -march are known before anything is built "
                        f"for it."),
                }
            kinds.append(entry)
        if kinds:
            group["groups"] = kinds
        groups.append(group)
    return groups


def _host_tree_cached() -> list[dict]:
    """The host tree, rebuilt the moment what it is drawn from changes.

    A plain TTL held a node profiled during a turn, or a scheduler recorded by one,
    off the panel for five minutes — only ⟳ showed it. The key is the registry and the
    mtime of each host's profile directory (a profile lands by rename), both a stat
    away; the per-host digest behind the tree stays memoised, so a rebuild is cheap.
    """
    hosts = slurm_hosts()
    stamps = []
    for host in hosts:
        try:
            stamps.append(os.stat(_profile_host_dir(host)).st_mtime_ns)
        except OSError:
            stamps.append(0)
    key = (tuple(hosts), tuple(stamps))
    entry = _PANEL_CACHE.get("hosts")
    now = time.time()
    if entry and entry.get("key") == key:
        unanswered = any(g.get("state") == "warn" for g in entry["value"])
        ttl = _PANEL_RETRY_SECS if unanswered else _PANEL_CLUSTER_TTL_SECS
        if now - entry["at"] <= ttl:
            return entry["value"]
    try:
        value = _host_tree()
    except Exception:
        return entry["value"] if entry else []
    _PANEL_CACHE["hosts"] = {"at": now, "key": key, "value": value}
    return value


@mcp.tool(**tool_caps(kind="slurm",
    caps=[PANEL_REPORT],
    read_only=True,
    panel={"section": "Cluster", "order": 30, "refresh_arg": "refresh"},
    label="Cluster status",
))
async def hpc_panel_report(refresh: bool = False) -> dict:
    """Where the work goes, in a form a panel can show: the queue, the room, the nodes.

    Two halves with two shapes. The lines are this moment — what is queued, where there
    is room — and the groups are the standing thing: every scheduler the workspace
    reaches, each with the kinds of node it is made of and what a probe found on one.
    A nested shape because the question is nested: which cluster, then which kind of
    machine on it, then what that machine actually is.

    Everything is served from the caches above, so a panel refreshing on a timer costs
    one ``squeue`` every few seconds at most, one ``sinfo`` every few minutes, and one
    digest per scheduler for the life of the server. A host with nothing to say returns
    neither lines nor groups, and the client then shows no section — a laptop should not
    grow an empty cluster panel.

    *refresh* is the reader asking, and it drops all of it: the per-TTL caches and the
    per-host digest alike. The digest is the one that matters — it is kept for the life
    of the server, so a scheduler that was unreachable at startup, or a partition added
    since, is invisible until something clears it. That is what ⟳ is for, and the only
    call that pays for the re-probe.

    Args:
        refresh: Drop every cache first and read the clusters again. The reader's own
            refresh; a panel on a timer leaves it alone.
    """
    # Off the server's loop: a cold read is seconds of ssh/sinfo, and a sync tool holds
    # every other call to this server for that long.
    return await asyncio.to_thread(_hpc_panel_report, refresh)


_PANEL_LOCK = threading.Lock()


def _hpc_panel_report(refresh: bool) -> dict:
    with _PANEL_LOCK:
        return _hpc_panel_report_locked(refresh)


def _hpc_panel_report_locked(refresh: bool) -> dict:
    if refresh:
        _PANEL_CACHE.clear()
        _forget_cluster_digest()
    lines: list[dict] = []
    queue = _queue_line()
    if queue:
        lines.append(queue)
    lines.extend(_partition_lines())
    groups = _host_tree_cached()
    if not lines and not groups:
        # No queue, no partitions and no registered scheduler: there is nothing here to
        # report on. Said with nothing rather than with a row saying "unavailable",
        # which is a section the user has to read to learn there is nothing to read.
        return ok({"title": "Cluster", "lines": []})
    return ok({"title": "Cluster", "lines": lines, "groups": groups,
               "detail": "Where submitted work runs — not the host MIMIR runs on."})


if __name__ == "__main__":
    mcp.run()
