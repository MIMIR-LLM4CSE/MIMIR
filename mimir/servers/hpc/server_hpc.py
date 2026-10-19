"""MCP HPC Server: Slurm queries, batch submission, cancellation, node profiles.

Submitters build the command from arguments and run it as argv (``CLUSTER_SUBMIT``);
``sbatch_submit`` returns a ``background_job`` the client polls through
``slurm_job_status``; ``slurm_cancel`` takes one job the user owns. Modules go through
the bash server.
"""

import asyncio
import json
import os
import re
import shlex
import socket
from typing import Any


from mcp.server.fastmcp import FastMCP
from ...shared.capabilities import (
    tool_caps,
    PLAN_BLOCKED,
    CLUSTER_SUBMIT,
    BACKGROUNDABLE,
    IRREVERSIBLE,
    PANEL_REPORT,
)
from ...shared.responses import err, ok
from ...shared.jobs import (
    SLURM_JOB_ID,
    slurm_descriptor,
    is_terminal,
    record_terminal_state,
    recorded_terminal_state,
)
from ...shared.slurm_site import (
    ADOPTED_FILE as _ADOPTED_FILE,
    HOST_FILE as _HOST_FILE,
    LOCAL as _LOCAL_HOST,
    LOCAL_NAME as _LOCAL_NAME,
    covered_by as _covered_by,
    hosts as slurm_hosts,
    name_of as _host_name,
    jobs_dir as _hpc_jobs_dir,
    new_job_dir as _new_job_dir,
    owned_job_hosts as _owned_job_hosts,
    record_job_dir as _record_job_dir,
    recorded_jobs as _recorded_jobs,
    record_host as _record_host,
    forget_host as _forget_host,
    unwatched as slurm_unwatched,
    unwatched_hint as slurm_unwatched_hint,
)
from .slurm_digest import (
    forget_cluster_digest as _forget_cluster_digest,
)
from .slurm_nodes import (
    SCONTROL_NODES_ARGV as _SCONTROL_NODES_ARGV,
    aggregate_node_types as _aggregate_node_types,
    hardware_key as _hardware_key,
    parse_pipe_table as _parse_pipe_table,
    parse_scontrol_nodes as _parse_scontrol_nodes,
)
from ...shared.slurm_script import sbatch_header, validate_target
from ...shared import build_progress
from ...shared import cpu_facts
from . import cli
from .cli import (
    TIMEOUT_ALLOC,
    TIMEOUT_READ,
    TIMEOUT_SUBMIT,
    chosen_host,
    no_scheduler_hint,
    sinfo_nodes,
    validate_comment,
    validate_mem,
    validate_time,
)
from . import jobs
from .jobs import (
    JOB_ID_RE,
    LOG_FILE,
    LOG_NAME,
    add_unwatched,
    find_job_dir,
    locate_job,
    record_job_host,
    recorded_log,
    salloc_argv,
)
from . import node_profiles, panel
from .node_profiles import (
    FULL_PROFILE_TYPES,
    PROBE_MARKER,
    compare_with_host,
    hardware_of,
    harvest_probes,
    probe_script,
    read_node_profile,
    slurm_facts,
    summary,
    write_node_profile,
)

mcp = FastMCP(
    "HPCServer",
    debug=False,
    log_level="ERROR",
)


# ── Which scheduler a call goes to ────────────────────────────────────────────
# The inventory is ``shared/slurm_site``. A tool that acts resolves a host from its
# arguments (``_resolve_host``) and refuses when the choice is open; one acting on an
# existing job finds where it is (``locate_job``). Every Slurm command goes through
# ``slurm_argv``/``slurm_exec_argv``.


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
        probe = cli.run_bash("command -v sbatch", TIMEOUT_READ, host=host)
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
        **hardware_of(host),
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
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    cmd = "sinfo -h -o '%P|%a|%l|%D|%t|%c|%m'"
    result = cli.run_bash(cmd, TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "sinfo failed"),
                    hint=no_scheduler_hint(host))
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["partition", "availability", "time_limit", "nodes", "state", "cpus_per_node", "mem_mb_per_node"],
    )
    return ok({"partitions": rows, "count": len(rows),
               "scheduler": _host_name(host)})


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
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    degraded = ""
    result = cli.run_argv(_SCONTROL_NODES_ARGV, TIMEOUT_READ, host=host)
    if result["status"] == "ok":
        nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    else:
        nodes = sinfo_nodes(host)
        degraded = "scontrol unavailable: architecture and live CPU occupancy are unknown."
    if not nodes:
        return err(result.get("stderr") or result.get("error", "node query failed"),
                   hint=no_scheduler_hint(host))

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
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    user_flag = " -u $USER" if user_only else ""
    state_flag = f" -t {shlex.quote(states)}" if states else ""
    cmd = f"squeue -h{user_flag}{state_flag} -o '%i|%u|%T|%M|%l|%D|%R|%j'"
    result = cli.run_bash(cmd, TIMEOUT_READ, host=host)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "squeue failed"),
                   hint=no_scheduler_hint(host))
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["job_id", "user", "state", "elapsed", "time_limit", "nodes", "reason_or_node", "name"],
    )
    # Which rows this workspace watches: a job is tracked by its descriptor, keyed by host
    # (ids are per cluster). The unwatched set goes to the client for its end-of-turn nudge.
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
    timeout_seconds: int = TIMEOUT_ALLOC,
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
    argv, error = salloc_argv(partition, account, qos, nodes, ntasks, cpus_per_task,
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
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    if host:
        return err(f"{host} is reached over ssh, and an interactive allocation needs a "
                   "terminal on the host that has the scheduler.",
                   hint="Submit the work as a batch job instead — sbatch_submit(..., "
                        f"scheduler='{host}') returns a handle this session watches to "
                        "completion.",
                   command=preview)

    result = cli.run_argv(argv, max(5, min(timeout_seconds, 120)), host=host)
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


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_job_status(job_id: str, scheduler: str = "") -> dict:
    """Normalized status of a single Slurm job (poll target for background jobs).

    Returns ``state`` in running|pending|done|crashed|unknown (squeue for active
    jobs, sacct for finished ones) plus the raw Slurm state string.

    A terminal answer also settles the submission's directory, so that the readers which
    cannot ask Slurm stop reading the job as live — see :func:`record_terminal_state`.

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
    job_dir, from_session = find_job_dir(job_id)
    host, state, raw, nodes = locate_job(job_id, scheduler, job_dir)
    payload: dict[str, Any] = {"job_id": job_id, "state": state, "raw_state": raw}
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
    if not JOB_ID_RE.fullmatch(job_id):
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
    located, located_raw, _nodes = jobs.normalized_job_state(job_id, host=host)
    if located == "unknown":
        return err(f"{_host_name(host)} no longer knows job {job_id}, "
                   "so there is nothing to cancel.",
                   hint="List what is running with slurm_queue(scheduler=...).")

    q = cli.run_argv(["squeue", "-h", "-j", job_id, "-o", "%u|%T"], TIMEOUT_READ,
                  host=host)
    line = (q.get("stdout") or "").strip().splitlines()
    if q.get("status") != "ok" or not line:
        return err(f"Job {job_id} is not in {_host_name(host)}'s queue, so there is "
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
    if wall_time and not validate_time(wall_time):
        return err("Invalid Slurm time format.", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not validate_mem(mem):
        return err("Invalid mem format.", hint="Use values like 8G, 32000M, 1T")
    if cpus_per_task < 1 or gpus < 0:
        return err("cpus_per_task must be >= 1 and gpus >= 0.")
    target_err = validate_target(constraint, nodelist, nodes or None, ntasks or None)
    if target_err:
        return err(target_err)
    comment_err = validate_comment(comment)
    if comment_err:
        return err(comment_err)
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    host, refusal = chosen_host(scheduler)
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
        wall_time=wall_time, log_file=os.path.join(work_dir, LOG_NAME),
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
    res = cli.run_bash("sbatch", TIMEOUT_SUBMIT, stdin=script, host=host)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account/qos, and requested resources. "
                        + no_scheduler_hint(host))
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return err("Could not parse a job id from sbatch output.",
                   hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    job_id = match.group(1)
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
        "background_job": slurm_descriptor(job_id),
    }
    # The submission is the other moment the model holds job ids: what else of the
    # user's is running unaccounted for can be acted on here, in the same turn.
    add_unwatched(payload)
    return ok(payload)


# ── compute-node profiles ─────────────────────────────────────────────────────
# What only the node can say (CPU model, ISA, ``-march=native``, caches, OS), read there by
# a short job with server_platform's collectors and kept until Slurm's description changes.


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
    host, refusal = chosen_host(scheduler)
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
    script = probe_script(work_dir, header, make_dir=bool(host))
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
    res = cli.run_bash("sbatch", TIMEOUT_SUBMIT, stdin=script, host=host)
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
    record_job_host(job_dir, host)
    with open(os.path.join(job_dir, PROBE_MARKER), "w") as fh:
        json.dump({"job_id": job_id, "partition": partition, "constraint": constraint,
                   "nodelist": nodelist, "submitted_at": stamp,
                   "host": host, "work_dir": work_dir}, fh)
    return ok({
        "job_id": job_id,
        "job_dir": job_dir,
        "partition": partition,
        "note": f"Probe job {job_id} submitted to '{partition}'. Its result is read "
                "with the node-profile tool once the job ends.",
        "background_job": slurm_descriptor(job_id),
    })


@mcp.tool(**tool_caps(kind="slurm"))
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
    sched_host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    result = cli.run_argv(["scontrol", "show", "node", "-o"], TIMEOUT_READ,
                       host=sched_host)
    if result.get("status") != "ok":
        return err(result.get("stderr") or result.get("error", "scontrol failed"),
                   hint=f"Needs Slurm's scontrol on {_host_name(sched_host)}.")
    all_nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    by_name = {n["node"]: n for n in all_nodes}
    pending = harvest_probes(by_name, sched_host)

    ctx = cpu_facts.execution_context(has_slurm=True)
    if ctx["context"] == "in_allocation":
        here = by_name.get(socket.gethostname().split(".")[0])
        if here:
            cached = read_node_profile(here["node"], sched_host)
            if not cached or cached.get("slurm") != slurm_facts(here):
                record = node_profiles.local_profile_record(here)
                if record:
                    write_node_profile(here["node"], record, sched_host)

    nodes = all_nodes
    if node:
        nodes = [n for n in nodes if n["node"] == node]
    if partition:
        nodes = [n for n in nodes if partition in n["partitions"]]
    if not nodes:
        return ok({"node_types": [], "note": "No node matched.", "pending_probes": pending})

    host = node_profiles.host_identity()
    members: dict[tuple, list[dict]] = {}
    for n in nodes:
        members.setdefault(_hardware_key(n), []).append(n)
    types = _aggregate_node_types(nodes)
    full = bool(node) or len(types) <= FULL_PROFILE_TYPES
    out = []
    for t in types:
        group = members.get(_hardware_key(t), [])
        fresh: list[dict] = []
        stale: list[dict] = []
        for n in group:
            rec = read_node_profile(n["node"], sched_host)
            if not rec:
                continue
            (fresh if rec.get("slurm") == slurm_facts(n) else stale).append(rec)
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
                "matches_this_host": compare_with_host(profile, host),
                "profile": profile if full else summary(profile),
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

    Everything is served from the panel's caches, so a panel refreshing on a timer costs
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


def _hpc_panel_report(refresh: bool) -> dict:
    return panel.report(refresh, read_queue=lambda: slurm_queue(user_only=True),
                        read_partitions=slurm_partitions)



if __name__ == "__main__":
    mcp.run()
