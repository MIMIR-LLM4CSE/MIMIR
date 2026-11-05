"""MCP HPC Server: Slurm queries, batch submission, cancellation, node profiles.

Submitters build the command from arguments and run it as argv (``CLUSTER_SUBMIT``);
``sbatch_submit`` returns a ``background_job`` the client polls through
``slurm_job_status``; ``slurm_cancel`` takes one job the user owns. Modules go through
the bash server.
"""

import asyncio
import os
import shlex
import socket


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
from ...shared import slurm_site
from . import slurm_digest
from .slurm_nodes import aggregate_node_types, out_of_service, parse_pipe_table, usable_nodes
from ...shared.slurm_script import sbatch_header, validate_target
from ...shared import cpu_facts
from . import cli
from .cli import (
    TIMEOUT_ALLOC,
    TIMEOUT_READ,
    chosen_host,
    no_scheduler_hint,
    validate_comment,
    validate_mem,
    validate_time,
)
from . import jobs
from .jobs import (
    JOB_ID_RE,
    LOG_NAME,
    salloc_argv,
)
from . import node_profiles, panel
from .node_profiles import (
    FULL_PROFILE_TYPES,
    hardware_of,
    harvest_probes,
)

mcp = FastMCP(
    "HPCServer",
    debug=False,
    log_level="ERROR",
)


# ── Which scheduler a call goes to ────────────────────────────────────────────
# The inventory is ``shared/slurm_site``. A tool that acts resolves a host from its
# arguments (``slurm_site.resolve_host``) and refuses when the choice is open; one acting on an
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
        forget: A host to drop, with its cached hardware and its node profiles: one
            that was a typo, or a cluster you no longer have. The local scheduler
            cannot be dropped, being a fact rather than a record.
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
        slurm_site.record_host(host)
        # The explicit re-probe: a host named again is one the user may have changed.
        slurm_digest.forget_cluster_digest(host)
        added = {"scheduler": host, "note": f"sbatch answers on {host}; added."}

    dropped: dict | None = None
    if forget.strip():
        host = forget.strip()
        if host == slurm_site.LOCAL_NAME:
            dropped = {"scheduler": host,
                       "note": "The local scheduler is not a record to drop: it is "
                               "this host answering sbatch, and it would be back on "
                               "the next call."}
        else:
            gone = slurm_site.forget_host(host)
            slurm_digest.forget_cluster_digest(host)
            node_profiles.forget_node_profiles(host)
            panel.forget(host)
            dropped = {"scheduler": host,
                       "note": f"{host} dropped." if gone
                               else f"{host} was not in this workspace's schedulers."}

    known = slurm_site.hosts()
    listed = [{
        "scheduler": slurm_site.name_of(host),
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
                           f"{slurm_site.name_of(known[0])}.")
    else:
        payload["note"] = ("More than one, so every submission names its scheduler. "
                           "The choice follows the job's resources — the hardware on "
                           "each entry above — and the filesystem its inputs are on.")
    return ok(payload)


# ── What this user can submit to ──────────────────────────────────────────────
# The rule the Cluster drawer applies, from the same digest: the partitions sinfo lists
# without --all, up, not root-only and open to one of the user's accounts, and the
# nodes in one of them. Unknown access (the digest did not answer) filters nothing.

_PRIVATE_JOBS_NOTE = ("Only your own jobs are visible on this scheduler (PrivateData=jobs): "
                      "other users' pending jobs are not counted.")


def _access_rule(nodes: list[dict], host: str, *,
                 keep_out_of_service: bool = False) -> tuple[list[dict], dict]:
    """*nodes* narrowed to where this user can submit, and what that left out."""
    digest = slurm_digest.cached_cluster_digest(host)
    kept = usable_nodes(nodes, slurm_digest.usable_set(digest))
    left_out: dict = {}
    if len(nodes) - len(kept):
        left_out["outside_your_partitions"] = len(nodes) - len(kept)
    if digest["left_out"]["partitions"]:
        left_out["partitions"] = digest["left_out"]["partitions"]
    if not keep_out_of_service:
        in_service = [n for n in kept if not out_of_service(n["state"])]
        if len(kept) - len(in_service):
            left_out["out_of_service"] = len(kept) - len(in_service)
        kept = in_service
    return kept, left_out


def _partition_refusal(partition: str, left_out: dict) -> str:
    """Why *partition* matched nothing, when it is one the rule left out."""
    why = (left_out.get("partitions") or {}).get(partition)
    return f"Partition '{partition}' is not usable by you: {why}." if why else ""


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_partitions(scheduler: str = "", all_partitions: bool = False) -> dict:
    """List the Slurm partitions this user can submit to, and what already waits on each.

    Only the partitions sinfo lists without --all, up, not root-only and open to one of
    the user's accounts; `left_out` names the others and why. `queue` is squeue's
    reading per partition, every user: `waiting` jobs (pending on Resources or
    Priority, with the cpus and GPUs they ask), `held` ones (a dependency, a limit),
    `running`. Where PrivateData hides other users' jobs, `note` says so: the counts
    are then the user's own.

    Args:
        scheduler: Which scheduler to ask — 'local' for this machine's own, or
            a hostname. Omitted is fine when this workspace reaches only one;
            slurm_schedulers() lists them.
        all_partitions: List every partition sinfo shows, usable or not.
    """
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    rows = slurm_digest.read_partitions(host, TIMEOUT_READ, states=True)
    if rows is None:
        return err(f"sinfo did not answer on {slurm_site.name_of(host)}.", hint=no_scheduler_hint(host))
    digest = slurm_digest.cached_cluster_digest(host)
    payload: dict = {"scheduler": slurm_site.name_of(host)}
    usable = slurm_digest.usable_set(digest)
    if not all_partitions and usable is not None:
        rows = [row for row in rows if row["partition"] in usable]
        if digest["left_out"]["partitions"]:
            payload["left_out"] = digest["left_out"]["partitions"]
    payload.update({"partitions": rows, "count": len(rows)})
    queued = slurm_digest.read_queue_load(host, TIMEOUT_READ)
    if queued is not None:
        names = {row["partition"] for row in rows}
        payload["queue"] = {name: q for name, q in queued.items() if name in names}
        if digest.get("private_jobs"):
            payload["note"] = _PRIVATE_JOBS_NOTE
    return ok(payload)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_nodes(partition: str = "", states: str = "", node: str = "",
                detail: bool = False, scheduler: str = "",
                all_nodes: bool = False) -> dict:
    """Inventory the cluster's compute nodes: hardware, GPUs, and what is free right now.

    Read-only and instant — it reads Slurm's own node database, which is what actually
    governs placement, so it allocates nothing. Use it before a submission to pick the
    partition and resources that fit: note that **architecture varies between nodes** on
    a mixed cluster, so a binary built on the login node will not necessarily run on the
    node you submit to.

    Returns node *types* (nodes collapsed onto their hardware signature, with a count
    per state) unless you ask for detail or name a node — a per-node listing of a large
    cluster is mostly noise.

    Only the nodes a job of this user can land on: in a partition they can submit to
    (listed by sinfo, up, not root-only, open to one of their accounts) and in service
    (not down, drained, reserved or unresponsive — kept when `states` asks for them).
    `left_out` counts the rest; a named node is always described.

    Args:
        partition: Only nodes in this partition.
        states: Comma-separated state filter, e.g. 'idle,mix,alloc'.
        node: A single node name; implies detail.
        detail: List every matching node individually instead of aggregating.
        scheduler: Which scheduler to ask — 'local' for this machine's own, or
            a hostname. Omitted is fine when this workspace reaches only one;
            slurm_schedulers() lists them.
        all_nodes: Every node Slurm knows, usable or not.
    """
    host, refusal = chosen_host(scheduler)
    if refusal:
        return refusal
    read, source = slurm_digest.read_nodes(host, TIMEOUT_READ)
    if not read:
        return err(f"Neither scontrol nor sinfo listed nodes on {slurm_site.name_of(host)}.",
                   hint=no_scheduler_hint(host))
    nodes = read

    left_out: dict = {}
    if node:
        wanted = {n.strip() for n in node.split(",") if n.strip()}
        nodes = [n for n in nodes if n["node"] in wanted]
    elif not all_nodes:
        nodes, left_out = _access_rule(nodes, host, keep_out_of_service=bool(states))
    if partition:
        nodes = [n for n in nodes if partition in n["partitions"]]
    if states:
        wanted = {s.strip().lower() for s in states.split(",") if s.strip()}
        nodes = [n for n in nodes if any(part.lower() in wanted for part in (n["state"] or "").split("+"))]

    if not nodes:
        return ok({"nodes": [], "count": 0,
                   "note": _partition_refusal(partition, left_out)
                           or "No node matched the filters."})
    if detail or node:
        payload = {"nodes": nodes, "count": len(nodes)}
    else:
        types = aggregate_node_types(nodes)
        payload = {
            "node_types":   types,
            "type_count":   len(types),
            "nodes_total":  len(nodes),
            "architectures": sorted({n["arch"] for n in nodes if n["arch"]}),
            "note": "Aggregated by hardware signature; pass detail=True or node='<name>' for individual nodes.",
        }
    if left_out:
        payload["left_out"] = left_out
    if source == "sinfo":
        payload["degraded"] = slurm_digest.DEGRADED
    payload["scheduler"] = slurm_site.name_of(host)
    return ok(payload)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_queue(user_only: bool = True, states: str = "", scheduler: str = "") -> dict:
    """List jobs from Slurm queue.

    One scheduler's queue, because that is what a queue is. What this workspace runs
    elsewhere is in ``unwatched`` when nothing is tracking it, and slurm_schedulers()
    says where else to look.

    Args:
        user_only: If true, show only jobs for the current user. Where PrivateData
            hides other users' jobs, false lists the same jobs and `note` says so.
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
        return cli.failed(result, "squeue", host)
    rows = parse_pipe_table(
        result.get("stdout", ""),
        ["job_id", "user", "state", "elapsed", "time_limit", "nodes", "reason_or_node", "name"],
    )
    # Which rows this workspace watches: a job is tracked by its descriptor, keyed by host
    # (ids are per cluster). The unwatched set goes to the client for its end-of-turn nudge.
    recorded = slurm_site.recorded_jobs()
    ids = [str(row.get("job_id") or "") for row in rows]
    for row, job_id in zip(rows, ids):
        row["watched"] = slurm_site.covered_by(job_id, recorded, host)
    payload = {"jobs": rows, "count": len(rows), "scheduler": slurm_site.name_of(host)}
    if not user_only and slurm_digest.cached_cluster_digest(host).get("private_jobs"):
        payload["note"] = _PRIVATE_JOBS_NOTE
    # Per row above, by array here: eight task rows name one descriptor to write, and
    # listing them one by one would ask for eight calls that all land on "already
    # recorded".
    missing = slurm_site.unwatched(ids, host=host)
    if missing:
        payload["unwatched"] = missing
        payload["unwatched_hint"] = slurm_site.unwatched_hint({host: missing})
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
    return jobs.status(job_id, scheduler)


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
    return jobs.adopt(job_id, note, scheduler)


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
    can be cancelled: through the cluster tools, or by a launcher or script one of its
    own commands ran (recognised by when it was submitted). One from another
    conversation or workspace, or one no longer in the queue, is refused.

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

    return jobs.cancel(job_id, scheduler)


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

    header = sbatch_header(
        job_name=job_name, partition=partition, cpus_per_task=cpus_per_task,
        wall_time=wall_time, log_file=os.path.join(work_dir, LOG_NAME),
        work_dir=work_dir, mem=mem, gpus=gpus, account=account,
        nodes=nodes or None, ntasks=ntasks or None, constraint=constraint,
        nodelist=nodelist, exclusive=exclusive, comment=comment,
    )
    return jobs.submit_batch(host, work_dir, command, header,
                             partition=partition, comment=comment)


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

    return node_profiles.submit_probe(host, partition, constraint, nodelist, account)


@mcp.tool(**tool_caps(kind="slurm"))
def slurm_node_profile(partition: str = "", node: str = "",
                       scheduler: str = "") -> dict:
    """What the compute nodes are, as the machines code will be built for and run on.

    Read-only and instant. For each kind of node (Slurm's hardware signature) in
    *partition* — or just *node* — among the partitions this user can submit to
    (`left_out` counts the rest), returns Slurm's facts plus, when a node of that
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
    read, source = slurm_digest.read_nodes(sched_host, TIMEOUT_READ)
    # The hardware is the question, and only scontrol gives the architecture.
    if read is None or source != "scontrol":
        return err(f"scontrol did not list the nodes on {slurm_site.name_of(sched_host)}.",
                   hint=f"Needs Slurm's scontrol on {slurm_site.name_of(sched_host)}.")
    all_nodes = read
    by_name = {n["node"]: n for n in all_nodes}
    pending = harvest_probes(by_name, sched_host)

    ctx = cpu_facts.execution_context(has_slurm=True)
    if ctx["context"] == "in_allocation":
        node_profiles.refresh_own_profile(by_name, sched_host)

    left_out: dict = {}
    if node:
        nodes = [n for n in all_nodes if n["node"] == node]
    else:
        # The hardware is the question, so a drained node of a kind still describes it.
        nodes, left_out = _access_rule(all_nodes, sched_host, keep_out_of_service=True)
    if partition:
        nodes = [n for n in nodes if partition in n["partitions"]]
    if not nodes:
        return ok({"node_types": [], "pending_probes": pending,
                   "note": _partition_refusal(partition, left_out) or "No node matched."})

    host = node_profiles.host_identity()
    types = aggregate_node_types(nodes)
    full = bool(node) or len(types) <= FULL_PROFILE_TYPES
    out = node_profiles.describe_node_types(nodes, types, sched_host, host, full)

    payload = {
        "execution_context": ctx,
        "host": {"hostname": socket.gethostname(), "arch": host["arch"],
                 "cpu_model": host["cpu"].get("model", ""), "os": host["os"]},
        "node_types": out,
        "type_count": len(out),
    }
    if left_out:
        payload["left_out"] = left_out
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

    The lines are the user's own jobs; the groups are every scheduler the workspace
    reaches, each with the room on the partitions this user may submit to — free cores
    and GPUs from the node listing, crossed with what squeue says already waits — and
    the kinds of node it is made of, down, drained, reserved or closed partitions left
    out and counted. A nested shape because the question is nested: which cluster, then
    which kind of machine on it, then what that machine actually is.

    Everything is served from the panel's caches, so a panel refreshing on a timer costs
    one ``squeue`` of the user's jobs every few seconds at most, one node listing and
    one ``squeue`` per scheduler every half minute, and one digest per scheduler for the
    life of the server. A host with nothing to say returns
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
    return panel.report(refresh, read_queue=lambda: slurm_queue(user_only=True))



if __name__ == "__main__":
    mcp.run()
