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
  the model never hands over a command string — and launch it as argv, never through
  a shell. They are approval-gated client-side (``CLUSTER_SUBMIT``). The read-only
  queries do use a shell, for ``$USER`` expansion; their filters are quoted.
- ``sbatch_submit`` returns a ``background_job`` descriptor that the client watcher
  polls to completion via ``slurm_job_status``.
- ``slurm_cancel`` takes one job ID, refuses a job the user does not own, and is
  approval-gated (irreversible) but not ``CLUSTER_SUBMIT``: stopping a job spends no
  allocation, so the local-validation hold has nothing to protect there.
"""

import json
import os
import re
import shlex
import socket
import subprocess
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, CLUSTER_SUBMIT, BACKGROUNDABLE, IRREVERSIBLE
from responses import err, ok
from state_paths import state_dir
from slurm_nodes import (
    HARDWARE_FIELDS as _HARDWARE_FIELDS,
    aggregate_node_types as _aggregate_node_types,
    as_int as _as_int,
    clean_gres as _clean_gres,
    hardware_key as _hardware_key,
    parse_scontrol_nodes as _parse_scontrol_nodes,
)
from slurm_script import node_python_lines, sbatch_header, validate_target
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

# Where async batch jobs stash their script + Slurm log: under the agent's own state
# dir like every other persistent artefact, not a second home-relative location
# (env-overridable for tests).
_HPC_JOBS_DIR = os.environ.get("MIMIR_HPC_JOBS_DIR", os.path.join(state_dir(), "hpc_jobs"))


def _run_bash(script: str, timeout: int) -> dict:
    try:
        res = subprocess.run(
            ["bash", "-lc", script],
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


@mcp.tool()
def slurm_partitions() -> dict:
    """List Slurm partitions with key scheduling attributes."""
    cmd = "sinfo -h -o '%P|%a|%l|%D|%t|%c|%m'"
    result = _run_bash(cmd, _TIMEOUT_READ)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "sinfo failed"),
                    hint="Ensure Slurm commands are available on this host.")
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


@mcp.tool()
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


@mcp.tool()
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
        return err(result.get("stderr") or result.get("error", "squeue failed"))
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["job_id", "user", "state", "elapsed", "time_limit", "nodes", "reason_or_node", "name"],
    )
    return ok({"jobs": rows, "count": len(rows)})


def _run_argv(argv: list[str], timeout: int) -> dict:
    """Run a command as argv — no shell, so no argument can inject a second command."""
    try:
        res = subprocess.run(
            argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=timeout,
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
                 job_name: str, extra_args: str) -> tuple[list[str], dict | None]:
    """Build a validated salloc argv, or return the rejection."""
    if not partition.strip():
        return [], err("partition is required.", hint="Use slurm_partitions() to list them.")
    if nodes < 1 or ntasks < 1 or cpus_per_task < 1:
        return [], err("nodes, ntasks, and cpus_per_task must be >= 1")
    if time and not _validate_time(time):
        return [], err("Invalid Slurm time format", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not _validate_mem(mem):
        return [], err("Invalid mem format", hint="Use values like 8G, 32000M, 1T")

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
                        ("gres", gres), ("constraint", constraint)):
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


@mcp.tool(**tool_caps(
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
        job_name: Slurm job name.
        confirm: Must be True to execute; False returns the command as a preview.
        timeout_seconds: Max time to wait for the allocation response.
        extra_args: Additional salloc flags; every token must start with '-'.
    """
    argv, error = _salloc_argv(partition, account, qos, nodes, ntasks, cpus_per_task,
                               mem, time, gres, constraint, job_name, extra_args)
    if error:
        return error
    preview = shlex.join(argv)
    if not confirm:
        return err("Execution not confirmed.",
                   hint="Review the command, then call again with confirm=True after user approval.",
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


def _normalized_job_state(job_id: str) -> tuple[str, str]:
    """Map a Slurm job's state to the shared vocab: running|pending|done|crashed|unknown.

    Active jobs come from ``squeue``; finished jobs have left the queue, so their
    terminal state is read from ``sacct``. Returns ``(state, raw)``. Missing tools
    or an unknown job yield ``("unknown", "")`` — a terminal signal for the watcher.
    """
    q = _run_bash(f"squeue -j {shlex.quote(job_id)} -h -o '%T'", _TIMEOUT_READ)
    raw = (q.get("stdout") or "").strip().upper() if q.get("status") == "ok" else ""
    if raw in ("RUNNING", "COMPLETING"):
        return "running", raw
    if raw in ("PENDING", "CONFIGURING", "REQUEUED"):
        return "pending", raw

    s = _run_bash(f"sacct -j {shlex.quote(job_id)} -n -X -o State", _TIMEOUT_READ)
    sraw = ""
    if s.get("status") == "ok":
        lines = [ln.strip().upper() for ln in (s.get("stdout") or "").splitlines() if ln.strip()]
        sraw = lines[0] if lines else ""
    if sraw.startswith("COMPLETED"):
        return "done", sraw
    if any(sraw.startswith(p) for p in _SACCT_CRASH_PREFIXES):
        return "crashed", sraw
    if raw:                       # in the queue with a non-standard active state
        return "running", raw
    if sraw:                      # some other terminal state we don't classify
        return "crashed", sraw
    return "unknown", ""


@mcp.tool()
def slurm_job_status(job_id: str) -> dict:
    """Normalized status of a single Slurm job (poll target for background jobs).

    Returns ``state`` in running|pending|done|crashed|unknown (squeue for active
    jobs, sacct for finished ones) plus the raw Slurm state string.

    Args:
        job_id: The Slurm job ID to poll, as returned when the job was submitted.
    """
    if not str(job_id).strip():
        return err("job_id is required.")
    state, raw = _normalized_job_state(str(job_id).strip())
    return ok({"job_id": str(job_id).strip(), "state": state, "raw_state": raw})


# A job, or one task of a job array. Anything wider — a user, a partition, a name — is
# what `scancel` would also take, and is not something to approve from one card.
_JOB_ID_RE = re.compile(r"\d+(_\d+)?")


def _current_user() -> str:
    import getpass
    try:
        return getpass.getuser()
    except Exception:
        return os.environ.get("USER", "")


@mcp.tool(**tool_caps(
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
                   mem: str, wall_time: str, account: str, log_file: str) -> list[str]:
    lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --output={log_file}",
        f"#SBATCH --error={log_file}",
    ]
    if mem:
        lines.append(f"#SBATCH --mem={mem}")
    if gpus > 0:
        lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if account:
        lines.append(f"#SBATCH --account={account}")
    return lines


@mcp.tool(**tool_caps(
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
    job_name: str = "mimir-batch",
    nodes: int = 0,
    ntasks: int = 0,
    constraint: str = "",
    nodelist: str = "",
    exclusive: bool = False,
    confirm: bool = False,
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
        job_name: Slurm job name (default 'mimir-batch').
        nodes: Nodes to allocate (0 = scheduler default).
        ntasks: Tasks to run, e.g. MPI ranks (0 = scheduler default).
        constraint: Slurm feature expression selecting the kind of node.
        nodelist: Specific node(s), as a Slurm hostlist.
        exclusive: Reserve whole nodes (for timings).
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
    target_err = validate_target(constraint, nodelist, nodes or None, ntasks or None)
    if target_err:
        return err(target_err)
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    import time as _time
    job_dir = os.path.join(_HPC_JOBS_DIR, _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime()))
    os.makedirs(job_dir, exist_ok=True)
    log_file    = os.path.join(job_dir, "slurm.log")
    script_path = os.path.join(job_dir, "batch_script.sh")
    header = sbatch_header(
        job_name=job_name, partition=partition, cpus_per_task=cpus_per_task,
        wall_time=wall_time, log_file=log_file, mem=mem, gpus=gpus, account=account,
        nodes=nodes or None, ntasks=ntasks or None, constraint=constraint,
        nodelist=nodelist, exclusive=exclusive,
    )
    script = "\n".join(header + ["", command, ""]) + "\n"
    try:
        with open(script_path, "w") as fh:
            fh.write(script)
    except OSError as exc:
        return err(f"Could not write batch script: {exc}")

    res = _run_argv(["sbatch", script_path], _TIMEOUT_SUBMIT)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account/qos, and requested resources.")
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return err("Could not parse a job id from sbatch output.",
                   hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    job_id = match.group(1)
    try:
        with open(os.path.join(job_dir, "slurm_job_id"), "w") as fh:
            fh.write(job_id)
    except OSError:
        pass

    return ok({
        "job_id":       job_id,
        "job_dir":      job_dir,
        "batch_script": script_path,
        "log":          log_file,
        "partition":    partition,
        "note":         f"Slurm job {job_id} submitted to '{partition}'.",
        "background_job": {
            "server":    "hpc",
            "job_key":   job_id,
            "kind":      "slurm-batch",
            "status_op": {"tool": "slurm_job_status", "args": {"job_id": job_id}},
        },
    })


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


def _probe_script(job_dir: str, header: list[str]) -> str:
    q = shlex.quote
    fallback = [f'  echo "=={name}"; {cmd}' for name, cmd in _FALLBACK_SECTIONS]
    return "\n".join(header + [
        "",
        f"cd {q(job_dir)} || exit 1",
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


def _fallback_profile(text: str) -> dict:
    """A profile from the raw sections — what the node could say without Python."""
    sections, current = {}, None
    for line in text.splitlines():
        if line.startswith("==") and line[2:].strip() in dict(_FALLBACK_SECTIONS):
            current = line[2:].strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    sec = {k: "\n".join(v) for k, v in sections.items()}
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


def _write_node_profile(node: str, record: dict) -> None:
    os.makedirs(_NODE_PROFILES_DIR, exist_ok=True)
    tmp = os.path.join(_NODE_PROFILES_DIR, f".{node}.json.tmp")
    with open(tmp, "w") as fh:
        json.dump(record, fh, indent=2, ensure_ascii=False)
    os.replace(tmp, os.path.join(_NODE_PROFILES_DIR, f"{node}.json"))


def _harvest_probes(nodes_by_name: dict[str, dict]) -> list[dict]:
    """Turn finished probe jobs into node profiles; report the ones not finished.

    A probe is finished when its script wrote ``mode`` — its own last act, so the
    answer never depends on squeue/sacct having caught up. Only a probe without it
    asks the scheduler where it stands.
    """
    pending = []
    try:
        entries = sorted(os.listdir(_HPC_JOBS_DIR))
    except OSError:
        return pending
    for entry in entries:
        job_dir = os.path.join(_HPC_JOBS_DIR, entry)
        meta = _read_json(os.path.join(job_dir, _PROBE_MARKER))
        if not meta or meta.get("harvested"):
            continue
        mode_path = os.path.join(job_dir, "mode")
        if not os.path.exists(mode_path):
            state, raw = _normalized_job_state(str(meta.get("job_id", "")))
            if state in ("running", "pending"):
                pending.append({"job_id": meta.get("job_id"), "state": state,
                                "partition": meta.get("partition", "")})
                continue
            meta.update(harvested=True, outcome=f"no result ({raw or state})")
        else:
            with open(mode_path) as fh:
                mode = fh.read().strip()
            with open(os.path.join(job_dir, "node")) as fh:
                node = fh.read().strip()
            if mode == "full":
                profile = _read_json(os.path.join(job_dir, "profile.json")) or {}
                partial = not profile
            else:
                try:
                    with open(os.path.join(job_dir, "fallback.txt")) as fh:
                        profile = _fallback_profile(fh.read())
                except OSError:
                    profile = {}
                partial = True
            record = {
                "node": node, "probed_at": meta.get("submitted_at", ""),
                "job_id": meta.get("job_id"), "source": "probe job",
                "slurm": _slurm_facts(nodes_by_name.get(node, {})),
                "partial": partial, "profile": profile,
            }
            if partial:
                record["partial_reason"] = (
                    "No MIMIR Python runs on this node (no .venv-<os>-<arch> for it), so "
                    "only what the shell can read was collected: CPU, -march, GPU, OS. "
                    "Toolchains, modules and Python environments are missing.")
            _write_node_profile(node, record)
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


@mcp.tool(**tool_caps(
    caps=[BACKGROUNDABLE], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="Submits a short Slurm job (a few seconds on one node) to read the node's hardware.",
    label="Slurm node probe",
))
def slurm_probe_node(partition: str, constraint: str = "", nodelist: str = "",
                     account: str = "", confirm: bool = False) -> dict:
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
        confirm: Must be True to submit.
    """
    if not partition.strip():
        return err("partition is required.")
    target_err = validate_target(constraint, nodelist)
    if target_err:
        return err(target_err)
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    import time as _time
    stamp = _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime())
    job_dir = os.path.join(_HPC_JOBS_DIR, f"{stamp}-probe")
    suffix = 1
    while os.path.exists(job_dir):
        suffix += 1
        job_dir = os.path.join(_HPC_JOBS_DIR, f"{stamp}-probe{suffix}")
    os.makedirs(job_dir)
    log_file = os.path.join(job_dir, "slurm.log")
    header = sbatch_header(
        job_name="mimir-probe", partition=partition, cpus_per_task=1,
        wall_time="00:05:00", log_file=log_file, account=account, nodes=1, ntasks=1,
        constraint=constraint, nodelist=nodelist,
    )
    script_path = os.path.join(job_dir, "probe.sh")
    try:
        with open(script_path, "w") as fh:
            fh.write(_probe_script(job_dir, header))
    except OSError as exc:
        return err(f"Could not write the probe script: {exc}")

    res = _run_argv(["sbatch", script_path], _TIMEOUT_SUBMIT)
    if res.get("status") != "ok":
        return err(res.get("stderr") or res.get("error", "sbatch failed"),
                   hint="Check partition, account and constraint.")
    match = re.search(r"(\d+)", res.get("stdout", ""))
    if not match:
        return err("Could not parse a job id from sbatch output.",
                   hint=f"sbatch said: {res.get('stdout', '').strip()[:200]}")
    job_id = match.group(1)
    with open(os.path.join(job_dir, _PROBE_MARKER), "w") as fh:
        json.dump({"job_id": job_id, "partition": partition, "constraint": constraint,
                   "nodelist": nodelist, "submitted_at": stamp}, fh)
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
def slurm_node_profile(partition: str = "", node: str = "") -> dict:
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

    Which partition: the one the user named; otherwise the one the code and the job
    call for; when that is not clear, ask the user.

    Args:
        partition: Only node kinds in this partition.
        node: A single node name.
    """
    result = _run_argv(["scontrol", "show", "node", "-o"], _TIMEOUT_READ)
    if result.get("status") != "ok":
        return err(result.get("stderr") or result.get("error", "scontrol failed"),
                   hint="Needs Slurm's scontrol on this host.")
    all_nodes = _parse_scontrol_nodes(result.get("stdout", ""))
    by_name = {n["node"]: n for n in all_nodes}
    pending = _harvest_probes(by_name)

    ctx = cpu_facts.execution_context(has_slurm=True)
    if ctx["context"] == "in_allocation":
        here = by_name.get(socket.gethostname().split(".")[0])
        if here:
            cached = _read_json(os.path.join(_NODE_PROFILES_DIR, f"{here['node']}.json"))
            if not cached or cached.get("slurm") != _slurm_facts(here):
                record = _local_profile_record(here)
                if record:
                    _write_node_profile(here["node"], record)

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
            rec = _read_json(os.path.join(_NODE_PROFILES_DIR, f"{n['node']}.json"))
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


if __name__ == "__main__":
    mcp.run()
