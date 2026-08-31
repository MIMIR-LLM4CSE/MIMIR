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


# Node facts we read out of `scontrol show node -o`. Values containing spaces (OS,
# Reason) are deliberately not among them, so a simple `KEY=<non-space>` scan is enough.
_NODE_FIELDS = (
    "NodeName", "Arch", "CPUTot", "CPUAlloc", "CPULoad", "RealMemory", "AllocMem",
    "FreeMem", "Sockets", "CoresPerSocket", "ThreadsPerCore", "Gres",
    "AvailableFeatures", "State", "Partitions",
)


def _as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _clean_gres(value: str) -> str:
    """'gpu:a100:8(S:1,3,5,7)' -> 'gpu:a100:8'; '(null)' -> ''."""
    if not value or value == "(null)":
        return ""
    return re.sub(r"\(S:[^)]*\)", "", value)


def _parse_scontrol_nodes(stdout: str) -> list[dict]:
    nodes = []
    for line in stdout.splitlines():
        if "NodeName=" not in line:
            continue
        raw = {k: v for k, v in re.findall(r"\b(\w+)=([^\s]+)", line)}
        gres = _clean_gres(raw.get("Gres", ""))
        cpu_tot, cpu_alloc = _as_int(raw.get("CPUTot", "")), _as_int(raw.get("CPUAlloc", ""))
        features = raw.get("AvailableFeatures", "")
        nodes.append({
            "node":             raw.get("NodeName", ""),
            "arch":             raw.get("Arch", ""),
            "state":            raw.get("State", ""),
            "partitions":       [p for p in raw.get("Partitions", "").split(",") if p],
            "cpus":             cpu_tot,
            "cpus_allocated":   cpu_alloc,
            "cpus_free":        (cpu_tot - cpu_alloc) if None not in (cpu_tot, cpu_alloc) else None,
            "cpu_load":         raw.get("CPULoad", ""),
            "sockets":          _as_int(raw.get("Sockets", "")),
            "cores_per_socket": _as_int(raw.get("CoresPerSocket", "")),
            "threads_per_core": _as_int(raw.get("ThreadsPerCore", "")),
            "mem_mb":           _as_int(raw.get("RealMemory", "")),
            "mem_free_mb":      _as_int(raw.get("FreeMem", "")),
            "gres":             gres,
            "features":         "" if features == "(null)" else features,
        })
    return nodes


def _aggregate_node_types(nodes: list[dict]) -> list[dict]:
    """Collapse nodes onto their hardware signature.

    A 124-node cluster listed one row per node buries the answer in noise; what the
    caller is choosing between is the handful of *kinds* of machine, and how much of
    each is free right now.
    """
    groups: dict[tuple, dict] = {}
    for n in nodes:
        key = (n["arch"], n["cpus"], n["mem_mb"], n["gres"],
               n["sockets"], n["cores_per_socket"], n["threads_per_core"], n["features"])
        g = groups.setdefault(key, {
            "arch": n["arch"], "cpus": n["cpus"], "mem_mb": n["mem_mb"],
            "mem_gb": round(n["mem_mb"] / 1024, 1) if n["mem_mb"] else None,
            "gres": n["gres"], "sockets": n["sockets"],
            "cores_per_socket": n["cores_per_socket"], "threads_per_core": n["threads_per_core"],
            "features": n["features"], "partitions": set(), "nodes_total": 0,
            "by_state": {}, "cpus_free_total": 0, "example_nodes": [],
        })
        g["partitions"].update(n["partitions"])
        g["nodes_total"] += 1
        state = (n["state"] or "UNKNOWN").split("+")[0].lower()
        g["by_state"][state] = g["by_state"].get(state, 0) + 1
        if n["cpus_free"]:
            g["cpus_free_total"] += n["cpus_free"]
        if len(g["example_nodes"]) < 3:
            g["example_nodes"].append(n["node"])

    out = []
    for g in groups.values():
        g["partitions"] = sorted(g["partitions"])
        out.append(g)
    # Most immediately usable first: idle nodes, then raw size.
    out.sort(key=lambda g: (-g["by_state"].get("idle", 0), -(g["cpus"] or 0)))
    return out


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
    """
    if not str(job_id).strip():
        return err("job_id is required.")
    state, raw = _normalized_job_state(str(job_id).strip())
    return ok({"job_id": str(job_id).strip(), "state": state, "raw_state": raw})


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
    confirm: bool = False,
) -> dict:
    """Submit *command* as a non-blocking Slurm batch job (sensitive, backgroundable).

    Unlike salloc_submit (synchronous, interactive), this returns immediately with a
    ``job_id`` and a ``background_job`` descriptor so the run is tracked off the
    critical path: end your turn and you are auto-resumed when the job finishes
    (poll manually with slurm_job_status(job_id) if needed).

    Args:
        command: The shell command line to run inside the batch job.
        partition: Slurm partition (required).
        cpus_per_task: CPU cores to allocate (default 1).
        gpus: GPUs per node (0 = CPU-only).
        mem: Memory in Slurm format (e.g. '8G'); empty = scheduler default.
        wall_time: Wall-clock limit HH:MM:SS or D-HH:MM:SS (default '01:00:00').
        account: Slurm account to charge (optional).
        job_name: Slurm job name (default 'mimir-batch').
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
    if not confirm:
        return err("Submission not confirmed.",
                   hint="Set confirm=True only after user approval.")

    import time as _time
    job_dir = os.path.join(_HPC_JOBS_DIR, _time.strftime("%Y%m%dT%H%M%SZ", _time.gmtime()))
    os.makedirs(job_dir, exist_ok=True)
    log_file    = os.path.join(job_dir, "slurm.log")
    script_path = os.path.join(job_dir, "batch_script.sh")
    header = _sbatch_header(job_name, partition, cpus_per_task, gpus, mem,
                            wall_time, account, log_file)
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


if __name__ == "__main__":
    mcp.run()
