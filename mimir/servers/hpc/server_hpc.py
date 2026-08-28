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
- Allocation/batch submission is separated and intended to be approval-gated by
  the client. ``sbatch_submit`` returns a ``background_job`` descriptor that the
  client watcher polls to completion via ``slurm_job_status``.
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

mcp = FastMCP(
    "HPCServer",
    debug=False,
    log_level="ERROR",
)

_TIMEOUT_READ = 15
_TIMEOUT_ALLOC = 20
_TIMEOUT_SUBMIT = 30
_MAX_OUTPUT = 128 * 1024

# Where async batch jobs stash their script + Slurm log (env-overridable for tests).
_HPC_JOBS_DIR = os.environ.get(
    "MIMIR_HPC_JOBS_DIR", os.path.expanduser("~/.cache/mimir_hpc/jobs"))


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


@mcp.tool()
def slurm_nodes(partition: str = "", states: str = "") -> dict:
    """List Slurm nodes and their resources/state.

    Args:
        partition: Optional partition filter.
        states: Optional state filter (comma-separated), e.g. 'idle,mix,alloc'.
    """
    p = f" -p {shlex.quote(partition)}" if partition else ""
    s = f" -t {shlex.quote(states)}" if states else ""
    cmd = f"sinfo -N -h{p}{s} -o '%N|%P|%t|%c|%m|%G'"
    result = _run_bash(cmd, _TIMEOUT_READ)
    if result["status"] != "ok":
        return err(result.get("stderr") or result.get("error", "sinfo node query failed"))
    rows = _parse_pipe_table(
        result.get("stdout", ""),
        ["node", "partition", "state", "cpus", "mem_mb", "gres"],
    )
    return ok({"nodes": rows, "count": len(rows)})


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


@mcp.tool()
def salloc_build_command(
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
    job_name: str = "mcp-interactive",
    extra_args: str = "",
) -> dict:
    """Build a validated salloc command without executing it.

    Use this first to preview requested resources before submitting.
    """
    if nodes < 1 or ntasks < 1 or cpus_per_task < 1:
        return err("nodes, ntasks, and cpus_per_task must be >= 1")
    if time and not _validate_time(time):
        return err("Invalid Slurm time format", hint="Use HH:MM:SS or D-HH:MM:SS")
    if mem and not _validate_mem(mem):
        return err("Invalid mem format", hint="Use values like 8G, 32000M, 1T")

    parts = ["salloc"]
    parts.append(f"--partition={shlex.quote(partition)}")
    parts.append(f"--nodes={nodes}")
    parts.append(f"--ntasks={ntasks}")
    parts.append(f"--cpus-per-task={cpus_per_task}")
    parts.append(f"--time={shlex.quote(time)}")
    parts.append(f"--job-name={shlex.quote(job_name)}")

    if account:
        parts.append(f"--account={shlex.quote(account)}")
    if qos:
        parts.append(f"--qos={shlex.quote(qos)}")
    if mem:
        parts.append(f"--mem={shlex.quote(mem)}")
    if gres:
        parts.append(f"--gres={shlex.quote(gres)}")
    if constraint:
        parts.append(f"--constraint={shlex.quote(constraint)}")
    if extra_args:
        # extra_args is pre-built by the user/agent so we pass it through,
        # but strip shell operators that could escape the salloc context.
        _FORBIDDEN = {";", "&&", "||", "$(", "`", ">", ">>"}
        if any(tok in extra_args for tok in _FORBIDDEN):
            return err("extra_args contains forbidden shell tokens.", hint="Pass plain salloc flags only.")
        parts.append(extra_args)

    cmd = " ".join(parts)
    return ok(
        {
            "command": cmd,
            "note": "Dry-run only. Use salloc_submit(command, confirm=True) to execute.",
        }
    )


@mcp.tool(**tool_caps(
    caps=[PLAN_BLOCKED, CLUSTER_SUBMIT], reversibility=IRREVERSIBLE, non_batch=True,
    risk_note="requests Slurm resource allocation",
))
def salloc_submit(command: str, confirm: bool = False, timeout_seconds: int = _TIMEOUT_ALLOC) -> dict:
    """Submit an salloc command.

    This is a sensitive action and should be approval-gated by the MCP client.

    Args:
        command: Full salloc command string.
        confirm: Must be true to execute.
        timeout_seconds: Max time to wait for command response.
    """
    if not command.strip().startswith("salloc "):
        return err("Only commands starting with 'salloc ' are allowed")
    if not confirm:
        return err(
            "Execution not confirmed.",
            hint="Set confirm=True only after user approval.",
        )

    timeout_seconds = max(5, min(timeout_seconds, 120))
    result = _run_bash(command, timeout_seconds)
    if result["status"] == "ok":
        return ok(
            {
                "stdout": result.get("stdout", ""),
                "stderr": result.get("stderr", ""),
                "returncode": result.get("returncode", 0),
            }
        )
    return err(
        result.get("stderr") or result.get("error", "salloc submission failed"),
        hint=(
            "Allocation may be pending/denied. Check partitions, account/qos, and requested resources. "
            "Use slurm_partitions() and slurm_queue() for diagnostics."
        ),
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

    res = _run_bash(f"sbatch {shlex.quote(script_path)}", _TIMEOUT_SUBMIT)
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
