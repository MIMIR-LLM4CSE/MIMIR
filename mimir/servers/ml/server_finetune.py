"""MCP LLM Fine-Tuning Server
==========================
Manage LoRA fine-tuning runs for causal language models.

All state lives under ~/.cache/ft_llm/:
  config.json         — active configuration
  runs/<timestamp>/   — one directory per run, containing:
      config.json     — snapshot of config used at launch
      start_time      — Unix timestamp written before launch
      pid             — PID of the training process (local runs only)
      slurm_job_id    — Slurm job ID (HPC runs only)
      batch_script.sh — generated sbatch script (HPC runs only)
      run.log         — combined stdout/stderr of the runner
      metrics.json    — final metrics written by runner on success
      model/          — saved adapter weights (default output_dir)
  runs/active         — symlink to the most recently launched run dir

Two execution backends are supported:
- ft_run          : local detached subprocess (Popen + start_new_session).
- ft_run_slurm    : Slurm batch job (sbatch); requires Slurm on the host.

Both backends write run.log in the same location, so ft_log_read(),
ft_metrics_parse(), and ft_runs_list() work identically regardless of how
the run was started.

Safety model:
- Read-only tools have no side effects.
- ft_config_set, ft_run, ft_run_slurm, and ft_stop are sensitive and
  approval-gated.
- ft_run and ft_run_slurm are non-blocking: both return immediately after
  launching or submitting. Use ft_status() and ft_log_read() to monitor.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, PLAN_BLOCKED, CLUSTER_SUBMIT, IRREVERSIBLE, RECOVERABLE
from responses import err, ok

mcp = FastMCP(
    "FinetuneServer",
    debug=False,
    log_level="ERROR",
)

_CACHE_DIR        = os.path.expanduser("~/.cache/ft_llm")
_RUNS_DIR         = os.path.join(_CACHE_DIR, "runs")
_CONFIG_FILE      = os.path.join(_CACHE_DIR, "config.json")
_SERVERS_DIR      = os.path.dirname(os.path.abspath(__file__))
_RUNNER           = os.path.join(_SERVERS_DIR, "_ft_runner.py")
# Canonical lives in ~/.cache — outside MCP_FILES_ROOT — so the agent
# cannot overwrite it via server_files write tools.
_RUNNER_CANONICAL = os.path.join(_CACHE_DIR, "_ft_runner.canonical.py")
# Seed file shipped with the package (used for first-run bootstrap only).
_RUNNER_SEED      = os.path.join(_SERVERS_DIR, "_ft_runner.canonical.py")
_MAX_LOG          = 128 * 1024  # bytes read at most from run.log


def _restore_runner() -> None:
    """Reset _ft_runner.py from the canonical copy at server startup.

    Called once at module load time so that any modifications made by the
    agent during a previous session are discarded when a new session begins.

    The canonical copy lives in ~/.cache/ft_llm/ which is outside
    MCP_FILES_ROOT, so server_files write tools cannot touch it.

    Bootstrap: if the ~/.cache canonical does not yet exist, it is seeded
    from the package-shipped _ft_runner.canonical.py (first run only).
    """
    import hashlib
    import shutil
    os.makedirs(_CACHE_DIR, exist_ok=True)
    # Seed canonical into ~/.cache on first run.
    if not os.path.isfile(_RUNNER_CANONICAL) and os.path.isfile(_RUNNER_SEED):
        try:
            shutil.copy2(_RUNNER_SEED, _RUNNER_CANONICAL)
        except OSError:
            return  # can't seed — skip everything
    if not os.path.isfile(_RUNNER_CANONICAL):
        return  # no canonical available — skip silently
    # Warn if the packaged seed has diverged from the cached canonical.
    if os.path.isfile(_RUNNER_SEED):
        try:
            seed_hash = hashlib.sha256(open(_RUNNER_SEED, "rb").read()).hexdigest()
            canon_hash = hashlib.sha256(open(_RUNNER_CANONICAL, "rb").read()).hexdigest()
            if seed_hash != canon_hash:
                import logging
                logging.getLogger(__name__).warning(
                    "Canonical runner in cache differs from packaged seed. "
                    "Use ft_runner_promote() to update or delete %s to re-seed.",
                    _RUNNER_CANONICAL,
                )
        except OSError:
            pass
    try:
        shutil.copy2(_RUNNER_CANONICAL, _RUNNER)
    except OSError:
        pass  # non-fatal


_restore_runner()

_DEFAULT_CONFIG: dict = {
    "model_id":          "distilgpt2",
    "train_data":        "",
    "val_data":          "",
    "lora_r":            8,
    "lora_alpha":        16,
    "target_modules":    ["c_attn", "c_proj"],
    "lora_dropout":      0.05,
    "batch_size":        8,
    "lr":                2e-4,
    "epochs":            3,
    "max_length":        256,
    "output_dir":        "",
    "precision":         "auto",
    "python_executable": "",
}


# ── helpers ─────────────────────────────────────────────────────────────────

def _active_run_dir() -> str | None:
    """Return the most-recently-launched run directory, or None."""
    link = os.path.join(_RUNS_DIR, "active")
    if os.path.islink(link):
        target = os.readlink(link)
        if not os.path.isabs(target):
            target = os.path.join(_RUNS_DIR, target)
        if os.path.isdir(target):
            return target
    return None


def _pid_path(run_dir: str) -> str:
    return os.path.join(run_dir, "pid")


def _log_path(run_dir: str) -> str:
    return os.path.join(run_dir, "run.log")


def _slurm_job_id_path(run_dir: str) -> str:
    return os.path.join(run_dir, "slurm_job_id")


def _read_slurm_job_id(run_dir: str) -> int | None:
    p = _slurm_job_id_path(run_dir)
    if os.path.isfile(p):
        try:
            return int(open(p).read().strip())
        except (ValueError, OSError):
            pass
    return None


def _squeue_state(job_id: int) -> str:
    """Query squeue for one job; return 'running'|'pending'|'done'|'crashed'."""
    try:
        res = subprocess.run(
            ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=10,
        )
        state = res.stdout.strip().upper()
        if state in ("RUNNING", "COMPLETING"):
            return "running"
        if state in ("PENDING", "CONFIGURING", "REQUEUED"):
            return "pending"
        if state in ("COMPLETED",):
            return "done"
        # FAILED, CANCELLED, TIMEOUT, NODE_FAIL, OUT_OF_MEMORY, etc.
        if state:
            return "crashed"
        # empty output means the job is no longer in the queue
        return "done"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return "unknown"

def _read_pid(run_dir: str) -> int | None:
    p = _pid_path(run_dir)
    if os.path.isfile(p):
        try:
            return int(open(p).read().strip())
        except (ValueError, OSError):
            pass
    return None


def _is_running(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _load_config() -> dict:
    base = dict(_DEFAULT_CONFIG)
    if os.path.isfile(_CONFIG_FILE):
        try:
            with open(_CONFIG_FILE) as fh:
                base.update(json.load(fh))
        except (json.JSONDecodeError, OSError):
            pass
    return base


def _tail_log(run_dir: str, n: int) -> list[str]:
    lp = _log_path(run_dir)
    if not os.path.isfile(lp):
        return []
    try:
        with open(lp, errors="replace") as fh:
            lines = fh.readlines()
        return [line.rstrip("\n") for line in lines[-n:]]
    except OSError:
        return []


def _run_status(run_dir: str) -> dict:
    """Return state ('running'|'pending'|'done'|'crashed'), PID/job_id, and elapsed_s."""
    metrics_path  = os.path.join(run_dir, "metrics.json")

    # ── Slurm run ─────────────────────────────────────────────────────────────
    job_id = _read_slurm_job_id(run_dir)
    if job_id is not None:
        slurm_state = _squeue_state(job_id)
        # Fall back to metrics.json for finished jobs that left the queue
        if slurm_state == "done" and not os.path.isfile(metrics_path):
            slurm_state = "crashed"
        state = slurm_state
    else:
        # ── local subprocess ──────────────────────────────────────────────────
        pid = _read_pid(run_dir)
        if pid and _is_running(pid):
            state = "running"
        elif os.path.isfile(metrics_path):
            state = "done"
        else:
            state = "crashed"

    elapsed: float | None = None
    start_file = os.path.join(run_dir, "start_time")
    if os.path.isfile(start_file):
        try:
            elapsed = round(time.time() - float(open(start_file).read().strip()), 1)
        except (ValueError, OSError):
            pass

    pid_val = _read_pid(run_dir) if job_id is None else None
    return {"state": state, "pid": pid_val, "slurm_job_id": job_id, "elapsed_s": elapsed}


# ── tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
def ft_config_get() -> dict:
    """Return current fine-tuning configuration.

    Returns defaults merged with any values set by ft_config_set.
    """
    cfg = _load_config()
    config_file = _CONFIG_FILE if os.path.isfile(_CONFIG_FILE) else None
    return ok({"config": cfg, "config_file": config_file})


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=RECOVERABLE))
def ft_config_set(
    model_id: str = "",
    train_data: str = "",
    val_data: str = "",
    lora_r: int = 0,
    lora_alpha: int = 0,
    target_modules: list[str] | None = None,
    lora_dropout: float = -1.0,
    batch_size: int = 0,
    lr: float = -1.0,
    epochs: int = 0,
    max_length: int = 0,
    output_dir: str = "",
    precision: str = "",
    python_executable: str = "",
) -> dict:
    """Update fine-tuning configuration (sensitive). Only provided fields are changed.

    Args:
        model_id: HuggingFace model identifier (e.g. 'distilgpt2', 'gpt2').
        train_data: Absolute path to the training text file (.txt).
        val_data: Absolute path to the validation text file (optional).
        lora_r: LoRA rank (positive integer, e.g. 8 or 16).
        lora_alpha: LoRA alpha scaling factor (e.g. 16 or 32).
        target_modules: Module names to inject LoRA into (e.g. ['c_attn', 'c_proj']).
        lora_dropout: Dropout probability inside LoRA layers (0.0–1.0).
        batch_size: Per-device training batch size.
        lr: Learning rate (e.g. 2e-4).
        epochs: Number of full passes over the training dataset.
        max_length: Maximum token sequence length for truncation.
        output_dir: Where to save adapter checkpoints (defaults to run_dir/model).
        precision: Training float precision. 'auto' (default) = fp16 on CUDA else fp32;
            'fp16' = fastest, ~half VRAM; 'bf16' = better range, needs Ampere+;
            'fp32' = most stable, highest VRAM; 'int8' = bitsandbytes quantized, lowest VRAM.
        python_executable: Absolute path to the Python interpreter used to launch
            _ft_runner.py (e.g. '/opt/conda/envs/train/bin/python'); useful for Slurm
            runs on a different environment. Empty = same interpreter as the server.
    """
    cfg = _load_config()

    if model_id:
        cfg["model_id"] = model_id
    if train_data:
        cfg["train_data"] = train_data
    if val_data:
        cfg["val_data"] = val_data
    if lora_r > 0:
        cfg["lora_r"] = lora_r
    if lora_alpha > 0:
        cfg["lora_alpha"] = lora_alpha
    if target_modules is not None:
        cfg["target_modules"] = target_modules
    if 0.0 <= lora_dropout <= 1.0:
        cfg["lora_dropout"] = lora_dropout
    if batch_size > 0:
        cfg["batch_size"] = batch_size
    if lr > 0:
        cfg["lr"] = lr
    if epochs > 0:
        cfg["epochs"] = epochs
    if max_length > 0:
        cfg["max_length"] = max_length
    if output_dir:
        cfg["output_dir"] = output_dir
    if precision and precision in ("auto", "fp16", "bf16", "fp32", "int8"):
        cfg["precision"] = precision
    elif precision:
        return err(
            f"Invalid precision '{precision}'.",
            hint="Use one of: auto, fp16, bf16, fp32, int8.",
        )
    if python_executable:
        if not os.path.isfile(python_executable):
            return err(
                f"python_executable not found: {python_executable}",
                hint="Provide an absolute path to an existing Python interpreter.",
            )
        cfg["python_executable"] = python_executable

    os.makedirs(_CACHE_DIR, exist_ok=True)
    with open(_CONFIG_FILE, "w") as fh:
        json.dump(cfg, fh, indent=2)

    return ok({"config": cfg, "config_file": _CONFIG_FILE})


@mcp.tool()
def ft_data_inspect(train_path: str, val_path: str = "") -> dict:
    """Inspect training (and optionally validation) data files.

    Reports existence, line count, file size, and a 3-line sample.

    Args:
        train_path: Path to training text file.
        val_path: Path to validation text file (optional).
    """
    def _inspect(path: str) -> dict:
        if not path:
            return {"provided": False}
        abs_path = os.path.abspath(path)
        if not os.path.isfile(abs_path):
            return {"path": abs_path, "exists": False}
        try:
            with open(abs_path, errors="replace") as fh:
                lines = fh.readlines()
            return {
                "path": abs_path,
                "exists": True,
                "line_count": len(lines),
                "size_bytes": os.path.getsize(abs_path),
                "sample": [line.rstrip("\n") for line in lines[:3]],
            }
        except OSError as exc:
            return {"path": abs_path, "exists": True, "error": str(exc)}

    result: dict = {"train": _inspect(train_path)}
    if val_path:
        result["validation"] = _inspect(val_path)
    return ok(result)


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True))
def ft_run(confirm: bool = False) -> dict:
    """Launch a LoRA fine-tuning run in the background (sensitive, confirm-gated).

    Creates a timestamped run directory, snapshots the config, starts
    _ft_runner.py as a detached subprocess, and returns immediately.
    Use ft_status() and ft_log_read() to monitor progress.

    Only one run may be active at a time. Call ft_stop(confirm=True) first
    to interrupt an in-progress run before starting a new one.

    Optimization loop: after the run completes, call ft_metrics_parse() to
    collect memory, loss, and performance metrics. If the results do not meet
    the user's constraints, you may read _ft_runner.py with read_file_lines (or
    search tools), modify it with replace_in_file to change any aspect of the
    training (model dtype, gradient checkpointing, optimizer, architecture,
    data preprocessing, etc.), then call ft_run again. Repeat until the
    constraints are satisfied. Configuration knobs (precision, batch_size,
    lora_r, lr, epochs) can be changed without file edits via ft_config_set.

    Args:
        confirm: Must be set to True to start training.
    """
    if not confirm:
        return err(
            "Execution not confirmed.",
            hint="Set confirm=True only after obtaining user approval.",
        )

    # ── guard: already running? ──────────────────────────────────────────────
    active = _active_run_dir()
    if active:
        pid = _read_pid(active)
        if pid and _is_running(pid):
            return err(
                f"Training already running (pid={pid}).",
                hint="Use ft_stop(confirm=True) to stop it first, or ft_status() to check.",
            )

    # ── validate config ───────────────────────────────────────────────────────
    cfg = _load_config()
    if not cfg.get("train_data"):
        return err(
            "train_data is not configured.",
            hint="Call ft_config_set(train_data='...') first.",
        )
    if not os.path.isfile(cfg["train_data"]):
        return err(f"train_data file not found: {cfg['train_data']}")
    if cfg.get("val_data") and not os.path.isfile(cfg["val_data"]):
        return err(f"val_data file not found: {cfg['val_data']}")

    # ── create run directory ──────────────────────────────────────────────────
    os.makedirs(_RUNS_DIR, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(_RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)

    # default output_dir when not set
    if not cfg.get("output_dir"):
        cfg["output_dir"] = os.path.join(run_dir, "model")

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(run_dir, "start_time"), "w") as fh:
        fh.write(str(time.time()))

    # ── launch ───────────────────────────────────────────────────────────────
    log_file = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable
    with open(log_file, "w") as log_fh:
        proc = subprocess.Popen(
            [python_exe, _RUNNER, "--run-dir", run_dir],
            stdout=log_fh,
            stderr=subprocess.STDOUT,
            close_fds=True,
            start_new_session=True,
        )

    with open(_pid_path(run_dir), "w") as fh:
        fh.write(str(proc.pid))

    # ── update 'active' symlink atomically ───────────────────────────────────
    link     = os.path.join(_RUNS_DIR, "active")
    tmp_link = link + ".tmp"
    if os.path.lexists(tmp_link):
        os.unlink(tmp_link)
    os.symlink(run_dir, tmp_link)
    os.replace(tmp_link, link)

    return ok({
        "run_dir": run_dir,
        "pid":     proc.pid,
        "log":     log_file,
        "config":  cfg,
        "note":    "Training started in background. Use ft_status() to monitor.",
    })


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED, CLUSTER_SUBMIT], reversibility=IRREVERSIBLE, non_batch=True))
def ft_run_slurm(
    partition: str,
    gpus: int = 1,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    job_name: str = "ft_llm",
    confirm: bool = False,
) -> dict:
    """Submit a LoRA fine-tuning run as a Slurm batch job (sensitive, confirm-gated).

    Generates a self-contained sbatch script that invokes _ft_runner.py with the
    current configuration, then submits it via `sbatch`. Training output is
    written to run.log inside the run directory, so ft_log_read(),
    ft_metrics_parse(), and ft_runs_list() work identically to local runs.

    Use ft_run(confirm=True) instead when Slurm is not available or the run is
    short enough to execute locally.

    Args:
        partition: Slurm partition name (required, e.g. 'gpu', 'A100').
        gpus: Number of GPUs to request per node (default 1).
        cpus_per_task: CPU cores to allocate (default 8).
        mem: Memory per node (default '32G'). Any Slurm-accepted format (e.g. '32G', '64000M').
        wall_time: Wall-clock time limit in HH:MM:SS format (default '04:00:00').
        account: Slurm account/project to charge (optional).
        job_name: Slurm job name shown in squeue (default 'ft_llm').
        confirm: Must be set to True to submit.
    """
    if not confirm:
        return err(
            "Submission not confirmed.",
            hint="Set confirm=True only after obtaining user approval.",
        )

    # ── validate partition ────────────────────────────────────────────────────
    if not partition.strip():
        return err("partition is required.")
    if gpus < 0 or cpus_per_task < 1:
        return err("gpus must be >= 0 and cpus_per_task >= 1.")
    if not re.fullmatch(r"\d+[KMGTP]?", mem.upper()):
        return err("Invalid mem format.", hint="Use values like '32G', '64000M', '1T'.")
    if not re.fullmatch(r"\d+(-\d{1,2}(:\d{2}(:\d{2})?)?|(:\d{2}){0,2})", wall_time):
        return err("Invalid wall_time format.", hint="Use HH:MM:SS or D-HH:MM:SS.")

    # ── guard: already running? ───────────────────────────────────────────────
    active = _active_run_dir()
    if active:
        status = _run_status(active)
        if status["state"] in ("running", "pending"):
            jid = status.get("slurm_job_id")
            pid = status.get("pid")
            tag = f"slurm_job_id={jid}" if jid else f"pid={pid}"
            return err(
                f"A run is already active ({tag}).",
                hint="Use ft_stop(confirm=True) first, or ft_status() to check.",
            )

    # ── validate config ───────────────────────────────────────────────────────
    cfg = _load_config()
    if not cfg.get("train_data"):
        return err("train_data is not configured.", hint="Call ft_config_set(train_data='...') first.")
    if not os.path.isfile(cfg["train_data"]):
        return err(f"train_data file not found: {cfg['train_data']}")
    if cfg.get("val_data") and not os.path.isfile(cfg["val_data"]):
        return err(f"val_data file not found: {cfg['val_data']}")

    # ── create run directory ──────────────────────────────────────────────────
    os.makedirs(_RUNS_DIR, exist_ok=True)
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(_RUNS_DIR, tag)
    os.makedirs(run_dir, exist_ok=True)

    if not cfg.get("output_dir"):
        cfg["output_dir"] = os.path.join(run_dir, "model")

    with open(os.path.join(run_dir, "config.json"), "w") as fh:
        json.dump(cfg, fh, indent=2)
    with open(os.path.join(run_dir, "start_time"), "w") as fh:
        fh.write(str(time.time()))

    # ── build sbatch script ───────────────────────────────────────────────
    log_file   = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable
    script_lines = [
        "#!/bin/bash",
        f"#SBATCH --job-name={job_name}",
        f"#SBATCH --partition={partition}",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --cpus-per-task={cpus_per_task}",
        f"#SBATCH --time={wall_time}",
        f"#SBATCH --mem={mem}",
        f"#SBATCH --output={log_file}",
        f"#SBATCH --error={log_file}",
    ]
    if gpus > 0:
        script_lines.append(f"#SBATCH --gres=gpu:{gpus}")
    if account:
        script_lines.append(f"#SBATCH --account={account}")
    script_lines += [
        "",
        f'"{python_exe}" "{_RUNNER}" --run-dir "{run_dir}"',
    ]
    script_content = "\n".join(script_lines) + "\n"

    batch_script_path = os.path.join(run_dir, "batch_script.sh")
    with open(batch_script_path, "w") as fh:
        fh.write(script_content)
    os.chmod(batch_script_path, 0o755)

    # ── submit ────────────────────────────────────────────────────────────────
    try:
        res = subprocess.run(
            ["sbatch", batch_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30,
        )
    except FileNotFoundError:
        return err("sbatch not found.", hint="Slurm is not available on this host. Use ft_run(confirm=True) instead.")
    except subprocess.TimeoutExpired:
        return err("sbatch timed out.", hint="Check Slurm availability on this host.")

    if res.returncode != 0:
        return err(
            f"sbatch failed: {res.stderr.strip()}",
            hint="Check partition name, account, and resource limits.",
        )

    # Parse job ID from "Submitted batch job 12345"
    m = re.search(r"(\d+)", res.stdout)
    if not m:
        return err(f"Could not parse job ID from sbatch output: {res.stdout.strip()}")
    job_id = int(m.group(1))

    with open(_slurm_job_id_path(run_dir), "w") as fh:
        fh.write(str(job_id))

    # ── update 'active' symlink atomically ───────────────────────────────────
    link     = os.path.join(_RUNS_DIR, "active")
    tmp_link = link + ".tmp"
    if os.path.lexists(tmp_link):
        os.unlink(tmp_link)
    os.symlink(run_dir, tmp_link)
    os.replace(tmp_link, link)

    return ok({
        "run_dir":       run_dir,
        "slurm_job_id":  job_id,
        "batch_script":  batch_script_path,
        "log":           log_file,
        "config":        cfg,
        "note": (
            f"Slurm job {job_id} submitted to partition '{partition}'. "
            "Use ft_status() to monitor."
        ),
    })


@mcp.tool()
def ft_status() -> dict:
    """Check status of the active or most-recent training run.

    Returns state ('running', 'pending', 'done', or 'crashed'), elapsed seconds,
    and the last few log lines. Works for both local and Slurm runs.
    """
    run_dir = _active_run_dir()
    if not run_dir:
        return ok({"state": "no_runs", "note": "No runs found. Use ft_run(confirm=True) to start one."})

    status = _run_status(run_dir)
    last_lines = _tail_log(run_dir, 3)
    return ok({
        "run_dir":        run_dir,
        "state":          status["state"],
        "pid":            status["pid"],
        "slurm_job_id":   status["slurm_job_id"],
        "elapsed_s":      status["elapsed_s"],
        "last_log_lines": last_lines,
    })


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True))
def ft_stop(confirm: bool = False) -> dict:
    """Stop the current training run (sensitive, confirm-gated).

    For local runs, sends SIGTERM to the training process.
    For Slurm runs, calls `scancel <job_id>`.

    Args:
        confirm: Must be set to True to send the stop signal.
    """
    if not confirm:
        return err(
            "Stop not confirmed.",
            hint="Set confirm=True only after obtaining user approval.",
        )

    run_dir = _active_run_dir()
    if not run_dir:
        return err("No active run directory found.")

    # ── Slurm run ─────────────────────────────────────────────────────────────
    job_id = _read_slurm_job_id(run_dir)
    if job_id is not None:
        slurm_state = _squeue_state(job_id)
        if slurm_state not in ("running", "pending"):
            return err(
                f"Slurm job {job_id} is not active (state: {slurm_state}).",
                hint="Check ft_status() — the job may have already finished or failed.",
            )
        try:
            res = subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True, text=True, timeout=15,
            )
            if res.returncode != 0:
                return err(f"scancel failed: {res.stderr.strip()}",
                           hint="Verify you own this job and Slurm is accessible.")
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            return err(f"scancel error: {exc}")
        return ok({"cancelled_job_id": job_id, "run_dir": run_dir, "note": "scancel sent."})

    # ── local run ─────────────────────────────────────────────────────────────
    pid = _read_pid(run_dir)
    if not pid:
        return err("No PID file found for the active run.")

    if not _is_running(pid):
        return err(
            f"Process {pid} is not running.",
            hint="The run may have already finished or crashed. Check ft_status().",
        )

    try:
        os.kill(pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError) as exc:
        return err(f"Failed to send SIGTERM to pid {pid}: {exc}")

    try:
        os.unlink(_pid_path(run_dir))
    except OSError:
        pass

    return ok({"stopped_pid": pid, "run_dir": run_dir, "note": "SIGTERM sent."})


@mcp.tool(**tool_caps(caps=[PLAN_BLOCKED], reversibility=RECOVERABLE, non_batch=True))
def ft_runner_promote(confirm: bool = False) -> dict:
    """Promote the current _ft_runner.py as the new canonical for future sessions (sensitive).

    After the agent has improved _ft_runner.py and verified the improvements
    produce better metrics, call this tool to persist those changes as the new
    canonical script. The promoted version will be restored at the start of every
    subsequent session instead of the original factory defaults.

    Args:
        confirm: Must be set to True to promote (prevents accidental overwrites).
    """
    if not confirm:
        return err(
            "Not confirmed.",
            hint="Set confirm=True after reviewing the changes to _ft_runner.py.",
        )
    try:
        import shutil
        shutil.copy2(_RUNNER, _RUNNER_CANONICAL)
        return ok({"promoted_to": _RUNNER_CANONICAL})
    except OSError as exc:
        return err(f"Could not promote runner: {exc}")


@mcp.tool()
def ft_log_read(tail_lines: int = 50) -> dict:
    """Read the tail of the training log from the active or most-recent run.

    Args:
        tail_lines: Number of lines to return from the end of the log (1–500).
    """
    tail_lines = max(1, min(tail_lines, 500))

    run_dir = _active_run_dir()
    if not run_dir:
        return err("No runs found.")

    lp = _log_path(run_dir)
    if not os.path.isfile(lp):
        return err("Log file not yet created — training may not have started writing output.")

    try:
        with open(lp, errors="replace") as fh:
            content = fh.read(_MAX_LOG)
    except OSError as exc:
        return err(f"Could not read log: {exc}")

    all_lines = content.splitlines()
    tail = all_lines[-tail_lines:]
    status = _run_status(run_dir)

    result = {
        "run_dir":     run_dir,
        "state":       status["state"],
        "total_lines": len(all_lines),
        "lines":       tail,
    }
    if len(content.encode("utf-8", errors="replace")) >= _MAX_LOG:
        result["truncated"] = True
    return ok(result)


@mcp.tool()
def ft_metrics_parse() -> dict:
    """Extract per-step metrics and final results from the active run.

    Parses Trainer JSON log lines from run.log for step, epoch, loss,
    eval_loss, and learning_rate. Also returns per-epoch memory snapshots
    (peak_vram_mb, cpu_ram_mb) and a structured summary with throughput,
    runtime, trainable parameters, and precision.

    After reviewing the results, decide whether the user's constraints are met:

    - If peak_vram_mb is too high: lower precision via ft_config_set
      (e.g. precision='int8' or 'bf16'), reduce batch_size or max_length,
      or read _ft_runner.py and add model.gradient_checkpointing_enable().
    - If eval_loss is too high: increase lora_r or lora_alpha, add more
      target_modules, raise epochs or lower lr, or modify the data pipeline
      in _ft_runner.py.
    - If throughput is too low: increase batch_size, switch precision, or
      rewrite the training loop in _ft_runner.py (e.g. use a custom optimizer).
    - For any other constraint (architecture, loss curve shape, regularization,
      etc.): read _ft_runner.py with read_file_lines, modify it freely with
      replace_in_file, then call ft_run again to observe the effect.

    The runner file (_ft_runner.py) is restored to its original state at the
    start of each new session, so modifications never persist across sessions.
    """
    run_dir = _active_run_dir()
    if not run_dir:
        return err("No runs found.")

    parsed: list[dict] = []
    per_epoch_memory: list[dict] = []
    summary: dict = {}
    lp = _log_path(run_dir)
    if os.path.isfile(lp):
        try:
            with open(lp, errors="replace") as fh:
                content = fh.read(_MAX_LOG)
        except OSError as exc:
            return err(f"Could not read log: {exc}")

        for line in content.splitlines():
            # ── HuggingFace Trainer step log ─────────────────────────────────
            # e.g. {'loss': 2.456, 'grad_norm': 0.12, 'learning_rate': 0.0002, 'epoch': 1.0}
            if re.search(r"['\"]loss['\"]", line):
                entry: dict = {}
                for key in ("step", "epoch", "loss", "eval_loss", "learning_rate"):
                    m = re.search(r"['\"]?" + key + r"['\"]?\s*:\s*([\d.eE+\-]+)", line)
                    if m:
                        entry[key] = float(m.group(1))
                if entry:
                    parsed.append(entry)

            # ── per-epoch memory line ─────────────────────────────────────────
            # [ft_runner] epoch=1 peak_vram_mb=4096 cpu_ram_mb=12288
            elif line.startswith("[ft_runner] epoch="):
                em: dict = {}
                for key in ("epoch", "peak_vram_mb", "cpu_ram_mb"):
                    m = re.search(key + r"=([\d.]+)", line)
                    if m:
                        em[key] = float(m.group(1))
                if em:
                    per_epoch_memory.append(em)

            # ── final structured summary line ─────────────────────────────────
            # [ft_runner] summary peak_vram_mb=... cpu_ram_mb=... ...
            elif line.startswith("[ft_runner] summary "):
                # numeric fields
                for key in (
                    "peak_vram_mb", "cpu_ram_mb", "throughput_sps",
                    "train_runtime_s", "trainable_params", "total_params",
                    "train_loss", "eval_loss",
                ):
                    m = re.search(key + r"=([\d.eE+\-]+)", line)
                    if m:
                        summary[key] = float(m.group(1))
                # string fields (e.g. precision=bf16)
                m = re.search(r"precision=(\w+)", line)
                if m:
                    summary["precision"] = m.group(1)

    final_metrics: dict | None = None
    mp = os.path.join(run_dir, "metrics.json")
    if os.path.isfile(mp):
        try:
            with open(mp) as fh:
                final_metrics = json.load(fh)
        except (json.JSONDecodeError, OSError):
            pass

    return ok({
        "run_dir":           run_dir,
        "parsed_steps":      parsed,
        "step_count":        len(parsed),
        "per_epoch_memory":  per_epoch_memory,
        "summary":           summary,
        "final_metrics":     final_metrics,
    })


@mcp.tool()
def ft_runs_list() -> dict:
    """List all fine-tuning runs (newest first) with state and final metrics."""
    if not os.path.isdir(_RUNS_DIR):
        return ok({"runs": [], "count": 0})

    names = sorted(
        [
            d for d in os.listdir(_RUNS_DIR)
            if d != "active" and os.path.isdir(os.path.join(_RUNS_DIR, d))
        ],
        reverse=True,
    )

    runs: list[dict] = []
    for name in names:
        run_dir = os.path.join(_RUNS_DIR, name)
        status  = _run_status(run_dir)
        entry: dict = {
            "name":      name,
            "run_dir":   run_dir,
            "state":     status["state"],
            "pid":       status["pid"],
            "elapsed_s": status["elapsed_s"],
        }
        mp = os.path.join(run_dir, "metrics.json")
        if os.path.isfile(mp):
            try:
                with open(mp) as fh:
                    entry["metrics"] = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass
        runs.append(entry)

    return ok({"runs": runs, "count": len(runs)})


@mcp.tool()
def ft_runs_diff(run_a: str = "", run_b: str = "") -> dict:
    """Compare config and metrics between two runs (newest-first if unspecified).

    Returns a structured diff showing which config fields changed and how each
    numeric metric shifted between the two runs. Use this after a series of
    experiments to understand which hyperparameter changes had the most impact.

    Args:
        run_a: Run directory name (e.g. '2024-01-01T12:00:00') or absolute path.
               Defaults to the second-newest run.
        run_b: Run directory name or absolute path. Defaults to the newest run.

    Returns a dict with:
        - run_a, run_b: the two run dirs being compared
        - config_diff: dict of fields that differ, as {field: [value_a, value_b]}
        - metrics_diff: dict of numeric fields present in at least one run,
            as {field: {a, b, delta, pct_change}}
    """
    if not os.path.isdir(_RUNS_DIR):
        return err("No runs found.", hint="Run ft_run() or ft_run_slurm() first.")

    all_runs = sorted(
        [
            d for d in os.listdir(_RUNS_DIR)
            if d != "active" and os.path.isdir(os.path.join(_RUNS_DIR, d))
        ],
        reverse=True,
    )
    if len(all_runs) < 2:
        return err("Need at least 2 runs to compare.", hint="Run more experiments first.")

    def _resolve(name: str, default_idx: int) -> str:
        if not name:
            return os.path.join(_RUNS_DIR, all_runs[default_idx])
        if os.path.isabs(name):
            return name
        return os.path.join(_RUNS_DIR, name)

    dir_a = _resolve(run_a, 1)   # second-newest by default
    dir_b = _resolve(run_b, 0)   # newest by default

    for d, label in ((dir_a, "run_a"), (dir_b, "run_b")):
        if not os.path.isdir(d):
            return err(f"{label} not found: {d}")

    def _load_json(path: str) -> dict:
        try:
            with open(path) as fh:
                return json.load(fh)
        except (json.JSONDecodeError, OSError):
            return {}

    cfg_a = _load_json(os.path.join(dir_a, "config.json"))
    cfg_b = _load_json(os.path.join(dir_b, "config.json"))
    met_a = _load_json(os.path.join(dir_a, "metrics.json"))
    met_b = _load_json(os.path.join(dir_b, "metrics.json"))

    # config diff — only fields that differ
    config_diff: dict = {}
    all_cfg_keys = set(cfg_a) | set(cfg_b)
    for k in sorted(all_cfg_keys):
        va, vb = cfg_a.get(k), cfg_b.get(k)
        if va != vb:
            config_diff[k] = [va, vb]

    # metrics diff — numeric fields only
    metrics_diff: dict = {}
    all_met_keys = set(met_a) | set(met_b)
    for k in sorted(all_met_keys):
        va, vb = met_a.get(k), met_b.get(k)
        if not isinstance(va, (int, float)) and not isinstance(vb, (int, float)):
            continue
        _a = float(va) if isinstance(va, (int, float)) else None
        _b = float(vb) if isinstance(vb, (int, float)) else None
        delta = (_b - _a) if (_a is not None and _b is not None) else None
        pct   = (delta / abs(_a) * 100) if (delta is not None and _a) else None
        metrics_diff[k] = {"a": _a, "b": _b, "delta": delta, "pct_change": pct}

    return ok({
        "run_a":        dir_a,
        "run_b":        dir_b,
        "config_diff":  config_diff,
        "metrics_diff": metrics_diff,
    })


if __name__ == "__main__":
    mcp.run()
