"""Slurm submission ops: single run, per-case suite jobs, and eval-session runs."""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime, timezone

from _ops import _PROXY_DIR, _with_next, err, ok
from _lib.command import _build_sbatch, _sbatch_header
from _lib.procs import (
    _log_path,
    _new_run_dir, _write_run_config, _update_run_config, _submit_sbatch,
    _update_active_link, _update_opt_active_link,
)
from _lib.store import (
    refs_dir,
    _load_registry_or_err, _load_suite,
    _proxy_runs_dir, _suite_results_dir,
)
from _ops.eval_session import _prepare_run, _background_descriptor
from _ops.suites import _prevalidate_suite

_OPT_RUNNER = os.path.join(_PROXY_DIR, "_proxy_runner.py")


def submit_run(
    proxy_name: str,
    partition: str,
    extra_params: str = "",
    param_overrides: dict | None = None,
    compare_to_reference: str = "",
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    job_name: str = "",
) -> dict:
    """Submit a single proxy run as a Slurm batch job (non-blocking)."""
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if proxy_name not in reg:
        return err(f"Proxy '{proxy_name}' not registered.",
                   hint="Call proxy_manage(op='register', ...) first.")
    entry = reg[proxy_name]

    if compare_to_reference and compare_to_reference not in (
        os.listdir(refs_dir()) if os.path.isdir(refs_dir()) else []
    ):
        return err(f"Reference '{compare_to_reference}' not found.",
                   hint="Call proxy_get(op='references') to see available references.")

    run_dir = _new_run_dir(_proxy_runs_dir(proxy_name))
    _write_run_config(run_dir, {
        "proxy_name":           proxy_name,
        "extra_params":         extra_params,
        "param_overrides":      param_overrides or {},
        "output_format":        entry.get("output_format", "npz"),
        "compare_to_reference": compare_to_reference,
        "partition":            partition,
        "started_at":           datetime.now(timezone.utc).isoformat(),
    })

    script = _build_sbatch(
        entry, run_dir, extra_params,
        partition=partition, gpus=gpus, cpus_per_task=cpus_per_task,
        mem=mem, wall_time=wall_time, account=account,
        job_name=job_name or f"proxy_{proxy_name}",
        compare_to_reference=compare_to_reference,
        python_exe=sys.executable,
        param_overrides=param_overrides,
    )
    job_id, error = _submit_sbatch(
        run_dir, script,
        local_alternative="proxy_exec(op='run', confirm=True)",
    )
    if error:
        return error
    _update_active_link(proxy_name, run_dir)

    return ok(_with_next({
        "run_dir":              run_dir,
        "slurm_job_id":         job_id,
        "batch_script":         os.path.join(run_dir, "batch_script.sh"),
        "log":                  _log_path(run_dir),
        "compare_to_reference": compare_to_reference or None,
        "note": f"Slurm job {job_id} submitted to partition '{partition}'.",
    }, "proxy_runs() to monitor completion"))


def submit_suite(
    suite_name: str,
    partition: str,
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
) -> dict:
    """Submit one Slurm job per (case × sweep point) in a suite (non-blocking)."""
    suite = _load_suite(suite_name)
    if suite is None:
        return err(f"Suite '{suite_name}' not found.",
                   hint="Call proxy_get(op='suites') to see defined suites.")

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)

    invalid = _prevalidate_suite(suite, reg)
    if invalid:
        return invalid

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = os.path.join(_suite_results_dir(suite_name), ts)
    os.makedirs(results_dir, exist_ok=True)

    submissions: list[dict] = []
    errors: list[dict] = []

    for case in suite.get("cases", []):
        case_id        = case["case_id"]
        proxy_name     = case["proxy_name"]
        reference_name = case.get("reference_name", "")
        param_sweeps   = case.get("param_sweeps") or [{}]
        extra_params   = case.get("extra_params", "")
        entry = reg[proxy_name]

        for idx, sweep_overrides in enumerate(param_sweeps):
            job_name = f"suite_{suite_name}_{case_id}_{idx}"[:60]
            run_dir = _new_run_dir(_proxy_runs_dir(proxy_name),
                                   tag_suffix=f"s{case_id}_{idx}")
            _write_run_config(run_dir, {
                "proxy_name":           proxy_name,
                "extra_params":         extra_params,
                "param_overrides":      sweep_overrides,
                "output_format":        entry.get("output_format", "npz"),
                "compare_to_reference": reference_name,
                "suite_name":           suite_name,
                "case_id":              case_id,
                "started_at":           datetime.now(timezone.utc).isoformat(),
            })

            script = _build_sbatch(
                entry, run_dir, extra_params,
                partition=partition, gpus=gpus, cpus_per_task=cpus_per_task,
                mem=mem, wall_time=wall_time, account=account, job_name=job_name,
                compare_to_reference=reference_name,
                python_exe=sys.executable,
                param_overrides=sweep_overrides,
            )

            # Record pointer before submission
            ptr_dir = os.path.join(results_dir, case_id, str(idx))
            os.makedirs(ptr_dir, exist_ok=True)
            with open(os.path.join(ptr_dir, "run_dir"), "w") as fh:
                fh.write(run_dir)

            job_id, error = _submit_sbatch(run_dir, script)
            if error:
                errors.append({"case_id": case_id, "sweep": sweep_overrides,
                               "error": error.get("error", "sbatch failed")})
                continue
            _update_active_link(proxy_name, run_dir)

            submissions.append({
                "case_id": case_id,
                "proxy":   proxy_name,
                "sweep":   sweep_overrides,
                "job_id":  job_id,
                "run_dir": run_dir,
                "log":     _log_path(run_dir),
            })

    return ok(_with_next({
        "suite":         suite_name,
        "run_timestamp": ts,
        "partition":     partition,
        "submissions":   submissions,
        "errors":        errors,
        "note": f"{len(submissions)} jobs submitted.",
    }, f"after completion, proxy_get(op='report', name='{suite_name}', "
       f"run_timestamp='{ts}') to aggregate results"))


def submit_eval(
    partition: str,
    proxy_name: str = "",
    background: bool = False,
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    job_name: str = "proxy_opt",
) -> dict:
    """Submit an optimization-session run as a Slurm batch job (non-blocking)."""
    cfg, error, run_dir = _prepare_run(proxy_name)
    if error:
        return error
    name       = cfg["proxy_name"]
    log_file   = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable

    # _prepare_run already wrote the full config (convergence included); only the
    # partition is Slurm-specific, so merge it in rather than rewriting the file.
    _update_run_config(run_dir, {"partition": partition})

    script_lines = _sbatch_header(
        job_name=job_name, partition=partition, cpus_per_task=cpus_per_task,
        wall_time=wall_time, mem=mem, log_file=log_file, gpus=gpus, account=account,
    )
    runner_cmd = shlex.join([python_exe, _OPT_RUNNER, "--run-dir", run_dir])
    script = "\n".join(script_lines + ["", runner_cmd, ""]) + "\n"

    job_id, error = _submit_sbatch(
        run_dir, script,
        local_alternative="proxy_eval(op='run', confirm=True)",
    )
    if error:
        return error
    _update_opt_active_link(name, run_dir)

    payload = {
        "run_dir":        run_dir,
        "slurm_job_id":   job_id,
        "batch_script":   os.path.join(run_dir, "batch_script.sh"),
        "log":            log_file,
        "proxy_name":     name,
        "benchmark_name": cfg["benchmark_name"],
        "note": f"Slurm job {job_id} submitted to '{partition}'.",
    }
    if background:
        # Same descriptor as the local run: the watcher polls proxy_eval_status,
        # whose _run_state reports Slurm state via squeue.
        payload["background_job"] = _background_descriptor(name, run_dir)
    return ok(_with_next(payload, "proxy_eval_status() to monitor"))
