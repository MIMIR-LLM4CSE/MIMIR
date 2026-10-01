"""Slurm submission ops: single run, per-case suite jobs, and eval-session runs."""

from __future__ import annotations

import os
import shlex
import sys
from datetime import datetime, timezone

from _ops import _PROXY_DIR, _with_next, err, ok
from _lib import build as build_mod
from _lib import placement
from _lib.command import _build_sbatch, _sbatch_header
from _lib.procs import (
    _log_path, _scancel, background_descriptor,
    _new_run_dir, _write_run_config, _update_run_config, _submit_sbatch,
    _update_active_link, _update_opt_active_link,
)
from _lib.store import (
    refs_dir,
    _entry_or_err, _load_registry_or_err, _load_suite,
    _proxy_runs_dir, _suite_results_dir,
)
from _ops.eval_session import _prepare_run, _background_descriptor
from _ops.suites import _prevalidate_suite

_OPT_RUNNER = os.path.join(_PROXY_DIR, "_proxy_runner.py")


def _run_background_descriptor(run_dir: str) -> dict:
    """The watcher handle for a plain submitted run.

    Never optional here, the way it is for a local run that can also be awaited
    in-turn: sbatch returns while the job is still queued, so no variant of this call
    carries a result. ``status_op`` is the single-run state op, whose ``state`` reaches
    'done'/'crashed' through squeue; ``summary_op`` reads the log, which is all a plain
    run records of itself.
    """
    return background_descriptor(
        run_dir, kind="proxy-slurm-run",
        status_op={"tool": "proxy_runs", "args": {"op": "status", "run_id": run_dir}},
        summary_op={"tool": "proxy_runs",
                    "args": {"op": "logs", "run_id": run_dir, "tail": 40}})


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
    constraint: str = "",
    nodelist: str = "",
    comment: str = "",
) -> dict:
    """Submit a single proxy run as a Slurm batch job (non-blocking)."""
    entry, error = _entry_or_err(proxy_name)
    if error:
        return error

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
        "comment":              comment,
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
        constraint=constraint, nodelist=nodelist, comment=comment,
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
        "comment":              comment or None,
        "background_job":       _run_background_descriptor(run_dir),
        "note": f"Slurm job {job_id} submitted to partition '{partition}'; "
                "this call returns while it is still queued.",
    }, "end your turn — you are resumed when the job finishes"))


def submit_suite(
    suite_name: str,
    partition: str,
    gpus: int = 0,
    cpus_per_task: int = 8,
    mem: str = "32G",
    wall_time: str = "04:00:00",
    account: str = "",
    constraint: str = "",
    nodelist: str = "",
    comment: str = "",
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
                "comment":              comment,
                "started_at":           datetime.now(timezone.utc).isoformat(),
            })

            script = _build_sbatch(
                entry, run_dir, extra_params,
                partition=partition, gpus=gpus, cpus_per_task=cpus_per_task,
                mem=mem, wall_time=wall_time, account=account, job_name=job_name,
                compare_to_reference=reference_name,
                python_exe=sys.executable,
                param_overrides=sweep_overrides,
                constraint=constraint, nodelist=nodelist, comment=comment,
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
        "comment":       comment or None,
        "submissions":   submissions,
        "errors":        errors,
        "note": f"{len(submissions)} jobs submitted.",
    }, f"after completion, proxy_get(op='report', name='{suite_name}', "
       f"run_timestamp='{ts}') to aggregate results"))


def _build_log_stdout_path(run_dir: str) -> str:
    """The build job's own stdout, kept apart from the run job's.

    Two jobs writing one stdout.log interleave into something neither of them said,
    and the run's log is what proxy_eval_status shows the model.
    """
    return os.path.join(run_dir, "build_stdout.log")


def _runner_script(
    place: dict, *, job_name: str, log_file: str, run_dir: str, python_exe: str,
    phase: str, dependency: str = "",
) -> str:
    """One sbatch script that runs _proxy_runner.py over *run_dir* for one phase."""
    lines = _sbatch_header(
        job_name=job_name,
        partition=place["partition"],
        cpus_per_task=place["cpus_per_task"],
        wall_time=place["wall_time"],
        mem=place["mem"],
        log_file=log_file,
        gpus=place.get("gpus", 0),
        account=place.get("account", ""),
        constraint=place.get("constraint", ""),
        nodelist=place.get("nodelist", ""),
        comment=place.get("comment", ""),
        dependency=dependency,
        kill_on_invalid_dep=bool(dependency),
    )
    argv = [python_exe, _OPT_RUNNER, "--run-dir", run_dir]
    if phase != "all":
        argv += ["--phase", phase]
    return "\n".join(lines + ["", shlex.join(argv), ""]) + "\n"


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
    constraint: str = "",
    nodelist: str = "",
    comment: str = "",
    build_partition: str = "",
    build_constraint: str = "",
    build_cpus_per_task: int = 0,
    build_mem: str = "",
    build_wall_time: str = "",
    build_gpus: int = 0,
) -> dict:
    """Submit an optimization-session run as Slurm batch job(s) (non-blocking).

    One job by default: it builds and measures, as it always has. When a build
    partition is resolved — from ``build_partition`` here, or from the proxy's
    ``build_partition`` metadata — the run is split into two jobs sharing this run
    directory, the second chained behind the first with ``--dependency=afterok``.
    That is what lets a compile land on a machine made for compiling while the
    measurement lands on the hardware under test.

    *background* is accepted for call compatibility with the local run and has no
    effect: a submission is detached by definition, so the response always carries a
    ``background_job`` handle.
    """
    cfg, error, run_dir, resume_notice = _prepare_run(proxy_name)
    if error:
        return error
    name       = cfg["proxy_name"]
    log_file   = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable

    run_place = placement.run_placement(
        partition=partition, gpus=gpus, cpus_per_task=cpus_per_task, mem=mem,
        wall_time=wall_time, account=account, constraint=constraint,
        nodelist=nodelist, comment=comment,
    )

    # Resolving the build's placement needs the registry entry (its standing
    # build_* metadata) and the suite (whether there is anything to build at all).
    # Neither is fatal if missing — _prepare_run has already established the session
    # is sound — so a lookup that comes up empty simply means one job.
    reg, _reg_err = _load_registry_or_err()
    entry = (reg or {}).get(name, {}) if not _reg_err else {}
    suite = _load_suite(cfg.get("benchmark_name", "")) or {}
    has_build = bool(build_mod.builds_for(reg or {}, suite, name, entry)) if entry else False
    build_place = placement.resolve_build(
        entry, run_place,
        build_partition=build_partition, build_constraint=build_constraint,
        build_cpus_per_task=build_cpus_per_task, build_mem=build_mem,
        build_wall_time=build_wall_time, build_gpus=build_gpus,
    ) if has_build else None

    # _prepare_run already wrote the full config (convergence included); only the
    # placement is Slurm-specific, so merge it in rather than rewriting the file.
    _update_run_config(run_dir, {
        "partition": partition,
        "placement": {"run": run_place, "build": build_place},
    })

    payload: dict = {
        "run_dir":        run_dir,
        "batch_script":   os.path.join(run_dir, "batch_script.sh"),
        "log":            log_file,
        "proxy_name":     name,
        "benchmark_name": cfg["benchmark_name"],
    }

    build_job_id = None
    if build_place is not None:
        build_log = _build_log_stdout_path(run_dir)
        build_script = _runner_script(
            build_place, job_name=f"{job_name}_build"[:60], log_file=build_log,
            run_dir=run_dir, python_exe=python_exe, phase="build",
        )
        build_job_id, error = _submit_sbatch(
            run_dir, build_script,
            local_alternative="proxy_eval(op='run', confirm=True)",
            script_name="build_batch_script.sh", id_file="build_slurm_job_id",
        )
        if error:
            return error
        payload.update({
            "build_job_id":     build_job_id,
            "build_partition":  build_place["partition"],
            "build_batch_script": os.path.join(run_dir, "build_batch_script.sh"),
            "build_log":        build_log,
        })

    script = _runner_script(
        run_place, job_name=job_name, log_file=log_file, run_dir=run_dir,
        python_exe=python_exe,
        phase="run" if build_place is not None else "all",
        dependency=f"afterok:{build_job_id}" if build_job_id is not None else "",
    )
    job_id, error = _submit_sbatch(
        run_dir, script,
        local_alternative="proxy_eval(op='run', confirm=True)",
    )
    if error:
        # The build is already queued and would compile for a run that will never
        # exist. Cancelling it is the only thing between this failure and a node
        # spending an hour on work nobody will read.
        if build_job_id is not None:
            cancel_err = _scancel(build_job_id)
            error["hint"] = (
                (error.get("hint", "") + " ") if error.get("hint") else ""
            ) + (f"The build job {build_job_id} could not be cancelled "
                 f"({cancel_err}) — cancel it by hand." if cancel_err
                 else f"The already-submitted build job {build_job_id} was cancelled.")
        return error
    _update_opt_active_link(name, run_dir)

    payload["slurm_job_id"] = job_id
    if build_job_id is not None:
        payload["note"] = (
            f"Build job {build_job_id} submitted to '{build_place['partition']}'; "
            f"run job {job_id} submitted to '{partition}' and held until the build "
            f"succeeds. A failed build kills the run job instead of leaving it queued."
        )
    else:
        payload["note"] = f"Slurm job {job_id} submitted to '{partition}'."
    if resume_notice:
        payload["resume_notice"] = resume_notice
    # Attached whatever *background* said. It distinguishes "wait for it here" from
    # "detach it" for the local run, and sbatch has no first option: the call returns
    # with the job still queued either way. Honouring the flag here only decided
    # whether anything was left watching the job — and on the False side, nothing was.
    payload["background_job"] = _background_descriptor(name, run_dir)
    return ok(_with_next(payload,
                         "end your turn — you are resumed when the job finishes"))
