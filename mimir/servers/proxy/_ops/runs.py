"""Run ops: history/logs/diff/compare/aggregate (read-only), local launch, cancel."""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone

from ....shared.launcher import PACKAGE_PARENT
from . import with_next, err, ok
from . import exec_jobs
from .._lib.command import output_ext, build_run_cmd
from .._lib.execute import compare_with_reference
from .._lib.procs import (
    run_log_path,
    run_state,
    new_run_dir,
    write_run_config,
    launch_detached,
    cancel_run,
    update_active_link,
)
from .._lib.report import diff_run_pair, row_with_extra
from .._lib.store import (
    refs_dir,
    runs_dir,
    entry_or_err,
    load_registry_or_err,
    proxy_runs_dir,
    read_json,
    run_dir_names,
)


def _resolve_run_dir(run_id: str) -> str:
    return run_id if os.path.isabs(run_id) else os.path.join(runs_dir(), run_id)


# ── read-only ─────────────────────────────────────────────────────────────────

def list_runs(proxy_name: str = "") -> dict:
    if not os.path.isdir(runs_dir()):
        return ok(with_next({"runs": [], "count": 0},
                             "proxy_exec(op='run', ...) to produce a first run."))

    proxy_dirs: list[str] = []
    if proxy_name:
        sd = proxy_runs_dir(proxy_name)
        if os.path.isdir(sd):
            proxy_dirs = [sd]
    else:
        proxy_dirs = [
            os.path.join(runs_dir(), d)
            for d in os.listdir(runs_dir())
            if os.path.isdir(os.path.join(runs_dir(), d))
        ]

    runs: list[dict] = []
    for sd in proxy_dirs:
        sname = os.path.basename(sd)
        for tag in sorted(
            [d for d in os.listdir(sd) if d != "active" and os.path.isdir(os.path.join(sd, d))],
            reverse=True,
        ):
            run_dir = os.path.join(sd, tag)
            rs = run_state(run_dir)
            entry: dict = {
                "proxy":     sname,
                "run_id":    f"{sname}/{tag}",
                "run_dir":   run_dir,
                "state":     rs["state"],
                "elapsed_s": rs["elapsed_s"],
            }
            m = read_json(os.path.join(run_dir, "metrics.json"))
            if m is not None:
                entry["metrics"] = m
            runs.append(entry)

    return ok(with_next(
        {"runs": runs, "count": len(runs)},
        "proxy_runs(op='logs', run_id=...) to read one, or op='diff' to compare two."))


def run_status(run_id: str) -> dict:
    """One run's state (pending, running, done, crashed, unknown) and progress: what a watcher polls."""
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")
    if exec_jobs.is_job_dir(run_dir):
        return exec_jobs.status(run_dir)
    rs = run_state(run_dir)
    payload = {"run_id": run_id, "run_dir": run_dir, "log": run_log_path(run_dir), **rs}
    metrics = read_json(os.path.join(run_dir, "metrics.json"))
    if metrics is not None:
        payload["metrics"] = metrics
    return ok(with_next(
        payload,
        f"proxy_runs(op='logs', run_id='{run_id}') to read its output."
        if rs["state"] in ("pending", "running") else
        f"proxy_runs(op='compare', run_id='{run_id}', reference_name=...) "
        "to score it against a reference."))


def run_logs(run_id: str, tail: int = 100) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")
    lp = run_log_path(run_dir)
    if not os.path.isfile(lp):
        # State even with no log yet: a queued Slurm job has nothing to read for as
        # long as it sits in the partition, and a reply that omits the state reads
        # like a run nobody can account for.
        return ok(with_next({"run_id": run_id, "state": run_state(run_dir)["state"],
                              "lines": [], "note": "Log not yet created."},
                             f"proxy_runs(op='logs', run_id='{run_id}') again once it starts."))
    try:
        with open(lp, errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        return err(f"Could not read log: {exc}")
    lines = [ln.rstrip("\n") for ln in all_lines[-max(1, tail):]]
    rs = run_state(run_dir)
    return ok(with_next(
        {"run_id": run_id, "state": rs["state"],
         "total_lines": len(all_lines), "tail": tail, "lines": lines},
        f"proxy_runs(op='compare', run_id='{run_id}', reference_name=...) "
        "to score it against a reference."))


def runs_diff(run_a: str = "", run_b: str = "") -> dict:
    if not os.path.isdir(runs_dir()):
        return err("No runs found.", hint="Launch one with proxy_exec(op='run', ...) first.")

    all_runs = [
        os.path.join(runs_dir(), sname, tag)
        for sname in sorted(os.listdir(runs_dir()))
        for tag in run_dir_names(os.path.join(runs_dir(), sname))
    ]

    def _resolve(run_id: str, default_idx: int) -> str:
        return all_runs[default_idx] if not run_id else _resolve_run_dir(run_id)

    payload, error = diff_run_pair(all_runs, run_a, run_b, _resolve)
    if error:
        return err(error, hint="Run more experiments first.")
    return ok(with_next(payload,
                         "proxy_runs(op='aggregate', ...) for a table across all runs."))


def compare(run_id: str, reference_name: str) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")

    cfg = read_json(os.path.join(run_dir, "config.json"), {}) or {}
    output_format = cfg.get("output_format", "npz")

    ext = output_ext(output_format)
    result = compare_with_reference(os.path.join(run_dir, f"output.{ext}"), reference_name, output_format)
    return ok(with_next(
        {"run_dir": run_dir, "reference": reference_name, "comparison": result},
        "proxy_runs(op='aggregate', ...) to compare this against the other runs."))


def aggregate(
    run_ids: list[str],
    metrics: list[str] | None = None,
    reference_name: str = "",
) -> dict:
    """Aggregate completed runs into one comparison table (read-only)."""
    if metrics is None:
        metrics = []

    reg, _reg_err = load_registry_or_err()
    if _reg_err:
        return err(_reg_err)

    rows: list[dict] = []
    errors: list[dict] = []

    for run_id in run_ids:
        run_dir = _resolve_run_dir(run_id)
        if not os.path.isdir(run_dir):
            errors.append({"run_id": run_id, "error": "directory not found"})
            continue

        cfg   = read_json(os.path.join(run_dir, "config.json"), {}) or {}
        entry = reg.get(cfg.get("proxy_name", ""), {})
        met   = read_json(os.path.join(run_dir, "metrics.json"), {}) or {}

        rs = run_state(run_dir)
        if rs["state"] != "done":
            errors.append({"run_id": run_id, "error": f"run state is '{rs['state']}'",
                           "skipped": True})
            continue

        comparison: dict = met.get("comparison_to_reference", {})
        if reference_name and not comparison:
            output_fmt = cfg.get("output_format", entry.get("output_format", "npz"))
            ext = output_ext(output_fmt)
            run_out = os.path.join(run_dir, f"output.{ext}")
            comparison = compare_with_reference(run_out, reference_name, output_fmt)

        row: dict = {
            "run_id":      run_id,
            "proxy":       cfg.get("proxy_name", ""),
            "arch":        entry.get("arch", ""),
            "backend":     entry.get("backend", ""),
            "parallelism": entry.get("parallelism", ""),
            "state":       rs["state"],
        }
        row.update(row_with_extra(met, comparison, metrics, entry, run_dir))
        rows.append(row)

    return ok(with_next({"rows": rows, "count": len(rows), "errors": errors},
                         "proxy_runs(op='diff', run_a=..., run_b=...) to detail two of them."))


# ── mutations (confirm already checked by the dispatch tool) ──────────────────

def launch_run(
    proxy_name: str,
    extra_params: str = "",
    param_overrides: dict | None = None,
    compare_to_reference: str = "",
) -> dict:
    """Launch a single proxy run in the background (non-blocking)."""
    entry, error = entry_or_err(proxy_name)
    if error:
        return error

    if compare_to_reference and compare_to_reference not in (
        os.listdir(refs_dir()) if os.path.isdir(refs_dir()) else []
    ):
        return err(f"Reference '{compare_to_reference}' not found.",
                   hint="Call proxy_get(op='references') to see available references.")

    run_dir = new_run_dir(proxy_runs_dir(proxy_name))
    write_run_config(run_dir, {
        "proxy_name":           proxy_name,
        "extra_params":         extra_params,
        "param_overrides":      param_overrides or {},
        "output_format":        entry.get("output_format", "npz"),
        "compare_to_reference": compare_to_reference,
        "started_at":           datetime.now(timezone.utc).isoformat(),
    })

    try:
        argv = build_run_cmd(entry, run_dir, extra_params, param_overrides)
    except Exception as exc:
        return err(f"Failed to expand run_cmd_template: {exc}")

    log_file = run_log_path(run_dir)
    # _post_run_finalize is called even when the launch itself fails, so the run
    # state resolves instead of stranding the reason on the server's stderr. Wall
    # time and exit code come from here: the timing guard and the crash state
    # both depend on them.
    wrapper = (
        f"import subprocess, sys, os, time, traceback\n"
        f"_log = {log_file!r}\n"
        f"_rc = None\n"
        f"_t0 = time.time()\n"
        f"try:\n"
        f"    with open(_log, 'w') as _lf:\n"
        f"        try:\n"
        f"            _p = subprocess.run({argv!r}, stdout=_lf, stderr=subprocess.STDOUT)\n"
        f"            _rc = _p.returncode\n"
        f"        except Exception:\n"
        f"            _lf.write('\\n[proxy_run] launch failed:\\n' + traceback.format_exc())\n"
        f"    _wall = round(time.time() - _t0, 3)\n"
        f"    sys.path.insert(0, {PACKAGE_PARENT!r})\n"
        f"    from mimir.servers.proxy._lib.execute import _post_run_finalize\n"
        f"    _post_run_finalize(\n"
        f"        run_dir={run_dir!r},\n"
        f"        proxy_name={proxy_name!r},\n"
        f"        compare_to_reference={compare_to_reference!r},\n"
        f"        output_format={entry.get('output_format', 'npz')!r},\n"
        f"        wall_s=_wall,\n"
        f"        returncode=_rc,\n"
        f"    )\n"
        f"except Exception:\n"
        f"    traceback.print_exc()\n"
    )

    pid = launch_detached([sys.executable, "-c", wrapper], run_dir)
    update_active_link(proxy_name, run_dir)
    return ok(with_next({
        "run_dir":              run_dir,
        "pid":                  pid,
        "log":                  log_file,
        "compare_to_reference": compare_to_reference or None,
    }, "proxy_runs() to monitor completion."))


def cancel(run_id: str) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")
    result = cancel_run(run_dir)
    if "error" in result:
        return err(result["error"])
    return ok(with_next({"run_id": run_id, **result},
                         "proxy_runs() to confirm the run is no longer active."))
