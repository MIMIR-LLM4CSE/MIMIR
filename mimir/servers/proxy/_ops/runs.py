"""Run ops: history/logs/diff/compare/aggregate (read-only), local launch, cancel."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone

from _ops import _PROXY_DIR, err, ok
from _lib.command import _output_ext, _build_run_cmd
from _lib.execute import _compare_to_reference
from _lib.procs import (
    _log_path, _run_state,
    _new_run_dir, _write_run_config, _launch_detached, _cancel_run,
    _update_active_link,
)
from _lib.report import _diff_run_dirs, _arch_label, _row_with_extra
from _lib.store import refs_dir, runs_dir, _load_registry_or_err, _proxy_runs_dir


def _resolve_run_dir(run_id: str) -> str:
    return run_id if os.path.isabs(run_id) else os.path.join(runs_dir(), run_id)


# ── read-only ─────────────────────────────────────────────────────────────────

def list_runs(proxy_name: str = "") -> dict:
    if not os.path.isdir(runs_dir()):
        return ok({"runs": [], "count": 0})

    proxy_dirs: list[str] = []
    if proxy_name:
        sd = _proxy_runs_dir(proxy_name)
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
            rs = _run_state(run_dir)
            entry: dict = {
                "proxy":     sname,
                "run_id":    f"{sname}/{tag}",
                "run_dir":   run_dir,
                "state":     rs["state"],
                "elapsed_s": rs["elapsed_s"],
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


def run_logs(run_id: str, tail: int = 100) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")
    lp = _log_path(run_dir)
    if not os.path.isfile(lp):
        return ok({"run_id": run_id, "lines": [], "note": "Log not yet created."})
    try:
        with open(lp, errors="replace") as fh:
            all_lines = fh.readlines()
    except OSError as exc:
        return err(f"Could not read log: {exc}")
    lines = [ln.rstrip("\n") for ln in all_lines[-max(1, tail):]]
    rs = _run_state(run_dir)
    return ok({
        "run_id": run_id, "state": rs["state"],
        "total_lines": len(all_lines), "tail": tail, "lines": lines,
    })


def runs_diff(run_a: str = "", run_b: str = "") -> dict:
    if not os.path.isdir(runs_dir()):
        return err("No runs found.", hint="Launch one with proxy_exec(op='run', ...) first.")

    def _all_run_dirs() -> list[str]:
        dirs = []
        for sname in os.listdir(runs_dir()):
            sd = os.path.join(runs_dir(), sname)
            if not os.path.isdir(sd):
                continue
            for tag in sorted(
                [d for d in os.listdir(sd) if d != "active" and os.path.isdir(os.path.join(sd, d))],
                reverse=True,
            ):
                dirs.append(os.path.join(sd, tag))
        return dirs

    all_runs = _all_run_dirs()
    if len(all_runs) < 2:
        return err("Need at least 2 runs to compare.", hint="Run more experiments first.")

    def _resolve(run_id: str, default_idx: int) -> str:
        if not run_id:
            return all_runs[default_idx]
        return _resolve_run_dir(run_id)

    dir_a = _resolve(run_a, 1)
    dir_b = _resolve(run_b, 0)
    for d, label in ((dir_a, "run_a"), (dir_b, "run_b")):
        if not os.path.isdir(d):
            return err(f"{label} not found: {d}")

    diff = _diff_run_dirs(dir_a, dir_b)
    return ok({"run_a": dir_a, "run_b": dir_b, **diff})


def compare(run_id: str, reference_name: str) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")

    output_format = "npz"
    cfg_p = os.path.join(run_dir, "config.json")
    if os.path.isfile(cfg_p):
        try:
            with open(cfg_p) as fh:
                output_format = json.load(fh).get("output_format", "npz")
        except (json.JSONDecodeError, OSError):
            pass

    ext = _output_ext(output_format)
    result = _compare_to_reference(os.path.join(run_dir, f"output.{ext}"), reference_name, output_format)
    return ok({"run_dir": run_dir, "reference": reference_name, "comparison": result})


def aggregate(
    run_ids: list[str],
    metrics: list[str] | None = None,
    reference_name: str = "",
) -> dict:
    """Aggregate completed runs into one comparison table (read-only)."""
    if metrics is None:
        metrics = []

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)

    rows: list[dict] = []
    errors: list[dict] = []

    for run_id in run_ids:
        run_dir = _resolve_run_dir(run_id)
        if not os.path.isdir(run_dir):
            errors.append({"run_id": run_id, "error": "directory not found"})
            continue

        cfg: dict = {}
        cfg_p = os.path.join(run_dir, "config.json")
        if os.path.isfile(cfg_p):
            try:
                with open(cfg_p) as fh:
                    cfg = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass

        proxy_name = cfg.get("proxy_name", "")
        entry = reg.get(proxy_name, {})

        met: dict = {}
        met_p = os.path.join(run_dir, "metrics.json")
        if os.path.isfile(met_p):
            try:
                with open(met_p) as fh:
                    met = json.load(fh)
            except (json.JSONDecodeError, OSError):
                pass

        rs = _run_state(run_dir)
        if rs["state"] != "done":
            errors.append({"run_id": run_id, "error": f"run state is '{rs['state']}'",
                           "skipped": True})
            continue

        comparison: dict = met.get("comparison_to_reference", {})
        if reference_name and not comparison:
            output_fmt = cfg.get("output_format", entry.get("output_format", "npz"))
            ext = _output_ext(output_fmt)
            run_out = os.path.join(run_dir, f"output.{ext}")
            comparison = _compare_to_reference(run_out, reference_name, output_fmt)

        row: dict = {
            "run_id":      run_id,
            "proxy":       proxy_name,
            "arch":        entry.get("arch") or _arch_label(entry),
            "backend":     entry.get("backend", ""),
            "parallelism": entry.get("parallelism", ""),
            "state":       rs["state"],
        }
        row.update(_row_with_extra(met, comparison, metrics, entry, run_dir))
        rows.append(row)

    return ok({"rows": rows, "count": len(rows), "errors": errors})


# ── mutations (confirm already checked by the dispatch tool) ──────────────────

def launch_run(
    proxy_name: str,
    extra_params: str = "",
    param_overrides: dict | None = None,
    compare_to_reference: str = "",
) -> dict:
    """Launch a single proxy run in the background (non-blocking)."""
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
        "started_at":           datetime.now(timezone.utc).isoformat(),
    })

    try:
        argv = _build_run_cmd(entry, run_dir, extra_params, param_overrides)
    except Exception as exc:
        return err(f"Failed to expand run_cmd_template: {exc}")

    log_file = _log_path(run_dir)
    # The child runs the proxy, capturing stdout/stderr to stdout.log.  A launch
    # failure (e.g. bad executable) is written into the log, and _post_run_finalize
    # is ALWAYS called so the run state resolves meaningfully instead of leaving
    # the failure reason on the server's stderr.  Wall time + exit code are
    # measured here and passed through so the time_s plausibility guard applies
    # and a non-zero exit reads as 'crashed'.
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
        f"    sys.path.insert(0, {_PROXY_DIR!r})\n"
        f"    from _lib.execute import _post_run_finalize\n"
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

    pid = _launch_detached([sys.executable, "-c", wrapper], run_dir)
    _update_active_link(proxy_name, run_dir)
    return ok({
        "run_dir":              run_dir,
        "pid":                  pid,
        "log":                  log_file,
        "compare_to_reference": compare_to_reference or None,
        "next_step": "proxy_runs() to monitor completion.",
    })


def cancel(run_id: str) -> dict:
    run_dir = _resolve_run_dir(run_id)
    if not os.path.isdir(run_dir):
        return err(f"Run directory not found: {run_dir}")
    result = _cancel_run(run_dir)
    if "error" in result:
        return err(result["error"])
    return ok({"run_id": run_id, **result})
