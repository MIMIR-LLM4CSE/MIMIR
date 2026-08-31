"""Result reporting helpers: roofline analysis, table rows, and run diffs."""

from __future__ import annotations

import os

from _lib import store


# ── roofline ──────────────────────────────────────────────────────────────────

def _resolve_roofline(entry: dict) -> tuple[float, float]:
    """Return (peak_gflops_per_s, peak_bandwidth_gbytes_per_s).

    Peaks come from the registration entry only; 0.0 means unavailable, and
    `_compute_roofline` then reports nothing rather than a ceiling nobody vouched for.
    """
    peak_gf = float(entry.get("peak_gflops_per_s", 0) or 0)
    peak_bw = float(entry.get("peak_bandwidth_gbytes_per_s", 0) or 0)
    return peak_gf, peak_bw


def _compute_roofline(metrics: dict, peak_gf: float, peak_bw: float) -> dict:
    """Compute roofline metrics from proxy-emitted flops/bytes_moved.

    Returns an empty dict if the required keys are missing or peaks are zero.
    """
    flops       = metrics.get("flops")
    bytes_moved = metrics.get("bytes_moved")
    time_s      = metrics.get("time_s")
    if not (flops and bytes_moved and time_s and peak_gf > 0 and peak_bw > 0):
        return {}
    try:
        flops       = float(flops)
        bytes_moved = float(bytes_moved)
        time_s      = float(time_s)
    except (TypeError, ValueError):
        return {}
    ai            = flops / bytes_moved
    actual_gf     = flops / time_s / 1e9
    actual_bw     = bytes_moved / time_s / 1e9
    roof          = min(peak_gf, ai * peak_bw)
    bound         = "memory" if ai * peak_bw < peak_gf else "compute"
    return {
        "arithmetic_intensity":      round(ai, 4),
        "actual_gflops_per_s":       round(actual_gf, 3),
        "actual_bandwidth_gbytes_s": round(actual_bw, 3),
        "flop_efficiency_pct":       round(actual_gf / peak_gf * 100, 2),
        "bandwidth_efficiency_pct":  round(actual_bw / peak_bw * 100, 2),
        "roofline_bound":            bound,
        "roofline_ceiling_gflops":   round(roof, 3),
    }


def _row_with_extra(
    metrics: dict,
    comparison: dict,
    extra_metrics: list[str],
    entry: dict,
    run_dir: str,
) -> dict:
    """Build a benchmark/sweep table row with optional extra metrics and roofline."""
    row: dict = {
        "time_s":   metrics.get("time_s"),
        "misfit":   metrics.get("misfit"),
        "l2_rel":   comparison.get("l2_rel"),
        "linf_rel": comparison.get("linf_rel"),
        "run_dir":  run_dir,
    }
    for k in extra_metrics:
        row[k] = metrics.get(k)
    peak_gf, peak_bw = _resolve_roofline(entry)
    rl = _compute_roofline(metrics, peak_gf, peak_bw)
    if rl:
        row.update(rl)
    return row


# ── run diffs ─────────────────────────────────────────────────────────────────

def _diff_run_pair(all_runs: list[str], run_a: str, run_b: str, resolve):
    """Diff two runs picked from a newest-first list; returns (payload, error).

    ``resolve(run_id, default_idx)`` maps a caller-supplied id (possibly empty)
    to an absolute run dir; the defaults are the two most recent runs.
    """
    if len(all_runs) < 2:
        return None, "Need at least 2 runs to compare."
    dir_a, dir_b = resolve(run_a, 1), resolve(run_b, 0)
    for d, label in ((dir_a, "run_a"), (dir_b, "run_b")):
        if not os.path.isdir(d):
            return None, f"{label} not found: {d}"
    return {"run_a": dir_a, "run_b": dir_b, **_diff_run_dirs(dir_a, dir_b)}, None


def _diff_run_dirs(dir_a: str, dir_b: str) -> dict:
    """Return ``{config_diff, metrics_diff}`` comparing two run directories.

    Numeric metrics get ``{a, b, delta, pct_change}``; non-numeric metrics get
    ``{a, b}`` only when they differ.  Config values are compared verbatim.
    """
    cfg_a = store._read_json(os.path.join(dir_a, "config.json"), {})
    cfg_b = store._read_json(os.path.join(dir_b, "config.json"), {})
    met_a = store._read_json(os.path.join(dir_a, "metrics.json"), {})
    met_b = store._read_json(os.path.join(dir_b, "metrics.json"), {})

    config_diff: dict = {}
    for k in sorted(set(cfg_a) | set(cfg_b)):
        va, vb = cfg_a.get(k), cfg_b.get(k)
        if va != vb:
            config_diff[k] = [va, vb]

    metrics_diff: dict = {}
    for k in sorted(set(met_a) | set(met_b)):
        va, vb = met_a.get(k), met_b.get(k)
        if not isinstance(va, (int, float)) and not isinstance(vb, (int, float)):
            if va != vb:
                metrics_diff[k] = {"a": va, "b": vb}
            continue
        _a = float(va) if isinstance(va, (int, float)) else None
        _b = float(vb) if isinstance(vb, (int, float)) else None
        delta = (_b - _a) if (_a is not None and _b is not None) else None
        pct   = (delta / abs(_a) * 100) if (delta is not None and _a) else None
        metrics_diff[k] = {"a": _a, "b": _b, "delta": delta, "pct_change": pct}

    return {"config_diff": config_diff, "metrics_diff": metrics_diff}
