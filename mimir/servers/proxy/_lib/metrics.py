"""Metrics parsing, numerical invariants, and requirement evaluation.

Everything that turns solver output into trusted numbers lives here: the
``PROXY_METRICS_BEGIN…END`` block parser, the reserved server-side invariants
(the code under optimization must not be able to satisfy its own acceptance
constraints), the ``time_s`` plausibility guard, field I/O and error norms,
and the pass/fail requirements engine.  No dependencies on the rest of
``_lib`` — numpy is imported lazily.
"""

from __future__ import annotations

import math
import os
import re

from numerics import RESERVED_METRICS

# ── metrics-block parsing ─────────────────────────────────────────────────────

_METRICS_BEGIN = "PROXY_METRICS_BEGIN"
_METRICS_END   = "PROXY_METRICS_END"
_TRUE_VALS     = {"true", "1", "yes"}
_FALSE_VALS    = {"false", "0", "no"}


def _coerce(value: str) -> bool | int | float | str:
    lv = value.lower()
    if lv in _TRUE_VALS:
        return True
    if lv in _FALSE_VALS:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _parse_metrics_block(stdout: str) -> dict:
    """Extract key=value pairs from the PROXY_METRICS_BEGIN…END block.

    Falls back to a single-token scan when no block markers are present.
    """
    metrics: dict = {}
    in_block = False
    block_found = False

    for line in stdout.splitlines():
        stripped = line.strip()
        if stripped == _METRICS_BEGIN:
            in_block = True
            block_found = True
            continue
        if stripped == _METRICS_END:
            in_block = False
            continue
        if in_block:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)", stripped)
            if m:
                metrics[m.group(1)] = _coerce(m.group(2).strip())

    if not block_found:
        # Fallback: scan only the last 20 lines with a strict single-token RHS pattern.
        # This further reduces false positives from verbose logs or compiler output.
        for line in stdout.splitlines()[-20:]:
            m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)=(\S+)$", line.strip())
            if m:
                metrics[m.group(1)] = _coerce(m.group(2))

    return metrics


# ── reserved server-side invariants ───────────────────────────────────────────

# Metrics computed server-side as integrity invariants. Any value the proxy prints for
# these keys is discarded before evaluation: the code under optimization must not be
# able to satisfy its own acceptance constraints. (Seen in the wild: an agent-edited
# proxy printing conservation_residual=<its own drift> to pass a requirement whose
# sealed reference was missing.) ``wall_time_s`` is the server's own wall-clock
# measurement — the tamper-proof twin of self-reported ``time_s``.
#
# The name set lives in _shared because the client's validation observer reads the same
# vocabulary for the opposite purpose: one of these keys in a validation command's
# output distinguishes "the code ran" from "the code was compared against something".
_RESERVED_METRICS = RESERVED_METRICS


def _strip_reserved_metrics(metrics: dict) -> list[str]:
    """Drop reserved keys the proxy tried to emit; return what was dropped."""
    forged = sorted(_RESERVED_METRICS.intersection(metrics))
    for key in forged:
        metrics.pop(key, None)
    return forged


def _normalize_time_metrics(metrics: dict, wall_s: float) -> None:
    """Record the server-measured wall time and guard the self-reported time_s.

    ``wall_time_s`` is always the server's own measurement of the solver
    process (reserved: proxy-printed values are stripped upstream).  ``time_s``
    stays proxy-reported so a kernel may legitimately exclude startup/IO, but a
    claim that is non-positive or exceeds the measured wall time is physically
    impossible: it is discarded (kept under ``time_s_ignored`` for the audit
    trail) and replaced by the wall time.  ``time_s`` is the default primary
    objective of the optimization ratchet, so this guard is what keeps a forged
    timer from ratcheting the best-so-far.
    """
    metrics["wall_time_s"] = wall_s
    reported = metrics.get("time_s")
    numeric = isinstance(reported, (int, float)) and not isinstance(reported, bool)
    # 2% + 50 ms slack for timer granularity; wall time always includes startup.
    if numeric and 0 < float(reported) <= wall_s * 1.02 + 0.05:
        return
    if numeric:
        metrics["time_s_ignored"] = float(reported)
    metrics["time_s"] = wall_s


# ── field I/O ─────────────────────────────────────────────────────────────────

def _load_field_npz(path: str):
    try:
        import numpy as np
        data = np.load(path)
        keys = list(data.keys())
        return data[keys[0]] if keys else None
    except Exception:
        return None


def _load_field_raw(path: str, shape: tuple | None = None):
    try:
        import numpy as np
        arr = np.fromfile(path, dtype=np.float64)
        if shape:
            try:
                arr = arr.reshape(shape)
            except ValueError:
                pass
        return arr
    except Exception:
        return None


def _load_field(path: str, output_format: str = "npz", shape: tuple | None = None):
    if not os.path.isfile(path):
        return None
    if output_format == "npz":
        return _load_field_npz(path)
    if output_format == "raw_float64":
        return _load_field_raw(path, shape)
    return None


def _field_norms(a, b) -> dict:
    try:
        import numpy as np
    except ImportError:
        return {"error": "numpy not available"}
    result: dict = {
        "shape_a": list(a.shape) if hasattr(a, "shape") else None,
        "shape_b": list(b.shape) if hasattr(b, "shape") else None,
    }
    if a.shape != b.shape:
        result["shape_error"] = (
            f"Shape mismatch: a={a.shape}, b={b.shape}. "
            "Norms computed on flattened arrays (min size)."
        )
        n = min(a.size, b.size)
        a = a.flat[:n]
        b = b.flat[:n]
    else:
        a = a.ravel()
        b = b.ravel()
    diff     = a - b
    l2_abs   = float(np.linalg.norm(diff))
    linf_abs = float(np.max(np.abs(diff)))
    norm_a   = float(np.linalg.norm(a))
    norm_b   = float(np.linalg.norm(b))
    result["l2_abs"]    = l2_abs
    result["l2_rel"]    = l2_abs / norm_a if norm_a > 0 else None
    result["linf_abs"]  = linf_abs
    result["linf_rel"]  = linf_abs / max(norm_a, norm_b) if max(norm_a, norm_b) > 0 else None
    return result


# ── numerical invariants ──────────────────────────────────────────────────────

def _finite_check(field) -> bool:
    """True iff *field* contains no NaN/Inf (an empty/None field is not finite)."""
    try:
        import numpy as np
    except ImportError:
        return False
    if field is None:
        return False
    arr = np.asarray(field)
    if arr.size == 0:
        return False
    return bool(np.all(np.isfinite(arr)))


def _conservation_residual(run_metrics: dict, ref_metrics: dict, key: str) -> float | None:
    """Relative discrepancy of a conserved scalar *key* between run and reference.

    ``|run[key] - ref[key]| / |ref[key]|`` (falls back to absolute error when the
    reference value is zero).  Returns ``None`` when either value is missing or
    non-numeric — the caller decides how to treat an uncomputable invariant.
    """
    rv = run_metrics.get(key)
    fv = (ref_metrics or {}).get(key)
    try:
        rv = float(rv)
        fv = float(fv)
    except (TypeError, ValueError):
        return None
    denom = abs(fv)
    return abs(rv - fv) / denom if denom > 0 else abs(rv - fv)


def _convergence_order(pairs: list[tuple[float, float]]) -> float | None:
    """Observed order of accuracy from ``(h, error)`` pairs via a log-log fit.

    Fits ``log(error) = p·log(h) + c`` by least squares and returns the slope
    ``p``.  Needs at least two pairs with strictly positive, finite ``h`` and
    ``error``; returns ``None`` otherwise (e.g. a single resolution, or an
    error that hit exact zero and cannot be log-scaled).
    """
    try:
        import numpy as np
    except ImportError:
        return None
    pts = [(h, e) for (h, e) in pairs
           if h is not None and e is not None and h > 0 and e > 0
           and math.isfinite(h) and math.isfinite(e)]
    if len(pts) < 2:
        return None
    logs_h = np.log([h for h, _ in pts])
    logs_e = np.log([e for _, e in pts])
    slope, _intercept = np.polyfit(logs_h, logs_e, 1)
    return float(slope)


# ── requirements engine ───────────────────────────────────────────────────────

_VALID_OPT_OPERATORS = {"lt", "gt", "lte", "gte", "eq"}


def _evaluate_requirements(metrics: dict, requirements: list[dict]) -> dict:
    """Evaluate a list of requirements against a metrics dict.

    Each requirement must have keys: metric (str), operator (str), threshold (float).
    Valid operators: lt (<), gt (>), lte (<=), gte (>=), eq (==).

    Returns:
        {
            "passed": bool,
            "results": [{"metric", "operator", "threshold", "actual", "met"}, ...]
        }
    """
    results = []
    all_met = True
    for req in requirements:
        metric    = req.get("metric", "")
        operator  = req.get("operator", "")
        threshold = req.get("threshold")

        actual_raw = metrics.get(metric)
        if actual_raw is None or threshold is None or operator not in _VALID_OPT_OPERATORS:
            results.append({
                "metric":    metric,
                "operator":  operator,
                "threshold": threshold,
                "actual":    actual_raw,
                "met":       False,
                "note":      (
                    (
                        "metric not found — computed server-side and requires a "
                        "sealed reference (proxy_exec op='benchmark_create') "
                        "and/or a registry conserved_metric; values printed by "
                        "the proxy for this metric are ignored"
                        if metric in _RESERVED_METRICS else
                        "metric not found in results"
                    ) if actual_raw is None else
                    "invalid operator" if operator not in _VALID_OPT_OPERATORS else
                    "missing threshold"
                ),
            })
            all_met = False
            continue

        try:
            actual = float(actual_raw)
            thr    = float(threshold)
        except (TypeError, ValueError):
            results.append({
                "metric": metric, "operator": operator,
                "threshold": threshold, "actual": actual_raw,
                "met": False, "note": "non-numeric value",
            })
            all_met = False
            continue

        met = (
            actual <  thr if operator == "lt"  else
            actual >  thr if operator == "gt"  else
            actual <= thr if operator == "lte" else
            actual >= thr if operator == "gte" else
            actual == thr
        )
        results.append({
            "metric": metric, "operator": operator,
            "threshold": thr, "actual": actual, "met": met,
        })
        if not met:
            all_met = False

    return {"passed": all_met, "results": results}
