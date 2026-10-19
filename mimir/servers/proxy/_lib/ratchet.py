"""The optimization ratchet's decisions and their persistence: verdicts, best-so-far,
ledger, and the measurement policy (replicates, margin). The session-level settle is in
``_ops/_eval_ratchet.py``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ....shared.atomic import write_json
from . import store


# ── case selection ────────────────────────────────────────────────────────────

def select_best_case(
    results: list[tuple[str, bool, float | None]],
) -> tuple[str | None, float | None]:
    """The fastest passing case of ``(case_id, passed, time_s)``, or ``(None, None)``.

    Across configurations, not replicates (those are reduced by median before).
    """
    best_case: str | None = None
    best_key: float | None = None
    best_time: float | None = None
    for case_id, passed, time_s in results:
        if not passed:
            continue
        key = time_s if time_s is not None else float("inf")
        if best_key is None or key < best_key:
            best_case, best_key, best_time = case_id, key, time_s
    return best_case, best_time


# ── verdict logic ─────────────────────────────────────────────────────────────

def _is_improvement(new, old, goal: str, min_improvement: float = 0.0) -> bool:
    """True iff *new* beats *old* by more than a relative *min_improvement* margin.

    ``goal == "max"`` treats larger as better; anything else minimizes.  A missing
    *old* (no incumbent) counts as an improvement; a missing *new* never does.
    """
    if new is None:
        return False
    if old is None:
        return True
    try:
        new = float(new)
        old = float(old)
    except (TypeError, ValueError):
        return False
    tol = abs(old) * float(min_improvement)
    return new > old + tol if goal == "max" else new < old - tol


def ratchet_verdict(feasible: bool, primary_value, best: dict | None,
                     goal: str, min_improvement: float = 0.0) -> str | None:
    """``"accept"`` (feasible and better by *min_improvement*, or first), ``"reject"``, or None
    (infeasible with no incumbent).
    """
    if not feasible:
        return "reject" if best is not None else None
    if best is None:
        return "accept"
    best_val = best.get("primary_value")
    return "accept" if _is_improvement(primary_value, best_val, goal, min_improvement) else "reject"


def run_primary_value(run_metrics: dict, primary_metric: str):
    """Scalar objective of a completed run for the given *primary_metric*.

    Prefers a run-level metric of that name (e.g. ``best_time_s``,
    ``convergence_order``); otherwise averages the per-case value across the run's
    cases.  Returns ``None`` when the metric appears nowhere.
    """
    top = run_metrics.get(primary_metric)
    if isinstance(top, (int, float)):
        return float(top)
    vals = []
    for r in run_metrics.get("results", []):
        v = (r.get("metrics") or {}).get(primary_metric)
        if isinstance(v, (int, float)):
            vals.append(float(v))
    return sum(vals) / len(vals) if vals else None


# ── best-so-far persistence + ledger ──────────────────────────────────────────

def load_best(proxy_name: str) -> dict | None:
    """Return the best-so-far record, or ``None`` if none has been recorded yet."""
    return store.read_json(store.opt_best_file(proxy_name))


def save_best(proxy_name: str, run_id: str, primary_value: float | None,
               tree_snapshot_id: str, wall_value: float | None = None) -> str | None:
    """Record *run_id* as best, pointing at the tree snapshot taken at its launch.

    One snapshot for the whole set, so a restore is a measured combination.
    """
    snapped = tree_snapshot_id or None
    record = {
        "run_id":        run_id,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        "tree_snapshot": snapped,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }
    try:
        write_json(store.opt_best_file(proxy_name), record, indent=2)
    except OSError:
        pass
    return snapped


def append_ledger(proxy_name: str, entry: dict) -> None:
    """Append one JSON entry as a line to the session ledger (best-effort)."""
    path = store.opt_ledger_file(proxy_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass


# ── measurement policy ────────────────────────────────────────────────────────

# Metrics whose value moves between two runs of identical code. A timing metric on a
# shared node is the whole reason this file needed a notion of noise; an accuracy metric
# against a sealed reference is reproducible to the bit, and repeating it buys nothing.
NOISY_METRICS = ("time_s", "wall_time_s", "elapsed_s", "runtime_s", "gflops_per_s",
                  "bandwidth_gbytes_per_s")

# Replicates per case when the primary metric is a noisy one and the caller expressed no
# preference. Three is the smallest number with a middle — enough for a median to mean
# something, cheap enough to be the default.
_AUTO_REPEAT = 3


def effective_repeat(cfg: dict) -> int:
    """How many times to measure each case: what was asked, else what the metric needs."""
    asked = int(cfg.get("repeat", 0) or 0)
    if asked > 0:
        return asked
    return _AUTO_REPEAT if cfg.get("primary_metric", "time_s") in NOISY_METRICS else 1


def effective_min_improvement(cfg: dict) -> tuple[float, str]:
    """``(margin, source)``: the configured margin, or the baseline's measured spread if larger."""
    configured = float(cfg.get("min_improvement", 0.0) or 0.0)
    floor = cfg.get("noise_floor")
    if not isinstance(floor, (int, float)) or floor <= configured:
        return configured, "configured"
    return float(floor), "measured noise floor"
