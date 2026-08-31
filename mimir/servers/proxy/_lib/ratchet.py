"""The optimization ratchet: verdicts, best-so-far tracking, and the ledger.

Pure decision logic plus its persistence.  A completed run is *accepted* when
it is feasible (all requirement constraints pass) and improves the primary
metric; regressions are *rejected*.  The session-level settle (stall counter,
frozen ratchet.json, flock serialization) lives in ``_ops/eval_session.py``;
this module owns the reusable pieces under it.
"""

from __future__ import annotations

import json
import os
import shutil
from datetime import datetime, timezone

from _lib import store


# ── case selection ────────────────────────────────────────────────────────────

def _select_best_case(
    results: list[tuple[str, bool, float | None]],
) -> tuple[str | None, float | None]:
    """Pick the best case from ``(case_id, passed, time_s)`` triples.

    The best case is the **passing** case with the smallest ``time_s``; a
    ``None`` time sorts as +infinity so any case with a real time wins over one
    without.  Returns ``(None, None)`` when no case passed — a failing case is
    never reported as best.
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


def _ratchet_verdict(feasible: bool, primary_value, best: dict | None,
                     goal: str, min_improvement: float = 0.0) -> str | None:
    """Decide whether a completed run should be accepted or rejected.

    * ``"accept"`` — run is feasible (all constraints pass) and either there is no
      incumbent, or it improves the primary metric beyond *min_improvement*.
    * ``"reject"`` — run regressed: infeasible while an incumbent exists, or feasible
      but no better than the incumbent.  The caller steers toward reset_to_best.
    * ``None`` — infeasible with no incumbent yet (nothing to revert to; keep editing).
    """
    if not feasible:
        return "reject" if best is not None else None
    if best is None:
        return "accept"
    best_val = best.get("primary_value")
    return "accept" if _is_improvement(primary_value, best_val, goal, min_improvement) else "reject"


def _run_primary_value(run_metrics: dict, primary_metric: str):
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

def _load_best(proxy_name: str) -> dict | None:
    """Return the best-so-far record, or ``None`` if none has been recorded yet."""
    return store._read_json(store._opt_best_file(proxy_name))


def _save_best(proxy_name: str, run_id: str, primary_value: float | None,
               source_path: str, wall_value: float | None = None) -> str | None:
    """Record *run_id* as best-so-far and snapshot its source for reset_to_best.

    ``wall_value`` is the run's server-measured wall time, kept alongside the
    (possibly self-reported) primary value so the timing audit can compare an
    accepted run's wall time against the incumbent's.  Returns the snapshot
    path on success, ``None`` if the source could not be snapshotted (the
    pointer is still written so best tracking survives).
    """
    snap = store._opt_best_source_path(proxy_name, source_path)
    snapped: str | None = None
    if source_path and os.path.isfile(source_path):
        try:
            os.makedirs(os.path.dirname(snap), exist_ok=True)
            shutil.copy2(source_path, snap)
            snapped = snap
        except OSError:
            snapped = None
    record = {
        "run_id":        run_id,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        "source_snapshot": snapped,
        "updated_at":    datetime.now(timezone.utc).isoformat(),
    }
    try:
        store._write_json_atomic(store._opt_best_file(proxy_name), record)
    except OSError:
        pass
    return snapped


def _append_ledger(proxy_name: str, entry: dict) -> None:
    """Append one JSON entry as a line to the session ledger (best-effort)."""
    path = store._opt_ledger_file(proxy_name)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps(entry) + "\n")
    except OSError:
        pass

