"""The optimization ratchet: verdicts, best-so-far tracking, ledger, and the
measurement policy underneath them.

Pure decision logic plus its persistence.  A completed run is *accepted* when
it is feasible (all requirement constraints pass) and improves the primary
metric; regressions are *rejected*.  The session-level settle (stall counter,
frozen ratchet.json, flock serialization) lives in ``_ops/_eval_ratchet.py``;
this module owns the reusable pieces under it — including how many times a case
is measured and what margin an improvement has to clear, which are decisions
about measurement rather than about any one session.
"""

from __future__ import annotations

import json
import os
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

    A minimum, and deliberately so: cases in a suite are different configurations
    — grid sizes, backends, thread counts — not repeated measurements of one, so
    "the fastest configuration" is a real answer to a real question. The selection
    bias that a minimum *does* carry lives one level down, between replicates of a
    single case, and is handled there: the runner reduces replicates by median
    before a case ever reaches this function. Reporting the minimum across
    replicates was how a headline 3.2x speed-up came out of a distribution whose
    middle said 2.8x.

    ``primary_value`` — what the ratchet actually compares — does not come from
    here in any case; see ``_run_primary_value``, which averages across cases
    rather than taking the best of them.
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
               tree_snapshot_id: str, wall_value: float | None = None) -> str | None:
    """Record *run_id* as best-so-far, pointing at the tree snapshot that produced it.

    *tree_snapshot_id* identifies the state of every tracked file at the moment the run
    was LAUNCHED — one id for the whole set, which is what makes reset_to_best restore a
    combination that was actually measured. Keeping a best per file would let a restore
    assemble file A from one run beside file B from another.

    ``wall_value`` is the run's server-measured wall time, kept alongside the (possibly
    self-reported) primary value so the timing audit can compare an accepted run's wall
    time against the incumbent's. Returns the snapshot id it recorded, or None when the
    run had none (the pointer is still written so best tracking survives).
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


# ── measurement policy ────────────────────────────────────────────────────────

# Metrics whose value moves between two runs of identical code. A timing metric on a
# shared node is the whole reason this file needed a notion of noise; an accuracy metric
# against a sealed reference is reproducible to the bit, and repeating it buys nothing.
_NOISY_METRICS = ("time_s", "wall_time_s", "elapsed_s", "runtime_s", "gflops_per_s",
                  "bandwidth_gbytes_per_s")

# Replicates per case when the primary metric is a noisy one and the caller expressed no
# preference. Three is the smallest number with a middle — enough for a median to mean
# something, cheap enough to be the default.
_AUTO_REPEAT = 3


def _effective_repeat(cfg: dict) -> int:
    """How many times to measure each case: what was asked, or what the metric needs.

    ``repeat=0`` means "decide for me", and the decision is made on the metric rather
    than on a constant: repeating a bit-reproducible accuracy check is pure cost, and
    not repeating a timing on a shared node is how an edit worth nothing gets accepted
    on that node's own noise.
    """
    asked = int(cfg.get("repeat", 0) or 0)
    if asked > 0:
        return asked
    return _AUTO_REPEAT if cfg.get("primary_metric", "time_s") in _NOISY_METRICS else 1


def _effective_min_improvement(cfg: dict) -> tuple[float, str]:
    """The margin a run must clear, and where that number came from.

    A caller-supplied constant annotated "guards timing noise" is an assertion about a
    machine nobody has measured; where the real floor is higher, the guard admits
    exactly the changes it exists to exclude.

    So the configured value is a floor, not the answer. Once the baseline has been
    measured more than once, the spread of those measurements is known, and the margin
    is whichever of the two is larger.
    """
    configured = float(cfg.get("min_improvement", 0.0) or 0.0)
    floor = cfg.get("noise_floor")
    if not isinstance(floor, (int, float)) or floor <= configured:
        return configured, "configured"
    return float(floor), "measured noise floor"
