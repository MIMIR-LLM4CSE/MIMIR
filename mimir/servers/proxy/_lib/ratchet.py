"""The optimization ratchet's decisions and their persistence: verdicts, best-so-far,
ledger, and the measurement policy (replicates, margin). The session-level settle is in
``_ops/_eval_ratchet.py``.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone

from ....shared.atomic import write_json
from . import metric_catalog, store
from ....shared.atomic import read_json


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


def case_times(results: list[tuple[str, bool, float | None]]) -> dict[str, float | None]:
    """Each case at its fastest passing configuration; None for a case that has none."""
    best: dict[str, float | None] = {}
    for case_id, passed, time_s in results:
        held = best.setdefault(case_id, None)
        if passed and time_s is not None and (held is None or time_s < held):
            best[case_id] = time_s
    return best


def total_time(results: list[tuple[str, bool, float | None]]) -> float | None:
    """The suite's cases end to end: each case at its fastest passing configuration,
    summed; None when a case has none. What a merge of several kernels is judged on,
    where the fastest case alone would speak for one of them."""
    best = case_times(results)
    times = [t for t in best.values() if t is not None]
    return sum(times) if best and len(times) == len(best) else None


def run_case_times(run_dir: str) -> dict[str, float | None]:
    """``case_times`` of a settled run, read off its ``metrics.json``; {} without one."""
    met = read_json(os.path.join(run_dir, "metrics.json")) or {}
    return case_times([(r.get("case_id", ""), bool(r.get("passed")), (r.get("metrics") or {}).get("time_s"))
                       for r in met.get("results", [])])


# ── verdict logic ─────────────────────────────────────────────────────────────

def is_improvement(new, old, goal: str, min_improvement: float = 0.0) -> bool:
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


def relative(value, base, goal: str) -> float | None:
    """*value* against *base*, lower is better whatever the goal: 0.8 is a fifth better, for
    a time or a throughput alike. None when either is missing or zero."""
    if not isinstance(value, (int, float)) or not isinstance(base, (int, float)) or not base or not value:
        return None
    return round(value / base if goal != "max" else base / value, 6)


def _within_noise(new, old, goal: str, margin: float) -> bool:
    """True iff *new* is no worse than *old* by more than a relative *margin*."""
    if new is None or old is None:
        return old is None and new is not None
    try:
        new, old = float(new), float(old)
    except (TypeError, ValueError):
        return False
    tol = abs(old) * float(margin)
    return new >= old - tol if goal == "max" else new <= old + tol


def objectives_verdict(feasible: bool, values: dict, best_values: dict | None,
                       objectives: list[dict], margins: dict, neutral: bool = False) -> tuple[str | None, list, list]:
    """The verdict against several objectives, with those bettered (past their margin, their
    noise) and those worsened (past their tolerance). Better on some and worse on none accepts;
    better on none rejects, a *neutral* refactor no worse on any kept; better on one and worse on
    another is ``tradeoff``, which waits for a judgement. One objective judges as before.
    """
    primary = objectives[0]["metric"]
    if not feasible:
        return ("reject" if best_values is not None else None), [], []
    if best_values is None:
        return ("accept" if values.get(primary) is not None else None), [], []
    if values.get(primary) is None:
        return "reject", [], []
    better, worse = [], []
    for objective in objectives:
        name, goal = objective["metric"], objective.get("goal", "min")
        value, base = values.get(name), best_values.get(name)
        if value is None or base is None:
            continue
        margin = margins.get(name, 0.0)
        tolerance = objective.get("tolerance")
        if is_improvement(value, base, goal, margin):
            better.append(name)
        elif not _within_noise(value, base, goal, max(margin, tolerance if tolerance is not None else margin)):
            worse.append(name)
    if better and worse:
        return "tradeoff", better, worse
    if better:
        return "accept", better, worse
    return ("accept" if neutral and not worse else "reject"), better, worse


# A loss this much worse than the best, or twice the margin if larger, is a regression:
# a change that should have helped and did the opposite has a cause worth finding.
REGRESSION = 0.05


def regression(new, best, goal: str, margin: float) -> float | None:
    """How much worse *new* is than *best* (0.37 for 37 %) when it is a regression; else None."""
    rel = relative(new, best, goal)
    return round(rel - 1, 4) if rel is not None and rel - 1 > max(2 * float(margin), REGRESSION) else None


def near_miss(new, best, goal: str, margin: float) -> float | None:
    """The gain of a run better than *best* but within the margin (-0.028 for 2.8 %); else None.

    Kept aside rather than lost: small gains on independent axes add up when combined.
    """
    rel = relative(new, best, goal)
    return round(rel - 1, 4) if rel is not None and rel < 1 and not is_improvement(new, best, goal, margin) else None


# ── hardware bound ────────────────────────────────────────────────────────────

# A best within this factor of the bound is near it: a stall there is a convergence.
NEAR_BOUND = 1.5

_TIME_METRICS = ("time_s", "best_time_s", "total_time_s", "wall_time_s")


def roofline_floor(run_metrics: dict, primary_metric: str, peaks: tuple[float, float]) -> float | None:
    """The fastest *primary_metric* the hardware allows for the cases of *run_metrics*.

    Per case, ``max(flops / peak_gflops, bytes_moved / peak_bandwidth)`` from what the
    code reports and the proxy's registered *peaks* ``(gflops, gbytes_per_s)``,
    aggregated as the metric is. None without both peaks, or when a case lacks a count.
    """
    gf, bw = peaks
    if primary_metric not in _TIME_METRICS or not (gf > 0 and bw > 0):
        return None
    floors = []
    for r in run_metrics.get("results") or []:
        m = r.get("metrics") or {}
        flops, moved = m.get("flops"), m.get("bytes_moved")
        if not (isinstance(flops, (int, float)) and isinstance(moved, (int, float))):
            return None
        floors.append((r.get("case_id", ""), bool(r.get("passed")),
                       max(flops / (gf * 1e9), moved / (bw * 1e9))))
    if not floors:
        return None
    if primary_metric == "best_time_s":
        return select_best_case(floors)[1]
    if primary_metric == "total_time_s":
        return total_time(floors)
    return sum(f for _, _, f in floors) / len(floors)


def bound_of(cfg: dict, best_metrics: dict, peaks: tuple[float, float]) -> tuple[float | None, str]:
    """``(bound, basis)``: the declared bound, else the roofline floor, else ``(None, "")``."""
    declared = cfg.get("bound")
    if isinstance(declared, (int, float)) and declared > 0:
        return float(declared), str(cfg.get("bound_basis") or "declared")
    floor = roofline_floor(best_metrics, cfg.get("primary_metric", "time_s"), peaks)
    if floor:
        return floor, "roofline: flops and bytes_moved against the registered peaks"
    return None, ""


def bound_ratio(best_value, bound, goal: str) -> float | None:
    """How far the best is from the bound: 1.0 at it, 5.7 for a time 5.7 times the floor."""
    if not isinstance(best_value, (int, float)) or not isinstance(bound, (int, float)) or not best_value or not bound:
        return None
    return round(best_value / bound if goal != "max" else bound / best_value, 3)


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
    return read_json(store.opt_best_file(proxy_name))


def save_best(proxy_name: str, run_id: str, primary_value: float | None,
               tree_snapshot_id: str, wall_value: float | None = None,
               memory: dict | None = None, measured: float | None = None,
               values: dict | None = None) -> str | None:
    """Record *run_id* as best, pointing at the tree snapshot taken at its launch.

    One snapshot for the whole set, so a restore is a measured combination. *measured* is
    the run's own value when *primary_value*, the value to beat, is another run's; *values*
    are each objective's values to beat.
    """
    snapped = tree_snapshot_id or None
    record = {
        "run_id":        run_id,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        **({"measured_value": measured} if measured is not None else {}),
        **(memory or {}),
        **({"values": values} if values else {}),
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


def rewrite_ledger(proxy_name: str, rows: list[dict]) -> None:
    """Write the whole ledger again: a judgement changes a row already settled."""
    path = store.opt_ledger_file(proxy_name)
    temp = f"{path}.{os.getpid()}.tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps(row) + "\n" for row in rows)
    os.replace(temp, path)


def read_ledger(proxy_name: str) -> list[dict]:
    """Every settled run of *proxy_name*, oldest first.

    One JSON object per line, appended as runs settle. A line that does not parse is
    dropped rather than failing the read: the file is written by another process and
    its last line can be half there. An entry may carry no axis.
    """
    rows = []
    try:
        with open(store.opt_ledger_file(proxy_name), encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except ValueError:
                    continue
    except OSError:
        return []
    return rows


def baseline_value(cfg: dict, rows: list[dict]):
    """The session's baseline metric, read off its ledger *rows*; None until it is measured."""
    run_id = cfg.get("baseline_run_id")
    return next((r.get("primary_value") for r in rows if run_id and r.get("run_id") == run_id), None)


# ── measurement policy ────────────────────────────────────────────────────────

# Metrics whose value moves between two runs of identical code: the catalogue says which.
NOISY_METRICS = metric_catalog.NOISY_METRICS

# Run-level metrics summed up from the cases' ``time_s``: a replicate of a case holds that.
_FROM_CASE_TIMES = ("best_time_s", "total_time_s")

# Replicates per case when the primary metric is a noisy one and the caller expressed no
# preference. Three is the smallest number with a middle — enough for a median to mean
# something, cheap enough to be the default.
_AUTO_REPEAT = 3


def replicate_metric(primary_metric: str) -> str:
    """The metric a case's replicates hold for *primary_metric*, which their spread is read on."""
    return "time_s" if primary_metric in _FROM_CASE_TIMES else primary_metric


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
