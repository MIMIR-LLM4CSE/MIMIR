"""Settling the ratchet: one run's verdict, written once (flock, frozen ``ratchet.json``).

Called by the results op and by the detached runner; the decisions are in ``_lib/ratchet.py``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .._lib.ratchet import (
    NOISY_METRICS,
    append_ledger,
    effective_min_improvement,
    load_best,
    ratchet_verdict,
    run_primary_value,
    save_best,
)
from ....shared.atomic import write_json
from .._lib.store import file_lock, opt_session_runs_dir, read_json, save_opt_config

def ratchet_state(proxy_name: str, run_dir: str, final_metrics: dict, cfg: dict) -> dict:
    """Compute the ratchet outcome for a completed run, persisting it once.

    Whichever of the runner or ``results`` gets here first wins: the verdict is
    frozen to ``<run_dir>/ratchet.json`` under a per-session flock and re-checked
    inside it, so best/ledger/stall never move twice for the same run.
    """
    name = cfg.get("proxy_name") or proxy_name
    outcome_path = os.path.join(run_dir, "ratchet.json")

    def _frozen() -> dict | None:
        frozen = read_json(outcome_path)  # None when missing or unreadable
        if frozen is None:
            return None  # recompute under the lock
        frozen["best"] = load_best(name)
        return frozen

    out = _frozen()
    if out is not None:
        return out

    with file_lock(os.path.join(opt_session_runs_dir(name), ".ratchet.lock")):
        out = _frozen()
        if out is not None:
            return out
        return _ratchet_settle_locked(name, run_dir, final_metrics, cfg, outcome_path)


def _timing_warning(verdict: str | None, primary_metric: str, best: dict | None,
                    wall_value) -> str | None:
    """Why an accepted time_s gain may not be one.

    A self-reported time that improved while the wall time the server measured
    regressed is plausible I/O noise — and also the exact signature of a tampered
    timer. The ratchet cannot tell them apart, so it says so rather than choosing.
    """
    if verdict != "accept" or primary_metric != "time_s" or best is None:
        return None
    prev_wall = best.get("wall_value")
    if not (isinstance(wall_value, (int, float))
            and isinstance(prev_wall, (int, float))
            and wall_value > prev_wall * 1.10):
        return None
    return ("Self-reported time_s improved but the server-measured wall time "
            f"regressed ({prev_wall:.3g}s -> {wall_value:.3g}s). Verify the "
            "proxy's timing instrumentation before trusting this acceptance, "
            "or optimize primary_metric='wall_time_s' instead.")


def _run_axis(run_dir: str) -> str:
    """What the run in *run_dir* said it was trying, or "".

    Read back from the run's own config at settle time, because settling happens in the
    runner process minutes or hours after the launch that knew it.
    """
    return str((read_json(os.path.join(run_dir, "config.json")) or {}).get("axis") or "")


def _ratchet_settle_locked(
    name: str, run_dir: str, final_metrics: dict, cfg: dict, outcome_path: str,
) -> dict:
    primary_metric  = cfg.get("primary_metric", "time_s")
    goal            = cfg.get("primary_goal", "min")
    max_stall       = int(cfg.get("max_stall", 5))
    # The tree as it stood when this run was LAUNCHED — what actually ran, immune to
    # edits made while it was in flight.
    launch = read_json(os.path.join(run_dir, "tree_at_launch.json")) or {}
    launch_tree = launch.get("snapshot_id", "")

    feasible      = bool(final_metrics.get("all_passed"))
    primary_value = run_primary_value(final_metrics, primary_metric)
    wall_value    = run_primary_value(final_metrics, "wall_time_s")
    run_id        = os.path.basename(os.path.normpath(run_dir))
    best          = load_best(name)

    # A timing describes the machine as much as the code. The session is pinned to the
    # machine its baseline ran on, and a run elsewhere is not compared at all: judged
    # against a baseline from the login node, a compute node's run would be accepted
    # or rejected on the hardware difference. An accuracy metric is reproducible
    # across machines, so it is never held back.
    machine = read_json(os.path.join(run_dir, "machine.json")) or {}
    pinned = (cfg.get("machine") or {}).get("machine_signature", "")
    if (primary_metric in NOISY_METRICS and pinned
            and machine.get("machine_signature") not in ("", None, pinned)):
        return _settle_incomparable(name, run_dir, cfg, outcome_path, machine, feasible,
                                    primary_value, wall_value, primary_metric, goal,
                                    max_stall, best)

    # The baseline is the one run whose spread describes the machine rather than the
    # change, so it is where the noise floor comes from — recorded once, then applied
    # to every comparison after it.
    spread = run_primary_value(final_metrics, "primary_spread")
    if not cfg.get("baseline_run_id") and isinstance(spread, (int, float)):
        cfg["noise_floor"] = float(spread)

    min_improvement, margin_source = effective_min_improvement(cfg)
    verdict = ratchet_verdict(feasible, primary_value, best, goal, min_improvement)

    timing_warning = _timing_warning(verdict, primary_metric, best, wall_value)

    # The launch gate in prepare_run guarantees the first run measured the untouched
    # tree; this records which run that was, so the gate opens exactly once.
    baseline_run = cfg.get("baseline_run_id", "")
    if not baseline_run:
        cfg["baseline_run_id"] = baseline_run = run_id
        if machine.get("machine_signature"):
            cfg["machine"] = machine

    stall = int(cfg.get("stall", 0))
    if verdict == "accept":
        save_best(name, run_id, primary_value, launch_tree, wall_value=wall_value)
        best  = load_best(name)
        stall = 0
    elif feasible:
        stall += 1  # feasible but not an improvement — a stalled iteration

    if feasible and stall >= max_stall:
        verdict = "converged"

    cfg["stall"] = stall
    save_opt_config(cfg)

    append_ledger(name, {
        "run_id":        run_id,
        "ts":            datetime.now(timezone.utc).isoformat(),
        # What this run was and where it started. The snapshot is the edge: a run whose
        # launch tree is the state another run produced descends from it, which is what
        # makes the session's attempts a shape rather than a list.
        "axis":          _run_axis(run_dir),
        "launch_tree":   launch_tree,
        "feasible":      feasible,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        "verdict":       verdict,
        "stall":          stall,
        "primary_spread": spread,
        "min_improvement": min_improvement,
        "best_run_id":   best.get("run_id") if best else None,
        **({"timing_warning": timing_warning} if timing_warning else {}),
    })

    outcome = {
        "baseline_run_id": baseline_run,
        "feasible":       feasible,
        "primary_value":  primary_value,
        "wall_value":     wall_value,
        "verdict":        verdict,
        "stall":          stall,
        "max_stall":      max_stall,
        "primary_metric": primary_metric,
        "goal":           goal,
        # Said on every run, not buried in the config: the margin is what decides
        # accept from reject, and a model that cannot see it cannot tell an edit
        # that did nothing from one the threshold was too loose to catch.
        "min_improvement": min_improvement,
        "margin_source":   margin_source,
        "primary_spread":  spread,
        "noise_floor":     cfg.get("noise_floor"),
        "timing_warning": timing_warning,
    }
    # Frozen to disk so a later results() replays this verdict instead of
    # recomputing one against a tree that has moved on since. Best-effort on
    # purpose: the verdict is already in the ledger and in best.json, so an
    # unwritable run dir must not turn a completed measurement into a crashed
    # tool call — it only costs the replay.
    try:
        write_json(outcome_path, outcome, indent=2)
    except OSError:
        pass
    outcome["best"] = best
    return outcome


def _settle_incomparable(
    name: str, run_dir: str, cfg: dict, outcome_path: str, machine: dict,
    feasible: bool, primary_value, wall_value, primary_metric: str, goal: str,
    max_stall: int, best: dict | None,
) -> dict:
    """Freeze the outcome of a run taken on another machine than the baseline's.

    Neither best nor stall moves: the number says nothing about the edit. The run is
    still ledgered, so where it ran and what it measured stay on record.
    """
    pinned = cfg.get("machine") or {}
    run_id = os.path.basename(os.path.normpath(run_dir))
    append_ledger(name, {
        "run_id": run_id, "ts": datetime.now(timezone.utc).isoformat(),
        "axis": _run_axis(run_dir),
        "launch_tree": (read_json(os.path.join(run_dir, "tree_at_launch.json"))
                        or {}).get("snapshot_id", ""),
        "feasible": feasible, "primary_value": primary_value, "wall_value": wall_value,
        "verdict": "incomparable", "stall": int(cfg.get("stall", 0)),
        "machine": machine.get("host", ""),
        "best_run_id": best.get("run_id") if best else None,
    })
    outcome = {
        "baseline_run_id": cfg.get("baseline_run_id", ""),
        "feasible": feasible, "primary_value": primary_value, "wall_value": wall_value,
        "verdict": "incomparable", "stall": int(cfg.get("stall", 0)),
        "max_stall": max_stall, "primary_metric": primary_metric, "goal": goal,
        "min_improvement": None, "margin_source": "", "primary_spread": None,
        "noise_floor": cfg.get("noise_floor"), "timing_warning": None,
        "machine": {"run": machine, "baseline": pinned},
    }
    # Best-effort for the same reason the settled outcome is: the verdict is already in
    # the ledger, so an unwritable run dir costs the replay and nothing else.
    try:
        write_json(outcome_path, outcome, indent=2)
    except OSError:
        pass
    outcome["best"] = best
    return outcome
