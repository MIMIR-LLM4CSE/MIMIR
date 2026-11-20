"""Settling the ratchet: one run's verdict, written once (flock, frozen ``ratchet.json``).

Called by the results op and by the detached runner; the decisions are in ``_lib/ratchet.py``.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

from .._lib.ratchet import (
    NEAR_BOUND,
    NOISY_METRICS,
    append_ledger,
    bound_of,
    bound_ratio,
    effective_min_improvement,
    load_best,
    near_miss,
    objectives_verdict,
    read_ledger,
    regression,
    rewrite_ledger,
    run_primary_value,
    save_best,
)
from ....shared.atomic import write_json
from .._lib import metric_catalog, spec as spec_lib
from .._lib.report import resolve_roofline
from .._lib.requests import is_refactor
from .._lib.store import (file_lock, load_opt_config, load_ref_metrics, load_registry, load_suite,
                          opt_best_file, save_opt_config)
from .._lib.layout import opt_session_runs_dir
from ....shared.atomic import read_json

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


_TRADEOFF_STEP = (
    "Run {run} bettered {better} but worsened {worse} past its tolerance: it is not the best "
    "until judged. Weigh the two against the primary objective and what was asked, then "
    "report_verdict(run='{run}', verdict='pass' to keep it or 'rejected', reason=...); when the "
    "call is close, ask the user first (ask_user_question, your choice first).")


def _values_of(final_metrics: dict, cfg: dict, primary_value) -> dict:
    """The run's value of every metric its spec names, and of its memory."""
    names = [o["metric"] for o in spec_lib.of(cfg)["objectives"]]
    names += [r["metric"] for r in cfg.get("requirements") or [] if r.get("metric") != "returncode"]
    names += ["wall_time_s", *MEMORY_KEYS]
    values = {name: run_primary_value(final_metrics, name) for name in dict.fromkeys(names)}
    values[cfg.get("primary_metric", "time_s")] = primary_value
    return {name: value for name, value in values.items() if value is not None}


def _relative_broken(cfg: dict, values: dict) -> list[dict]:
    """The constraints relative to the baseline this run breaks: each with its threshold. The
    baseline's own value of an error to its sealed start is its replicates' spread."""
    rel = spec_lib.relative(cfg.get("requirements") or [])
    if not rel:
        return []
    run_id = cfg.get("baseline_run_id")
    row = next((r for r in read_ledger(cfg.get("proxy_name", "")) if run_id and r.get("run_id") == run_id), {})
    base = {**({cfg.get("primary_metric", "time_s"): row.get("primary_value")}), **(row.get("values") or {})}
    broken = []
    for req in rel:
        metric, operator = req["metric"], req["operator"]
        reference, actual, floor = base.get(metric), values.get(metric), req.get("threshold")
        if req.get("precision_floor"):
            floor = max(float(floor or 0.0), _precision_floor(cfg)) or None
        bounds = [float(b) for b in (None if reference is None else float(reference) * float(req["of_baseline"]),
                                     floor) if b is not None]
        if not bounds or actual is None:
            broken.append({**req, "actual": actual, "note": "no value to compare"})
            continue
        # With a threshold and a factor of the baseline, the looser holds: the floor a code
        # reproducible to the bit is held to, or the spread its own replicates showed.
        threshold = max(bounds) if operator in ("lt", "lte") else min(bounds)
        met = {"lt": actual < threshold, "lte": actual <= threshold, "gt": actual > threshold,
               "gte": actual >= threshold, "eq": actual == threshold}[operator]
        if not met:
            broken.append({**req, "threshold": threshold, "actual": actual})
    return broken


# How close a field reordered by an optimisation can stay to its start, by its type: summing in
# another order moves the last bits, which a float32 field holds far fewer of.
_PRECISION = {"float64": 1e-10, "float32": 1e-5, "float16": 1e-2}


def _precision_floor(cfg: dict) -> float:
    """The loosest floor the types of the session's sealed starts call for; 0 when none says."""
    cases = (load_suite(cfg.get("benchmark_name", "")) or {}).get("cases") or []
    found = [(load_ref_metrics(c["reference_name"]) or {}).get("dtype") for c in cases if c.get("reference_name")]
    return max((_PRECISION.get(str(d), 0.0) for d in found), default=0.0)


def _timing_warning(verdict: str | None, primary_metric: str, best: dict | None,
                    wall_value, measured: bool = False) -> str | None:
    """Why an accepted time_s gain may not be one: a self-reported time that improved
    while the measured wall time regressed is noise or a tampered timer, and the ratchet
    cannot tell which. A time *measured* over the harness's regions raises no doubt.
    """
    if verdict != "accept" or primary_metric != "time_s" or best is None or measured:
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


# How much more memory an accepted run may take than the best it replaces before the
# reply says so: allocators and page rounding move a peak by a few percent on their own.
MEMORY_GROWTH_WARNED = 0.25
# The peaks a warning compares, the code's own first.
MEMORY_KEYS = ("region_peak_rss_total_mb", "region_peak_rss_mb", "peak_rss_total_mb", "peak_rss_mb")


def _memory_warning(verdict: str | None, primary_metric: str, best: dict | None, memory: dict) -> str | None:
    """Why an accepted gain may have a cost the objective does not see: a peak memory
    well above the best's it replaces; over the harness's regions when both runs have
    them (the harness's own memory is not the code's), every process's summed (an MPI
    run's ranks) before the largest process's."""
    key = next((k for k in MEMORY_KEYS if (best or {}).get(k) and memory.get(k)), "peak_rss_mb")
    peak, before = memory.get(key), (best or {}).get(key)
    if (verdict != "accept" or primary_metric in MEMORY_KEYS
            or not isinstance(peak, (int, float)) or not isinstance(before, (int, float)) or before <= 0
            or peak <= before * (1 + MEMORY_GROWTH_WARNED)):
        return None
    return (f"Accepted, but its {key} grew {peak / before - 1:+.0%} ({before:.0f} -> {peak:.0f} MB). "
            f"If memory matters here, add a requirement on {key}; "
            "proxy_eval(op='profile', level='memory') shows what allocates it.")


def _bound(name: str, cfg: dict, best: dict | None) -> tuple[float | None, str]:
    """The session's hardware bound for the best run's cases: declared, else its roofline."""
    best_dir = os.path.join(opt_session_runs_dir(name), str((best or {}).get("run_id") or ""))
    try:
        peaks = resolve_roofline(load_registry().get(name) or {})
    except Exception:
        peaks = (0.0, 0.0)
    return bound_of(cfg, read_json(os.path.join(best_dir, "metrics.json")) or {}, peaks)


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

    # The first run that measured the metric is the baseline. A run without a value leaves
    # the launch gate closed: nothing could be compared to it. Its spread describes the
    # machine rather than the change, so it sets the noise floor every comparison uses.
    spread = run_primary_value(final_metrics, "primary_spread")
    is_baseline = not cfg.get("baseline_run_id") and primary_value is not None
    if is_baseline and isinstance(spread, (int, float)):
        cfg["noise_floor"] = float(spread)

    min_improvement, margin_source = effective_min_improvement(cfg)
    # A run tagged with a refactor request's axis is kept when it costs nothing measurable.
    neutral = is_refactor(_run_axis(run_dir))
    objectives = spec_lib.of(cfg)["objectives"]
    values = _values_of(final_metrics, cfg, primary_value)
    broken = [] if is_baseline else _relative_broken(cfg, values)
    feasible = feasible and not broken
    best_values = None if best is None else (best.get("values") or {primary_metric: best.get("primary_value")})
    # The margin is the noise's: of the primary, and of any objective two runs of one code move.
    margins = {o["metric"]: (min_improvement if o is objectives[0] or metric_catalog.is_noisy(cfg, o["metric"])
                             else 0.0) for o in objectives}
    verdict, better, worse = objectives_verdict(feasible, values, best_values, objectives, margins, neutral=neutral)
    kept_neutral = neutral and verdict == "accept" and best is not None and not better

    # A metric the spec already holds, as an objective or a constraint, is judged: no warning on it.
    covered = {o["metric"] for o in objectives} | {r.get("metric") for r in cfg.get("requirements") or []}
    measured = any((r.get("metrics") or {}).get("regions") for r in final_metrics.get("results") or [])
    timing_warning = (None if "wall_time_s" in covered
                      else _timing_warning(verdict, primary_metric, best, wall_value, measured))
    memory = {k: final_metrics[k] for k in MEMORY_KEYS if final_metrics.get(k) is not None}
    memory_warning = (None if covered & set(MEMORY_KEYS)
                      else _memory_warning(verdict, primary_metric, best, memory))

    # The launch gate in prepare_run guarantees this run measured the untouched tree;
    # recording it opens the gate, exactly once.
    baseline_run = cfg.get("baseline_run_id", "")
    if is_baseline:
        cfg["baseline_run_id"] = baseline_run = run_id
        if machine.get("machine_signature"):
            cfg["machine"] = machine

    # A rejected run says what its code did, not whether its idea is wrong: a regression
    # is a version to diagnose, a gain under the margin one to combine with others.
    best_val = best.get("primary_value") if best else None
    regressed = missed = None
    if verdict == "reject" and feasible:
        regressed = regression(primary_value, best_val, goal, min_improvement)
        missed = near_miss(primary_value, best_val, goal, min_improvement)

    stall = int(cfg.get("stall", 0))
    if verdict == "accept":
        # A refactor kept as neutral moves the code, not the value to beat: the faster stays
        # the reference, or neutral runs would let the time drift up a margin at a time.
        if kept_neutral:
            save_best(name, run_id, best_val, launch_tree, wall_value=best.get("wall_value"),
                      memory=memory, measured=primary_value, values=best_values)
        else:
            save_best(name, run_id, primary_value, launch_tree, wall_value=wall_value, memory=memory,
                      values=values)
        best  = load_best(name)
        stall = 0
        cfg["stall_reviewed"] = False
    elif verdict != "tradeoff" and feasible and primary_value is not None:
        stall += 1  # feasible but not an improvement; a tradeoff counts once judged

    bound = basis = ratio = None
    if feasible and stall >= max_stall:
        bound, basis = _bound(name, cfg, best)
        ratio = bound_ratio(best.get("primary_value") if best else None, bound, goal) if bound else None
        if (ratio is not None and ratio <= NEAR_BOUND) or cfg.get("stall_reviewed"):
            verdict = "converged"
        else:
            # Far from the bound, or with none known: the stop rule alone does not end the
            # search. One more streak follows a review of what limits the code.
            verdict = "stalled"
            cfg["stall_reviewed"] = True
            stall = 0

    cfg["stall"] = stall
    save_opt_config(cfg)
    bound_fields = ({"bound": bound, "bound_basis": basis, "bound_ratio": ratio}
                    if verdict in ("stalled", "converged") else {})
    judged = {
        "values": values,
        **({"better": better, "worse": worse} if better or worse else {}),
        **({"broken": broken} if broken else {}),
        **({"regressed": regressed} if regressed is not None else {}),
        **({"near_miss": missed} if missed is not None else {}),
        **bound_fields,
    }

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
        **({"neutral": True} if kept_neutral else {}),
        **judged,
        # What the best was once this run settled: a judgement taking it back restores it.
        **({"best_after": best} if verdict == "accept" else {}),
        **memory,
        **({"timing_warning": timing_warning} if timing_warning else {}),
        **({"memory_warning": memory_warning} if memory_warning else {}),
    })

    outcome = {
        "baseline_run_id": baseline_run,
        "feasible":       feasible,
        "primary_value":  primary_value,
        "wall_value":     wall_value,
        "verdict":        verdict,
        **({"neutral": "kept as a refactor: no better, and no worse than the best beyond the margin"}
           if kept_neutral else {}),
        "stall":          stall,
        "max_stall":      max_stall,
        **judged,
        **({"next_step": _TRADEOFF_STEP.format(run=run_id, better=", ".join(better), worse=", ".join(worse))}
           if verdict == "tradeoff" else {}),
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
        **memory,
        "memory_warning": memory_warning,
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


def judge(proxy_name: str, run_id: str, decision: str, reason: str) -> dict:
    """The agent's verdict on a run (``report_verdict``): settle a ``tradeoff`` ("accept" makes it
    the best), or take back the best ("reject" of the best, say for breaking a directive: the best
    before it comes back). The ledger keeps the judgement and why; the user's word reaches it
    through the agent's question."""
    from . import err, ok
    if decision not in ("accept", "reject"):
        return err("decision is 'accept' or 'reject'.")
    if not reason.strip():
        return err("A judgement says why: give reason.")
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.")
    name = cfg.get("proxy_name") or proxy_name
    with file_lock(os.path.join(opt_session_runs_dir(name), ".ratchet.lock")):
        rows = read_ledger(name)
        at = next((i for i, r in enumerate(rows) if r.get("run_id") == run_id), None)
        if at is None:
            return err(f"No settled run {run_id} in {name}.", hint="proxy_eval_status(op='runs') lists them.")
        row, best = rows[at], load_best(name)
        is_best = bool(best and best.get("run_id") == run_id)
        if row.get("verdict") != "tradeoff" and not (decision == "reject" and is_best):
            return err(f"{run_id} is {row.get('verdict') or 'unjudged'}: only a tradeoff is judged, "
                       "or the best taken back.")
        stall = int(cfg.get("stall", 0))
        if decision == "accept":
            best = {"run_id": run_id, "primary_value": row.get("primary_value"), "wall_value": row.get("wall_value"),
                    **{k: row[k] for k in ("peak_rss_mb", "peak_rss_total_mb") if row.get(k) is not None},
                    "values": row.get("values") or {}, "tree_snapshot": row.get("launch_tree") or None,
                    "updated_at": datetime.now(timezone.utc).isoformat()}
            stall = 0
            row["best_after"] = best
        else:
            if is_best:
                before = next((r.get("best_after") for r in reversed(rows[:at])
                               if r.get("verdict") == "accept" and r.get("best_after")), None)
                best = before
            else:
                stall += 1
        if best is None:
            try:
                os.remove(opt_best_file(name))
            except OSError:
                pass
        else:
            write_json(opt_best_file(name), best, indent=2)
        row.update({"verdict": decision, "judged_by": "agent", "reason": reason.strip()})
        rows[at] = row
        rewrite_ledger(name, rows)
        cfg["stall"] = stall
        save_opt_config(cfg)
        frozen_path = os.path.join(opt_session_runs_dir(name), run_id, "ratchet.json")
        frozen = read_json(frozen_path)
        if frozen is not None:
            frozen.update({"verdict": decision, "judged_by": "agent", "reason": reason.strip()})
            frozen.pop("next_step", None)
            try:
                write_json(frozen_path, frozen, indent=2)
            except OSError:
                pass
    restore = ("" if decision == "accept" and not is_best else
               " The files still hold its code: proxy_eval(op='reset_to_best', confirm=True) puts the best back."
               if is_best else "")
    return ok({"run_id": run_id, "verdict": decision, "judged_by": "agent", "best": best,
               "note": f"{run_id} judged {decision}.{restore}"})


def retire_failing_best(name: str, cfg: dict, reason: str) -> dict | None:
    """Under constraints just changed, retire the best that no longer meets them: the last
    kept run before it that does is the best again, else there is none. Each kept run passed
    over is marked ``retired`` in its ledger row, with why. None when the best still holds.
    """
    from .._lib import metrics as metrics_mod
    with file_lock(os.path.join(opt_session_runs_dir(name), ".ratchet.lock")):
        best = load_best(name)
        if not best:
            return None
        rows = read_ledger(name)
        absolute = spec_lib.absolute(cfg.get("requirements") or [])

        def fails(run_id: str, values: dict) -> list[dict]:
            met = read_json(os.path.join(opt_session_runs_dir(name), run_id, "metrics.json")) or {}
            out = []
            for r in met.get("results") or []:
                checked = metrics_mod.evaluate_requirements(r.get("metrics") or {}, absolute)
                out += [{**x, "case_id": r.get("case_id", "")} for x in checked["results"] if not x["met"]]
            return out + _relative_broken(cfg, values)

        failed = fails(best["run_id"], best.get("values") or {})
        if not failed:
            return None
        # Walk back the kept runs, newest first: each accept's row holds the best it made.
        retired, restored = {best["run_id"]: failed}, None
        for row in reversed(rows):
            made = row.get("best_after")
            if row.get("verdict") != "accept" or not made or row.get("retired") or made["run_id"] in retired:
                continue
            why = fails(made["run_id"], made.get("values") or {})
            if not why:
                restored = made
                break
            retired[made["run_id"]] = why
        for row in rows:
            if row.get("run_id") in retired:
                row["retired"] = {"reason": reason, "failed": retired[row["run_id"]][:8]}
        rewrite_ledger(name, rows)
        if restored is None:
            try:
                os.remove(opt_best_file(name))
            except OSError:
                pass
        else:
            write_json(opt_best_file(name), restored, indent=2)
        cfg["stall"], cfg["stall_reviewed"] = 0, False
        save_opt_config(cfg)
    return {"retired": best["run_id"], "also_retired": sorted(set(retired) - {best["run_id"]}),
            "failed": failed[:8], "best": restored}
