"""Observe ops of an optimization session (``proxy_eval_status``): status, results,
log, runs, diff, config, the graph and the axes tried. Read-only."""

from __future__ import annotations

import json
import os

from . import with_next, err, ok
from .._lib.procs import (
    run_log_path,
    read_log,
    run_state,
    opt_active_run_dir,
)
from .._lib.ratchet import load_best
from .._lib.report import diff_run_pair, parse_runner_log
from ._eval_ratchet import ratchet_state
from ..._shared import build_progress
from .._lib import build, procs
from .._lib.store import (
    load_registry,
    load_opt_config,
    opt_session_runs_dir,
    opt_ledger_file,
    resolve_proxy_name,
    read_json,
    run_dir_names,
)
from .eval_common import (
    LONG_BUILD_S,
    NEXT_RESET_BEST,
    NEXT_RESULTS,
    NEXT_RUN,
    eval_background_descriptor,
    opt_tail_log,
)


# The edge of the attempt graph is `best_run_id` (the incumbent a run was measured
# against), not `launch_tree` (a node's own identity, the state a restore returns to).


def _read_ledger(proxy_name: str) -> list[dict]:
    """Every settled run of *proxy_name*, oldest first.

    One JSON object per line, appended as runs settle. A line that does not parse is
    dropped rather than failing the read: the file is written by another process and
    its last line can be half there. An entry may carry no axis.
    """
    rows = []
    try:
        with open(opt_ledger_file(proxy_name), encoding="utf-8") as fh:
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


def graph(proxy_name: str = "") -> dict:
    """Every axis tried in every proxy's session, with the incumbent each was measured against.

    Structure, never causality: attempts on one incumbent are only candidates to combine.
    """
    names = [proxy_name] if proxy_name else sorted(load_registry().keys())
    axes, nodes, edges = [], [], []
    for name in names:
        cfg = load_opt_config(name) or {}
        ledger = _read_ledger(name)
        if not cfg and not ledger:
            continue
        best = load_best(name) or {}
        accepted = sum(1 for row in ledger if row.get("verdict") == "accept")
        axes.append({
            "proxy":           name,
            "branch":          cfg.get("branch", ""),
            "metric":          cfg.get("primary_metric", ""),
            "goal":            cfg.get("primary_goal", ""),
            "baseline_run_id": cfg.get("baseline_run_id", ""),
            "best_run_id":     best.get("run_id", ""),
            "best_value":      best.get("primary_value"),
            "runs":            len(ledger),
            "accepted":        accepted,
        })
        for row in ledger:
            run_id = row.get("run_id", "")
            nodes.append({
                "proxy":         name,
                "run_id":        run_id,
                "axis":          row.get("axis", ""),
                "verdict":       row.get("verdict", ""),
                "feasible":      bool(row.get("feasible")),
                "primary_value": row.get("primary_value"),
                "launch_tree":   row.get("launch_tree", ""),
                "ts":            row.get("ts", ""),
            })
            parent = row.get("best_run_id")
            if parent and parent != run_id:
                edges.append({"proxy": name, "from": parent, "to": run_id})

    # Attempts that were judged against the same incumbent: each was made without the
    # others in the tree, so none of them contains another's edit.
    from_same: dict[tuple, list[dict]] = {}
    for edge in edges:
        from_same.setdefault((edge["proxy"], edge["from"]), []).append(edge["to"])
    by_id = {(n["proxy"], n["run_id"]): n for n in nodes}
    siblings = [
        {"proxy": proxy, "incumbent": parent,
         "attempts": [{k: by_id[(proxy, run)][k]
                       for k in ("run_id", "axis", "verdict", "primary_value")}
                      for run in runs if (proxy, run) in by_id]}
        for (proxy, parent), runs in sorted(from_same.items()) if len(runs) > 1
    ]

    return ok({
        "axes":  axes,
        "nodes": nodes,
        "edges": edges,
        "from_the_same_point": siblings,
        "note": (
            "An edge points from the incumbent a run was measured against to the run "
            "itself. Attempts under one incumbent were each made without the others, "
            "so they are independent — candidates for being combined, never a promise "
            "that they add up. Only a run that combines them measures that. Use the "
            "rejected axes too: they are the ideas already ruled out on this code."
        ),
    })


# How much of the picture rides back with every settled run. Small on purpose: this is
# appended to a payload the agent reads after each run, not a report it asked for.
_AXES_SHOWN = 8


def axes_tried(proxy_name: str = "") -> dict:
    """A compact account of this session's attempts, attached to results; {} on any failure."""
    try:
        rows = _read_ledger(resolve_proxy_name(proxy_name) or "")
    except Exception:
        return {}
    if not rows:
        return {}
    accepted = [{"run_id": r.get("run_id", ""), "axis": r.get("axis", ""),
                 "primary_value": r.get("primary_value")}
                for r in rows if r.get("verdict") == "accept"]
    # Names only, and deduplicated: what this is for is stopping an idea being paid
    # for twice, so the same word said three times is one fact.
    rejected, seen = [], set()
    for row in rows:
        axis = (row.get("axis") or "").strip()
        if row.get("verdict") == "reject" and axis and axis not in seen:
            seen.add(axis)
            rejected.append(axis)
    out = {
        "tried": len(rows),
        "accepted": accepted[-_AXES_SHOWN:],
        "rejected": rejected[-_AXES_SHOWN:],
    }
    # Attempts judged against one incumbent were each made without the others in the
    # tree, so no one of them contains another's edit.
    from_same: dict[str, list[str]] = {}
    for row in rows:
        parent, axis = row.get("best_run_id"), (row.get("axis") or "").strip()
        if parent and parent != row.get("run_id") and row.get("verdict") == "accept" and axis:
            from_same.setdefault(parent, []).append(axis)
    pairs = [sorted(set(axes)) for axes in from_same.values() if len(set(axes)) > 1]
    if pairs:
        out["independent"] = pairs[-_AXES_SHOWN:]
        out["note"] = (
            "Axes under one incumbent were each measured without the others, so none "
            "contains another's edit — worth a run that combines them, never a promise "
            "that they add up. The rejected ones are ideas already ruled out on this "
            "code: choose something else rather than paying for them again."
        )
    elif rejected:
        out["note"] = ("Already ruled out on this code — choose something else rather "
                       "than paying for them again.")
    return out


def status(proxy_name: str = "") -> dict:
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(with_next(
            {"state": "no_session", "note": "No proxy name or active session."},
            "proxy_eval(op='init', ...)"))
    run_dir = opt_active_run_dir(resolved)
    if not run_dir:
        return ok(with_next(
            {"state": "no_runs", "note": "No optimization runs found."}, NEXT_RUN))
    rs         = run_state(run_dir)
    last_lines = opt_tail_log(run_dir, 3)
    next_step = {
        "running": "poll proxy_eval_status() again until state is 'done'",
        "pending": "poll proxy_eval_status() again until state is 'done'",
        "done":    NEXT_RESULTS,
        "crashed": "proxy_eval_status(op='log', tail=100) to diagnose the failure",
    }.get(rs["state"], NEXT_RESULTS)
    payload = {
        "run_dir":        run_dir,
        "state":          rs["state"],
        "pid":            rs["pid"],
        "slurm_job_id":   rs["slurm_job_id"],
        "elapsed_s":      rs["elapsed_s"],
        "last_log_lines": last_lines,
    }
    # What the run is doing, when it says. Carried here and not only on the blocking
    # wait's own channel because this op is what a *detached* run is watched through:
    # without it, a run moved to the background goes quiet the moment it is moved.
    for key in ("phase", "percent"):
        if rs.get(key) is not None:
            payload[key] = rs[key]
    # A handle for a run still going, so asking where it is at puts a watcher back on
    # it: the watchers belong to the agent that made them and do not outlive it, while
    # the run — a detached process with its own run dir — carries on regardless.
    # Plural, never ``background_job``: that key says *this call launched it*, and this
    # one only looked.
    if rs["state"] in ("running", "pending"):
        payload["background_jobs"] = [eval_background_descriptor(resolved, run_dir)]
    return ok(with_next(payload, next_step))


def _recommend(r: dict, summary: dict, proxy_source: str) -> tuple[str, str]:
    """``(meaning, next_step)`` for the settled verdict: the loop's script."""
    verdict  = r["verdict"]
    best     = r.get("best")
    best_id  = best.get("run_id") if best else None
    best_val = best.get("primary_value") if best else None
    pm, pv   = r["primary_metric"], r["primary_value"]

    if verdict == "incomparable":
        ran, base = r["machine"]["run"], r["machine"]["baseline"]
        return (
            f"Not compared: this run executed on {ran.get('host', '?')} "
            f"({ran.get('cpu_model') or ran.get('arch', '?')}), a different machine from "
            f"the baseline's {base.get('host', '?')} "
            f"({base.get('cpu_model') or base.get('arch', '?')}). "
            f"A {pm} measured on other hardware says nothing about the edit, so best and "
            "stall are unchanged. Run on the baseline's kind of machine again, or — if "
            "this machine is the real target — re-measure the baseline here.",
            "run again where the baseline ran, or proxy_eval(op='rebaseline', "
            "confirm=True) then run on the target")
    if verdict == "converged":
        return (
            f"Converged: all requirements pass and {pm} did not improve for "
            f"{r['max_stall']} iterations (best run {best_id}, {pm}={best_val}). "
            "Restore the best implementation and summarize.",
            NEXT_RESET_BEST + ", then summarize the results.")
    if verdict == "accept":
        return (
            f"Accepted — new best for {pm} ({pv}) with all requirements passing. "
            f"Try to improve {pm} further, or restore the best and summarize if "
            f"satisfied.",
            f"edit '{proxy_source}' to improve {pm}, then {NEXT_RUN} "
            f"(or {NEXT_RESET_BEST} + summarize)")
    if verdict == "reject":
        if r["feasible"]:
            why = (f"Requirements still pass but {pm}={pv} did not beat the best "
                   f"({best_val}). This edit is not an improvement — revert to the "
                   f"best and try a different approach.")
        else:
            why = ("This change regressed: requirements no longer pass. Revert to "
                   "the best-so-far and try a different modification.")
        return why, NEXT_RESET_BEST + f", then edit '{proxy_source}' and {NEXT_RUN}"

    # verdict is None — not feasible yet, and no best to fall back on.
    passed = summary.get("cases_passed", 0)
    total  = summary.get("cases_total", 0)
    return (
        f"{total - passed} of {total} case(s) failed requirements. "
        f"Read and modify '{proxy_source}' with the file tools, then run again.",
        f"edit '{proxy_source}', then {NEXT_RUN}")


def results(proxy_name: str = "") -> dict:
    """Per-case benchmark results + requirements check, parsed from the run log."""
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    cases, summary = parse_runner_log(read_log(run_dir))

    final_metrics = read_json(os.path.join(run_dir, "metrics.json"))

    cfg          = load_opt_config(proxy_name)
    proxy_source = cfg.get("proxy_source_path", "")
    complete     = isinstance(final_metrics, dict) and "all_passed" in final_metrics

    # A run stopped by its build has no metrics.json, which without this would be
    # reported as "may still be in progress" — the one answer that hides the cause
    # and invites a wait for a run that is already over.
    bld = build.read_report(run_dir)
    if not complete and bld and bld.get("status") not in (None, "ok"):
        log_path = procs.build_log_path(run_dir)
        return err(
            build.failure_summary(bld) + " — no measurement was taken.",
            hint="Fix the build and run again; nothing was accepted or recorded. "
                 + build.SHELL_HINT,
            run_dir=run_dir,
            state="crashed",
            build=bld,
            build_log=log_path,
            build_log_tail=build_progress.read_tail(log_path, 8192),
            next_step="read build_log_tail, fix the build, then "
                      "proxy_eval(op='run', confirm=True)",
        )

    if not complete:
        # Run has not produced a summary yet — still in progress.
        return ok(with_next({
            "run_dir":        run_dir,
            "state":          run_state(run_dir)["state"],
            "cases":          cases,
            "summary":        summary,
            "final_metrics":  final_metrics,
            "recommendation": "No summary yet — the run may still be in progress.",
        }, "proxy_eval_status() to check the run state"))

    r        = ratchet_state(proxy_name, run_dir, final_metrics, cfg)
    verdict  = r["verdict"]
    best     = r.get("best")
    best_id  = best.get("run_id") if best else None
    best_val = best.get("primary_value") if best else None
    pm, pv   = r["primary_metric"], r["primary_value"]

    recommendation, next_step = _recommend(r, summary, proxy_source)

    if r.get("timing_warning"):
        recommendation += " WARNING: " + r["timing_warning"]

    # Said only once the run has measured how long its own build takes: a constant
    # cannot know whether this project compiles in two seconds or half an hour, and
    # advising a detach on a two-second build would be noise on every reply.
    build_s = (final_metrics or {}).get("build_s") or 0.0
    recommendation += _long_build_note(build_s)

    tried = axes_tried(proxy_name)

    return ok(with_next({
        "run_dir":        run_dir,
        **({"build_s": build_s} if build_s else {}),
        "state":          run_state(run_dir)["state"],
        "cases":          cases,
        "summary":        summary,
        "final_metrics":  final_metrics,
        "verdict":        verdict,
        "feasible":       r["feasible"],
        "primary_metric": pm,
        "primary_value":  pv,
        "wall_value":     r.get("wall_value"),
        "best":           {"run_id": best_id, "primary_value": best_val},
        "stall":          r["stall"],
        "recommendation": recommendation,
        # Attached rather than asked for: the moment the next edit is chosen is right
        # here, and a call the agent has to remember to make is one it will not make.
        **({"axes_tried": tried} if tried else {}),
        **({"timing_warning": r["timing_warning"]} if r.get("timing_warning") else {}),
    }, next_step))


def _long_build_note(build_s: float) -> str:
    """Advice to detach the next run, or "" when the build is not worth it."""
    if not build_s or build_s < LONG_BUILD_S:
        return ""
    return (f" This run spent {build_s:.0f}s building. Pass background=True to the "
            "next run: it detaches immediately and you are resumed with the verdict, "
            "so the wait is not yours to sit through.")


def log(proxy_name: str = "", tail: int = 50) -> dict:
    tail = max(1, min(tail, 500))
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    if not os.path.isfile(run_log_path(run_dir)):
        return err("Log file not yet created — run may not have started writing output.")
    content = read_log(run_dir)

    all_lines = content.splitlines()
    rs        = run_state(run_dir)
    return ok(with_next({
        "run_dir":     run_dir,
        "state":       rs["state"],
        "total_lines": len(all_lines),
        "lines":       all_lines[-tail:],
    }, "proxy_eval(op='status') for the ratchet verdict on this run."))


def runs_list(proxy_name: str = "") -> dict:
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(with_next({"runs": [], "count": 0,
                              "note": "No proxy name or active session."},
                             "proxy_eval(op='init', ...) to start a session."))
    session_dir = opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return ok(with_next({"runs": [], "count": 0},
                             "proxy_eval(op='run', confirm=True) to produce a first run."))

    names = run_dir_names(session_dir)
    best        = load_best(resolved)
    best_run_id = best.get("run_id") if best else None
    runs: list[dict] = []
    for name in names:
        run_dir = os.path.join(session_dir, name)
        rs      = run_state(run_dir)
        entry: dict = {"run_id": name, "run_dir": run_dir,
                       "state": rs["state"], "elapsed_s": rs["elapsed_s"],
                       "is_best": name == best_run_id}
        m = read_json(os.path.join(run_dir, "metrics.json"))
        if isinstance(m, dict):
            for k in ("cases_passed", "cases_total", "all_passed",
                      "best_case", "best_time_s", "convergence_order"):
                entry[k] = m.get(k)
        runs.append(entry)
    return ok(with_next(
        {"runs": runs, "count": len(runs), "best_run_id": best_run_id},
        "proxy_eval(op='diff') to compare the last two runs."))


def runs_diff(run_a: str = "", run_b: str = "", proxy_name: str = "") -> dict:
    resolved = resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    session_dir = opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return err("No optimization runs found.")

    all_runs = run_dir_names(session_dir)

    def _resolve(rid: str, default_idx: int) -> str:
        rid = rid or all_runs[default_idx]
        return rid if os.path.isabs(rid) else os.path.join(session_dir, rid)

    payload, error = diff_run_pair(all_runs, run_a, run_b, _resolve)
    if error:
        return err(error, hint="Run proxy_eval(op='run', confirm=True) more times first.")
    return ok(with_next(payload,
                         "proxy_eval(op='results') for the ratchet view across all runs."))


def config_get(proxy_name: str = "") -> dict:
    cfg = load_opt_config(proxy_name)
    if not cfg:
        return ok(with_next({"config": None, "note": "No optimization session initialized."},
                             "proxy_eval(op='init', ...)"))
    return ok(with_next({"config": cfg},
                         "proxy_eval(op='configure', ...) to change it, or op='run' to use it."))
