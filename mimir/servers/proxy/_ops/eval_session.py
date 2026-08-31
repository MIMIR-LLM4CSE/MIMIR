"""Optimization-session ops.

Act ops (dispatched by ``proxy_eval``): init, configure, run, stop, reset,
reset_to_best, end.
Observe ops (dispatched by ``proxy_eval_status``): status, results, log,
runs, diff, config.  Every response carries a ``next_step`` hint naming the
exact next call, so the loop is self-describing.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
from datetime import datetime, timezone

from _ops import _PROXY_DIR, _with_next, err, ok
from _lib.metrics import _VALID_OPT_OPERATORS, _coerce
from _lib.procs import (
    _log_path, _read_log, _run_state,
    _new_run_dir, _write_run_config, _launch_detached, _cancel_run,
    _opt_active_run_dir, _update_opt_active_link,
)
from _lib.ratchet import (
    _load_best, _save_best, _append_ledger, _ratchet_verdict,
    _run_primary_value,
)
from _lib.report import _diff_run_pair
from _lib.store import (
    opt_canonical_dir,
    _load_registry_or_err, _load_suite,
    _opt_config_file, _opt_session_runs_dir, _opt_best_source_path,
    _resolve_proxy_name, _write_active_session, _clear_active_session,
    _read_json, _write_json_atomic, _file_lock, _run_dir_names,
)

_OPT_RUNNER = os.path.join(_PROXY_DIR, "_proxy_runner.py")

# Computed server-side from a sealed reference, so a requirement on one of these
# can never be satisfied when a case has no reference — refuse it up front.
_REFERENCE_METRICS = ("conservation_residual", "l2_abs", "l2_rel",
                      "linf_abs", "linf_rel")

_NEXT_RUN       = "proxy_eval(op='run', confirm=True)"
_NEXT_STATUS    = "proxy_eval_status() to monitor (state 'done' means finished)"
_NEXT_RESULTS   = "proxy_eval_status(op='results')"
_NEXT_RESET_BEST = "proxy_eval(op='reset_to_best', confirm=True)"


# ── session config helpers ────────────────────────────────────────────────────

def _load_opt_config(proxy_name: str = "") -> dict:
    name = _resolve_proxy_name(proxy_name)
    if not name:
        return {}
    return _read_json(_opt_config_file(name), {}) or {}


def _save_opt_config(cfg: dict, proxy_name: str = "") -> None:
    name = proxy_name or cfg.get("proxy_name", "")
    if not name:
        return
    _write_json_atomic(_opt_config_file(name), cfg)


def _opt_tail_log(run_dir: str, n: int) -> list[str]:
    lines = _read_log(run_dir).splitlines()
    return lines[-n:]


def _opt_canonical_path(proxy_name: str, source_path: str) -> str:
    ext = os.path.splitext(source_path)[1] or ".py"
    return os.path.join(opt_canonical_dir(), proxy_name + ext)


def _check_requirements(requirements: list[dict]) -> str | None:
    for i, req in enumerate(requirements):
        if not req.get("metric"):
            return f"requirements[{i}] is missing 'metric'."
        if req.get("operator") not in _VALID_OPT_OPERATORS:
            return (f"requirements[{i}] has invalid operator '{req.get('operator')}'. "
                    "Use one of: lt, gt, lte, gte, eq.")
        if req.get("threshold") is None:
            return f"requirements[{i}] is missing 'threshold'."
    return None


def _check_reference_requirements(
    requirements: list[dict], suite: dict, entry: dict,
) -> str | None:
    """Reject requirements that can never be satisfied with this setup.

    Reference-dependent metrics (see ``_REFERENCE_METRICS``) are computed
    server-side against a sealed reference; ``conservation_residual``
    additionally needs the registration to name the conserved scalar. Failing
    fast here turns a dead-end session into an actionable setup error.
    """
    needed = sorted({r.get("metric") for r in requirements}
                    & set(_REFERENCE_METRICS))
    if not needed:
        return None
    cases = suite.get("cases") or []
    no_ref = [str(c.get("case_id", "?")) for c in cases
              if not c.get("reference_name")]
    if not cases or no_ref:
        which = ", ".join(no_ref) if no_ref else "(no cases defined)"
        return (f"Requirement(s) {', '.join(needed)} are computed server-side "
                f"against a sealed reference, but benchmark case(s) {which} "
                "have no reference_name — they could never pass. Create the "
                "benchmark with proxy_exec(op='benchmark_create', "
                "reference_params=...), which seals a reference, or add "
                "reference_name to every case. Values printed by the proxy "
                "for these metrics are ignored.")
    if "conservation_residual" in needed and not entry.get("conserved_metric"):
        return ("Requirement conservation_residual needs the proxy "
                "registration to declare which scalar is conserved: "
                "proxy_manage(op='update', metadata="
                "{'conserved_metric': '<metric name>'}, confirm=True).")
    return None


# ── act ops (confirm already checked by the dispatch tool) ────────────────────

def init(
    proxy_name: str,
    benchmark_name: str,
    requirements: list[dict],
    proxy_source_path: str,
    python_executable: str = "",
    max_hours: float = 0.0,
    primary_metric: str = "time_s",
    primary_goal: str = "min",
    min_improvement: float = 0.02,
    max_stall: int = 5,
    convergence: dict | None = None,
) -> dict:
    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if proxy_name not in reg:
        return err(f"Proxy '{proxy_name}' not registered.",
                   hint="Call proxy_manage(op='register', ...) first.")

    suite = _load_suite(benchmark_name)
    if suite is None:
        return err(f"Benchmark suite '{benchmark_name}' not found.",
                   hint="Create one with proxy_exec(op='benchmark_create', ...) "
                        "or proxy_manage(op='suite_define', ...) first.")

    req_err = _check_requirements(requirements)
    if req_err:
        return err(req_err)

    ref_err = _check_reference_requirements(requirements, suite, reg[proxy_name])
    if ref_err:
        return err(ref_err)

    if primary_goal not in ("min", "max"):
        return err(f"primary_goal must be 'min' or 'max', got '{primary_goal}'.")
    if max_stall < 1:
        return err(f"max_stall must be >= 1, got {max_stall}.")
    if min_improvement < 0:
        return err(f"min_improvement must be >= 0, got {min_improvement}.")

    abs_src = os.path.abspath(proxy_source_path)
    if not os.path.isfile(abs_src):
        return err(f"proxy_source_path not found: {abs_src}")
    if python_executable and not os.path.isfile(python_executable):
        return err(f"python_executable not found: {python_executable}")

    os.makedirs(opt_canonical_dir(), exist_ok=True)
    canon_path        = _opt_canonical_path(proxy_name, abs_src)
    canonical_existed = os.path.isfile(canon_path)
    if not canonical_existed:
        try:
            shutil.copy2(abs_src, canon_path)
        except OSError as exc:
            return err(f"Could not snapshot canonical: {exc}")

    cfg = {
        "proxy_name":        proxy_name,
        "benchmark_name":    benchmark_name,
        "requirements":      requirements,
        "proxy_source_path": abs_src,
        "canonical_path":    canon_path,
        "python_executable": python_executable,
        "max_hours":         max_hours if max_hours > 0 else 24.0,
        "primary_metric":    primary_metric,
        "primary_goal":      primary_goal,
        "min_improvement":   float(min_improvement),
        "max_stall":         int(max_stall),
        "stall":             0,
        "convergence":       convergence or {},
        "initialized_at":    datetime.now(timezone.utc).isoformat(),
    }
    _save_opt_config(cfg)
    _write_active_session(proxy_name)

    return ok(_with_next({
        "proxy_name":        proxy_name,
        "benchmark_name":    benchmark_name,
        "proxy_source_path": abs_src,
        "canonical_path":    canon_path,
        "canonical_existed": canonical_existed,
        "requirements":      requirements,
        "objective":         f"{primary_goal}imize {primary_metric} subject to the requirements",
        "note": (
            "Canonical snapshot taken. " if not canonical_existed
            else "Canonical already existed — not overwritten. "
        ) + f"Modify '{abs_src}' between runs to test new implementations. "
            "Feasible runs that improve the objective are accepted; regressions are "
            "rejected — revert them with proxy_eval(op='reset_to_best', confirm=True).",
    }, _NEXT_RUN))


def configure(
    proxy_name: str = "",
    requirements: list[dict] | None = None,
    benchmark_name: str = "",
    python_executable: str = "",
    max_hours: float = 0.0,
) -> dict:
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    if requirements is not None:
        req_err = _check_requirements(requirements)
        if req_err:
            return err(req_err)
        cfg["requirements"] = requirements

    if benchmark_name:
        if _load_suite(benchmark_name) is None:
            return err(f"Benchmark suite '{benchmark_name}' not found.")
        cfg["benchmark_name"] = benchmark_name

    if python_executable:
        if not os.path.isfile(python_executable):
            return err(f"python_executable not found: {python_executable}")
        cfg["python_executable"] = python_executable

    if max_hours > 0:
        cfg["max_hours"] = max_hours

    # Re-check reference-dependent requirements against the (possibly new)
    # suite and registration — same dead-end guard as init.
    final_suite = _load_suite(cfg.get("benchmark_name", "")) or {}
    reg, reg_err = _load_registry_or_err()
    if reg_err:
        return err(reg_err)
    entry = (reg or {}).get(cfg.get("proxy_name", ""), {})
    ref_err = _check_reference_requirements(
        cfg.get("requirements") or [], final_suite, entry)
    if ref_err:
        return err(ref_err)

    cfg["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_opt_config(cfg)
    return ok(_with_next({"config": cfg}, _NEXT_RUN))


def _prepare_run(proxy_name: str) -> tuple[dict | None, dict | None, str | None]:
    """Shared preamble for local/Slurm eval runs.

    Returns ``(cfg, err_response, run_dir)``: on failure *err_response* is set;
    on success *cfg* holds the session config and *run_dir* the fresh run dir
    (config.json + start_time already written).
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return None, err("No optimization session found.",
                         hint="Call proxy_eval(op='init', ...) first."), None

    name           = cfg.get("proxy_name", "")
    benchmark_name = cfg.get("benchmark_name", "")
    if not name or not benchmark_name:
        return None, err("proxy_name and benchmark_name must be set.",
                         hint="Call proxy_eval(op='init', ...) first."), None

    active = _opt_active_run_dir(name)
    if active:
        rs = _run_state(active)
        if rs["state"] in ("running", "pending"):
            pid = rs.get("pid")
            jid = rs.get("slurm_job_id")
            tag = f"slurm_job_id={jid}" if jid else f"pid={pid}"
            return None, err(f"An optimization run is already active ({tag}).",
                             hint="Use proxy_eval(op='stop', confirm=True) first."), None

    run_dir = _new_run_dir(_opt_session_runs_dir(name))
    _write_run_config(run_dir, {
        "proxy_name":        name,
        "benchmark_name":    benchmark_name,
        "requirements":      cfg.get("requirements", []),
        "python_executable": cfg.get("python_executable", ""),
        "deadline_s":        (cfg.get("max_hours") or 24.0) * 3600,
        "convergence":       cfg.get("convergence") or {},
        "started_at":        datetime.now(timezone.utc).isoformat(),
    })
    # Freeze what is about to run: the best-source snapshot must capture the
    # code that produced the run, not whatever the file contains when the run
    # is settled (the agent may edit while a run is in flight).
    src = cfg.get("proxy_source_path", "")
    if src and os.path.isfile(src):
        try:
            shutil.copy2(src, os.path.join(
                run_dir, "source_at_launch" + os.path.splitext(src)[1]))
        except OSError:
            pass  # legacy fallback in the settle path handles the absence
    return cfg, None, run_dir


def _background_descriptor(name: str, run_dir: str) -> dict:
    """Data-only handle a client watcher polls to completion (no tool name in loop code).

    ``status_op``/``summary_op`` name the read-only ops the watcher calls generically;
    the client augments the model's view and, on completion, auto-resumes.
    """
    return {
        "server":     "proxy",
        "run_dir":    run_dir,
        "job_key":    name,
        "kind":       "proxy-optimization",
        "status_op":  {"tool": "proxy_eval_status", "args": {"proxy_name": name}},
        "summary_op": {"tool": "proxy_eval_status",
                       "args": {"op": "results", "proxy_name": name}},
    }


def run(proxy_name: str = "", background: bool = False) -> dict:
    """Launch a background optimization run (non-blocking).

    ``background=True`` attaches a ``background_job`` descriptor so a client watcher
    monitors completion off the agent's critical path and auto-resumes the agent with
    the results — the agent should end its turn instead of polling.
    """
    cfg, error, run_dir = _prepare_run(proxy_name)
    if error:
        return error
    name       = cfg["proxy_name"]
    log_file   = _log_path(run_dir)
    python_exe = cfg.get("python_executable") or sys.executable

    pid = _launch_detached(
        [python_exe, _OPT_RUNNER, "--run-dir", run_dir], run_dir, log_file=log_file,
    )
    _update_opt_active_link(name, run_dir)

    payload = {
        "run_dir":        run_dir,
        "pid":            pid,
        "log":            log_file,
        "proxy_name":     name,
        "benchmark_name": cfg["benchmark_name"],
        "note": "Optimization run started in background.",
    }
    if background:
        payload["background_job"] = _background_descriptor(name, run_dir)
    return ok(_with_next(payload, _NEXT_STATUS))


def stop(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No active optimization run found.")

    result = _cancel_run(run_dir)
    if "error" in result:
        return err(result["error"])
    return ok(_with_next({"run_dir": run_dir, **result}, _NEXT_RUN + " to start a new run"))


def reset(proxy_name: str = "") -> dict:
    """Restore the proxy source file from the canonical snapshot."""
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")

    source_path = cfg.get("proxy_source_path", "")
    canon_path  = cfg.get("canonical_path", "")
    if not source_path:
        return err("proxy_source_path not set in config.",
                   hint="Call proxy_eval(op='init', ...) again.")
    if not canon_path or not os.path.isfile(canon_path):
        return err(f"Canonical not found at: {canon_path}",
                   hint="Call proxy_eval(op='init', ...) to create it.")

    try:
        shutil.copy2(canon_path, source_path)
    except OSError as exc:
        return err(f"Could not restore canonical: {exc}")

    return ok(_with_next({
        "restored_to":    source_path,
        "from_canonical": canon_path,
        "note": "Proxy source restored to canonical. Try a different modification approach.",
    }, _NEXT_RUN + " to verify the baseline"))


def reset_to_best(proxy_name: str = "") -> dict:
    """Restore the proxy source file from the best-so-far snapshot.

    Unlike ``reset`` (which restores the original canonical baseline), this reverts
    to the source of the best *accepted* run — the way to undo a regression without
    discarding the improvements found so far.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Call proxy_eval(op='init', ...) first.")
    name = cfg.get("proxy_name", "")
    best = _load_best(name)
    if not best:
        return err("No best-so-far recorded yet — no feasible run has been accepted.",
                   hint="Run at least once until requirements pass, or use "
                        "proxy_eval(op='reset', confirm=True) to restore the baseline.")

    source_path = cfg.get("proxy_source_path", "")
    snapshot    = best.get("source_snapshot") or _opt_best_source_path(name, source_path)
    if not source_path:
        return err("proxy_source_path not set in config.")
    if not snapshot or not os.path.isfile(snapshot):
        return err(f"Best snapshot not found at: {snapshot}",
                   hint="Use proxy_eval(op='reset', confirm=True) to restore the baseline.")

    try:
        shutil.copy2(snapshot, source_path)
    except OSError as exc:
        return err(f"Could not restore best snapshot: {exc}")

    return ok(_with_next({
        "restored_to":  source_path,
        "from_best":    best.get("run_id"),
        "primary_value": best.get("primary_value"),
        "note": "Proxy source restored to the best-so-far run. Try a different "
                "modification approach from this baseline.",
    }, _NEXT_RUN + " to verify, or summarize if converged"))


def end(proxy_name: str = "") -> dict:
    """End the optimization session: drop the active-session pointer.

    A clean close for the session. The proxy source, snapshots, ledger and best
    stay on disk for history — only the "current session" marker is cleared, so
    subsequent nameless ops no longer resolve to it and the client's direct-exec
    guard lifts (the proxy may be run by hand again). Refuses while a run is still
    in flight so a live job is never orphaned. Re-``init`` to optimize again.
    """
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return err("No optimization session found.",
                   hint="Nothing to end. Call proxy_eval(op='init', ...) to start one.")
    name = cfg.get("proxy_name", "") or _resolve_proxy_name(proxy_name)

    active = _opt_active_run_dir(name)
    if active and _run_state(active).get("state") in ("running", "pending"):
        return err("A run is still active — stop it before ending the session.",
                   hint="proxy_eval(op='stop', confirm=True), then proxy_eval(op='end', confirm=True).")

    _clear_active_session()
    return ok(_with_next({
        "ended":  name,
        "note": "Optimization session ended; source/snapshots/ledger kept for history. "
                "The proxy can be run directly again. Re-init to optimize further.",
    }, "proxy_eval(op='init', ...) to start a new session, or you are done."))


def _ratchet_state(proxy_name: str, run_dir: str, final_metrics: dict, cfg: dict) -> dict:
    """Compute the ratchet outcome for a completed run, persisting it once.

    Whichever of the runner or ``results`` gets here first wins: the verdict is
    frozen to ``<run_dir>/ratchet.json`` under a per-session flock and re-checked
    inside it, so best/ledger/stall never move twice for the same run.
    """
    name = cfg.get("proxy_name") or proxy_name
    outcome_path = os.path.join(run_dir, "ratchet.json")

    def _frozen() -> dict | None:
        frozen = _read_json(outcome_path)  # None when missing or unreadable
        if frozen is None:
            return None  # recompute under the lock
        frozen["best"] = _load_best(name)
        return frozen

    out = _frozen()
    if out is not None:
        return out

    with _file_lock(os.path.join(_opt_session_runs_dir(name), ".ratchet.lock")):
        out = _frozen()
        if out is not None:
            return out
        return _ratchet_settle_locked(name, run_dir, final_metrics, cfg, outcome_path)


def _ratchet_settle_locked(
    name: str, run_dir: str, final_metrics: dict, cfg: dict, outcome_path: str,
) -> dict:
    primary_metric  = cfg.get("primary_metric", "time_s")
    goal            = cfg.get("primary_goal", "min")
    min_improvement = cfg.get("min_improvement", 0.0)
    max_stall       = int(cfg.get("max_stall", 5))
    proxy_source    = cfg.get("proxy_source_path", "")
    # Snapshot taken at launch: what actually ran, immune to edits made while
    # the run was in flight. Fall back to the live source for legacy runs.
    launch_snapshot = os.path.join(
        run_dir, "source_at_launch" + os.path.splitext(proxy_source)[1])
    if os.path.isfile(launch_snapshot):
        proxy_source = launch_snapshot

    feasible      = bool(final_metrics.get("all_passed"))
    primary_value = _run_primary_value(final_metrics, primary_metric)
    wall_value    = _run_primary_value(final_metrics, "wall_time_s")
    run_id        = os.path.basename(os.path.normpath(run_dir))
    best          = _load_best(name)

    verdict = _ratchet_verdict(feasible, primary_value, best, goal, min_improvement)

    # An accepted time_s improvement whose measured wall time regressed is
    # plausible I/O noise, but also the signature of a tampered timer: warn.
    timing_warning: str | None = None
    if verdict == "accept" and primary_metric == "time_s" and best is not None:
        prev_wall = best.get("wall_value")
        if (isinstance(wall_value, (int, float)) and isinstance(prev_wall, (int, float))
                and wall_value > prev_wall * 1.10):
            timing_warning = (
                "Self-reported time_s improved but the server-measured wall time "
                f"regressed ({prev_wall:.3g}s -> {wall_value:.3g}s). Verify the "
                "proxy's timing instrumentation before trusting this acceptance, "
                "or optimize primary_metric='wall_time_s' instead."
            )

    stall = int(cfg.get("stall", 0))
    if verdict == "accept":
        _save_best(name, run_id, primary_value, proxy_source, wall_value=wall_value)
        best  = _load_best(name)
        stall = 0
    elif feasible:
        stall += 1  # feasible but not an improvement — a stalled iteration

    if feasible and stall >= max_stall:
        verdict = "converged"

    cfg["stall"] = stall
    _save_opt_config(cfg)

    _append_ledger(name, {
        "run_id":        run_id,
        "ts":            datetime.now(timezone.utc).isoformat(),
        "feasible":      feasible,
        "primary_value": primary_value,
        "wall_value":    wall_value,
        "verdict":       verdict,
        "stall":         stall,
        "best_run_id":   best.get("run_id") if best else None,
        **({"timing_warning": timing_warning} if timing_warning else {}),
    })

    outcome = {
        "feasible":       feasible,
        "primary_value":  primary_value,
        "wall_value":     wall_value,
        "verdict":        verdict,
        "stall":          stall,
        "max_stall":      max_stall,
        "primary_metric": primary_metric,
        "goal":           goal,
        "timing_warning": timing_warning,
    }
    try:
        with open(outcome_path, "w") as fh:
            json.dump(outcome, fh, indent=2)
    except OSError:
        pass
    outcome["best"] = best
    return outcome


# ── observe ops ───────────────────────────────────────────────────────────────

def status(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(_with_next(
            {"state": "no_session", "note": "No proxy name or active session."},
            "proxy_eval(op='init', ...)"))
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return ok(_with_next(
            {"state": "no_runs", "note": "No optimization runs found."}, _NEXT_RUN))
    rs         = _run_state(run_dir)
    last_lines = _opt_tail_log(run_dir, 3)
    next_step = {
        "running": "poll proxy_eval_status() again until state is 'done'",
        "pending": "poll proxy_eval_status() again until state is 'done'",
        "done":    _NEXT_RESULTS,
        "crashed": "proxy_eval_status(op='log', tail=100) to diagnose the failure",
    }.get(rs["state"], _NEXT_RESULTS)
    return ok(_with_next({
        "run_dir":        run_dir,
        "state":          rs["state"],
        "pid":            rs["pid"],
        "slurm_job_id":   rs["slurm_job_id"],
        "elapsed_s":      rs["elapsed_s"],
        "last_log_lines": last_lines,
    }, next_step))


def results(proxy_name: str = "") -> dict:
    """Per-case benchmark results + requirements check, parsed from the run log."""
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    content = _read_log(run_dir)

    cases:   list[dict] = []
    summary: dict       = {}
    for line in content.splitlines():
        line = line.strip()
        if line.startswith("[proxy_runner] case="):
            row: dict = {}
            for part in line.split()[1:]:
                if "=" in part:
                    k, _, v = part.partition("=")
                    row[k] = _coerce(v)
            if row:
                cases.append(row)
        elif line.startswith("[proxy_runner] summary "):
            for part in line.split()[2:]:
                if "=" in part:
                    k, _, v = part.partition("=")
                    summary[k] = _coerce(v)

    final_metrics = _read_json(os.path.join(run_dir, "metrics.json"))

    cfg          = _load_opt_config(proxy_name)
    proxy_source = cfg.get("proxy_source_path", "")
    complete     = isinstance(final_metrics, dict) and "all_passed" in final_metrics

    if not complete:
        # Run has not produced a summary yet — still in progress.
        return ok(_with_next({
            "run_dir":        run_dir,
            "state":          _run_state(run_dir)["state"],
            "cases":          cases,
            "summary":        summary,
            "final_metrics":  final_metrics,
            "recommendation": "No summary yet — the run may still be in progress.",
        }, "proxy_eval_status() to check the run state"))

    r        = _ratchet_state(proxy_name, run_dir, final_metrics, cfg)
    verdict  = r["verdict"]
    best     = r.get("best")
    best_id  = best.get("run_id") if best else None
    best_val = best.get("primary_value") if best else None
    pm, pv   = r["primary_metric"], r["primary_value"]

    if verdict == "converged":
        recommendation = (
            f"Converged: all requirements pass and {pm} did not improve for "
            f"{r['max_stall']} iterations (best run {best_id}, {pm}={best_val}). "
            "Restore the best implementation and summarize."
        )
        next_step = _NEXT_RESET_BEST + ", then summarize the results."
    elif verdict == "accept":
        recommendation = (
            f"Accepted — new best for {pm} ({pv}) with all requirements passing. "
            f"Try to improve {pm} further, or restore the best and summarize if satisfied."
        )
        next_step = (f"edit '{proxy_source}' to improve {pm}, then {_NEXT_RUN} "
                     f"(or {_NEXT_RESET_BEST} + summarize)")
    elif verdict == "reject":
        if r["feasible"]:
            recommendation = (
                f"Requirements still pass but {pm}={pv} did not beat the best "
                f"({best_val}). This edit is not an improvement — revert to the best "
                "and try a different approach."
            )
        else:
            recommendation = (
                "This change regressed: requirements no longer pass. Revert to the "
                "best-so-far and try a different modification."
            )
        next_step = _NEXT_RESET_BEST + f", then edit '{proxy_source}' and {_NEXT_RUN}"
    else:  # verdict is None — not feasible yet and no best to fall back on
        passed = summary.get("cases_passed", 0)
        total  = summary.get("cases_total", 0)
        recommendation = (
            f"{total - passed} of {total} case(s) failed requirements. "
            f"Read and modify '{proxy_source}' with the file tools, then run again."
        )
        next_step = f"edit '{proxy_source}', then {_NEXT_RUN}"

    if r.get("timing_warning"):
        recommendation += " WARNING: " + r["timing_warning"]

    return ok(_with_next({
        "run_dir":        run_dir,
        "state":          _run_state(run_dir)["state"],
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
        **({"timing_warning": r["timing_warning"]} if r.get("timing_warning") else {}),
    }, next_step))


def log(proxy_name: str = "", tail: int = 50) -> dict:
    tail = max(1, min(tail, 500))
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    run_dir = _opt_active_run_dir(resolved)
    if not run_dir:
        return err("No optimization runs found.")

    if not os.path.isfile(_log_path(run_dir)):
        return err("Log file not yet created — run may not have started writing output.")
    content = _read_log(run_dir)

    all_lines = content.splitlines()
    rs        = _run_state(run_dir)
    return ok(_with_next({
        "run_dir":     run_dir,
        "state":       rs["state"],
        "total_lines": len(all_lines),
        "lines":       all_lines[-tail:],
    }, "proxy_eval(op='status') for the ratchet verdict on this run."))


def runs_list(proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return ok(_with_next({"runs": [], "count": 0,
                              "note": "No proxy name or active session."},
                             "proxy_eval(op='init', ...) to start a session."))
    session_dir = _opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return ok(_with_next({"runs": [], "count": 0},
                             "proxy_eval(op='run', confirm=True) to produce a first run."))

    names = _run_dir_names(session_dir)
    best        = _load_best(resolved)
    best_run_id = best.get("run_id") if best else None
    runs: list[dict] = []
    for name in names:
        run_dir = os.path.join(session_dir, name)
        rs      = _run_state(run_dir)
        entry: dict = {"run_id": name, "run_dir": run_dir,
                       "state": rs["state"], "elapsed_s": rs["elapsed_s"],
                       "is_best": name == best_run_id}
        m = _read_json(os.path.join(run_dir, "metrics.json"))
        if isinstance(m, dict):
            for k in ("cases_passed", "cases_total", "all_passed",
                      "best_case", "best_time_s", "convergence_order"):
                entry[k] = m.get(k)
        runs.append(entry)
    return ok(_with_next(
        {"runs": runs, "count": len(runs), "best_run_id": best_run_id},
        "proxy_eval(op='diff') to compare the last two runs."))


def runs_diff(run_a: str = "", run_b: str = "", proxy_name: str = "") -> dict:
    resolved = _resolve_proxy_name(proxy_name)
    if not resolved:
        return err("No proxy name or active session.")
    session_dir = _opt_session_runs_dir(resolved)
    if not os.path.isdir(session_dir):
        return err("No optimization runs found.")

    all_runs = _run_dir_names(session_dir)

    def _resolve(rid: str, default_idx: int) -> str:
        rid = rid or all_runs[default_idx]
        return rid if os.path.isabs(rid) else os.path.join(session_dir, rid)

    payload, error = _diff_run_pair(all_runs, run_a, run_b, _resolve)
    if error:
        return err(error, hint="Run proxy_eval(op='run', confirm=True) more times first.")
    return ok(_with_next(payload,
                         "proxy_eval(op='results') for the ratchet view across all runs."))


def config_get(proxy_name: str = "") -> dict:
    cfg = _load_opt_config(proxy_name)
    if not cfg:
        return ok(_with_next({"config": None, "note": "No optimization session initialized."},
                             "proxy_eval(op='init', ...)"))
    return ok(_with_next({"config": cfg},
                         "proxy_eval(op='configure', ...) to change it, or op='run' to use it."))
