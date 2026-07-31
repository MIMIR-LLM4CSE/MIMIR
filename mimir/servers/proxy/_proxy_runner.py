"""Standalone proxy optimization runner.

Invoked as a subprocess by the ``proxy_eval``/``proxy_slurm`` eval ops of
server_proxy.py; not an MCP server.

Usage:
    python _proxy_runner.py --run-dir <path>

Reads  <run-dir>/config.json, executes the full benchmark suite for the
configured proxy, evaluates requirements, and writes
<run-dir>/metrics.json on completion.
All output goes to stdout (captured as stdout.log by the server).

config.json schema
------------------
{
  "proxy_name":       str,          # registered proxy to run
  "benchmark_name":   str,          # defined benchmark suite name
  "requirements":     list[dict],   # [{metric, operator, threshold}, ...]
  "python_executable": str          # (optional) python used to rebuild cmd
}

Structured log lines emitted for agent observation
---------------------------------------------------
Every structured line starts with "[proxy_runner]" and uses key=value format
so it is parseable by proxy_eval_status(op='results'):

  [proxy_runner] case=<case_id> sweep=<idx> time_s=<n> passed=<T/F> [<metric>=<v> ...]
  [proxy_runner] summary cases_passed=<n> cases_total=<m> all_passed=<T/F> \
      best_case=<case_id> best_time_s=<n>

The agent can therefore:
  1. proxy_eval(op='run') → proxy_eval_status()
  2. proxy_eval_status(op='results') / (op='log') to see results
  3. Read the proxy source file (proxy_source_path from opt_config.json)
  4. Modify the proxy source with the file-edit tools
  5. proxy_eval(op='run') again to measure the effect
  6. Repeat until requirements are satisfied
  7. proxy_eval(op='reset') if a change makes things worse
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone


def _log(msg: str) -> None:
    print(msg, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Proxy optimization runner")
    parser.add_argument(
        "--run-dir", required=True,
        help="Run directory (must contain config.json; receives metrics.json).",
    )
    args = parser.parse_args()

    run_dir    = os.path.abspath(args.run_dir)
    config_path = os.path.join(run_dir, "config.json")

    if not os.path.isfile(config_path):
        _log(f"[proxy_runner] ERROR: config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as fh:
        cfg = json.load(fh)

    proxy_name     = cfg.get("proxy_name", "")
    benchmark_name = cfg.get("benchmark_name", "")
    requirements   = cfg.get("requirements", [])

    _log(f"[proxy_runner] proxy_name={proxy_name}")
    _log(f"[proxy_runner] benchmark_name={benchmark_name}")
    _log(f"[proxy_runner] requirements={len(requirements)}")

    # ── locate the _lib helper package ───────────────────────────────────────
    _this_dir = os.path.dirname(os.path.abspath(__file__))
    _shared_dir = os.path.join(_this_dir, '..', '_shared')
    if _this_dir not in sys.path:
        sys.path.insert(0, _this_dir)
    if _shared_dir not in sys.path:
        sys.path.insert(0, _shared_dir)

    try:
        from _lib.execute import _run_benchmark_case
        from _lib.metrics import _evaluate_requirements, _convergence_order
        from _lib.ratchet import _select_best_case
        from _lib.store import _load_registry, _load_suite
    except ImportError as exc:
        _log(f"[proxy_runner] ERROR: cannot import _lib: {exc}")
        sys.exit(1)

    # ── load registry + proxy entry ──────────────────────────────────────────
    try:
        reg = _load_registry()
    except RuntimeError as exc:
        _log(f"[proxy_runner] ERROR: registry corrupt: {exc}")
        sys.exit(1)

    if proxy_name not in reg:
        _log(f"[proxy_runner] ERROR: proxy '{proxy_name}' not in registry")
        sys.exit(1)

    entry = reg[proxy_name]

    # ── load benchmark suite ─────────────────────────────────────────────────
    suite = _load_suite(benchmark_name)
    if suite is None:
        _log(f"[proxy_runner] ERROR: suite '{benchmark_name}' not found")
        sys.exit(1)

    cases = suite.get("cases", [])
    _log(f"[proxy_runner] suite_cases={len(cases)}")

    # ── iterate suite ────────────────────────────────────────────────────────
    deadline    = time.monotonic() + cfg.get("deadline_s", 86400.0)  # default 24 h
    per_case_timeout_s = cfg.get("per_case_timeout_s") or None
    all_results: list[dict] = []
    cases_passed = 0
    cases_total  = 0
    best_inputs: list[tuple[str, bool, float | None]] = []

    # Optional convergence study: gather (step h, error) pairs across the sweep so
    # an observed order of accuracy can be fitted after the loop.
    conv_cfg      = cfg.get("convergence") or {}
    conv_h_param  = conv_cfg.get("h_param", "")
    conv_err_key  = conv_cfg.get("error_metric", "l2_rel")
    conv_pairs: list[tuple[float, float]] = []

    # convergence_order is fitted across the whole sweep, so it can never
    # appear in a single case's metrics — evaluating it per case would fail
    # every case whenever it is required. Keep it out of the per-case set;
    # it is folded into all_passed at run level below.
    case_requirements = [
        r for r in requirements if r.get("metric") != "convergence_order"]

    for case in cases:
        case_id        = case.get("case_id", "?")
        case_proxy     = case.get("proxy_name", proxy_name)
        reference_name = case.get("reference_name", "")
        param_sweeps   = case.get("param_sweeps") or [{}]
        extra_params   = case.get("extra_params", "")
        extra_metrics  = case.get("metrics", [])

        case_entry = reg.get(case_proxy, entry)

        for idx, sweep_overrides in enumerate(param_sweeps):
            cases_total += 1
            tag_suffix = f"opt_{proxy_name}_{case_id}_{idx}"

            _log(f"[proxy_runner] running case={case_id} sweep={idx} overrides={sweep_overrides}")

            run_case_dir, row = _run_benchmark_case(
                entry=case_entry,
                proxy_name=case_proxy,
                reference_name=reference_name,
                extra_params=extra_params,
                param_overrides=sweep_overrides,
                extra_metrics=extra_metrics,
                deadline=deadline,
                tag_suffix=tag_suffix,
                per_case_timeout_s=per_case_timeout_s,
            )

            if "error" in row:
                _log(f"[proxy_runner] case={case_id} sweep={idx} error={row['error']}")
                best_inputs.append((case_id, False, None))
                all_results.append({
                    "case_id": case_id, "sweep_idx": idx,
                    "sweep": sweep_overrides, "run_dir": run_case_dir,
                    "error": row["error"], "passed": False,
                })
                continue

            # Evaluate requirements against this case's metrics
            case_metrics = {}
            metrics_path = os.path.join(run_case_dir, "metrics.json")
            if os.path.isfile(metrics_path):
                try:
                    with open(metrics_path) as fh:
                        case_metrics = json.load(fh)
                except (json.JSONDecodeError, OSError):
                    pass

            req_result = _evaluate_requirements(case_metrics, case_requirements)
            passed     = req_result["passed"]
            if passed:
                cases_passed += 1

            time_s = case_metrics.get("time_s")
            best_inputs.append((case_id, passed, time_s))

            if conv_h_param:
                h_val = sweep_overrides.get(conv_h_param, case_metrics.get(conv_h_param))
                err_val = case_metrics.get(conv_err_key)
                if isinstance(h_val, (int, float)) and isinstance(err_val, (int, float)):
                    conv_pairs.append((float(h_val), float(err_val)))

            # Emit structured log line
            metric_tokens = " ".join(
                f"{k}={v}" for k, v in case_metrics.items()
                if isinstance(v, (int, float)) and k != "comparison_to_reference"
            )
            _log(
                f"[proxy_runner] case={case_id} sweep={idx} "
                f"passed={passed} {metric_tokens}"
            )

            result_entry: dict = {
                "case_id":          case_id,
                "sweep_idx":        idx,
                "sweep":            sweep_overrides,
                "run_dir":          run_case_dir,
                "metrics":          case_metrics,
                "requirements":     req_result,
                "passed":           passed,
            }
            all_results.append(result_entry)

    # ── run-level numerical invariant: observed order of accuracy ────────────
    # convergence_order is inherently cross-case, so it is evaluated here (not
    # per-case) and, when a requirement targets it, folded into all_passed.
    cases_all_passed = cases_passed == cases_total and cases_total > 0
    convergence_order = _convergence_order(conv_pairs) if conv_pairs else None
    conv_reqs = [r for r in requirements if r.get("metric") == "convergence_order"]
    run_level_passed = True
    if conv_reqs:
        run_level = _evaluate_requirements(
            {"convergence_order": convergence_order}, conv_reqs)
        run_level_passed = run_level["passed"]
        _log(f"[proxy_runner] convergence_order={convergence_order} "
             f"passed={run_level_passed}")

    all_passed = cases_all_passed and run_level_passed
    best_case, best_time = _select_best_case(best_inputs)
    _log(
        f"[proxy_runner] summary "
        f"cases_passed={cases_passed} cases_total={cases_total} "
        f"all_passed={all_passed} "
        f"best_case={best_case or 'none'} "
        f"best_time_s={best_time if best_time is not None else 'n/a'}"
    )

    # ── write metrics.json ───────────────────────────────────────────────────
    output = {
        "proxy_name":    proxy_name,
        "benchmark_name": benchmark_name,
        "cases_passed":  cases_passed,
        "cases_total":   cases_total,
        "all_passed":    all_passed,
        "best_case":     best_case,
        "best_time_s":   best_time,
        "convergence_order": convergence_order,
        "results":       all_results,
        "completed_at":  datetime.now(timezone.utc).isoformat(),
    }
    metrics_path = os.path.join(run_dir, "metrics.json")
    try:
        with open(metrics_path, "w") as fh:
            json.dump(output, fh, indent=2)
    except OSError as exc:
        _log(f"[proxy_runner] WARNING: could not write metrics.json: {exc}")

    # ── settle the ratchet server-side ───────────────────────────────────────
    # Ledger / best-so-far must not depend on the agent ever calling
    # results(): settle here, at completion. results() replays the frozen
    # outcome (idempotent, flock-serialized).
    try:
        from _ops.eval_session import _load_opt_config, _ratchet_state
        cfg = _load_opt_config(proxy_name)
        if cfg and cfg.get("proxy_name") == proxy_name:
            outcome = _ratchet_state(proxy_name, run_dir, output, cfg)
            _log(f"[proxy_runner] ratchet verdict={outcome.get('verdict')} "
                 f"feasible={outcome.get('feasible')} stall={outcome.get('stall')}")
            if outcome.get("timing_warning"):
                _log(f"[proxy_runner] WARNING: {outcome['timing_warning']}")
    except Exception as exc:  # never fail the run over bookkeeping
        _log(f"[proxy_runner] WARNING: ratchet settle failed: {exc}")

    if not all_passed:
        _log(
            f"[proxy_runner] NOTE: {cases_total - cases_passed} case(s) did not meet "
            "requirements. Read the proxy source and modify the implementation, "
            "then call proxy_eval(op='run', confirm=True) again. "
            "Use proxy_eval(op='reset', confirm=True) to revert to the original."
        )
    else:
        _log("[proxy_runner] All requirements satisfied.")


if __name__ == "__main__":
    main()
