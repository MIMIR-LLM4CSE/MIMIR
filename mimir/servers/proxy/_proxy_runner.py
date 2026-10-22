"""The standalone optimization runner (not an MCP server), launched by the eval ops.

``--run-dir <dir> [--phase all|build|run]``: reads ``config.json``, runs the suite,
evaluates requirements, writes ``metrics.json``; a split submission runs the two phases
as two jobs. Its config and log lines: docs/SERVERS_DETAILED.md, "proxy".
"""

from __future__ import annotations

from ._lib import layout
import argparse
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NoReturn

from ._lib import build as build_mod
from ._lib import execute, metrics, ratchet, store, tree_snapshot


def _log(msg: str) -> None:
    print(msg, flush=True)


def _write_phase(run_dir: str, kind: str, text: str, **fields) -> None:
    """Record what this run is doing, atomically, for whoever waits on it (never fatal)."""
    try:
        from ...shared.atomic import write_json
        write_json(os.path.join(run_dir, "phase.json"),
                           {"kind": kind, "text": text, **fields}, indent=2)
    except Exception:
        pass


def _aggregate_replicates(replicates: list[dict]) -> dict:
    """One set of metrics from replicates: numeric values by median, others from the first.

    The minimum of N draws improves with N whatever the code does.
    """
    if not replicates:
        return {}
    if len(replicates) == 1:
        return dict(replicates[0])
    out: dict = {}
    for key in replicates[0]:
        values = [r.get(key) for r in replicates]
        numeric = [v for v in values if isinstance(v, (int, float))
                   and not isinstance(v, bool)]
        if len(numeric) == len(values) and numeric:
            out[key] = statistics.median(numeric)
        else:
            out[key] = values[0]
    return out


def _relative_spread(replicates: list[dict], metric: str) -> float | None:
    """The full range of *metric* across replicates of the same code, as a fraction."""
    values = [r.get(metric) for r in replicates]
    numeric = [float(v) for v in values
               if isinstance(v, (int, float)) and not isinstance(v, bool)]
    if len(numeric) < 2:
        return None
    mid = statistics.median(numeric)
    if not mid:
        return None
    return abs(max(numeric) - min(numeric)) / abs(mid)


def _fail(msg: str, code: int = 1) -> NoReturn:
    _log(f"[proxy_runner] ERROR: {msg}")
    sys.exit(code)


def _build_phase(reg: dict, suite: dict, proxy_name: str, entry: dict, run_dir: str,
                 cfg: dict) -> bool:
    """Build every proxy this suite measures. False means the run must stop.

    A failed build must not reach the ratchet at all: the caller exits before
    metrics.json is written, so there is no verdict, no ledger entry, and the
    baseline slot is not spent on a run that measured nothing.
    """
    builds = build_mod.builds_for(reg, suite, proxy_name, entry)
    if not builds:
        return True

    records = []
    for i, (name, ent) in enumerate(builds, start=1):
        total = len(builds)
        _write_phase(run_dir, "build",
                     f"building {name}" + (f" ({i}/{total})" if total > 1 else ""),
                     index=i, total=total)
        _log(f"[proxy_runner] build proxy={name} cmd={ent.get('build_cmd')}")
        rec = build_mod.run_build(ent, run_dir, proxy=name)
        records.append(rec)
        _log(f"[proxy_runner] build proxy={name} status={rec['status']} "
             f"returncode={rec.get('returncode')} duration_s={rec.get('duration_s')}")
        if rec["status"] not in ("ok", "skipped"):
            _log(f"[proxy_runner] ERROR: {rec.get('error', rec['status'])}")
            build_mod.write_report(run_dir, records)
            return False

        # A build that reports success but leaves no executable has not built the
        # thing that was registered, and every later error would point elsewhere.
        exe = ent.get("executable_path", "")
        if exe and not os.path.isfile(exe):
            rec["status"] = "failed"
            rec["error"] = (f"build succeeded but executable_path is still missing: "
                            f"{exe}. The build does not produce the file registered "
                            f"for proxy '{name}'.")
            _log(f"[proxy_runner] ERROR: {rec['error']}")
            build_mod.write_report(run_dir, records)
            return False

    # A build that rewrites a tracked source (a code generator, a formatter in the
    # Makefile, a generated header listed in optimize_paths) leaves the measured
    # tree different from the one snapshotted at launch, so the run would be
    # recorded against code that is not what ran.
    launch = layout.read_json(os.path.join(run_dir, "tree_at_launch.json"), {}) or {}
    opt_paths = (cfg.get("optimize_paths") or launch.get("paths") or [])
    expected = launch.get("fingerprint", "")
    if expected and opt_paths:
        root = os.environ.get("MCP_FILES_ROOT") or os.getcwd()
        current = tree_snapshot.fingerprint(os.path.realpath(root), opt_paths)
        if current != expected:
            msg = ("the build modified a file listed in optimize_paths, so the tree "
                   "measured would not be the tree snapshotted at launch. Keep "
                   "generated files out of optimize_paths.")
            _log(f"[proxy_runner] ERROR: {msg}")
            records.append({"proxy": proxy_name, "status": "failed",
                            "error": msg, "duration_s": 0.0})
            build_mod.write_report(run_dir, records)
            return False

    build_mod.write_report(run_dir, records)
    _write_phase(run_dir, "measure_start", "preparing measurements")
    return True


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Proxy optimization runner")
    parser.add_argument(
        "--run-dir", required=True,
        help="Run directory (must contain config.json; receives metrics.json).",
    )
    parser.add_argument(
        "--phase", choices=("all", "build", "run"), default="all",
        help="Which half of the run to do. 'all' (the default) builds then measures "
             "in one process, which is what a local run does. 'build' and 'run' split "
             "that across two Slurm jobs sharing this run directory, so a build can go "
             "to a machine made for compiling and the measurement to the machine being "
             "measured.",
    )
    return parser.parse_args()


def _load_config(run_dir: str) -> dict:
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.isfile(config_path):
        _fail(f"config not found: {config_path}")
    with open(config_path) as fh:
        return json.load(fh)


def _record_machine(run_dir: str) -> None:
    """Write ``machine.json`` from the machine running this, so the ratchet compares
    timings taken on the same kind of machine. Never fatal."""
    try:
        from ...shared.cpu_facts import local_machine
        machine = local_machine()
        with open(os.path.join(run_dir, "machine.json"), "w") as fh:
            json.dump(machine, fh, indent=2)
        _log(f"[proxy_runner] machine={machine['host']} "
             f"context={machine['execution_context']} cpu={machine['cpu_model']!r}")
    except Exception as exc:
        _log(f"[proxy_runner] WARNING: could not record the machine: {exc}")


def _load_target(proxy_name: str, benchmark_name: str) -> tuple[dict, dict, dict]:
    """The registry, the proxy's entry and the suite; exits when one is missing."""
    try:
        reg = store.load_registry()
    except RuntimeError as exc:
        _fail(f"registry corrupt: {exc}")
    if proxy_name not in reg:
        _fail(f"proxy '{proxy_name}' not in registry")
    suite = store.load_suite(benchmark_name)
    if suite is None:
        _fail(f"suite '{benchmark_name}' not found")
    return reg, reg[proxy_name], suite


def _check_split_build(reg: dict, suite: dict, proxy_name: str, entry: dict,
                       run_dir: str) -> None:
    """The run half of a split run: the build job must have succeeded, or nothing is
    measured (a hand-edited dependency would otherwise measure a stale binary)."""
    report = build_mod.read_report(run_dir)
    if report is not None and report.get("status") not in ("ok", "skipped"):
        _fail(build_mod.failure_summary(report), 2)
    if report is None and build_mod.builds_for(reg, suite, proxy_name, entry):
        _fail("this run was split into a build job and a run job, but no build report "
              "is present — the build job did not run. Nothing has been measured.", 2)


@dataclass
class _Plan:
    """How the suite is measured, read from the run config."""
    proxy_name: str
    cases: list[dict]
    requirements: list[dict]
    deadline: float
    per_case_timeout_s: float | None
    repeat: int
    primary_metric: str
    conv_h_param: str
    conv_err_key: str

    @classmethod
    def from_config(cls, cfg: dict, cases: list[dict]) -> _Plan:
        conv = cfg.get("convergence") or {}
        return cls(
            proxy_name=cfg.get("proxy_name", ""),
            cases=cases,
            requirements=cfg.get("requirements", []),
            # The budget starts after the build, so compiling never eats measurement time.
            deadline=time.monotonic() + cfg.get("deadline_s", 86400.0),
            per_case_timeout_s=cfg.get("per_case_timeout_s") or None,
            repeat=max(1, int(cfg.get("repeat", 1) or 1)),
            primary_metric=cfg.get("primary_metric", "time_s"),
            conv_h_param=conv.get("h_param", ""),
            conv_err_key=conv.get("error_metric", "l2_rel"),
        )

    @property
    def planned(self) -> int:
        """Every sweep of every case is one measurement."""
        return sum(len(c.get("param_sweeps") or [{}]) for c in self.cases)

    @property
    def case_requirements(self) -> list[dict]:
        # convergence_order is fitted across the sweep, never in one case's metrics.
        return [r for r in self.requirements if r.get("metric") != "convergence_order"]

    @property
    def run_requirements(self) -> list[dict]:
        return [r for r in self.requirements if r.get("metric") == "convergence_order"]


@dataclass
class _Tally:
    """What the measurements produced so far."""
    results: list[dict] = field(default_factory=list)
    best_inputs: list[tuple[str, bool, float | None]] = field(default_factory=list)
    conv_pairs: list[tuple[float, float]] = field(default_factory=list)
    passed: int = 0
    total: int = 0


def _measure_replicates(plan: _Plan, case: dict, case_entry: dict, idx: int,
                        overrides: dict, label: str, index: int,
                        run_dir: str) -> tuple[str, dict, list[dict]]:
    """Run one sweep *repeat* times; stop at the first error. Returns the last case dir,
    the last row and each replicate's metrics."""
    case_id = case.get("case_id", "?")
    tag = f"opt_{plan.proxy_name}_{case_id}_{idx}"
    replicates: list[dict] = []
    run_case_dir: str = ""
    row: dict = {}
    for rep in range(plan.repeat):
        if plan.repeat > 1:
            _write_phase(run_dir, "measure", f"{label} rep {rep + 1}/{plan.repeat}",
                         index=index, total=plan.planned,
                         replicate=rep + 1, replicates=plan.repeat)
        run_case_dir, row = execute.run_benchmark_case(
            entry=case_entry,
            proxy_name=case.get("proxy_name", plan.proxy_name),
            reference_name=case.get("reference_name", ""),
            extra_params=case.get("extra_params", ""),
            param_overrides=overrides,
            extra_metrics=case.get("metrics", []),
            deadline=plan.deadline,
            tag_suffix=tag if plan.repeat == 1 else f"{tag}_r{rep}",
            per_case_timeout_s=plan.per_case_timeout_s,
        )
        if "error" in row:
            break
        rep_metrics = layout.read_json(os.path.join(run_case_dir, "metrics.json"), {}) or {}
        replicates.append(rep_metrics)
        if plan.repeat > 1:
            _log(f"[proxy_runner] case={case_id} sweep={idx} replicate={rep} "
                 f"{plan.primary_metric}={rep_metrics.get(plan.primary_metric)}")
    return run_case_dir, row, replicates


def _measure_sweep(plan: _Plan, tally: _Tally, case: dict, case_entry: dict, idx: int,
                   overrides: dict, run_dir: str) -> None:
    """Measure one sweep of *case* and add its outcome to *tally*."""
    case_id = case.get("case_id", "?")
    tally.total += 1
    label = f"case {case_id}" + (f" ({tally.total}/{plan.planned})" if plan.planned > 1 else "")
    _write_phase(run_dir, "measure", label, index=tally.total, total=plan.planned)
    _log(f"[proxy_runner] running case={case_id} sweep={idx} "
         f"overrides={overrides} repeat={plan.repeat}")

    run_case_dir, row, replicates = _measure_replicates(
        plan, case, case_entry, idx, overrides, label, tally.total, run_dir)
    if "error" in row:
        _log(f"[proxy_runner] case={case_id} sweep={idx} error={row['error']}")
        tally.best_inputs.append((case_id, False, None))
        tally.results.append({"case_id": case_id, "sweep_idx": idx, "sweep": overrides,
                              "run_dir": run_case_dir, "error": row["error"],
                              "passed": False})
        return

    # With replicates, the median of each metric, never the best: the minimum of
    # several draws measures how lucky the node was, not the code.
    case_metrics = _aggregate_replicates(replicates)
    spread = _relative_spread(replicates, plan.primary_metric)
    if spread is not None:
        case_metrics["primary_spread"] = spread

    req_result = metrics.evaluate_requirements(case_metrics, plan.case_requirements)
    passed = req_result["passed"]
    tally.passed += int(passed)
    tally.best_inputs.append((case_id, passed, case_metrics.get("time_s")))

    if plan.conv_h_param:
        h_val = overrides.get(plan.conv_h_param, case_metrics.get(plan.conv_h_param))
        err_val = case_metrics.get(plan.conv_err_key)
        if isinstance(h_val, (int, float)) and isinstance(err_val, (int, float)):
            tally.conv_pairs.append((float(h_val), float(err_val)))

    metric_tokens = " ".join(
        f"{k}={v}" for k, v in case_metrics.items()
        if isinstance(v, (int, float)) and k != "comparison_to_reference")
    _log(f"[proxy_runner] case={case_id} sweep={idx} passed={passed} {metric_tokens}")
    tally.results.append({
        "case_id": case_id, "sweep_idx": idx, "sweep": overrides,
        "run_dir": run_case_dir, "metrics": case_metrics,
        "requirements": req_result, "passed": passed,
    })


def _measure_suite(plan: _Plan, reg: dict, entry: dict, run_dir: str) -> _Tally:
    tally = _Tally()
    for case in plan.cases:
        case_entry = reg.get(case.get("proxy_name", plan.proxy_name), entry)
        for idx, overrides in enumerate(case.get("param_sweeps") or [{}]):
            _measure_sweep(plan, tally, case, case_entry, idx, overrides, run_dir)
    return tally


def _run_level(plan: _Plan, tally: _Tally) -> tuple[float | None, bool]:
    """The observed order of accuracy across the sweep, and whether its requirement holds."""
    order = metrics.observed_convergence_order(tally.conv_pairs) if tally.conv_pairs else None
    if not plan.run_requirements:
        return order, True
    passed = metrics.evaluate_requirements(
        {"convergence_order": order}, plan.run_requirements)["passed"]
    _log(f"[proxy_runner] convergence_order={order} passed={passed}")
    return order, passed


def _write_metrics(run_dir: str, output: dict) -> None:
    try:
        with open(os.path.join(run_dir, "metrics.json"), "w") as fh:
            json.dump(output, fh, indent=2)
    except OSError as exc:
        _log(f"[proxy_runner] WARNING: could not write metrics.json: {exc}")


def _settle_ratchet(proxy_name: str, run_dir: str, output: dict) -> None:
    """Settle the ledger and best now, so they never depend on the agent calling
    ``results()`` (which replays the frozen outcome). Never fatal."""
    try:
        from ._ops._eval_ratchet import ratchet_state
        opt_cfg = store.load_opt_config(proxy_name)
        if opt_cfg and opt_cfg.get("proxy_name") == proxy_name:
            outcome = ratchet_state(proxy_name, run_dir, output, opt_cfg)
            _log(f"[proxy_runner] ratchet verdict={outcome.get('verdict')} "
                 f"feasible={outcome.get('feasible')} stall={outcome.get('stall')}")
            if outcome.get("timing_warning"):
                _log(f"[proxy_runner] WARNING: {outcome['timing_warning']}")
    except Exception as exc:
        _log(f"[proxy_runner] WARNING: ratchet settle failed: {exc}")


def main() -> None:
    args = _parse_args()
    run_dir = os.path.abspath(args.run_dir)
    cfg = _load_config(run_dir)
    proxy_name = cfg.get("proxy_name", "")
    benchmark_name = cfg.get("benchmark_name", "")
    _log(f"[proxy_runner] proxy_name={proxy_name}")
    _log(f"[proxy_runner] benchmark_name={benchmark_name}")
    _log(f"[proxy_runner] requirements={len(cfg.get('requirements', []))}")

    _record_machine(run_dir)
    reg, entry, suite = _load_target(proxy_name, benchmark_name)
    cases = suite.get("cases", [])
    _log(f"[proxy_runner] suite_cases={len(cases)}")

    # Built once per run, before any case.
    if args.phase in ("all", "build"):
        if not _build_phase(reg, suite, proxy_name, entry, run_dir, cfg):
            sys.exit(2)
        if args.phase == "build":
            # No metrics.json: it is what makes a run read as finished.
            _write_phase(run_dir, "build_done", "build finished; waiting for the run job")
            _log("[proxy_runner] build phase complete")
            return
    else:
        _check_split_build(reg, suite, proxy_name, entry, run_dir)
        _write_phase(run_dir, "measure_start", "preparing measurements")

    plan = _Plan.from_config(cfg, cases)
    tally = _measure_suite(plan, reg, entry, run_dir)
    convergence_order, run_level_passed = _run_level(plan, tally)

    _write_phase(run_dir, "settle", "settling")
    all_passed = tally.passed == tally.total and tally.total > 0 and run_level_passed
    best_case, best_time = ratchet.select_best_case(tally.best_inputs)
    _log(f"[proxy_runner] summary cases_passed={tally.passed} cases_total={tally.total} "
         f"all_passed={all_passed} best_case={best_case or 'none'} "
         f"best_time_s={best_time if best_time is not None else 'n/a'}")

    build_report = build_mod.read_report(run_dir) or {}
    output = {
        "proxy_name": proxy_name,
        "benchmark_name": benchmark_name,
        "cases_passed": tally.passed,
        "cases_total": tally.total,
        "all_passed": all_passed,
        "best_case": best_case,
        "best_time_s": best_time,
        "convergence_order": convergence_order,
        # Run-level, so no requirement can gate on how long a compiler took.
        "build_s": build_report.get("total_duration_s", 0.0),
        "results": tally.results,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_metrics(run_dir, output)
    _settle_ratchet(proxy_name, run_dir, output)

    if all_passed:
        _log("[proxy_runner] All requirements satisfied.")
    else:
        _log(f"[proxy_runner] NOTE: {tally.total - tally.passed} case(s) did not meet "
             "requirements. Read the proxy source and modify the implementation, "
             "then call proxy_eval(op='run', confirm=True) again. "
             "Use proxy_eval(op='reset', confirm=True) to revert to the original.")


if __name__ == "__main__":
    main()
