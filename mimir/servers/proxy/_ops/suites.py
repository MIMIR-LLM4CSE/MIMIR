"""Benchmark-suite ops: list/inspect/report (read-only), define/update/delete,
local suite runs, and one-call benchmark creation (reference + suite)."""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timezone

from _ops import err, ok
from _lib.execute import _REF_RUN_TIMEOUT, _run_benchmark_case, _seal_reference
from _lib.store import (
    refs_dir, suites_dir,
    _load_registry_or_err,
    _suite_path, _load_suite, _save_suite,
    _suite_results_dir, _latest_suite_results,
)


# ── read-only ─────────────────────────────────────────────────────────────────

def list_suites() -> dict:
    if not os.path.isdir(suites_dir()):
        return ok({"suites": [], "count": 0})
    suites = []
    for name in sorted(os.listdir(suites_dir())):
        suite = _load_suite(name)
        if suite is None:
            continue
        suites.append({
            "name":        suite.get("name"),
            "description": suite.get("description", ""),
            "case_count":  len(suite.get("cases", [])),
            "created_at":  suite.get("created_at"),
        })
    return ok({"suites": suites, "count": len(suites)})


def inspect_suite(name: str) -> dict:
    """Full suite definition + validation status + past result runs."""
    suite = _load_suite(name)
    if suite is None:
        return err(f"Suite '{name}' not found.",
                   hint="Call proxy_get(op='suites') to see defined suites.")

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)

    existing_refs = set(os.listdir(refs_dir())) if os.path.isdir(refs_dir()) else set()
    validation: list[dict] = []
    for case in suite.get("cases", []):
        issues = []
        if case.get("proxy_name") not in (reg or {}):
            issues.append(f"proxy '{case.get('proxy_name')}' not in registry")
        if case.get("reference_name") and case["reference_name"] not in existing_refs:
            issues.append(f"reference '{case.get('reference_name')}' not found — "
                          "create with proxy_exec(op='reference', ...) first")
        validation.append({
            "case_id": case.get("case_id"),
            "ok":      len(issues) == 0,
            "issues":  issues,
        })

    results_dir = _suite_results_dir(name)
    result_runs = []
    if os.path.isdir(results_dir):
        result_runs = sorted(os.listdir(results_dir), reverse=True)

    return ok({
        "suite":       suite,
        "validation":  validation,
        "result_runs": result_runs,
    })


def report(suite_name: str, run_timestamp: str = "") -> dict:
    if run_timestamp:
        results_dir = os.path.join(_suite_results_dir(suite_name), run_timestamp)
    else:
        results_dir = _latest_suite_results(suite_name)

    if not results_dir or not os.path.isdir(results_dir):
        return err(f"No results found for suite '{suite_name}'.",
                   hint="Run it with proxy_exec(op='suite', ...) first.")

    summary_path = os.path.join(results_dir, "summary.json")
    if not os.path.isfile(summary_path):
        return err("summary.json not found in results directory.",
                   hint=f"Check: {results_dir}")

    try:
        with open(summary_path) as fh:
            summary = json.load(fh)
    except (json.JSONDecodeError, OSError) as exc:
        return err(f"Could not read summary.json: {exc}")

    return ok({"suite": suite_name, "results_dir": results_dir, "summary": summary})


# ── suite definition helpers ──────────────────────────────────────────────────

def _validate_cases(cases: list[dict], reg: dict, *, require_proxy: bool) -> str | None:
    seen_ids: set[str] = set()
    for i, case in enumerate(cases):
        case_id = case.get("case_id", "")
        if not case_id:
            return f"cases[{i}] is missing 'case_id'."
        if case_id in seen_ids:
            return f"Duplicate case_id '{case_id}' in cases."
        seen_ids.add(case_id)
        sname = case.get("proxy_name", "")
        if require_proxy and not sname:
            return f"cases[{i}] (case_id='{case_id}') is missing 'proxy_name'."
        if sname and sname not in reg:
            return f"Proxy '{sname}' not found in registry."
        ps = case.get("param_sweeps")
        if ps is not None and (not isinstance(ps, list) or not ps):
            return (f"cases[{i}] (case_id='{case_id}'): "
                    "'param_sweeps' must be a non-empty list if provided.")
    return None


# ── mutations (confirm already checked by the dispatch tool) ──────────────────

def define(name: str, cases: list[dict], description: str = "") -> dict:
    if not name or not re.match(r"^[A-Za-z0-9_\-]+$", name):
        return err("name must be non-empty and contain only [A-Za-z0-9_-].")
    if not cases:
        return err("cases must be a non-empty list.")

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    case_err = _validate_cases(cases, reg, require_proxy=True)
    if case_err:
        return err(case_err, hint="Call proxy_manage(op='register', ...) first, or check for typos.")

    suite = {
        "name":        name,
        "description": description,
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "cases":       cases,
    }
    _save_suite(name, suite)
    return ok({"registered": suite,
               "next_step": f"proxy_exec(op='suite', suite_name='{name}', confirm=True) to run it."})


def update(name: str, cases: list[dict] | None = None, description: str = "") -> dict:
    suite = _load_suite(name)
    if suite is None:
        return err(f"Suite '{name}' not found.")

    if description:
        suite["description"] = description
    if cases is not None:
        if not cases:
            return err("cases must be a non-empty list.")
        reg, _reg_err = _load_registry_or_err()
        if _reg_err:
            return err(_reg_err)
        case_err = _validate_cases(cases, reg, require_proxy=False)
        if case_err:
            return err(case_err)
        suite["cases"] = cases
    suite["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_suite(name, suite)
    return ok({"updated": suite})


def delete(name: str) -> dict:
    p = _suite_path(name)
    if not os.path.isfile(p):
        return err(f"Suite '{name}' not found.")
    os.remove(p)
    return ok({"unregistered": name,
               "note": "Suite definition removed. Run history under suites/"
                       + name + "/results/ is preserved."})


def _prevalidate_suite(suite: dict, reg: dict) -> dict | None:
    """Check every case's proxy and reference exist before starting any run."""
    existing_refs = set(os.listdir(refs_dir())) if os.path.isdir(refs_dir()) else set()
    for case in suite.get("cases", []):
        sname = case.get("proxy_name", "")
        if sname not in reg:
            return err(f"Proxy '{sname}' not registered (case '{case.get('case_id')}').",
                       hint="Call proxy_manage(op='register', ...) first.")
        ref = case.get("reference_name", "")
        if ref and ref not in existing_refs:
            return err(f"Reference '{ref}' not found (case '{case.get('case_id')}').",
                       hint="Call proxy_exec(op='reference', ...) first.")
    return None


def run_suite(
    suite_name: str,
    extra_metrics: list[str] | None = None,
    timeout_s: int = 7200,
    per_case_timeout_s: int = 0,
) -> dict:
    """Run all cases in a suite locally (blocking); write and return the summary."""
    suite = _load_suite(suite_name)
    if suite is None:
        return err(f"Suite '{suite_name}' not found.",
                   hint="Call proxy_get(op='suites') to see defined suites.")

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)

    if extra_metrics is None:
        extra_metrics = []

    invalid = _prevalidate_suite(suite, reg)
    if invalid:
        return invalid

    deadline = time.monotonic() + max(1, timeout_s)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    results_dir = os.path.join(_suite_results_dir(suite_name), ts)
    os.makedirs(results_dir, exist_ok=True)

    all_rows: list[dict] = []

    for case in suite.get("cases", []):
        case_id        = case["case_id"]
        proxy_name     = case["proxy_name"]
        reference_name = case.get("reference_name", "")
        param_sweeps   = case.get("param_sweeps") or [{}]
        extra_params   = case.get("extra_params", "")
        case_extra_metrics = list(set(extra_metrics) | set(case.get("metrics", [])))

        entry = reg[proxy_name]

        for idx, sweep_overrides in enumerate(param_sweeps):
            tag_suffix = f"suite_{suite_name}_{case_id}_{idx}"
            run_dir, row = _run_benchmark_case(
                entry=entry,
                proxy_name=proxy_name,
                reference_name=reference_name,
                extra_params=extra_params,
                param_overrides=sweep_overrides,
                extra_metrics=case_extra_metrics,
                deadline=deadline,
                tag_suffix=tag_suffix,
                per_case_timeout_s=(per_case_timeout_s or None),
            )
            row["case_id"] = case_id
            row["proxy"]   = proxy_name
            row["sweep"]   = sweep_overrides
            row["scope"]   = case.get("scope", "full")

            # Record a pointer: suites/<suite>/results/<ts>/<case_id>/<idx>/run_dir
            ptr_dir = os.path.join(results_dir, case_id, str(idx))
            os.makedirs(ptr_dir, exist_ok=True)
            with open(os.path.join(ptr_dir, "run_dir"), "w") as fh:
                fh.write(run_dir)

            all_rows.append(row)

    summary = {
        "suite":         suite_name,
        "run_timestamp": ts,
        "rows":          all_rows,
    }
    with open(os.path.join(results_dir, "summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)

    return ok(summary)


def benchmark_create(
    proxy_name: str,
    benchmark_name: str,
    param_sweeps: list[dict] | None = None,
    reference_params: dict | None = None,
    extra_metrics: list[str] | None = None,
    description: str = "",
    timeout_s: int = _REF_RUN_TIMEOUT,
) -> dict:
    """Create a reference and register a one-case suite in a single blocking call."""
    if not benchmark_name or not re.match(r"^[A-Za-z0-9_\-]+$", benchmark_name):
        return err("benchmark_name must be non-empty and contain only [A-Za-z0-9_-].")

    reg, _reg_err = _load_registry_or_err()
    if _reg_err:
        return err(_reg_err)
    if proxy_name not in reg:
        return err(f"Proxy '{proxy_name}' not registered.",
                   hint="Call proxy_manage(op='register', ...) first.")
    entry = reg[proxy_name]

    ps     = param_sweeps or [{}]
    ref_po = reference_params if reference_params is not None else (ps[0] if ps else {})
    ref_name = benchmark_name + "_ref"

    # ── Step A: create reference ──────────────────────────────────────────────
    ref_result, ref_error = _seal_reference(
        entry, ref_name, param_overrides=ref_po, timeout_s=timeout_s,
    )
    if ref_error:
        return ref_error
    rd            = ref_result["ref_dir"]
    ref_metrics   = ref_result["metrics"]
    output_stored = ref_result["output_stored"]

    # ── Step B: register suite ────────────────────────────────────────────────
    suite = {
        "name":        benchmark_name,
        "description": description or f"Benchmark for {proxy_name}",
        "created_at":  datetime.now(timezone.utc).isoformat(),
        "cases": [
            {
                "case_id":        "main",
                "description":    description or f"Benchmark sweep for {proxy_name}",
                "scope":          "full",
                "proxy_name":     proxy_name,
                "reference_name": ref_name,
                "param_sweeps":   ps,
                "extra_params":   "",
                "metrics":        extra_metrics or [],
            }
        ],
    }
    _save_suite(benchmark_name, suite)

    return ok({
        "benchmark_name":    benchmark_name,
        "reference_name":    ref_name,
        "reference_dir":     rd,
        "reference_metrics": ref_metrics,
        "output_stored":     output_stored,
        "suite":             suite,
        "next_step": (
            f"proxy_exec(op='suite', suite_name='{benchmark_name}', confirm=True) "
            "to run the benchmark."
        ),
    })
