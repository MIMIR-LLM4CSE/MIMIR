"""``python -m mimir.evals``: run a benchmark under prose variants, then compare the reports.

The code's own texts always run, as the reference. Runs are interleaved — each repetition
runs every task under every variant before the next — so a backend that drifts over the
session drifts for all of them alike. An adapter whose grader runs over every answer at
once (a ``BatchAdapter``) is graded at the end of each repetition, one batch per variant.
``--resume`` carries on an interrupted report. ``--help`` lists the options.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import os
import sys
import time
from collections import defaultdict
from typing import Any

from mimir.client.prompt.variants import load_variant, use_variant
from mimir.runner import RunResult, grade_pending, run_one, use_backend

from .report import REFERENCE, Report, compare, load_runs

# What a resumed report must share with the run that resumes it.
_SAME_ON_RESUME = ("adapter", "adapter_args", "model", "backend", "enforcement")


def _adapter_args(pairs: list[str]) -> dict[str, str]:
    """``KEY=VALUE`` pairs as keyword arguments, left as strings for the adapter to read."""
    kwargs: dict[str, str] = {}
    for pair in pairs:
        key, sep, value = pair.partition("=")
        if not sep or not key:
            raise SystemExit(f"--adapter-arg is KEY=VALUE, not {pair!r}")
        kwargs[key] = value
    return kwargs


def _adapter(spec: str, kwargs: dict[str, str] | None = None) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--adapter is module:Class, not {spec!r}")
    return getattr(importlib.import_module(module_name), attr)(**(kwargs or {}))


def _resumed(path: str, header: dict[str, Any]) -> list[dict[str, Any]]:
    """The runs of the report at *path*, refused if it was made by a different run."""
    with open(path, encoding="utf-8") as fh:
        earlier = json.load(fh)
    differs = [k for k in _SAME_ON_RESUME if earlier.get(k) != header.get(k)]
    if differs:
        raise SystemExit(f"{path} was made with another {', '.join(differs)}: "
                         "resume it with the same options, or write to a new --out")
    return list(earlier.get("runs", []))


async def _run(args: argparse.Namespace, get_backend_override: Any = None) -> None:
    """Every task, under every variant, *repeat* times; *get_backend_override* is for CI."""
    if get_backend_override is None:
        use_backend(args.backend)
    adapter_args = _adapter_args(args.adapter_arg)
    adapter = _adapter(args.adapter, adapter_args)
    variants: list[dict[str, Any] | None] = [None, *(load_variant(p) for p in args.variant)]
    tasks = adapter.load_tasks(args.limit)
    header = {"adapter": args.adapter, "adapter_args": adapter_args, "model": args.model,
              "backend": args.backend, "enforcement": args.enforcement, "repeat": args.repeat,
              "variants": [REFERENCE, *(v["name"] for v in variants[1:] if v)]}
    earlier = _resumed(args.out, header) if args.resume and os.path.exists(args.out) else []
    report = Report(args.out, header, earlier)
    done = {(r["task_id"], r["variant"], r["repetition"]) for r in earlier}
    # The runs collected and not yet graded, by repetition and variant: one batch each.
    batches: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for run in earlier:
        if run.get("pending"):
            batches[(run["repetition"], run["variant"])].append(run)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for repetition in range(args.repeat):
        for task in tasks:
            for variant in variants:
                with use_variant(variant) as name:
                    if (task.id, name or REFERENCE, repetition) in done:
                        continue
                    result = await run_one(task, adapter, args.model,
                                           get_backend_override=get_backend_override,
                                           enforcement=args.enforcement,
                                           keep_workspace=args.keep_workspaces)
                run = {**result.as_dict(), "variant": name or REFERENCE,
                       "repetition": repetition}
                report.add(run)
                if run["pending"]:
                    batches[(repetition, run["variant"])].append(run)
                print(f"[{repetition + 1}/{args.repeat}] {task.id} · {run['variant']}: "
                      f"{_outcome(run)}", flush=True)
        for (rep, variant), runs in sorted(batches.items()):
            if rep == repetition:
                await _grade_batch(adapter, runs, f"{adapter.name}-{variant}-r{rep}-{stamp}")
                report.save()


def _outcome(run: dict[str, Any]) -> str:
    if run["pending"]:
        return "collected"
    return "skipped" if run["skipped"] else "pass" if run["passed"] else "fail"


async def _grade_batch(adapter: Any, runs: list[dict[str, Any]], label: str) -> None:
    """Grade the pending *runs* as one batch named *label*, their report entries updated."""
    results = [RunResult.from_dict(run) for run in runs]
    print(f"grading {len(results)} run(s) as {label}", flush=True)
    await grade_pending(adapter, results, label)
    for run, result in zip(runs, results):
        run.update(result.as_dict())
        print(f"  {run['task_id']} · {run['variant']}: {_outcome(run)} — {run['detail']}",
              flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mimir.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a benchmark under the code's prose and variants")
    run.add_argument("--adapter", required=True,
                     help="module:Class of a BenchmarkAdapter or BatchAdapter")
    run.add_argument("--adapter-arg", action="append", default=[], metavar="KEY=VALUE",
                     help="a keyword argument for the adapter (repeatable)")
    run.add_argument("--variant", action="append", default=[], help="a prose variant (JSON)")
    run.add_argument("--repeat", type=int, default=3)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--model", required=True)
    run.add_argument("--backend", default="vllm")
    run.add_argument("--enforcement", default=None, choices=["strict", "light", "off"])
    run.add_argument("--out", required=True)
    run.add_argument("--resume", action="store_true",
                     help="carry on the report at --out: its runs are kept, the rest run")
    run.add_argument("--keep-workspaces", action="store_true",
                     help="leave each task's workspace on disk")
    cmp = sub.add_parser("compare", help="compare the variants of one or more reports")
    cmp.add_argument("reports", nargs="+")
    args = parser.parse_args(argv)
    if args.command == "run":
        asyncio.run(_run(args))
        print(f"report: {args.out}")
    else:
        sys.stdout.write(compare(load_runs(args.reports)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
