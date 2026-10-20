"""``python -m mimir.evals``: run a benchmark under prose variants, then compare the reports.

The code's own texts always run, as the reference. Runs are interleaved — each repetition
runs every task under every variant before the next — so a backend that drifts over the
session drifts for all of them alike. ``--help`` lists the options.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib
import os
import sys
from typing import Any

from mimir.client.prompt.variants import load_variant, use_variant
from mimir.runner import run_one

from .report import REFERENCE, Report, compare, load_runs


def _adapter(spec: str) -> Any:
    module_name, _, attr = spec.partition(":")
    if not attr:
        raise SystemExit(f"--adapter is module:Class, not {spec!r}")
    return getattr(importlib.import_module(module_name), attr)()


async def _run(args: argparse.Namespace, get_backend_override: Any = None) -> None:
    """Every task, under every variant, *repeat* times; *get_backend_override* is for CI."""
    if get_backend_override is None:
        os.environ["LLM_BACKEND"] = args.backend
        from mimir.client.llm.factory import clear_backend_cache
        clear_backend_cache()
    adapter = _adapter(args.adapter)
    variants: list[dict[str, Any] | None] = [None, *(load_variant(p) for p in args.variant)]
    tasks = adapter.load_tasks(args.limit)
    report = Report(args.out, {"adapter": args.adapter, "model": args.model,
                               "backend": args.backend, "enforcement": args.enforcement,
                               "repeat": args.repeat,
                               "variants": [REFERENCE, *(v["name"] for v in variants[1:] if v)]})
    for repetition in range(args.repeat):
        for task in tasks:
            for variant in variants:
                with use_variant(variant) as name:
                    result = await run_one(task, adapter, args.model,
                                           get_backend_override=get_backend_override,
                                           enforcement=args.enforcement)
                run = {**result.as_dict(), "variant": name or REFERENCE,
                       "repetition": repetition}
                report.add(run)
                print(f"[{repetition + 1}/{args.repeat}] {task.id} · {run['variant']}: "
                      f"{'skipped' if run['skipped'] else 'pass' if run['passed'] else 'fail'}",
                      flush=True)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m mimir.evals")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="run a benchmark under the code's prose and variants")
    run.add_argument("--adapter", required=True, help="module:Class of a BenchmarkAdapter")
    run.add_argument("--variant", action="append", default=[], help="a prose variant (JSON)")
    run.add_argument("--repeat", type=int, default=3)
    run.add_argument("--limit", type=int, default=None)
    run.add_argument("--model", required=True)
    run.add_argument("--backend", default="vllm")
    run.add_argument("--enforcement", default=None, choices=["strict", "light", "off"])
    run.add_argument("--out", required=True)
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
