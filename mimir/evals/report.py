"""A report of repeated runs, written as it grows, and the comparison of its variants.

A report is ``{"adapter", "model", "enforcement", "repeat", "runs": [...]}``; each run is a
``RunResult`` with the variant it ran under and its repetition. The comparison reads the
first variant as the reference and states, for each other, whether the gap is measurable
(intervals widened for the variants compared at once, and McNemar's test over the runs the
two share), then what each variant cost: steps, tokens, context, time.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from statistics import NormalDist
from typing import Any, Iterable

from .stats import Z95, mcnemar, newcombe, wilson

REFERENCE = "code"
# The chance, over all the variants compared at once, that any is called measurably
# different by luck alone.
FAMILY_ERROR = 0.05


def joint_z(comparisons: int) -> float:
    """The z that holds *comparisons* intervals together at ``FAMILY_ERROR`` (Bonferroni)."""
    if comparisons <= 1:
        return Z95
    return NormalDist().inv_cdf(1 - FAMILY_ERROR / (2 * comparisons))


class Report:
    """The runs so far, rewritten whole after each: an interrupted benchmark keeps them.

    *runs* are those an earlier, interrupted report already holds, kept as they were.
    """

    def __init__(self, path: str, header: dict[str, Any],
                 runs: list[dict[str, Any]] | None = None) -> None:
        self.path = path
        self.data: dict[str, Any] = {**header, "runs": list(runs or [])}

    def add(self, run: dict[str, Any]) -> None:
        self.data["runs"].append(run)
        self.save()

    def save(self) -> None:
        """Write the report as it stands, the runs added being changed in place since."""
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(self.data, fh, indent=2)
        os.replace(tmp, self.path)


def load_runs(paths: Iterable[str]) -> list[dict[str, Any]]:
    """Every run of the reports at *paths*, in order."""
    runs: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            runs.extend(json.load(fh).get("runs", []))
    return runs


def _variant(run: dict[str, Any]) -> str:
    return run.get("variant") or REFERENCE


def _mean(values: list[float]) -> str:
    return f"{sum(values) / len(values):.1f}" if values else "—"


def _pct(x: float) -> str:
    return f"{100 * x:.0f} %"


def _total(run: dict[str, Any], key: str) -> float | None:
    value = run.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def _known(values: list[float | None]) -> list[float]:
    return [v for v in values if v is not None]


def _cost_rows(names: list[str], scored: dict[str, list[dict[str, Any]]]) -> list[str]:
    """What each variant spent per run, on the runs it was scored on."""
    rows = ["| Variant | Steps | Side calls | Prompt tokens | Completion tokens | Cached "
            "| Peak context | Answer time (s) | Compactions | Refused |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        runs = scored[v]
        prompt = _known([_total(r, "prompt_tokens") for r in runs])
        cached = _known([_total(r, "cached_tokens") for r in runs])
        share = (f"{100 * sum(cached) / sum(prompt):.0f} %"
                 if prompt and cached and sum(prompt) else "—")
        rows.append(
            f"| {v} | {_mean([float(r.get('model_calls', 0)) for r in runs])} "
            f"| {_mean([float(sum((r.get('side_calls') or {}).values())) for r in runs])} "
            f"| {_mean(prompt)} "
            f"| {_mean(_known([_total(r, 'completion_tokens') for r in runs]))} "
            f"| {share} "
            f"| {_mean(_known([_total(r, 'peak_prompt_tokens') for r in runs]))} "
            f"| {_mean(_known([_total(r, 'answer_secs') for r in runs]))} "
            f"| {_mean([float((r.get('context_events') or {}).get('compact', 0)) for r in runs])} "
            f"| {_mean([float(len(r.get('refused') or [])) for r in runs])} |")
    return rows


def _paired(reference: list[dict[str, Any]], other: list[dict[str, Any]]) -> tuple[int, int, int]:
    """The runs both variants made of one task at one repetition: ``(pairs, only the
    reference passed, only the other passed)``."""
    def key(run: dict[str, Any]) -> tuple[str, int]:
        return run["task_id"], int(run.get("repetition", 0))

    mine = {key(r): bool(r.get("passed")) for r in reference}
    pairs = only_ref = only_other = 0
    for run in other:
        if key(run) not in mine:
            continue
        pairs += 1
        ref_passed, other_passed = mine[key(run)], bool(run.get("passed"))
        only_ref += ref_passed and not other_passed
        only_other += other_passed and not ref_passed
    return pairs, only_ref, only_other


def compare(runs: list[dict[str, Any]]) -> str:
    """A markdown comparison of the variants found in *runs*, the first being the reference."""
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        by_variant[_variant(run)].append(run)
    if not by_variant:
        return "No run to compare."
    names = list(by_variant)
    scored = {v: [r for r in by_variant[v] if not r.get("skipped")] for v in names}
    passed = {v: sum(1 for r in scored[v] if r.get("passed")) for v in names}

    lines = ["| Variant | Pass rate | 95 % interval | Runs | Tool calls | Model calls "
             "| Prompt tokens | Reminders | Incomplete |",
             "|---|---|---|---|---|---|---|---|---|"]
    for v in names:
        n = len(scored[v])
        lo, hi = wilson(passed[v], n)
        rate = passed[v] / n if n else 0.0
        tokens = [float(r["prompt_tokens"]) for r in scored[v] if r.get("prompt_tokens") is not None]
        lines.append(
            f"| {v} | {_pct(rate)} ({passed[v]}/{n}) | {_pct(lo)}–{_pct(hi)} | {n} "
            f"| {_mean([float(r.get('tool_calls', 0)) for r in scored[v]])} "
            f"| {_mean([float(r.get('model_calls', 0)) for r in scored[v]])} "
            f"| {_mean(tokens)} "
            f"| {_mean([float(sum((r.get('nudges') or {}).values())) for r in scored[v]])} "
            f"| {_pct(sum(1 for r in scored[v] if r.get('incomplete')) / n) if n else '—'} |")

    reference = names[0]
    comparisons = len(names) - 1
    z = joint_z(comparisons)
    verdicts = []
    if comparisons > 1:
        # Five wordings each held at 95 % alone would crown one by luck about once in four.
        verdicts.append(f"{comparisons} variants are compared with the reference at once: each "
                        f"interval below is widened (Bonferroni) so that the chance of any of them "
                        f"being called different by luck stays under {FAMILY_ERROR:.0%}.")
        verdicts.append("")
    for v in names[1:]:
        gap, lo, hi = newcombe(passed[reference], len(scored[reference]), passed[v], len(scored[v]), z)
        if lo <= 0 <= hi:
            verdict = "no measurable difference"
        else:
            verdict = "measurably better" if gap > 0 else "measurably worse"
        level = "95 %" if comparisons <= 1 else "joint 95 %"
        verdicts.append(f"- **{v}** against **{reference}**: {gap * 100:+.0f} points "
                        f"({level} interval {lo * 100:+.0f} to {hi * 100:+.0f}) — {verdict}.")
        pairs, only_ref, only_other = _paired(scored[reference], scored[v])
        if pairs:
            p = mcnemar(only_ref, only_other)
            alpha = FAMILY_ERROR / max(1, comparisons)
            verdicts.append(f"  Paired over {pairs} runs of the same tasks: {only_other} passed "
                            f"only under **{v}**, {only_ref} only under **{reference}** "
                            f"(exact McNemar p = {p:.3f}"
                            + (f", measurable at {alpha:.3g})." if p < alpha else ", not measurable)."))

    tasks = sorted({r["task_id"] for r in runs})
    per_task = ["| Task | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    for task in tasks:
        cells = []
        for v in names:
            mine = [r for r in scored[v] if r["task_id"] == task]
            cells.append(f"{sum(1 for r in mine if r.get('passed'))}/{len(mine)}" if mine else "—")
        per_task.append(f"| {task} | " + " | ".join(cells) + " |")

    return "\n".join(["## Variants", "", *lines, "", "## Cost per run", "",
                      *_cost_rows(names, scored), "", "## Verdict", "",
                      *(verdicts or ["- Only one variant: nothing to compare."]),
                      "", "## By task", "", *per_task, ""])
