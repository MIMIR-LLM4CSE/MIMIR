"""Placing the runs the watchers hold onto the panel's host and node rows, and reading
the calls its rows offer.

Groups offer ``host``, ``local``, ``nodes`` and ``call``; runs carry ``scheduler`` and
``nodes``. No tool or server is named.
"""

from __future__ import annotations

import json

# A run nothing claimed is not dropped: it goes back to the caller for the flat list the
# panel already shows, because a run the user cannot see is worse than one shown out of
# place.


def _run_nodes(run: dict) -> set[str]:
    """The nodes a run says it is on. ``node`` as well as ``nodes``: a status op that
    reports a single machine is as entitled to its own spelling as one that reports a
    list."""
    out: set[str] = set()
    for key in ("nodes", "node"):
        value = run.get(key)
        if isinstance(value, str) and value.strip():
            out.add(value.strip())
        elif isinstance(value, (list, tuple)):
            out.update(str(v).strip() for v in value if str(v).strip())
    return out


def _claims_host(group: dict, run: dict) -> bool:
    """Whether this host group is where *run* is running.

    A run carrying a scheduler belongs to the host of that name. One carrying none is
    on this machine — a detached shell command has no scheduler to name, and ``local``
    is how a group says it is this machine.
    """
    scheduler = str(run.get("scheduler") or "").strip()
    if scheduler:
        return str(group.get("host") or "") == scheduler
    return bool(group.get("local"))


def _badge(run: dict) -> dict:
    """What a badge shows, and what its disclosure shows when opened.

    A projection rather than the run itself: the panel renders what it is given, and
    giving it the whole descriptor would have it deciding which fields matter.
    """
    origin = run.get("origin") if isinstance(run.get("origin"), dict) else {}
    badge = {
        "job_key": run.get("job_key", "?"),
        "kind": run.get("kind", ""),
        "state": run.get("state", ""),
        "origin": origin,
        "cancellable": bool(run.get("cancellable")),
    }
    for key in ("phase", "percent", "command", "pid", "job_id", "elapsed_s", "scheduler"):
        if run.get(key) not in (None, "", []):
            badge[key] = run[key]
    nodes = sorted(_run_nodes(run))
    if nodes:
        badge["nodes"] = nodes
    return badge


def offered_calls(sections: list[dict]) -> set[str]:
    """Every ``call`` a group of *sections* offers, at any depth, as ``[tool, args]`` JSON."""
    out: set[str] = set()
    for section in sections:
        for top in section.get("groups") or []:
            if not isinstance(top, dict):
                continue
            for group in (top, *_descendants(top)):
                call = group.get("call")
                if isinstance(call, dict) and call.get("tool"):
                    args = call.get("args") if isinstance(call.get("args"), dict) else {}
                    out.add(json.dumps([str(call["tool"]), args], sort_keys=True))
    return out


def _descendants(group: dict):
    """Every group under *group*, at any depth: a kind may sit under a heading."""
    for child in group.get("groups") or []:
        if isinstance(child, dict):
            yield child
            yield from _descendants(child)


def attach_runs(sections: list[dict], runs: list[dict]) -> tuple[list[dict], list[dict]]:
    """Hang each run on its host row and its node kind's row; ``(sections, unplaced)``."""
    placed: set[str] = set()
    for section in sections:
        for group in section.get("groups") or []:
            if not isinstance(group, dict):
                continue
            host_badges = []
            for run in runs:
                if not _claims_host(group, run):
                    continue
                host_badges.append(_badge(run))
                placed.add(str(run.get("job_key") or ""))
                for kind in _descendants(group):
                    known = {str(n) for n in (kind.get("nodes") or [])}
                    if known & _run_nodes(run):
                        kind.setdefault("badges", []).append(_badge(run))
            if host_badges:
                group["badges"] = host_badges
    unplaced = [run for run in runs
                if str(run.get("job_key") or "") not in placed]
    return sections, unplaced
