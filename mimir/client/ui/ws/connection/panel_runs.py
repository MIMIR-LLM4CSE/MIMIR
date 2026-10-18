"""Placing the runs the watchers hold onto the panel's host and node rows.

Groups offer ``host``, ``local`` or ``nodes``; runs carry ``scheduler`` and ``nodes``.
No tool or server is named.
"""

from __future__ import annotations

import os

# The workspace registry servers/_shared/slurm_site.py appends to, one host per line.
_HOSTS_FILE = "slurm_hosts"

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
    }
    for key in ("phase", "percent", "command", "pid", "job_id", "elapsed_s", "scheduler"):
        if run.get(key) not in (None, "", []):
            badge[key] = run[key]
    nodes = sorted(_run_nodes(run))
    if nodes:
        badge["nodes"] = nodes
    return badge


def recorded_hosts_sections(state_dir: str) -> list[dict]:
    """The Cluster section as the workspace registry has it, before any server is up.

    Names only: what each host is made of is a server's probe, and the drawer asks again
    once the agent is ready. Ids match the server's so open rows survive the swap.
    """
    try:
        with open(os.path.join(state_dir, _HOSTS_FILE)) as fh:
            names = list(dict.fromkeys(line.strip() for line in fh if line.strip()))
    except OSError:
        return []
    if not names:
        return []
    return [{
        "section": "Cluster",
        "title": "Cluster",
        "lines": [],
        "groups": [{"id": f"host:{name}", "label": name, "host": name, "local": False,
                    "lines": [{"label": "reached", "value": f"over ssh to {name}"}]}
                   for name in names],
        "detail": "Recorded for this workspace. Hardware appears once this "
                  "conversation's agent is up.",
    }]


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
                for kind in group.get("groups") or []:
                    if not isinstance(kind, dict):
                        continue
                    known = {str(n) for n in (kind.get("nodes") or [])}
                    if known & _run_nodes(run):
                        kind.setdefault("badges", []).append(_badge(run))
            if host_badges:
                group["badges"] = host_badges
    unplaced = [run for run in runs
                if str(run.get("job_key") or "") not in placed]
    return sections, unplaced
