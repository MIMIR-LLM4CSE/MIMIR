"""Parsing and aggregation of Slurm's node inventory.

Shared because two servers need the same view of the cluster's hardware from two
angles: ``hpc/server_hpc`` answers "what is free right now", while
``hpc/server_platform`` records the *stable* half — the kinds of machine the site
has — in its persisted digest. Duplicating a ``scontrol`` parser to serve both is
how the two would drift apart.
"""

import hashlib
import json
import re


# The node facts read below are all single-token values: the fields that can contain
# spaces (OS, Reason) are deliberately left alone, which is what lets a simple
# `KEY=<non-space>` scan stand in for a real parser.


def as_int(value: str) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def clean_gres(value: str) -> str:
    """'gpu:a100:8(S:1,3,5,7)' -> 'gpu:a100:8'; '(null)' -> ''."""
    if not value or value == "(null)":
        return ""
    return re.sub(r"\(S:[^)]*\)", "", value)


def parse_scontrol_nodes(stdout: str) -> list[dict]:
    nodes = []
    for line in stdout.splitlines():
        if "NodeName=" not in line:
            continue
        raw = {k: v for k, v in re.findall(r"\b(\w+)=([^\s]+)", line)}
        gres = clean_gres(raw.get("Gres", ""))
        cpu_tot, cpu_alloc = as_int(raw.get("CPUTot", "")), as_int(raw.get("CPUAlloc", ""))
        features = raw.get("AvailableFeatures", "")
        nodes.append({
            "node":             raw.get("NodeName", ""),
            "arch":             raw.get("Arch", ""),
            "state":            raw.get("State", ""),
            "partitions":       [p for p in raw.get("Partitions", "").split(",") if p],
            "cpus":             cpu_tot,
            "cpus_allocated":   cpu_alloc,
            "cpus_free":        (cpu_tot - cpu_alloc) if None not in (cpu_tot, cpu_alloc) else None,
            "cpu_load":         raw.get("CPULoad", ""),
            "sockets":          as_int(raw.get("Sockets", "")),
            "cores_per_socket": as_int(raw.get("CoresPerSocket", "")),
            "threads_per_core": as_int(raw.get("ThreadsPerCore", "")),
            "mem_mb":           as_int(raw.get("RealMemory", "")),
            "mem_free_mb":      as_int(raw.get("FreeMem", "")),
            "gres":             gres,
            "features":         "" if features == "(null)" else features,
        })
    return nodes


def aggregate_node_types(nodes: list[dict]) -> list[dict]:
    """Collapse nodes onto their hardware signature.

    A 124-node cluster listed one row per node buries the answer in noise; what the
    caller is choosing between is the handful of *kinds* of machine, and how much of
    each is free right now.
    """
    groups: dict[tuple, dict] = {}
    for n in nodes:
        key = (n["arch"], n["cpus"], n["mem_mb"], n["gres"],
               n["sockets"], n["cores_per_socket"], n["threads_per_core"], n["features"])
        g = groups.setdefault(key, {
            "arch": n["arch"], "cpus": n["cpus"], "mem_mb": n["mem_mb"],
            "mem_gb": round(n["mem_mb"] / 1024, 1) if n["mem_mb"] else None,
            "gres": n["gres"], "sockets": n["sockets"],
            "cores_per_socket": n["cores_per_socket"], "threads_per_core": n["threads_per_core"],
            "features": n["features"], "partitions": set(), "nodes_total": 0,
            "by_state": {}, "cpus_free_total": 0, "example_nodes": [],
        })
        g["partitions"].update(n["partitions"])
        g["nodes_total"] += 1
        state = (n["state"] or "UNKNOWN").split("+")[0].lower()
        g["by_state"][state] = g["by_state"].get(state, 0) + 1
        if n["cpus_free"]:
            g["cpus_free_total"] += n["cpus_free"]
        if len(g["example_nodes"]) < 3:
            g["example_nodes"].append(n["node"])

    out = []
    for g in groups.values():
        g["partitions"] = sorted(g["partitions"])
        out.append(g)
    # Most immediately usable first: idle nodes, then raw size.
    out.sort(key=lambda g: (-g["by_state"].get("idle", 0), -(g["cpus"] or 0)))
    return out



# Fields of an aggregated node type that describe the *machine*, not its current
# occupancy. The digest's freshness signal is computed over these alone: including
# by_state or cpus_free_total would make the signal change every few seconds as jobs
# start and end, which would turn a deterministic fingerprint into a busy-loop.
_STABLE_TYPE_FIELDS = (
    "arch", "cpus", "mem_mb", "gres", "sockets",
    "cores_per_socket", "threads_per_core", "features", "partitions",
)


def stable_types(types: list[dict]) -> list[dict]:
    """The occupancy-free projection of :func:`aggregate_node_types` output."""
    return [{k: t.get(k) for k in _STABLE_TYPE_FIELDS} for t in types]


def stable_signature(nodes: list[dict]) -> str:
    """A SHA-1 over the cluster's hardware makeup, blind to what is running on it.

    Two calls a second apart return the same digest on a busy cluster; adding a node
    of a new kind, or changing a partition's membership, changes it.
    """
    types = stable_types(aggregate_node_types(nodes))
    payload = json.dumps(sorted(types, key=lambda t: json.dumps(t, sort_keys=True)),
                         sort_keys=True, ensure_ascii=False)
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()
