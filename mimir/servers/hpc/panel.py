"""What the scientific-computing drawer shows about the cluster.

Deliberately not the host MIMIR runs on — that is usually a login node, the one machine
nothing will be measured on. What is worth a line is where the work goes: the queue the
user waits in, the room on the partitions they may submit to, and the hardware of the
nodes their code will be built for. Every line is served from a cache: the panel
refreshes on a timer.
"""

import concurrent.futures
import contextlib
import hashlib
import json
import os
import re
import threading
import time
from typing import Any, Callable

from ...shared.responses import ok
from ...shared.slurm_site import hosts as slurm_hosts, name_of as host_name
from .node_profiles import hardware_of, profile_host_dir, read_node_profile
from .slurm_digest import (
    cached_cluster_digest as cluster_digest,
    cluster_load,
    forget_cluster_digest,
    usable_set,
)
from .slurm_nodes import HARDWARE_FIELDS, hardware_key, sum_room

# The panel refreshes on a timer while a turn runs, so every line below is served from
# a cache. Two clocks, because the facts age at different speeds: the user's queue moves
# between two refreshes; what is free on a cluster is a node listing and a squeue away.
# The hardware itself is the digest's, kept for the life of the process.
_QUEUE_TTL_SECS = 15
_LOAD_TTL_SECS = 30
_CACHE: dict = {}


def _cached(key: str, ttl: int, produce):
    """*produce()* at most once every *ttl* seconds, per server process.

    A failure serves the last good answer rather than a blank: a transient sinfo is a
    reason to show what was true a minute ago, not to empty the section.
    """
    entry = _CACHE.get(key)
    now = time.time()
    if entry and now - entry["at"] <= ttl:
        return entry["value"]
    try:
        value = produce()
    except Exception:
        return entry["value"] if entry else None
    _CACHE[key] = {"at": now, "value": value}
    return value


def _queue_line(read_queue: Callable[[], dict]) -> dict | None:
    """The user's own jobs, counted by state. None when the queue cannot be read."""
    payload = _cached("queue", _QUEUE_TTL_SECS, read_queue)
    if not payload or payload.get("status") != "ok":
        return None
    counts: dict[str, int] = {}
    for job in payload.get("jobs") or []:
        counts[str(job.get("state") or "?")] = counts.get(str(job.get("state") or "?"), 0) + 1
    if not counts:
        return {"label": "your jobs", "value": "none queued"}
    order = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    return {
        "label": "your jobs",
        "value": ", ".join(f"{n} {state.lower()}" for state, n in order),
        "state": "running" if counts.get("RUNNING") else "",
    }


def _node_kind_label(kind: dict) -> str:
    """A node kind as a person would name it: what you would say to pick one.

    The GPUs first when there are any, because that is what a kind is chosen for; then
    the core count and the memory, which is what a job is sized against. Slurm's own
    hardware signature is what grouped them, and it is unreadable.
    """
    parts = []
    gpu = _gpu_of(kind)
    if gpu:
        parts.append(gpu)
    if kind.get("cpus"):
        parts.append(f"{kind['cpus']} cpus")
    mem_mb = kind.get("mem_mb")
    if isinstance(mem_mb, (int, float)) and mem_mb:
        parts.append(f"{round(mem_mb / 1024)} GB")
    return " · ".join(parts) or str(kind.get("arch") or "node")


def _gpu_of(kind: dict) -> str:
    """``gpu:a100:4`` as ``4× a100``; empty when the kind has no GPU."""
    gres = str(kind.get("gres") or "")
    if not gres.startswith("gpu:"):
        return ""
    bits = gres.split(":")
    model = bits[1] if len(bits) > 2 else ""
    count = bits[-1] if bits[-1].isdigit() else ""
    return " ".join(x for x in (f"{count}×" if count else "", model) if x).strip() or gres


def _node_names_label(kind: dict) -> str:
    """The kind's nodes by name, ``+N`` for those beyond the sample."""
    names = list(kind.get("example_nodes") or [])
    more = (kind.get("nodes_total") or len(names)) - len(names)
    return ", ".join(names) + (f" +{more}" if more > 0 else "")


def _kind_spec_lines(kind: dict, gpu_mem: str = "") -> list[dict]:
    """What Slurm says a node of this kind is, one fact per line.

    The GPUs and the sizes are already in the row's label; these are the rest.
    *gpu_mem* comes from a probe: Slurm's gres names GPUs but never their memory.
    """
    lines = []
    cpu = [kind.get("arch") or "arch unknown"]
    topo = [kind.get(f) for f in ("sockets", "cores_per_socket", "threads_per_core")]
    if kind.get("cpus") and all(topo):
        cpu.append(f"{kind['cpus']} = {topo[0]} sockets × {topo[1]} cores × {topo[2]} threads")
    lines.append({"label": "cpus", "value": " · ".join(cpu)})
    if kind.get("gres") and not _gpu_of(kind):
        lines.append({"label": "gres", "value": str(kind["gres"])})
    if _gpu_of(kind):
        lines.append({"label": "gpu mem",
                      "value": gpu_mem or "unknown until a node is profiled"})
    if kind.get("features"):
        lines.append({"label": "features", "value": kind["features"]})
    return lines


def _count(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _free_label(room: dict) -> str:
    """What *room* could give a job now, the GPUs first when it has any: ``2 GPUs · 96 cpus free``."""
    if not room["nodes"]:
        return "out of service"
    parts = []
    if room["gpus"]:
        parts.append("? GPUs" if room["gpus_free"] is None else _count(room["gpus_free"], "GPU"))
    parts.append("? cpus" if room["cpus_free"] is None else _count(room["cpus_free"], "cpu"))
    if not room["gpus"] and room["cpus_free"] == 0:
        return "full"
    if room["gpus"] and room["gpus_free"] == 0 and room["cpus_free"] == 0:
        return "full"
    return " · ".join(parts) + " free"


def _waiting_label(queue: dict | None) -> str:
    """The jobs in line for that room, ``12 waiting (480 cpus, 8 GPUs)``; "" when none."""
    if not queue or not queue["waiting"]:
        return ""
    asks = [_count(queue["waiting_cpus"], "cpu")]
    if queue["waiting_gpus"]:
        asks.append(_count(queue["waiting_gpus"], "GPU"))
    return f"{queue['waiting']} waiting ({', '.join(asks)})"


def _partition_room_lines(digest: dict, load: dict, limit: int = 8) -> list[dict]:
    """One line per usable partition: what is free on it now, and what already waits for it.

    The partitions with the most room first, because that is the question being asked.
    """
    queue = load.get("queue") or {}
    rows = []
    for name in dict.fromkeys(str(p.get("partition") or "") for p in digest["partitions"]):
        members = [n for n in load["nodes"] if name in n["partitions"]]
        if not members:
            continue
        room = sum_room(members)
        parts = [f"{room['idle']}/{room['nodes']} nodes idle", _free_label(room)]
        waiting = _waiting_label(queue.get(name))
        if waiting:
            parts.append(waiting)
        rows.append(((-(room["cpus_free"] or 0), name),
                     {"label": name, "value": " · ".join(parts)}))
    rows.sort(key=lambda row: row[0])
    lines = [line for _, line in rows[:limit]]
    if len(rows) > limit:
        lines.append({"label": "more", "value": f"{len(rows) - limit} other partitions"})
    return lines


def _left_out_line(digest: dict, out_of_service: int) -> dict | None:
    """What the tree does not show, and why — so a missing node reads as left out, not unseen."""
    parts = []
    hidden = digest.get("left_out") or {}
    if hidden.get("nodes"):
        named = ", ".join(f"{name} ({why})"
                          for name, why in sorted((hidden.get("partitions") or {}).items()))
        parts.append(_count(hidden["nodes"], "node") + " outside your partitions"
                     + (f": {named}" if named else ""))
    elif hidden.get("partitions"):
        parts.append("partitions " + ", ".join(
            f"{name} ({why})" for name, why in sorted(hidden["partitions"].items())))
    if out_of_service:
        parts.append(_count(out_of_service, "node") + " down, drained or reserved")
    return {"label": "not shown", "value": "; ".join(parts)} if parts else None


def _gpu_mem_label(raw: str) -> str:
    """nvidia-smi's ``81920 MiB`` as ``80 GB per GPU``; anything else as it came."""
    m = re.fullmatch(r"\s*(\d+)\s*MiB\s*", raw)
    return f"{round(int(m.group(1)) / 1024)} GB per GPU" if m else raw


def _kind_id(host: str, kind: dict) -> str:
    """A stable id for one node kind of one scheduler, from its hardware (the panel keys on it)."""
    payload = json.dumps([kind.get(f) for f in HARDWARE_FIELDS], sort_keys=True,
                         default=str)
    digest = hashlib.sha1(payload.encode()).hexdigest()[:8]
    return f"host:{host_name(host)}/kind:{digest}"


def _profiled_kind_lines(kind: dict, host: str) -> tuple[list[dict], str]:
    """What a profiled node of this kind is (CPU, ISA, ``-march=native``), from the profile cache."""
    for node in kind.get("example_nodes") or []:
        record = read_node_profile(node, host)
        if not record or not record.get("profile"):
            continue
        profile = record["profile"]
        cpu = profile.get("cpu") or {}
        lines = []
        if cpu.get("model"):
            lines.append({"label": "cpu", "value": cpu["model"]})
        simd = sorted(k for k, v in (cpu.get("simd") or {}).items() if v)
        if simd:
            lines.append({"label": "vector isa", "value": ", ".join(simd[-3:])})
        march = (profile.get("march") or {}).get("native")
        if march:
            lines.append({"label": "-march=native", "value": march})
        node_os = profile.get("os") or {}
        if node_os.get("id"):
            value = f"{node_os['id']} {node_os.get('version', '')}".strip()
            if node_os.get("glibc"):
                value += f" · glibc {node_os['glibc']}"
            lines.append({"label": "os", "value": value})
        devices = (profile.get("gpu") or {}).get("devices") or []
        if devices and devices[0].get("memory"):
            lines.append({"label": "gpu mem",
                          "value": _gpu_mem_label(str(devices[0]["memory"]))})
        if devices and devices[0].get("compute_cap"):
            lines.append({"label": "compute capability",
                          "value": str(devices[0]["compute_cap"])})
        detail = f"Profiled on {record.get('node', node)}"
        if record.get("probed_at"):
            detail += f", {record['probed_at']}"
        if record.get("partial"):
            detail += ". " + str(record.get("partial_reason") or "")
        return lines, detail.strip()
    return [], ""


def _read_host(host: str) -> tuple[dict, dict | None]:
    """*host*'s digest (memoised) and its load (cached for ``_LOAD_TTL_SECS``)."""
    digest = cluster_digest(host)
    if not digest["source"]:
        return digest, None
    load = _cached(f"load:{host}", _LOAD_TTL_SECS,
                   lambda: cluster_load(host, usable=usable_set(digest)))
    return digest, load


def _host_tree() -> list[dict]:
    """Every reachable scheduler, the room on it, and its node kinds.

    A host that did not answer is listed as such, never as empty. A kind whose nodes
    are all out of service is left out and counted; one with room comes first.
    """
    hosts = slurm_hosts()
    # Every host at once. Each miss is an ssh away and waits on a scheduler rather than
    # on this process, so one after another the drawer cost the sum of the clusters it
    # reaches; together it costs the slowest of them.
    reads: dict[str, tuple[dict, dict | None]] = {}
    if len(hosts) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            futures = {h: pool.submit(_read_host, h) for h in hosts}
            for h, done in futures.items():
                with contextlib.suppress(Exception):
                    reads[h] = done.result()
    groups = []
    for host in hosts:
        name = host_name(host)
        digest, load = reads.get(host) or _read_host(host)
        reached = "Slurm answers on this host" if not host else f"Reached over ssh to {host}"
        group: dict[str, Any] = {"id": f"host:{name}", "label": name, "host": name,
                                 "local": not host, "lines": []}
        if not digest["source"]:
            group["state"] = "warn"
            group["value"] = "did not answer"
            group["lines"].append(
                {"label": "hardware",
                 "value": "unknown, not absent", "state": "warn"})
            group["detail"] = reached + "."
            groups.append(group)
            continue

        members: dict[tuple, list[dict]] = {}
        for node in (load or {}).get("nodes") or []:
            members.setdefault(hardware_key(node), []).append(node)
        out_of_service = 0
        kinds = []
        for kind in digest["node_types"]:
            room = sum_room(members.get(hardware_key(kind), [])) if load else None
            if room is not None and not room["nodes"]:
                out_of_service += room["out"]
                continue
            if room is not None:
                out_of_service += room["out"]
            kinds.append((kind, room))
        kinds.sort(key=lambda kr: (-(kr[1] or {}).get("idle", 0),
                                   -((kr[1] or {}).get("cpus_free") or 0),
                                   -(kr[0].get("cpus") or 0)))

        if load:
            total = sum_room(load["nodes"])
            waiting = sum(q["waiting"] for q in (load.get("queue") or {}).values())
            group["value"] = _free_label(total) + (f" · {waiting} waiting" if waiting else "")
            group["lines"].extend(_partition_room_lines(digest, load))
            detail = (f"{reached}. Free: not allocated to any job, read at "
                      f"{load['read_at'][11:16]} UTC. Waiting: jobs of every user pending "
                      "on Resources or Priority, as squeue shows them.")
            if load.get("queue") is None:
                detail += " squeue did not answer, so what waits is unknown."
            elif digest.get("private_jobs"):
                group["lines"].append({
                    "label": "queue", "state": "warn",
                    "value": "only your own jobs are visible (PrivateData=jobs): "
                             "other users' waiting jobs are not counted"})
        else:
            group["value"] = hardware_of(host).get("hardware", "")
            group["lines"].append({"label": "room", "value": "unknown: the node listing "
                                   "did not answer", "state": "warn"})
            detail = reached + "."
        left_out = _left_out_line(digest, out_of_service)
        if left_out:
            group["lines"].append(left_out)
        if digest["degraded"]:
            group["lines"].append({"label": "reading", "value": digest["degraded"],
                                   "state": "warn"})
        group["detail"] = detail

        partitions = [str(p.get("partition") or "") for p in digest["partitions"]]
        labels = [_node_kind_label(kind) for kind, _ in kinds]
        entries = []
        for (kind, room), label in zip(kinds, labels):
            if labels.count(label) > 1 and kind.get("features"):
                label += f" · {kind['features']}"
            entries.append(_kind_entry(host, name, kind, room, label, partitions))
        if entries:
            group["groups"] = entries
        groups.append(group)
    return groups


def _kind_entry(host: str, name: str, kind: dict, room: dict | None, label: str,
                partitions: list[str]) -> dict:
    """One kind of machine: its room on the row, then where it is and what it is."""
    profiled, detail = _profiled_kind_lines(kind, host)
    # GPU memory sits with the spec lines, not among the probe's lines.
    gpu_mem = next((line["value"] for line in profiled if line["label"] == "gpu mem"), "")
    profiled = [line for line in profiled if line["label"] != "gpu mem"]
    lines = []
    if room is not None:
        lines.append({"label": "idle", "value": f"{room['idle']} of {_count(room['nodes'], 'node')}"
                      + (f" ({room['out']} out of service)" if room["out"] else "")})
    if kind.get("partitions"):
        lines.append({"label": "partitions", "value": ", ".join(kind["partitions"])})
    lines.append({"label": "nodes", "value": _node_names_label(kind)})
    entry: dict[str, Any] = {
        "id": _kind_id(host, kind),
        "label": label,
        "lines": lines + _kind_spec_lines(kind, gpu_mem) + profiled,
        "nodes": list(kind.get("example_nodes") or []),
    }
    if room is not None:
        entry["value"] = _free_label(room)
    if detail:
        entry["detail"] = detail
    if not profiled:
        # No "not profiled" badge (the ordinary case): the row offers the request that would
        # profile it, worded here by the end that knows the scheduler and partition.
        part = next((p for p in partitions if p in (kind.get("partitions") or [])),
                    partitions[0] if partitions else "")
        entry["action"] = {
            "kind": "probe",
            "label": "Profile a node",
            "prompt": (
                f"Probe one node of the '{_node_kind_label(kind)}' kind on "
                f"scheduler '{name}', partition '{part}', so its CPU model, "
                f"vector ISA and -march are known before anything is built "
                f"for it."),
        }
    return entry


def _host_tree_cached() -> list[dict]:
    """The host tree, rebuilt when the registry or a profile directory changes."""
    hosts = slurm_hosts()
    stamps = []
    for host in hosts:
        try:
            stamps.append(os.stat(profile_host_dir(host)).st_mtime_ns)
        except OSError:
            stamps.append(0)
    key = (tuple(hosts), tuple(stamps))
    entry = _CACHE.get("hosts")
    now = time.time()
    if entry and entry.get("key") == key:
        if now - entry["at"] <= _LOAD_TTL_SECS:
            return entry["value"]
    try:
        value = _host_tree()
    except Exception:
        return entry["value"] if entry else []
    _CACHE["hosts"] = {"at": now, "key": key, "value": value}
    return value


_LOCK = threading.Lock()


def report(refresh: bool, read_queue: Callable[[], dict]) -> dict:
    """The Cluster section: *read_queue* is the server's own read of the user's jobs.

    *refresh* drops every cache first, the per-host digest included.
    """
    with _LOCK:
        if refresh:
            _CACHE.clear()
            forget_cluster_digest()
        return _report(read_queue)


def forget() -> None:
    """Drop what the panel holds, so the next report reads everything again."""
    with _LOCK:
        _CACHE.clear()


def _report(read_queue: Callable[[], dict]) -> dict:
    lines: list[dict] = []
    queue = _queue_line(read_queue)
    if queue:
        lines.append(queue)
    groups = _host_tree_cached()
    if not lines and not groups:
        # No queue, no partitions and no registered scheduler: there is nothing here to
        # report on. Said with nothing rather than with a row saying "unavailable",
        # which is a section the user has to read to learn there is nothing to read.
        return ok({"title": "Cluster", "lines": []})
    return ok({"title": "Cluster", "lines": lines, "groups": groups,
               "detail": "Where submitted work runs — not the host MIMIR runs on."})
