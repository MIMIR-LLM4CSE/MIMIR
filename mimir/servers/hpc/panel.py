"""What the scientific-computing drawer shows about the cluster.

Deliberately not the host MIMIR runs on — that is usually a login node, the one machine
nothing will be measured on. What is worth a line is where the work goes: the queue the
user waits in, the partitions with room, and the hardware of the nodes their code will
be built for. Every line is served from a cache: the panel refreshes on a timer.
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
    forget_cluster_digest,
)
from .slurm_nodes import HARDWARE_FIELDS, as_int

# The panel refreshes on a timer while a turn runs, so every line below is served from
# a cache. Two clocks, because the facts age at different speeds: a queue moves between
# two refreshes, a partition's shape and a node's hardware do not.
_QUEUE_TTL_SECS = 15
_CLUSTER_TTL_SECS = 300
# A tree holding a host that did not answer is asked again this soon instead.
_RETRY_SECS = 30
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


def _partition_lines(read_partitions: Callable[[], dict], limit: int = 4) -> list[dict]:
    """Idle nodes per partition — whether a job submitted now starts now.

    sinfo emits one row per (partition, state), so the rows are folded back into one
    entry per partition here. The busiest are not the interesting ones: the partitions
    with room come first, because that is the question being asked.
    """
    payload = _cached("partitions", _CLUSTER_TTL_SECS, read_partitions)
    if not payload or payload.get("status") != "ok":
        return []
    totals: dict[str, list[int]] = {}
    for row in payload.get("partitions") or []:
        name = str(row.get("partition") or "").rstrip("*")
        nodes = as_int(row.get("nodes")) or 0
        if not name:
            continue
        seen = totals.setdefault(name, [0, 0])
        seen[0] += nodes
        if str(row.get("state") or "").startswith("idle"):
            seen[1] += nodes
    ranked = sorted(totals.items(), key=lambda kv: (-kv[1][1], kv[0]))
    # No state on these: the value already says how much room there is, and the one
    # colour the panel has means "something of yours is running".
    return [{"label": name, "value": f"{idle}/{total} nodes idle"}
            for name, (total, idle) in ranked[:limit]]


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
    if not names:
        return _node_kind_label(kind)
    more = (kind.get("nodes_total") or len(names)) - len(names)
    return ", ".join(names) + (f" +{more}" if more > 0 else "")


def _kind_spec_lines(kind: dict, gpu_mem: str = "") -> list[dict]:
    """What Slurm says a node of this kind is, one fact per line.

    *gpu_mem* comes from a probe: Slurm's gres names GPUs but never their memory.
    """
    lines = [{"label": "arch", "value": kind.get("arch") or "?"}]
    if kind.get("cpus"):
        value = str(kind["cpus"])
        topo = [kind.get(f) for f in ("sockets", "cores_per_socket", "threads_per_core")]
        if all(topo):
            value += f" ({topo[0]} sockets × {topo[1]} cores × {topo[2]} threads)"
        lines.append({"label": "cpus", "value": value})
    gpu = _gpu_of(kind)
    if gpu:
        lines.append({"label": "gpu", "value": gpu})
    elif kind.get("gres"):
        lines.append({"label": "gres", "value": str(kind["gres"])})
    mem_mb = kind.get("mem_mb")
    if isinstance(mem_mb, (int, float)) and mem_mb:
        lines.append({"label": "cpu mem", "value": f"{round(mem_mb / 1024)} GB"})
    if gpu:
        lines.append({"label": "gpu mem",
                      "value": gpu_mem or "unknown until a node is profiled"})
    if kind.get("partitions"):
        lines.append({"label": "partitions", "value": ", ".join(kind["partitions"])})
    if kind.get("features"):
        lines.append({"label": "features", "value": kind["features"]})
    if kind.get("nodes_total"):
        lines.append({"label": "nodes", "value": str(kind["nodes_total"])})
    return lines


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


def _host_tree() -> list[dict]:
    """Every reachable scheduler and its node kinds, from the memoised digest.

    A host that did not answer is listed as such, never as empty.
    """
    hosts = slurm_hosts()
    # Fill the memo for every host at once. Each miss is an ssh away and waits on a
    # scheduler rather than on this process, so one after another the drawer cost the
    # sum of the clusters it reaches; together it costs the slowest of them. Reads
    # below still go through cluster_digest, which now finds them already there.
    if len(hosts) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(hosts)) as pool:
            for done in [pool.submit(cluster_digest, h) for h in hosts]:
                with contextlib.suppress(Exception):
                    done.result()
    groups = []
    for host in hosts:
        name = host_name(host)
        digest = cluster_digest(host)
        group: dict[str, Any] = {"id": f"host:{name}", "label": name, "host": name,
                 "local": not host,
                 "lines": [{"label": "reached",
                            "value": "Slurm answers on this host" if not host
                                     else f"over ssh to {host}"}]}
        if not digest["source"]:
            group["state"] = "warn"
            group["value"] = "did not answer"
            group["lines"].append(
                {"label": "hardware",
                 "value": "unknown, not absent", "state": "warn"})
            groups.append(group)
            continue
        hardware = hardware_of(host)
        group["value"] = hardware.get("hardware", "")
        partitions = [str(p.get("partition") or "") for p in digest["partitions"]]
        if partitions:
            group["lines"].append({"label": "partitions",
                                   "value": ", ".join(partitions[:6])})
        if digest["degraded"]:
            group["lines"].append({"label": "reading", "value": digest["degraded"],
                                   "state": "warn"})
        kinds = []
        for kind in digest["node_types"]:
            profiled, detail = _profiled_kind_lines(kind, host)
            # GPU memory sits with the other memory line, not among the probe's lines.
            gpu_mem = next((line["value"] for line in profiled if line["label"] == "gpu mem"), "")
            profiled = [line for line in profiled if line["label"] != "gpu mem"]
            entry: dict[str, Any] = {
                "id": _kind_id(host, kind),
                "label": _node_names_label(kind),
                "lines": _kind_spec_lines(kind, gpu_mem) + profiled,
                "nodes": list(kind.get("example_nodes") or []),
            }
            if detail:
                entry["detail"] = detail
            if not profiled:
                # No "not profiled" badge (the ordinary case): the row offers the request that would
                # profile it, worded here by the end that knows the scheduler and partition.
                part = partitions[0] if partitions else ""
                for candidate in partitions:
                    if candidate in (kind.get("partitions") or []):
                        part = candidate
                        break
                entry["action"] = {
                    "kind": "probe",
                    "label": "Profile a node",
                    "prompt": (
                        f"Probe one node of the '{_node_kind_label(kind)}' kind on "
                        f"scheduler '{name}', partition '{part}', so its CPU model, "
                        f"vector ISA and -march are known before anything is built "
                        f"for it."),
                }
            kinds.append(entry)
        if kinds:
            group["groups"] = kinds
        groups.append(group)
    return groups


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
        unanswered = any(g.get("state") == "warn" for g in entry["value"])
        ttl = _RETRY_SECS if unanswered else _CLUSTER_TTL_SECS
        if now - entry["at"] <= ttl:
            return entry["value"]
    try:
        value = _host_tree()
    except Exception:
        return entry["value"] if entry else []
    _CACHE["hosts"] = {"at": now, "key": key, "value": value}
    return value


_LOCK = threading.Lock()


def report(refresh: bool, read_queue: Callable[[], dict],
           read_partitions: Callable[[], dict]) -> dict:
    """The Cluster section: *read_queue* and *read_partitions* are the server's own reads.

    *refresh* drops every cache first, the per-host digest included.
    """
    with _LOCK:
        if refresh:
            _CACHE.clear()
            forget_cluster_digest()
        return _report(read_queue, read_partitions)


def forget() -> None:
    """Drop what the panel holds, so the next report reads everything again."""
    with _LOCK:
        _CACHE.clear()


def _report(read_queue: Callable[[], dict], read_partitions: Callable[[], dict]) -> dict:
    lines: list[dict] = []
    queue = _queue_line(read_queue)
    if queue:
        lines.append(queue)
    lines.extend(_partition_lines(read_partitions))
    groups = _host_tree_cached()
    if not lines and not groups:
        # No queue, no partitions and no registered scheduler: there is nothing here to
        # report on. Said with nothing rather than with a row saying "unavailable",
        # which is a section the user has to read to learn there is nothing to read.
        return ok({"title": "Cluster", "lines": []})
    return ok({"title": "Cluster", "lines": lines, "groups": groups,
               "detail": "Where submitted work runs — not the host MIMIR runs on."})
