"""Parsing and aggregation of Slurm's node inventory (``scontrol`` and ``sinfo``). Pure."""

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


def gpus_of(gres: str) -> list[tuple[str, int]]:
    """The GPUs a gres string names, as (model, count): ``gpu:a100:4,gpu:2`` ->
    ``[("a100", 4), ("", 2)]``; the socket bindings and other resources are dropped."""
    out = []
    for item in re.sub(r"\([^)]*\)", "", gres or "").split(","):
        bits = item.strip().split(":")
        if bits[0] != "gpu":
            continue
        count = int(bits[-1]) if len(bits) > 1 and bits[-1].isdigit() else 1
        out.append((bits[1] if len(bits) > 2 else "", count))
    return out


def gpu_count(gres: str) -> int:
    """The GPUs a gres string names: ``gpu:a100:4,gpu:v100:2`` -> 6, ``gpu:2`` -> 2."""
    return sum(n for _, n in gpus_of(gres))


def gpu_label(gres: str) -> str:
    """The GPUs a gres string names, as a label: ``4× a100``, ``4× a100 + 2× v100``; empty
    when it names none."""
    return " + ".join(f"{n}× {model}".strip() for model, n in gpus_of(gres))


def _allocated_gpus(line: str, raw: dict) -> int | None:
    """``gres/gpu=N`` of AllocTRES; 0 when the field is there and empty, None when absent."""
    if "AllocTRES=" not in line:
        return None
    m = re.search(r"gres/gpu=(\d+)", raw.get("AllocTRES", ""))
    return int(m.group(1)) if m else 0


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
            "cpus_free":        (cpu_tot - cpu_alloc) if cpu_tot is not None and cpu_alloc is not None else None,
            "cpu_load":         raw.get("CPULoad", ""),
            "sockets":          as_int(raw.get("Sockets", "")),
            "cores_per_socket": as_int(raw.get("CoresPerSocket", "")),
            "threads_per_core": as_int(raw.get("ThreadsPerCore", "")),
            "mem_mb":           as_int(raw.get("RealMemory", "")),
            "mem_free_mb":      as_int(raw.get("FreeMem", "")),
            "mem_alloc_mb":     as_int(raw.get("AllocMem", "")),
            "gres":             gres,
            "gpus":             gpu_count(gres),
            "gpus_allocated":   _allocated_gpus(line, raw),
            "features":         "" if features == "(null)" else features,
        })
    return nodes


# What makes two nodes "the same kind of machine" in Slurm's eyes. Deliberately blind
# to the CPU model, which Slurm does not report: two generations with the same core
# count and memory share a key, which is why a probed node profile is per node.
HARDWARE_FIELDS = ("arch", "cpus", "mem_mb", "gres", "sockets",
                   "cores_per_socket", "threads_per_core", "features")


def hardware_key(node: dict) -> tuple:
    return tuple(node.get(k) for k in HARDWARE_FIELDS)


def aggregate_node_types(nodes: list[dict]) -> list[dict]:
    """Collapse nodes onto their hardware signature.

    A 124-node cluster listed one row per node buries the answer in noise; what the
    caller is choosing between is the handful of *kinds* of machine, and how much of
    each is free right now.
    """
    groups: dict[tuple, dict] = {}
    for n in nodes:
        g = groups.setdefault(hardware_key(n), {
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
_STABLE_TYPE_FIELDS = HARDWARE_FIELDS + ("partitions",)


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


# ── sinfo, the degraded spelling ──────────────────────────────────────────────
#
# ``scontrol show node`` is restricted on some sites. ``sinfo`` is readable everywhere
# and carries most of the same fields, so a cluster that refuses the first is still
# described — with the gaps named rather than defaulted, since a wrong architecture is
# worse than an unknown one.

# Asked for as one argv so the format string and the column list below cannot drift
# apart: they are one fact in two halves, and separated they break in silence.
SCONTROL_NODES_ARGV = ["scontrol", "show", "node", "-o"]

SINFO_NODES_FORMAT = "%N|%P|%t|%c|%m|%e|%G|%X|%Y|%Z"

_SINFO_NODES_COLUMNS = [
    "node", "partition", "state", "cpus", "mem_mb", "mem_free_mb", "gres",
    "sockets", "cores_per_socket", "threads_per_core",
]


def parse_pipe_table(stdout: str, columns: list[str]) -> list[dict]:
    """Slurm's ``-o '%a|%b'`` output as dicts, one per line.

    A row whose cell count disagrees with *columns* is dropped rather than padded. A
    short line is a wrapped or truncated one, not a node with missing fields, and
    padding it would invent values a caller cannot tell from real ones.
    """
    rows = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) != len(columns):
            continue
        rows.append({columns[i]: parts[i] for i in range(len(columns))})
    return rows


def parse_sinfo_nodes(stdout: str) -> list[dict]:
    """``sinfo -N`` output in :func:`parse_scontrol_nodes`'s shape, unknown fields left empty.

    Rows are merged per node (one per partition otherwise).
    """
    nodes: list[dict] = []
    by_name: dict[str, dict] = {}
    for r in parse_pipe_table(stdout, _SINFO_NODES_COLUMNS):
        part = r["partition"].rstrip("*")
        if r["node"] in by_name:
            if part not in by_name[r["node"]]["partitions"]:
                by_name[r["node"]]["partitions"].append(part)
            continue
        by_name[r["node"]] = {
            "node": r["node"], "arch": "", "state": r["state"].upper(),
            "partitions": [part],
            "cpus": as_int(r["cpus"].rstrip("+")),
            "cpus_allocated": None, "cpus_free": None,
            "cpu_load": "", "sockets": as_int(r["sockets"]),
            "cores_per_socket": as_int(r["cores_per_socket"]),
            "threads_per_core": as_int(r["threads_per_core"]),
            "mem_mb": as_int(r["mem_mb"]), "mem_free_mb": as_int(r["mem_free_mb"]),
            "mem_alloc_mb": None,
            "gres": clean_gres(r["gres"]), "gpus": gpu_count(clean_gres(r["gres"])),
            "gpus_allocated": None, "features": "",
        }
        nodes.append(by_name[r["node"]])
    return nodes


# ── Who may use a partition ───────────────────────────────────────────────────
#
# ``sinfo`` without ``--all`` already leaves out hidden partitions and those closed to
# the user's groups; what it keeps can still be down, root-only or closed to the user's
# accounts, which only ``scontrol show partition`` says. A node in no partition left is
# not somewhere a job can go — a login or service node declared to Slurm, mostly.

SCONTROL_PARTITIONS_ARGV = ["scontrol", "show", "partition", "-o"]


def _names(value: str) -> set[str] | None:
    """``AllowAccounts=a,b`` as a set; None for ``ALL`` or an absent field (no restriction)."""
    if not value or value.upper() == "ALL":
        return None
    return {v for v in value.split(",") if v and v != "(null)"}


def parse_scontrol_partitions(stdout: str) -> dict[str, dict]:
    """``scontrol show partition -o`` as ``{name: restrictions}``."""
    out: dict[str, dict] = {}
    for line in stdout.splitlines():
        if "PartitionName=" not in line:
            continue
        raw = {k: v for k, v in re.findall(r"\b(\w+)=([^\s]+)", line)}
        out[raw["PartitionName"]] = {
            "allow_accounts": _names(raw.get("AllowAccounts", "")),
            "deny_accounts": _names(raw.get("DenyAccounts", "")) or set(),
            "root_only": raw.get("RootOnly", "NO").upper() == "YES",
        }
    return out


def parse_private_data(stdout: str) -> bool | None:
    """Whether ``scontrol show config``'s ``PrivateData`` hides other users' jobs; None if unsaid."""
    m = re.search(r"^\s*PrivateData\s*=\s*(\S*)", stdout or "", re.MULTILINE | re.IGNORECASE)
    if not m:
        return None
    return "jobs" in m.group(1).lower().split(",")


def usable_partitions(visible: list[dict], restrictions: dict[str, dict],
                      accounts: set[str] | None, user: str) -> tuple[list[str], dict[str, str]]:
    """``(usable names, {left-out name: why})`` from sinfo's rows and scontrol's restrictions.

    *accounts* None is "not known" (no sacctmgr): the account rules are then not applied,
    since a partition wrongly hidden is worse than one wrongly shown.
    """
    usable: list[str] = []
    left_out: dict[str, str] = {}
    for row in visible:
        name = str(row.get("partition") or "").rstrip("*")
        if not name or name in usable or name in left_out:
            continue
        rules = restrictions.get(name, {})
        availability = str(row.get("availability") or "up").lower()
        if availability != "up":
            left_out[name] = availability
        elif rules.get("root_only") and user != "root":
            left_out[name] = "root only"
        elif accounts is not None and (
                (rules.get("allow_accounts") is not None
                 and not accounts & rules["allow_accounts"])
                or (accounts and accounts <= rules.get("deny_accounts", set()))):
            left_out[name] = "not open to your accounts"
        else:
            usable.append(name)
    return usable, left_out


def usable_nodes(nodes: list[dict], usable: set[str] | None) -> list[dict]:
    """The nodes a job of this user can land on, each narrowed to its usable partitions.

    *usable* None is "not known": only the nodes in no partition at all are left out.
    """
    out = []
    for n in nodes:
        parts = [p for p in n["partitions"] if usable is None or p in usable]
        if parts:
            out.append({**n, "partitions": parts})
    return out


# ── What is free right now ────────────────────────────────────────────────────
#
# A node can be in a partition and still take nothing: down, drained or draining,
# reserved, in maintenance, not responding. Those are counted, never offered as room.

_OUT_OF_SERVICE = frozenset({
    "DOWN", "DRAIN", "DRAINED", "DRAINING", "DRNG", "FAIL", "FAILING", "FUTURE", "FUTR",
    "MAINT", "MAINTENANCE", "NOT_RESPONDING", "INVAL", "UNKNOWN", "UNK", "RESERVED",
    "RESV", "REBOOT_ISSUED", "REBOOT_REQUESTED", "POWERING_DOWN", "POWER_DOWN",
})


def out_of_service(state: str) -> bool:
    """Whether a node in *state* (scontrol's ``IDLE+DRAIN`` or sinfo's ``down*``) takes no job."""
    if "*" in (state or ""):            # sinfo's mark for a node that does not respond
        return True
    words = {w.strip("~#!%$@^-*").upper() for w in (state or "UNKNOWN").split("+")}
    return bool(words & _OUT_OF_SERVICE)


# Below this much unallocated memory a node's free cores cannot hold a job.
_MIN_FREE_MEM_MB = 1024


def node_room(node: dict) -> dict | None:
    """What *node* could give a job now (None when it is out of service).

    ``cpus_free`` and ``gpus_free`` are None where Slurm did not say (sinfo on a mixed
    node). Free cores on a node whose memory is all allocated are not room.
    """
    if out_of_service(node.get("state", "")):
        return None
    base = (node.get("state") or "").split("+")[0].strip("~#!%$@^-*").upper()
    cpus = node.get("cpus") or 0
    alloc = node.get("cpus_allocated")
    if alloc is not None:
        cpus_free = max(cpus - alloc, 0)
    else:
        cpus_free = cpus if base == "IDLE" else 0 if base in ("ALLOC", "ALLOCATED") else None
    mem, mem_alloc = node.get("mem_mb"), node.get("mem_alloc_mb")
    if mem and mem_alloc is not None and mem - mem_alloc < _MIN_FREE_MEM_MB:
        cpus_free = 0
    gpus = node.get("gpus") or 0
    gpus_alloc = node.get("gpus_allocated")
    gpus_free: int | None
    if not gpus:
        gpus_free = 0
    elif gpus_alloc is not None:
        gpus_free = max(gpus - gpus_alloc, 0)
    else:
        gpus_free = gpus if base == "IDLE" else None
    return {"idle": (alloc == 0) if alloc is not None else base == "IDLE",
            "cpus": cpus, "cpus_free": cpus_free, "gpus": gpus, "gpus_free": gpus_free}


def sum_room(nodes: list[dict]) -> dict:
    """The room of *nodes* together; a free count is None when any node's was unknown."""
    total: dict[str, int | None] = {"nodes": 0, "out": 0, "idle": 0, "cpus": 0, "cpus_free": 0,
                                    "gpus": 0, "gpus_free": 0}
    counted = out = idle = 0
    for n in nodes:
        room = node_room(n)
        if room is None:
            out += 1
            continue
        counted += 1
        idle += int(room["idle"])
        for key in ("cpus", "cpus_free", "gpus", "gpus_free"):
            have = total[key]
            total[key] = None if have is None or room[key] is None else have + room[key]
    total.update(nodes=counted, out=out, idle=idle)
    return total


# ── What the queue already wants ──────────────────────────────────────────────
#
# Free cores are only room for a job that would be next. squeue says what is waiting in
# each partition, all users together; the jobs pending on Resources or Priority are the
# ones in line for that room, those held on a dependency, a limit or a begin time not.

SQUEUE_LOAD_FORMAT = "%P|%T|%C|%D|%b|%r"
_SQUEUE_LOAD_COLUMNS = ["partition", "state", "cpus", "nodes", "tres", "reason"]
_IN_LINE_REASONS = frozenset({"Resources", "Priority", "None", ""})


def _gpus_requested(tres: str) -> int:
    """GPUs per node of ``%b`` (``gres/gpu:a100:2``, ``gres:gpu:2``, ``N/A``)."""
    if "gpu" not in (tres or ""):
        return 0
    found = re.findall(r"gpu(?::[A-Za-z][^:,=]*)?[:=](\d+)", tres)
    return sum(int(n) for n in found) if found else 1


def parse_squeue_load(stdout: str) -> dict[str, dict]:
    """``{partition: {running, waiting, waiting_cpus, waiting_gpus, held}}`` from squeue.

    A job pending on ``p1,p2`` is waiting in both: either may be where it lands.
    """
    out: dict[str, dict] = {}
    for r in parse_pipe_table(stdout, _SQUEUE_LOAD_COLUMNS):
        for part in r["partition"].split(","):
            q = out.setdefault(part.rstrip("*"), {"running": 0, "waiting": 0,
                                                 "waiting_cpus": 0, "waiting_gpus": 0,
                                                 "held": 0})
            if r["state"] in ("RUNNING", "COMPLETING"):
                q["running"] += 1
            elif r["state"] == "PENDING":
                if r["reason"] not in _IN_LINE_REASONS:
                    q["held"] += 1
                    continue
                q["waiting"] += 1
                q["waiting_cpus"] += as_int(r["cpus"]) or 0
                q["waiting_gpus"] += _gpus_requested(r["tres"]) * (as_int(r["nodes"]) or 1)
    return out
