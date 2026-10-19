"""What is known about a cluster's compute nodes: one profile per node kind, probed on
the node by a short job running the platform profile, filed per scheduler host."""
from __future__ import annotations

import json
import os
import re
import shlex
import socket
import sys
import time
from ...shared.atomic import write_json
from ...shared.jobs import slurm_descriptor
from ...shared.responses import err, ok
from ...shared.launcher import launch_argv
from ...shared.state_paths import state_dir
from ...shared.slurm_site import (
    LOCAL as _LOCAL_HOST,
    LOCAL_NAME as _LOCAL_NAME,
    name_of as _host_name,
    jobs_dir as _hpc_jobs_dir,
)
from .slurm_digest import (
    cached_cluster_digest as _cluster_digest,
)
from .slurm_nodes import (
    HARDWARE_FIELDS as _HARDWARE_FIELDS,
    as_int as _as_int,
    hardware_key,
)
from ...shared.slurm_script import node_python_lines, sbatch_header
from ...shared import cpu_facts
from . import cli
from .cli import (
    TIMEOUT_READ,
)
from . import jobs

_PLATFORM_PROFILE = launch_argv("mimir.servers.hpc.server_platform", "--profile-json")


NODE_PROFILES_DIR = os.environ.get(
    "MIMIR_NODE_PROFILES_DIR", os.path.join(state_dir(), "hpc", "node_profiles"))


PROBE_MARKER = "probe.json"


def profile_host_dir(host: str) -> str:
    """Where a scheduler's node profiles are filed: one directory per host (node names are per cluster)."""
    name = _host_name(host) or _LOCAL_NAME
    return os.path.join(NODE_PROFILES_DIR, re.sub(r"[^A-Za-z0-9._-]", "_", name))


def _node_profile_path(node: str, host: str) -> str:
    return os.path.join(profile_host_dir(host), f"{node}.json")


def read_node_profile(node: str, host: str) -> dict | None:
    return _read_json(_node_profile_path(node, host))


# Fallback when no MIMIR Python runs on the node: raw sections, parsed server-side
# by the same cpu_facts parsers. Everything that needs no Python is still read.
_FALLBACK_SECTIONS = (
    ("uname", "uname -m"),
    ("lscpu", "LC_ALL=C lscpu"),
    ("march", cpu_facts.MARCH_QUERY + " 2>/dev/null"),
    ("gpu", cpu_facts.GPU_QUERY + " 2>/dev/null"),
    ("os", "cat /etc/os-release 2>/dev/null"),
    ("ldd", "ldd --version 2>&1 | head -n 1"),
    ("nproc", "nproc --all"),
)


def probe_script(work_dir: str, header: list[str], *, make_dir: bool = False) -> str:
    """The probe as a script writing into *work_dir* (created first when *make_dir*, for remote nodes)."""
    q = shlex.quote
    fallback = [f'  echo "=={name}"; {cmd}' for name, cmd in _FALLBACK_SECTIONS]
    return "\n".join(header + [
        "",
        *([f"mkdir -p {q(work_dir)} || exit 1"] if make_dir else []),
        f"cd {q(work_dir)} || exit 1",
        'echo "${SLURMD_NODENAME:-$(hostname -s)}" > node',
        *node_python_lines(),
        f'if [ -n "$_MIMIR_PY" ] && "$_MIMIR_PY" {shlex.join(_PLATFORM_PROFILE)}'
        " > profile.json 2> profile.err; then",
        "  echo full > mode",
        "else",
        "  rm -f profile.json",
        "  {",
        *fallback,
        "  } > fallback.txt 2>&1",
        "  echo partial > mode",
        "fi",
        "",
    ])


def _split_sections(text: str, names) -> dict[str, str]:
    """Split output that labels its parts with ``==<name>`` lines.

    The probe script writes its fallback sections this way, and a remote harvest reads
    a probe's whole directory the same way — one round trip instead of one per file.
    Lines before the first marker, and markers nobody asked for, are dropped.
    """
    wanted = set(names)
    sections: dict[str, list[str]] = {}
    current = None
    for line in text.splitlines():
        if line.startswith("==") and line[2:].strip() in wanted:
            current = line[2:].strip()
            sections[current] = []
        elif current is not None:
            sections[current].append(line)
    return {k: "\n".join(v) for k, v in sections.items()}


# What a probe leaves behind, read whether its directory is on this filesystem or on
# the far side of an ssh. ``mode`` is written last by the script, so its presence is
# what says the probe finished — no scheduler has to have caught up.
_PROBE_FILES = ("mode", "node", "profile.json", "fallback.txt")


def _probe_results(work_dir: str, host: str) -> dict[str, str]:
    """The probe's own files, by name. Absent files are simply missing keys."""
    if not host:
        out = {}
        for name in _PROBE_FILES:
            try:
                with open(os.path.join(work_dir, name)) as fh:
                    out[name] = fh.read()
            except OSError:
                continue
        return out
    # One round trip for all of them: a file per call would be four ssh connections
    # per probe, per panel refresh. Missing files contribute an empty section rather
    # than an error, which is the same answer the local branch gives.
    q = shlex.quote
    script = "; ".join(
        [f"cd {q(work_dir)} || exit 0"] +
        [f'echo "=={name}"; cat {q(name)} 2>/dev/null' for name in _PROBE_FILES])
    res = cli.run_bash(script, TIMEOUT_READ, host=host)
    if res.get("status") != "ok":
        return {}
    found = _split_sections(res.get("stdout", ""), _PROBE_FILES)
    return {k: v for k, v in found.items() if v.strip()}


def _fallback_profile(text: str) -> dict:
    """A profile from the raw sections — what the node could say without Python."""
    sec = _split_sections(text, dict(_FALLBACK_SECTIONS))
    arch = sec.get("uname", "").strip()
    gpus = cpu_facts.parse_nvidia_csv(sec.get("gpu", ""))
    os_facts = cpu_facts.parse_os_release(sec.get("os", ""))
    glibc = cpu_facts.parse_ldd_version(sec.get("ldd", ""))
    if glibc:
        os_facts["glibc"] = glibc
    cpu = {"arch": arch, "logical_cpus": _as_int(sec.get("nproc", "").strip()),
           **cpu_facts.cpu_facts(arch, sec.get("lscpu", ""))}
    return {
        "os": os_facts,
        "cpu": cpu,
        "march": cpu_facts.parse_march(sec.get("march", "")),
        "gpu": {"available": bool(gpus), "count": len(gpus), "devices": gpus},
        "machine_signature": cpu_facts.machine_signature(arch, cpu, [g["name"] for g in gpus]),
    }


def slurm_facts(node: dict) -> dict:
    return {k: node.get(k) for k in _HARDWARE_FIELDS}


def _read_json(path: str) -> dict | None:
    try:
        with open(path) as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return None


def write_node_profile(node: str, record: dict, host: str = _LOCAL_HOST) -> None:
    write_json(os.path.join(profile_host_dir(host), f"{node}.json"), record,
               indent=2, ensure_ascii=False)


def harvest_probes(nodes_by_name: dict[str, dict],
                    host: str = _LOCAL_HOST) -> list[dict]:
    """Turn finished probes into node profiles (filed under their own host); return those pending.

    A probe is finished when it wrote ``mode``.
    """
    pending: list[dict] = []
    try:
        entries = sorted(os.listdir(_hpc_jobs_dir()))
    except OSError:
        return pending
    for entry in entries:
        job_dir = os.path.join(_hpc_jobs_dir(), entry)
        meta = _read_json(os.path.join(job_dir, PROBE_MARKER))
        if not meta or meta.get("harvested"):
            continue
        # The scheduler this probe was submitted to, which is not necessarily the one
        # being asked about now, and the directory its result is in — on this
        # filesystem for a local scheduler, on the far side for one reached over ssh.
        probe_host = meta.get("host", host)
        work_dir = meta.get("work_dir") or job_dir
        files = _probe_results(work_dir, probe_host)
        if "mode" not in files:
            state, raw, _nodes = jobs.normalized_job_state(str(meta.get("job_id", "")),
                                                       host=probe_host)
            if state in ("running", "pending"):
                pending.append({"job_id": meta.get("job_id"), "state": state,
                                "partition": meta.get("partition", ""),
                                "scheduler": _host_name(probe_host)})
                continue
            meta.update(harvested=True, outcome=f"no result ({raw or state})")
        else:
            mode = files["mode"].strip()
            node = files.get("node", "").strip()
            if not node:
                meta.update(harvested=True, outcome="no node name recorded")
                with open(os.path.join(job_dir, PROBE_MARKER), "w") as fh:
                    json.dump(meta, fh)
                continue
            if mode == "full":
                try:
                    profile = json.loads(files.get("profile.json") or "")
                except ValueError:
                    profile = {}
                partial = not profile
            else:
                profile = _fallback_profile(files.get("fallback.txt") or "")
                partial = True
            record = {
                "node": node, "probed_at": meta.get("submitted_at", ""),
                "job_id": meta.get("job_id"), "source": "probe job",
                "scheduler": _host_name(probe_host),
                "slurm": slurm_facts(nodes_by_name.get(node, {})),
                "partial": partial, "profile": profile,
            }
            if partial:
                record["partial_reason"] = (
                    "No MIMIR Python runs on this node (no .venv-<os>-<arch> for it), so "
                    "only what the shell can read was collected: CPU, -march, GPU, OS. "
                    "Toolchains, modules and Python environments are missing."
                    + (" This scheduler is reached over ssh, where MIMIR's own "
                       "collectors are not installed unless the checkout is shared."
                       if probe_host else ""))
            write_node_profile(node, record, probe_host)
            meta.update(harvested=True, outcome=f"profiled {node}")
        with open(os.path.join(job_dir, PROBE_MARKER), "w") as fh:
            json.dump(meta, fh)
    return pending


def local_profile_record(node: dict) -> dict | None:
    """This host's own profile, when this host is an allocated compute node."""
    res = cli.run_argv([sys.executable, *_PLATFORM_PROFILE], 90)
    try:
        profile = json.loads(res.get("stdout") or "")
    except ValueError:
        return None
    return {"node": node.get("node", ""), "probed_at": profile.get("timestamp", ""),
            "source": "this host (inside the allocation)", "slurm": slurm_facts(node),
            "partial": False, "profile": profile}


def host_identity() -> dict:
    """What the host MIMIR runs on is, in the terms a node profile is compared on."""
    lscpu = cli.run_argv(["env", "LC_ALL=C", "lscpu"], TIMEOUT_READ)
    import platform as _platform
    arch = _platform.machine()
    cpu = cpu_facts.cpu_facts(arch, lscpu.get("stdout", "") if lscpu.get("status") == "ok" else "")
    return {"arch": arch, "cpu": cpu, "os": cpu_facts.local_os()}


def compare_with_host(profile: dict, host: dict) -> dict:
    """Which of the facts that decide a build differ between this host and the node."""
    cpu = profile.get("cpu") or {}
    node_os, host_os = profile.get("os") or {}, host["os"]
    checks = {
        "arch": (cpu.get("arch"), host["arch"]),
        "cpu_model": (cpu.get("model"), host["cpu"].get("model")),
        "simd": (sorted(k for k, v in (cpu.get("simd") or {}).items() if v),
                 sorted(k for k, v in (host["cpu"].get("simd") or {}).items() if v)),
        "os": ((node_os.get("id"), node_os.get("version")), (host_os.get("id"), host_os.get("version"))),
        "glibc": (node_os.get("glibc"), host_os.get("glibc")),
    }
    differs = [k for k, (a, b) in checks.items() if a and b and a != b]
    return {"same": not differs, "differs": differs}


# Above this many node types, only the facts that decide a build are shown per type;
# a full profile each (toolchains, environments, modules) would drown the answer.
FULL_PROFILE_TYPES = 3


def summary(profile: dict) -> dict:
    cpu = profile.get("cpu") or {}
    return {
        "cpu_model": cpu.get("model", ""),
        "simd": sorted(k for k, v in (cpu.get("simd") or {}).items() if v),
        "caches": cpu.get("caches", {}),
        "march": profile.get("march", {}),
        "gpu": [{k: d.get(k) for k in ("name", "compute_cap") if d.get(k)}
                for d in (profile.get("gpu") or {}).get("devices", [])],
        "os": profile.get("os", {}),
    }


def hardware_of(host: str) -> dict:
    """What *host*'s machines are, for the listing a scheduler is chosen from.

    A summary and the stable node types, never occupancy. A host that said nothing
    carries neither and says so, because a cluster described as having no hardware
    reads as a cluster not worth submitting to.
    """
    digest = _cluster_digest(host)
    if not digest["source"]:
        return {"hardware": "",
                "hardware_note": "Did not answer; its hardware is unknown, not absent."}
    types = digest["node_types"]
    arches = sorted({t["arch"] for t in types if t["arch"]})
    # The model, not the whole gres: a site with ten GPU shapes makes a line nobody
    # scans, and the counts are on each node type below for whoever needs them.
    gpus = sorted({t["gres"].split(":")[1] for t in types
                   if t["gres"].startswith("gpu:") and t["gres"].count(":") >= 1})
    parts = [f"{len(types)} node type{'s' if len(types) != 1 else ''}"]
    if arches:
        parts.append("/".join(arches))
    if gpus:
        parts.append(", ".join(gpus))
    parts.append(f"{len(digest['partitions'])} partitions")
    summary = {"hardware": " · ".join(parts), "node_types": types}
    if digest["degraded"]:
        summary["degraded"] = digest["degraded"]
    return summary


def submit_probe(host: str, partition: str, constraint: str, nodelist: str,
                 account: str) -> dict:
    """Submit a few-second job that profiles one node with the platform collectors.

    The job's directory carries ``PROBE_MARKER``, which ``harvest_probes`` reads once
    the job has ended.
    """
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    job_dir = os.path.join(_hpc_jobs_dir(), f"{stamp}-probe")
    suffix = 1
    while os.path.exists(job_dir):
        suffix += 1
        job_dir = os.path.join(_hpc_jobs_dir(), f"{stamp}-probe{suffix}")
    os.makedirs(job_dir)
    # Where the job writes, on the node that runs it. A scheduler on this machine
    # writes into the submission's own directory; one reached over ssh has no access
    # to this filesystem, so it writes beside its own state and the harvest reads it
    # back over the same connection. The expansion is the script's, not ours — only
    # the far side knows what $HOME is there.
    if host:
        work_dir = ('"${MIMIR_STATE_HOME:-$HOME/.mimir}"/node_probes/'
                    + os.path.basename(job_dir))
    else:
        work_dir = job_dir
    log_file = os.path.join(work_dir, "slurm.log")
    header = sbatch_header(
        job_name="mimir-probe", partition=partition, cpus_per_task=1,
        wall_time="00:05:00", log_file=log_file, account=account, nodes=1, ntasks=1,
        constraint=constraint, nodelist=nodelist,
    )
    script = probe_script(work_dir, header, make_dir=bool(host))
    # Written locally whatever the scheduler, for a reader: the submission's directory
    # is where someone goes to see what was actually asked for.
    script_path = os.path.join(job_dir, "probe.sh")
    try:
        with open(script_path, "w") as fh:
            fh.write(script)
    except OSError as exc:
        return err(f"Could not write the probe script: {exc}")

    job_id, refusal = jobs.sbatch_on(host, script, "Check partition, account and constraint.")
    if refusal:
        return refusal
    # The host is recorded twice on purpose: in the marker the harvest reads, and in
    # the descriptor's own host file, which is what every other cluster tool resolves a
    # job's scheduler from.
    jobs.record_job_host(job_dir, host)
    with open(os.path.join(job_dir, PROBE_MARKER), "w") as fh:
        json.dump({"job_id": job_id, "partition": partition, "constraint": constraint,
                   "nodelist": nodelist, "submitted_at": stamp,
                   "host": host, "work_dir": work_dir}, fh)
    return ok({
        "job_id": job_id,
        "job_dir": job_dir,
        "partition": partition,
        "note": f"Probe job {job_id} submitted to '{partition}'. Its result is read "
                "with the node-profile tool once the job ends.",
        "background_job": slurm_descriptor(job_id),
    })


def refresh_own_profile(by_name: dict[str, dict], host: str) -> None:
    """Profile the node this runs on when it is an allocated node and its profile is stale.

    Free here: inside an allocation the collectors run on the node itself, no job needed.
    """
    here = by_name.get(socket.gethostname().split(".")[0])
    if here:
        cached = read_node_profile(here["node"], host)
        if not cached or cached.get("slurm") != slurm_facts(here):
            record = local_profile_record(here)
            if record:
                write_node_profile(here["node"], record, host)


def describe_node_types(nodes: list[dict], types: list[dict], sched_host: str,
                        host: dict, full: bool) -> list[dict]:
    """Each kind in *types* (the kinds *nodes* fall into), with what a fresh profile of
    one of its nodes says it is.

    A profile is fresh while Slurm still describes its node the same way. *full* gives
    each kind its whole profile rather than a summary.
    """
    members: dict[tuple, list[dict]] = {}
    for n in nodes:
        members.setdefault(hardware_key(n), []).append(n)
    out = []
    for t in types:
        group = members.get(hardware_key(t), [])
        fresh: list[dict] = []
        stale: list[dict] = []
        for n in group:
            rec = read_node_profile(n["node"], sched_host)
            if not rec:
                continue
            (fresh if rec.get("slurm") == slurm_facts(n) else stale).append(rec)
        entry = {k: t.get(k) for k in ("arch", "cpus", "mem_gb", "gres", "sockets",
                                       "cores_per_socket", "threads_per_core", "features",
                                       "partitions", "nodes_total", "by_state", "example_nodes")}
        if not fresh:
            entry["profiled"] = False
            entry["note"] = ("No node of this kind profiled"
                             + (" since Slurm's description of it changed" if stale else "")
                             + "; probe one to know its CPU, ISA and -march.")
        else:
            rec = fresh[0]
            profile = rec.get("profile") or {}
            entry.update({
                "profiled": True,
                "profiled_on": rec.get("node"),
                "profiled_at": rec.get("probed_at"),
                "source": rec.get("source"),
                "matches_this_host": compare_with_host(profile, host),
                "profile": profile if full else summary(profile),
            })
            if rec.get("partial"):
                entry["partial"] = rec.get("partial_reason", True)
            models = {(r.get("profile") or {}).get("cpu", {}).get("model") for r in fresh}
            models.discard(None)
            if len(models) > 1:
                entry["cpu_models_differ"] = sorted(models)
            if len(fresh) < len(group):
                entry["unprofiled_nodes"] = len(group) - len(fresh)
                entry["caveat"] = ("Slurm's signature does not include the CPU model: "
                                   "the unprofiled nodes of this kind are assumed, not "
                                   "known, to be the same machine.")
        out.append(entry)
    return out
