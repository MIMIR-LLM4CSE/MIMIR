"""What a machine is, read the same way wherever the reading happens.

Shared because the same facts are read in three places: the platform server describing
the host it runs on, a Slurm probe job describing a compute node, and the proxy runner
recording where a timed run executed. If each parsed ``lscpu`` its own way, a node
profile and the host profile would disagree about the same machine, and two timings
taken on identical hardware would look incomparable.

Everything here is a pure parser over command output, except the few ``local_*``
helpers, which read the current machine directly and name themselves accordingly.
"""

import hashlib
import json
import os
import platform
import re
import shutil
import socket
import subprocess


# ── lscpu ─────────────────────────────────────────────────────────────────────

def parse_lscpu(text: str) -> dict:
    """``lscpu`` output as a dict keyed by the lower-cased field name."""
    info = {}
    for line in (text or "").splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        # Lower-case the key so lookups do not depend on the distro's capitalisation.
        info[k.strip().lower()] = v.strip()
    return info


# ISA extensions worth reporting, per architecture. There is no portable name for
# "the vector unit": asking an aarch64 host whether it has AVX-512 always answers no,
# which reads as "no SIMD" rather than "a different SIMD". lscpu prints these under
# "Flags:" on x86 and "Features:" on aarch64, so both keys are read.
ISA_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "x86_64":  ("avx2", "avx512f", "avx512bw", "avx512vl", "fma", "amx_tile"),
    "aarch64": ("asimd", "sve", "sve2", "bf16", "i8mm"),
    "ppc64le": ("vsx",),
}

# Cache levels as lscpu names them. Their size decides a blocking factor, and that
# is exactly the number that differs between a login node and a compute node.
_CACHE_KEYS = (("l1d", "l1d cache"), ("l1i", "l1i cache"),
               ("l2", "l2 cache"), ("l3", "l3 cache"))


def cpu_facts(arch: str, lscpu_text: str) -> dict:
    """The CPU facts read from one ``lscpu`` output, for a machine of *arch*."""
    ls = parse_lscpu(lscpu_text)
    flags = set((ls.get("flags") or ls.get("features") or "").split())
    known = ISA_EXTENSIONS.get(arch, ())
    data = {
        "model": ls.get("model name", ""),
        "sockets": ls.get("socket(s)", ""),
        "cores_per_socket": ls.get("core(s) per socket", ""),
        "threads_per_core": ls.get("thread(s) per core", ""),
        "numa_nodes": ls.get("numa node(s)", ""),
        "simd": {name: name in flags for name in known},
    }
    caches = {name: ls[key] for name, key in _CACHE_KEYS if ls.get(key)}
    if caches:
        data["caches"] = caches
    if not known:
        data["simd_note"] = f"No ISA extension list known for {arch}."
    return data


# ── GPU ───────────────────────────────────────────────────────────────────────

# compute_cap is queried alongside the descriptive fields because it is the one that
# decides a build: on a CUDA/Kokkos project the arch flag comes from it, and nothing
# else here substitutes. Reported without it, a host is described by its marketing
# name, and the gap gets filled by inference from that name — which is exactly where
# it breaks (B200 is sm_100, B300 is sm_103), silently, until the first kernel launch.
GPU_FIELDS_BASE = ("name", "memory", "driver")
GPU_FIELDS = GPU_FIELDS_BASE + ("compute_cap",)
GPU_QUERY_BASE = ("nvidia-smi --query-gpu=name,memory.total,driver_version"
                  " --format=csv,noheader")
GPU_QUERY = ("nvidia-smi --query-gpu=name,memory.total,driver_version,compute_cap"
             " --format=csv,noheader")

# Vendor CLI -> the tool that proves a GPU of that vendor is present. Only the NVIDIA
# output is parsed into devices; the others are detected and reported as such.
GPU_PROBES = (("nvidia", "nvidia-smi"), ("amd", "rocm-smi"), ("intel", "xpu-smi"))


def parse_nvidia_csv(stdout: str, fields: tuple[str, ...] = GPU_FIELDS) -> list[dict]:
    """Devices from ``nvidia-smi --query-gpu=... --format=csv,noheader``."""
    gpus = []
    need = min(len(fields), len(GPU_FIELDS_BASE))
    for line in (stdout or "").splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < need:
            continue
        # nvidia-smi prints '[N/A]' / '[Not Supported]' for a field this driver cannot
        # answer. Dropping it says "unknown"; keeping it would read as a real value.
        gpu = {k: v for k, v in zip(fields, parts) if not v.startswith("[")}
        if gpu.get("name"):
            gpus.append(gpu)
    return gpus


# ── compiler target ───────────────────────────────────────────────────────────

MARCH_QUERY = "gcc -march=native -Q --help=target"


def parse_march(stdout: str) -> dict:
    """``-march``/``-mtune`` that ``gcc -march=native`` resolves to on this machine.

    This is the name to hand the compiler when building *elsewhere* for this machine:
    ``native`` on the login node means the login node.
    """
    out = {}
    for line in (stdout or "").splitlines():
        m = re.match(r"\s*-m(arch|tune)=\s+(\S+)", line)
        if m and m.group(1) not in out:
            out[m.group(1)] = m.group(2)
    return {"march": out["arch"], "mtune": out.get("tune", "")} if "arch" in out else {}


# ── operating system ──────────────────────────────────────────────────────────

def parse_os_release(text: str) -> dict:
    raw = {}
    for line in (text or "").splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            raw[k.strip()] = v.strip().strip('"')
    return {k: raw[src] for k, src in (("id", "ID"), ("version", "VERSION_ID"),
                                        ("name", "PRETTY_NAME")) if raw.get(src)}


def parse_ldd_version(text: str) -> str:
    """'ldd (GNU libc) 2.28' -> '2.28'."""
    m = re.search(r"(\d+\.\d+)\s*$", (text or "").splitlines()[0] if text else "")
    return m.group(1) if m else ""


def local_os() -> dict:
    """OS release and glibc of this machine.

    Both decide whether a binary built here runs on another machine: a login node on
    a newer OS links against a glibc that an older compute node does not have.
    """
    data = {}
    try:
        with open("/etc/os-release") as fh:
            data.update(parse_os_release(fh.read()))
    except OSError:
        pass
    lib, ver = platform.libc_ver()
    if lib == "glibc" and ver:
        data["glibc"] = ver
    return data


# ── where am I ────────────────────────────────────────────────────────────────

def expand_hostlist(expr: str) -> list[str]:
    """Expand a Slurm hostlist: 'n[01-03,07],gpu1' -> n01 n02 n03 n07 gpu1.

    Covers one bracket group per name, which is what sites write in practice. A form
    it cannot read comes back as-is, so a membership test degrades to "not found".
    """
    names = []
    for part in re.findall(r"[^,\[]+(?:\[[^\]]*\])?[^,]*", expr or ""):
        m = re.fullmatch(r"([^\[]*)\[([^\]]*)\](.*)", part)
        if not m:
            names.append(part)
            continue
        prefix, body, suffix = m.groups()
        for rng in body.split(","):
            lo, _, hi = rng.partition("-")
            if hi and lo.isdigit() and hi.isdigit():
                width = len(lo)
                names.extend(f"{prefix}{i:0{width}d}{suffix}" for i in range(int(lo), int(hi) + 1))
            else:
                names.append(f"{prefix}{lo}{suffix}")
    return [n for n in names if n]


def execution_context(env: dict | None = None, hostname: str = "",
                      has_slurm: bool | None = None) -> dict:
    """Where this process sits relative to the scheduler, from deterministic signals.

    ``none``          no Slurm here: the host is the only machine.
    ``login``         Slurm is here, but this host is not inside an allocation it runs
                      on — a login node (possibly holding an allocation elsewhere, via
                      salloc: then ``job_id`` is set and a job step lands on it).
    ``in_allocation`` this host is one of the allocated nodes: it *is* the compute node.
    """
    env = os.environ if env is None else env
    hostname = hostname or socket.gethostname()
    if has_slurm is None:
        has_slurm = shutil.which("sinfo") is not None or bool(env.get("SLURM_JOB_ID"))
    if not has_slurm:
        return {"context": "none"}
    job_id = env.get("SLURM_JOB_ID", "")
    nodelist = env.get("SLURM_JOB_NODELIST") or env.get("SLURM_NODELIST") or ""
    if not job_id:
        return {"context": "login"}
    short = hostname.split(".")[0]
    on_node = (env.get("SLURMD_NODENAME") in (hostname, short)
               or short in expand_hostlist(nodelist))
    return {"context": "in_allocation" if on_node else "login",
            "job_id": job_id, "nodelist": nodelist}


# ── machine signature ─────────────────────────────────────────────────────────

def machine_signature(arch: str, cpu: dict, gpu_names: list[str]) -> str:
    """A short hash of the hardware a timing depends on.

    Computed on the machine where the code runs, from the same facts whichever way it
    got there (local, inside an allocation, in a batch job), so two timings carry the
    same signature exactly when they were taken on the same kind of machine. Unlike
    Slurm's own node signature it includes the CPU model, which is what tells two
    generations with the same core count apart.
    """
    payload = {
        "arch": arch,
        "model": cpu.get("model", ""),
        "sockets": str(cpu.get("sockets", "")),
        "cores_per_socket": str(cpu.get("cores_per_socket", "")),
        "threads_per_core": str(cpu.get("threads_per_core", "")),
        "simd": sorted(k for k, v in (cpu.get("simd") or {}).items() if v),
        "gpus": sorted(gpu_names),
    }
    blob = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()[:16]


def _quiet(cmd: list[str], timeout: int = 8) -> str:
    try:
        # C locale: lscpu translates its field names, and the parser reads English ones.
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                             text=True, timeout=timeout, env={**os.environ, "LC_ALL": "C"})
        return res.stdout if res.returncode == 0 else ""
    except (OSError, subprocess.SubprocessError):
        return ""


def local_machine() -> dict:
    """This machine's identity: host, context and signature. Cheap enough per run."""
    arch = platform.machine()
    cpu = cpu_facts(arch, _quiet(["lscpu"]) if shutil.which("lscpu") else "")
    gpus = []
    if shutil.which("nvidia-smi"):
        gpus = [g["name"] for g in parse_nvidia_csv(
            _quiet(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]), ("name",))]
    ctx = execution_context()
    return {
        "host": socket.gethostname(),
        "execution_context": ctx["context"],
        "arch": arch,
        "cpu_model": cpu.get("model", ""),
        "machine_signature": machine_signature(arch, cpu, gpus),
    }
