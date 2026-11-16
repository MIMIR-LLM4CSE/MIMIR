"""What a profiled command runs, from what its measured run loaded (``proc_run``'s
fingerprint), the languages of the code optimized, else its command line; and how a
profiler is put in front of it: past the wrappers (``env``, ``uv run``...) and, under
an MPI launcher, in front of rank 0 only (``rank0.sh``), whose ranks are the launcher's
daemons' children, not the profiler's.
"""
from __future__ import annotations

import os
import re
import shlex
import subprocess
from dataclasses import dataclass, field

from ..device_memory import vendor_of

_PYTHON_RE = re.compile(r"^(python|pypy)[\d.]*$")
LAUNCHERS = {"mpirun", "mpiexec", "mpiexec.hydra", "orterun", "srun", "aprun", "jsrun", "prun", "ibrun"}
# Launcher options that take a value as the next word (a word with "=" carries its own).
_LAUNCHER_VALUE = {"-n", "-np", "-N", "-c", "-p", "-t", "-A", "-w", "-H", "-x", "-ppn", "-npernode",
                   "--npernode", "--host", "--hostfile", "-hostfile", "-machinefile", "--machinefile",
                   "-f", "--map-by", "--bind-to", "--rank-by", "-wdir", "--wdir", "-d", "-cc", "-a",
                   "--ntasks", "--nodes", "--cpus-per-task", "--partition", "--time", "--account",
                   "--nodelist", "--mpi", "--distribution", "-m", "-r", "-rs", "-g", "-e", "-o"}
_LAUNCHER_TWO = {"--mca", "-mca", "-genv", "-env", "--gmca"}
# Wrappers before the program, and how many words each takes before it.
_RUNNERS = {("uv", "run"), ("conda", "run"), ("pixi", "run"), ("poetry", "run"), ("pipenv", "run"),
            ("micromamba", "run"), ("mamba", "run")}
_RUNNER_VALUE = {"-n", "--name", "-p", "--prefix", "-e", "--environment", "--with", "--python",
                 "--project", "--directory"}
_WRAPPERS = {"time", "nice", "nohup", "stdbuf", "taskset", "numactl", "chrt", "ionice", "env",
             "command", "exec", "ltrace"}
_SHELLS = {"sh", "bash", "dash", "zsh", "ksh"}
_RANK_VARS = ("OMPI_COMM_WORLD_RANK", "PMI_RANK", "PMIX_RANK", "MV2_COMM_WORLD_RANK",
              "SLURM_PROCID", "ALPS_APP_PE", "PALS_RANKID", "JSM_NAMESPACE_RANK")
_FORTRAN_RT = {"gfortran", "ifcore", "ifport"}


@dataclass
class Target:
    """A command split into its launcher (``mpirun -n 4``, or nothing), what runs before
    the program (wrappers), and the program with its arguments; and what it runs."""
    argv: list[str]
    launcher: list[str] = field(default_factory=list)
    wrappers: list[str] = field(default_factory=list)
    command: list[str] = field(default_factory=list)
    python: bool = False
    native: set[str] = field(default_factory=set)       # "c", "cpp", "fortran", "cuda"
    gpu: str = ""
    runtimes: set[str] = field(default_factory=set)

    @property
    def mpi(self) -> bool:
        return bool(self.launcher)

    @property
    def interpreter(self) -> str:
        """The Python the command runs, when its program is one."""
        return self.command[0] if self.command and _PYTHON_RE.match(os.path.basename(self.command[0])) else ""

    @property
    def script(self) -> str:
        """The script a Python command runs, or ""."""
        if self.interpreter:
            rest = [a for a in self.command[1:] if not a.startswith("-")]
            return rest[0] if rest and rest[0].endswith(".py") else ""
        return self.command[0] if self.command and self.command[0].endswith(".py") else ""

    def wrap(self, tool: list[str], out_dir: str, inside: bool = False) -> list[str]:
        """*tool* (a profiler's words up to the program) put in front of the program,
        after the launcher and on rank 0 only when there is one. *inside*: the tool is
        the interpreter's own (``python -m cProfile ...``), its first word the program's."""
        rest = self.command[1:] if inside else self.command
        words = [*self.wrappers, *tool, *rest]
        if not self.launcher:
            return words
        keep = 1 if inside else 0
        return [*self.launcher, rank0_script(out_dir), str(len(self.wrappers) + keep),
                str(len(tool) - keep), *words]


def split(argv: list[str]) -> tuple[list[str], list[str], list[str]]:
    """*argv* as (launcher, wrappers, program and its arguments)."""
    i, launcher = 0, []
    if argv and os.path.basename(argv[0]) in LAUNCHERS:
        i = 1
        while i < len(argv) and argv[i].startswith("-"):
            word = argv[i]
            if word == "--":
                i += 1
                break
            i += 3 if word in _LAUNCHER_TWO else 2 if (word in _LAUNCHER_VALUE and "=" not in word) else 1
        launcher, argv, i = argv[:i], argv[i:], 0
    start = i
    while i < len(argv):
        base = os.path.basename(argv[i])
        if (base, argv[i + 1] if i + 1 < len(argv) else "") in _RUNNERS:
            i += 2
            while i < len(argv) and argv[i].startswith("-"):
                i += 2 if argv[i] in _RUNNER_VALUE else 1
        elif base in _WRAPPERS:
            i += 1
            while i < len(argv) and (argv[i].startswith("-") or "=" in argv[i]
                                      or (base == "taskset" and re.fullmatch(r"[0-9a-fx,\-]+", argv[i]))
                                      or (base == "nice" and argv[i].lstrip("-").isdigit())):
                takes = base in ("numactl", "taskset", "chrt", "ionice") and not argv[i].startswith("--") \
                    and argv[i] in ("-C", "-m", "-N", "-c", "-p", "-n", "-l", "-i", "-P")
                i += 2 if takes or (base == "env" and argv[i] in ("-u", "-C", "--unset", "--chdir")) else 1
        else:
            break
    return launcher, argv[start:i], argv[i:]


def _shell_words(command: list[str]) -> list[str]:
    """The words of a shell command's script (``bash x.sh``, ``sh -c '...'``), to see
    what it runs."""
    if not command or os.path.basename(command[0]) not in _SHELLS:
        return []
    rest = command[1:]
    if "-c" in rest and rest.index("-c") + 1 < len(rest):
        text = rest[rest.index("-c") + 1]
    else:
        script = next((a for a in rest if not a.startswith("-")), "")
        try:
            with open(script, encoding="utf-8", errors="replace") as fh:
                text = fh.read(65536)
        except OSError:
            return []
    try:
        return shlex.split(text, comments=True)
    except ValueError:
        return text.split()


def describe(argv: list[str], fingerprint: dict | None = None, languages: set[str] | None = None) -> Target:
    """What *argv* runs: its parts (``split``), whether Python runs, the native languages
    in play (the code's *languages*, the runtimes its measured run loaded) and its GPU."""
    launcher, wrappers, command = split(argv)
    t = Target(argv=list(argv), launcher=launcher, wrappers=wrappers, command=command)
    words = [command[0]] if command else []
    words += [w for w in _shell_words(command) if "/" in w or _PYTHON_RE.match(os.path.basename(w))]
    fp = fingerprint or {}
    programs = [os.path.basename(p) for p in fp.get("programs") or []]
    t.runtimes = set(fp.get("runtimes") or [])
    t.python = (any(_PYTHON_RE.match(os.path.basename(w)) or w.endswith(".py") for w in words)
                or any(_PYTHON_RE.match(p) for p in programs) or any(r.startswith("python") for r in t.runtimes))
    langs = set(languages or ())
    t.native = {lang for lang in langs if lang in ("c", "cpp", "fortran", "cuda")}
    if t.runtimes & _FORTRAN_RT or any(r.startswith("flang") for r in t.runtimes):
        t.native.add("fortran")
    if not t.python and command and not t.native:
        t.native.add("c")        # a compiled program of unknown language
    t.gpu = vendor_of(t.runtimes)
    return t


def python_exposes_frames(python: str) -> bool:
    """Whether *python* can show its frames to perf (``PYTHONPERFSUPPORT``, 3.12 and later
    on Linux)."""
    if not python:
        return False
    try:
        out = subprocess.run([python, "-c", "import sys; print(sys.version_info >= (3, 12))"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return False
    return out.strip() == "True"


def rank0_script(out_dir: str) -> str:
    """A script that runs its words on rank 0, and on every other rank without the
    profiler's: ``rank0.sh FROM N words...`` drops the N words from the FROM-th."""
    path = os.path.join(out_dir, "rank0.sh")
    rank = "".join("${%s:-" % v for v in _RANK_VARS) + "0" + "}" * len(_RANK_VARS)
    with open(path, "w") as fh:
        fh.write("#!/bin/sh\n"
                 "from=$1; n=$2; shift 2\n"
                 f"rank={rank}\n"
                 'if [ "$rank" = 0 ]; then exec "$@"; fi\n'
                 "i=0\n"
                 'for a do shift; if [ "$i" -lt "$from" ] || [ "$i" -ge $((from + n)) ]; then set -- "$@" "$a"; fi; '
                 "i=$((i + 1)); done\n"
                 'exec "$@"\n')
    os.chmod(path, 0o755)
    return path


def missing(tool: str) -> str:
    """Why *tool* is not here, with where a cluster may keep it."""
    hint = f" (on a cluster: `module avail {tool}`, then load it in the session's environment)" \
        if os.environ.get("MODULEPATH") else ""
    return f"{tool} is not installed{hint}"
