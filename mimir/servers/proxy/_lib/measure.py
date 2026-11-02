"""What the server measures of a run where no MIMIR process waits for it (a Slurm job):
its exit code, wall time, largest process's peak memory and the sum of every process's,
written beside its output for the post-run step to read, as a local run hands them over."""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

from . import proc_run

# sacct's figures arrive once the step's accounting is flushed: a few tries, after the run.
_SACCT_TRIES, _SACCT_WAIT_S = 3, 2.0
_UNITS = {"": 1 / 1024, "K": 1, "M": 1024, "G": 1024**2, "T": 1024**3}


def main(args: list[str]) -> int:
    """``main([run_dir, *argv])``: run *argv*, record it in *run_dir*, return its exit code."""
    run_dir, argv = args[0], args[1:]
    t0 = time.monotonic()
    proc = None
    try:
        proc = proc_run.run(argv, rusage=True)
        code = proc.returncode
    except OSError as exc:
        print(f"[measure] could not start {argv[:1]}: {exc}", file=sys.stderr)
        code = 127
    wall_ms = int((time.monotonic() - t0) * 1000)
    total, processes = getattr(proc, "peak_rss_total_kb", None), getattr(proc, "processes", None)
    # Ranks srun starts belong to Slurm's step daemons, not to this process, on whatever
    # node: Slurm's own accounting is what sees all of them.
    steps = slurm_steps(os.environ.get("SLURM_JOB_ID", ""))
    if steps and steps[0] > (total or 0):
        total, processes = steps
    # A signal is -N here and 128+N to the shell, which is what the job script reads.
    code = code if code >= 0 else 128 - code
    for name, value in (("returncode", code), ("wall_time_ms", wall_ms),
                        ("peak_rss_kb", getattr(proc, "peak_rss_kb", None)),
                        ("peak_rss_total_kb", total), ("processes", processes)):
        if value is not None:
            with open(os.path.join(run_dir, name), "w") as fh:
                fh.write(f"{value}\n")
    return code


def slurm_steps(job_id: str) -> tuple[int, int] | None:
    """The sum of every task's peak memory (KB) over the job's steps, and the task count,
    from Slurm's accounting; None outside a job or without accounting."""
    if not job_id:
        return None
    for attempt in range(_SACCT_TRIES):
        if attempt:
            time.sleep(_SACCT_WAIT_S)
        try:
            out = subprocess.run(["sacct", "-j", job_id, "-n", "-P", "--units=K",
                                  "-o", "JobID,NTasks,AveRSS"], capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return None
        found = sacct_total(out)
        if found:
            return found
    return None


def sacct_total(text: str) -> tuple[int, int] | None:
    """``sacct -P -o JobID,NTasks,AveRSS`` lines: AveRSS is the mean of the tasks' peaks,
    so times NTasks it is their sum; summed over the srun steps (``<job>.<n>``)."""
    total = tasks = 0
    for line in text.splitlines():
        cols = line.split("|")
        if len(cols) < 3 or not re.fullmatch(r"\d+(?:_\d+)?\.\d+", cols[0]) or not cols[1].isdigit():
            continue
        m = re.fullmatch(r"([\d.]+)([KMGT]?)", cols[2].strip())
        if not m:
            continue
        total += int(float(m.group(1)) * _UNITS[m.group(2)] * int(cols[1]))
        tasks += int(cols[1])
    return (total, tasks) if tasks else None
