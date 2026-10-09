---
name: run-on-cluster
description: Run work on a Slurm cluster — pick the partition and node type, build for the hardware that will run it, size and submit the job, read what came back.
disable-model-invocation: false
---

The machine you are on is not the machine that will run the job. Everything below
follows from that one fact: a login node answers your probes, and a compute node runs
your binary, and on a mixed cluster they are not the same hardware, do not have the
same modules loaded, and do not have the same devices.

## Before submitting anything

1. `platform_get_profile()` — what *this* host is: cores, SIMD, memory, GPUs,
   toolchains, whether Slurm is even present. A host with no Slurm is a host where
   this skill does not apply; run the work locally and say so.
2. `slurm_partitions()` — the queues, their limits and their defaults. A wall time
   above the partition maximum is rejected at submission; a job with no account on a
   partition that requires one pends with a reason you have to go read.
3. `slurm_nodes(partition=...)` — the hardware you will actually get, collapsed onto
   node types with a free count per state. This is the sizing input: `cpus_per_task`
   that exceeds a node's cores, a `mem` above its memory, or a `gres` no node offers
   does not fail — **it pends forever**, and a job that never starts looks exactly
   like a job that is queued. Read the inventory, then ask for what exists.
4. `constraint=` when the partition is heterogeneous. Without it the scheduler picks
   any node in the queue, and two measurements taken on two node types are not a
   comparison — they are two numbers.

## Building for the node, not for here

Build with the toolchain and flags the *target* node accepts. `-march=native` resolved
on a login node and run on an older compute node is an illegal instruction at the first
vectorized loop, after the job waited in the queue. Prefer the explicit architecture the
node inventory reports over `native`, or build inside the allocation.

Modules are part of the build, so they are part of the job. `platform_search("cuda")`
or `platform_search("parallel hdf5")` gives the exact `load` string for this site —
never guess a module name or a version suffix, they are site-specific. Then put the
`module load` lines in the command you submit: a batch job gets a fresh shell, and
what you loaded in your own session is not what it will have. Build modules and run
modules must be the same set, or the run dies on a missing `.so` having consumed its
allocation.

A compile is not a measurement. If the partition you are measuring is scarce — GPU
nodes especially — compile somewhere else: a dedicated build partition if the site has
one, or the login node when the architectures match. Burning a GPU allocation on
`make` is the most common way to spend a reservation on nothing.

## Submitting

- `salloc_submit(..., confirm=False)` returns the exact command without executing it.
  Read it. The resources you *meant* and the resources you *typed* diverge silently.
- `sbatch_submit(command=..., partition=..., confirm=True)` for anything that is not
  interactive. It returns immediately with a `job_id` and a `background_job`
  descriptor: when the result says the run is being watched, **end your turn** — you
  are resumed when the job finishes, and polling it meanwhile buys nothing. Otherwise
  `slurm_job_status(job_id)`.
- `wall_time` is a guess with two failure modes. Too short and the job is killed at
  the limit with its output half-written; too long and it sits behind every shorter
  job in the queue. Time the smallest case locally or interactively, multiply by the
  problem-size ratio, then add margin — do not default to an hour because it is the
  default.
- `comment=` what the job is for. The queue is read later, by you and by whoever else
  is on the allocation, and `mimir-batch` tells them nothing.

## When the work belongs to a registered proxy

A proxy under measurement does not go through `sbatch_submit`. `proxy_slurm` submits it
instead, and the difference is not convenience — it is whether the result counts.

- `proxy_slurm(op='eval', partition=..., confirm=True)` for an optimization-session run:
  it submits the ratchet's own run, so the sealed references, the numerical invariants
  and the accept/reject margin apply to what comes back. A hand-submitted run of the
  same code measures the same seconds and establishes nothing, because nothing compared
  it to the baseline or checked it against the reference.
- `proxy_slurm(op='run', proxy_name=..., ...)` for a single run outside a session, and
  `op='suite'` for one job per case×sweep point of a benchmark suite — aggregate that
  one yourself with `proxy_get(op='report', ...)`, since it submits many jobs.
- Say where the **build** goes, separately from the run: `build_partition` (with
  `build_constraint`, `build_cpus_per_task`) sends the compile to a partition of its own
  and chains the two jobs. This is where the GPU-allocation waste above actually bites,
  and declaring it once at registration beats passing it per call.
- `constraint=` and `nodelist=` pin the hardware, for the same reason as any other
  submission: two runs of a ratchet on two node types are not a comparison, and the
  ratchet will happily accept the difference between the nodes as an improvement.
- Validate locally with `proxy_exec` before submitting. A batch job that fails on a
  missing module consumed its allocation to tell you that.

Every `proxy_slurm` op returns while the job is still queued and is watched from there:
**end your turn**, and you are resumed with the results.

While an optimization session is active, running the proxy directly through the shell is
blocked outright — that route bypasses the ratchet, so it cannot produce a valid result.
Submitting it as a batch job is the same bypass, and that one no guard catches: it is
yours not to do. `proxy_eval(op='end', confirm=True)` when the proxy is genuinely no
longer the subject.

## Scale up, do not start at the top

The first submission is the smallest case that exercises the whole path — the real
code, the real input format, the real output write — on the shortest queue that will
take it. It answers "does this run there at all", which is the question that actually
fails, and it answers it in minutes instead of after a night in the queue. Scale the
problem only once a small case has come back clean.

## Reading what came back

A Slurm job that ended with state `COMPLETED` ran to its end. That is all it says. Read
the log, find the numbers, and `report_verdict(...)` on what they showed — `pass` with
the value you read, `unknown` when the output settles nothing, `fail` when it shows the
run was wrong. A crashed job: read the tail of the log before resubmitting, because the
same submission will fail the same way, and a resubmission costs another wait.

An allocation you hold and a job you queued are obligations, not artefacts. Before
concluding, say what is still running or still allocated, and `slurm_cancel(job_id)`
what should not outlive the task — ask first; a job you did not submit is never yours
to cancel.
