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
   toolchains, whether Slurm is even present. No Slurm *here* is not the end of it: the
   scheduler often lives on another host. If reading the queue over `ssh` works
   (`ssh <host> "squeue -u $USER"`), that one call is enough — MIMIR notes where the
   scheduler is — for the whole workspace, so a cluster another conversation reached is
   one you can submit to without reaching it again. Only a site where neither answers is
   one where this skill does not apply; run the work locally and say so.
2. `slurm_schedulers()` — which clusters this workspace can submit to, and what each
   one is **made of**: how many node types, their architectures, the GPU models, how
   many partitions. One reachable and there is nothing to choose; the calls below may
   omit `scheduler=`. Several and every one of them takes it — they refuse rather than
   defaulting, because which cluster a job belongs on follows the hardware it needs
   and, above all, the filesystem its inputs and its `work_dir` are on: a job can only
   run where its files are. Ask the user when the task does not settle it. The hardware
   is on each entry so that choosing costs no extra call — what is free *right now* is
   step 4's question, not this one.
3. `slurm_partitions(scheduler=...)` — the queues, their limits and their defaults. A
   wall time above the partition maximum is rejected at submission; a job with no
   account on a partition that requires one pends with a reason you have to go read.
4. `slurm_nodes(partition=..., scheduler=...)` — the hardware you will actually get,
   collapsed onto node types with a free count per state. This is the sizing input:
   `cpus_per_task` that exceeds a node's cores, a `mem` above its memory, or a `gres`
   no node offers does not fail — **it pends forever**, and a job that never starts
   looks exactly like a job that is queued. Read the inventory, then ask for what
   exists.
5. `constraint=` when the partition is heterogeneous. Without it the scheduler picks
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

- `scheduler=` on every one of these when the workspace reaches more than one — the
  same name step 2 settled. They refuse an unnamed submission rather than sending it to
  whichever cluster came first.
- `salloc_submit(..., confirm=False)` returns the exact command without executing it.
  Read it. The resources you *meant* and the resources you *typed* diverge silently. It
  is also the one tool that cannot go to a cluster behind `ssh`: an interactive
  allocation needs a terminal on that host, so it is refused there rather than taking
  the allocation and handing it straight back. Use `sbatch_submit` on a remote
  scheduler.
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
- `work_dir=` the directory the work belongs to, when it is not the workspace — a
  scratch filesystem, the run's own tree. The job starts there *and* writes its log
  there, which is what makes the output reachable at all: a compute node writes what it
  can see, and the directory the job started in is the one thing it certainly could.

## A job you did not submit through MIMIR is not being watched

Some work cannot go through `sbatch_submit`: a site launcher that submits a server plus
a client array with its own dependencies, a submission script of the user's, a `cd` and
a wrapper on a shared filesystem. Those are submitted from a shell and they are the
right answer — but nothing of MIMIR's is involved, and tracking is a *descriptor* a
submission writes, not a job id you have read. With no descriptor nothing watches the
job, no wake is owed when it ends, and a server left running overnight has — as far as
it can tell — nothing to do, so it stops. The job runs on; nobody is ever told.

So for **every id the submission printed**, the server job and the client array alike:

- `slurm_watch(job_id, note="what this is")` — adopts it. An array's base id covers all
  of its tasks. It refuses an id Slurm has never heard of, so it is also the check that
  the submission actually landed.
- If what you started is a *process* on this host rather than a scheduler job — a driver
  a script left running in the background — the same applies with `bash_watch(pid)`.
- Then **end your turn**, exactly as after `sbatch_submit`: you are resumed when the job
  finishes, with the window closed or not.

Do this in the same turn as the submission, before you report anything. A job submitted
and not adopted is a job whose result reaches nobody — and if the user detaches on the
turn you were about to adopt it in, their night goes to a run nothing is watching.

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
