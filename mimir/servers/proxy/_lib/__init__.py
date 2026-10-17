"""Shared helper library for the proxy MCP server.

  store.py    — storage root/layout, atomic IO, registry/suite/reference/opt persistence
  procs.py    — process & run state, launch/sbatch/cancel lifecycle, log access
  metrics.py  — metrics parsing, numerical invariants, requirement evaluation
  command.py  — param-file rendering and run/sbatch command building
  build.py    — building a registered proxy before a run measures it
  report.py   — roofline, result rows, run-dir diffs
  execute.py  — synchronous case runs, reference sealing, detached-run finalize
  ratchet.py  — the optimization ratchet (verdicts, best-so-far, ledger, measurement policy)
  tree_snapshot.py — atomic snapshots of the file set under optimization
  placement.py — where each phase of a Slurm eval run is sent

Dependency direction: store <- procs/metrics/command/build/report/tree_snapshot/placement
<- execute/ratchet.
"""
