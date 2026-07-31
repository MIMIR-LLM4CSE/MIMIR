"""Shared helper library for the proxy MCP server.

Split of the historical ``_shared_proxy.py`` god-module into cohesive units:

  store.py    — storage root/layout, atomic IO, registry/suite/reference/opt persistence
  procs.py    — process & run state, launch/sbatch/cancel lifecycle, log access
  metrics.py  — metrics parsing, numerical invariants, requirement evaluation
  command.py  — param-file rendering and run/sbatch command building
  report.py   — roofline, result rows, run-dir diffs
  execute.py  — synchronous case runs, reference sealing, detached-run finalize
  ratchet.py  — the optimization ratchet (verdicts, best-so-far, ledger)

Dependency direction: store <- procs/metrics/command/report <- execute/ratchet.
``_shared_proxy.py`` remains as a minimal legacy facade (generated ``postrun.py``
scripts import ``_post_run_finalize`` from it by name).
"""

import os
import sys

_PROXY_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (os.path.join(_PROXY_DIR, "..", "_shared"), _PROXY_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)
