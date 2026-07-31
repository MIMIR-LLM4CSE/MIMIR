# Custom policies & nudges

A plugin module registers `PolicyCheck` / `NudgeRule` descriptors at import time (see
`mimir.client.extensions`). Policies are locked (mandatory); nudges are toggleable via
`/nudges`.

**To add one:** drop a Python module in `.mimir/plugins/` in your workspace (override the
location with the `MIMIR_PLUGINS_DIR` env var). It is imported on startup.

Ready-to-copy templates in this folder:

- [`policy_example.py`](policy_example.py) — a locked outbound-secret guard
- [`nudge_example.py`](nudge_example.py) — a toggleable authorization reminder

See [`PLUGINS_DETAILED.md`](../../../PLUGINS_DETAILED.md) for the full authoring guide.
