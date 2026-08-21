# Custom policies, post-tool hooks & nudges

A plugin module registers `PolicyCheck` / `PostToolRule` / `NudgeRule` descriptors at import
time (see `mimir.client.extensions`) — one per moment in a call's life: before it runs,
after it produced something, at the end of the turn. Policies are locked (mandatory);
nudges are toggleable via `/nudges`.

**To add one:** drop a Python module in `.mimir/plugins/` in your workspace (override the
location with the `MIMIR_PLUGINS_DIR` env var). It is imported on startup.

Ready-to-copy templates in this folder:

- [`policy_example.py`](policy_example.py) — a locked outbound-secret guard
- [`post_tool_example.py`](post_tool_example.py) — the project's own tests, run after an edit
- [`nudge_example.py`](nudge_example.py) — a toggleable authorization reminder

See [`PLUGINS_DETAILED.md`](../../../PLUGINS_DETAILED.md) for the full authoring guide.
