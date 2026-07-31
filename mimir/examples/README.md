# MIMIR examples

Canonical, copy-to-customize examples for every MIMIR extension type. This folder
**mirrors the `.mimir/` layout** (`skills/`, `servers/`, `plugins/`, plus the base-prompt
file at the root) and is the **single source of truth** for every example.

> **Nothing is auto-created in your workspace.** MIMIR never writes into `.mimir/` — that
> directory is yours alone. To add an extension, create the file yourself under `.mimir/`
> (or point the matching `MIMIR_*_DIR` env var elsewhere), using these files as a template.

| Example | Extension type | Where it goes in your workspace |
|---------|----------------|---------------------------------|
| `skills/example-skill/SKILL.md` | Skill (methodology prompt) | `.mimir/skills/<skill-name>/SKILL.md` |
| `servers/server_example.py` | MCP server | `.mimir/servers/server_<name>.py` |
| `plugins/policy_example.py` | Policy (locked — blocks a call) | `.mimir/plugins/<name>.py` |
| `plugins/nudge_example.py` | Nudge (toggleable reminder) | `.mimir/plugins/<name>.py` |
| `system_prompt.md` | Base system prompt | `.mimir/system_prompt.md` (or `$MIMIR_SYSTEM_PROMPT_FILE`) |

Per-type notes live in the `README.md` inside each subfolder. See
[`PLUGINS_DETAILED.md`](../../PLUGINS_DETAILED.md) for the full authoring guide.
