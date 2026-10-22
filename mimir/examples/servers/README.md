# Custom MCP servers

A custom server exposes extra tools to the agent over stdio MCP.

**To add one:** create `.mimir/servers/server_<name>.py` (or `.js`) in your workspace
(override the location with the `MIMIR_SERVERS_DIR` env var). It is auto-connected on
startup; the tool namespace is the filename stem with any `server_` prefix stripped
(`server_weather.py` → `weather`). A name that collides with a bundled (core) server is
ignored to protect the core.

Tool capabilities (which policies/nudges apply) come from each tool's declaration — see
[`server_example.py`](server_example.py) here and
[`mimir/shared/capabilities.py`](../../shared/capabilities.py).

See [`PLUGINS_DETAILED.md`](../../../docs/PLUGINS_DETAILED.md) for the full authoring guide.
