"""Example custom MCP server.

Copy this into your servers directory to activate it — `$MIMIR_SERVERS_DIR` or
`<workspace>/.mimir/servers/` — and rename it to `server_<name>.py` (drop the `.example`
suffix). The tool namespace is the filename stem with any `server_` prefix stripped
(`server_weather.py` -> `weather`). A name that collides with a bundled core server is
ignored. It is NOT auto-loaded from `mimir/examples/`.

Declaring **capabilities** with `tool_caps(...)` is OPTIONAL — MIMIR runs undeclared tools
fine — but RECOMMENDED: capabilities are what let MIMIR's safety policies, result caching,
and nudges treat your tool correctly, and they are essential if you write policies/nudges
that key off a capability. See the capability table in PLUGINS_DETAILED.md.

See PLUGINS_DETAILED.md for the full authoring guide.
"""

from mcp.server.fastmcp import FastMCP
from mimir.servers._shared.capabilities import tool_caps, READ, CACHEABLE

mcp = FastMCP("example")


@mcp.tool()
def greet(name: str) -> str:
    """A pure tool — no side effects, so no capabilities are needed."""
    return f"Hello, {name}!"


@mcp.tool(**tool_caps(caps=[READ, CACHEABLE], path_args=["path"]))
def read_note(path: str) -> str:
    """Return the contents of a text file.

    Declaring capabilities lets MIMIR treat this tool correctly:
    - READ      — marks it as discovery evidence (a read-before-edit precondition);
    - CACHEABLE — lets MIMIR reuse the result within a query, invalidated on a write;
    - path_args — tells MIMIR which argument carries a filesystem path.
    """
    with open(path, encoding="utf-8") as f:
        return f.read()


if __name__ == "__main__":
    mcp.run()
