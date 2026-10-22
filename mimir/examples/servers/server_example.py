"""Example custom MCP server.

Copy into ``.mimir/servers/`` (or ``$MIMIR_SERVERS_DIR``) as ``server_<name>.py``; the
namespace is ``<name>``, and a bundled name is ignored. Declaring ``tool_caps(...)`` is
optional but recommended. See docs/PLUGINS_DETAILED.md.
"""

from mcp.server.fastmcp import FastMCP
from mimir.shared.capabilities import tool_caps, READ, CACHEABLE

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
