"""The bundled MCP servers, and the names the client keys off them."""
from __future__ import annotations

from .paths import SERVER_BASE

# The tool the model loads a skill's instructions with — an op of the bundled
# ``mimir_api`` server. Named here because three layers key off it and none of them
# owns it: the system prompt gates the skills index on it being connected, the policy
# engine gates the pull on the live toggles, and the dispatcher pins the result.
SKILL_LOAD_TOOL: str = "load_skill"

# Tool classification (sensitive / plan-blocked / path args / …) is no longer a
# static list here: each server declares its tools' capabilities and the client
# reads them from the per-agent live registry (agent.tool_caps). See
# context/capabilities.py and POLICY.md.

SERVERS: dict[str, str] = {
    "math": SERVER_BASE + "utilities/server_math.py",
    "strings": SERVER_BASE + "utilities/server_strings.py",
    "datetime": SERVER_BASE + "utilities/server_datetime.py",
    "memory": SERVER_BASE + "agent_state/server_memory.py",
    "files": SERVER_BASE + "workspace/server_files.py",
    "search": SERVER_BASE + "workspace/server_search.py",
    "web": SERVER_BASE + "external/server_web.py",
    "github": SERVER_BASE + "external/server_github.py",
    "hpc": SERVER_BASE + "hpc/server_hpc.py",
    "platform": SERVER_BASE + "hpc/server_platform.py",
    "env": SERVER_BASE + "hpc/server_env.py",
    "system": SERVER_BASE + "external/server_system.py",
    "code_intel": SERVER_BASE + "workspace/server_code_intel.py",
    "bash": SERVER_BASE + "workspace/server_bash.py",
    "todo": SERVER_BASE + "agent_state/server_todo.py",
    "mimir_api": SERVER_BASE + "agent_state/server_mimir_api.py",
    "symbolic_math": SERVER_BASE + "utilities/server_symbolic_math.py",
    "proxy": SERVER_BASE + "proxy/server_proxy.py",
    "agent": SERVER_BASE + "agent_state/server_spawn_agent.py",
    "interaction": SERVER_BASE + "interaction/server_interaction.py",
}

# One-line, human-readable purpose per server. Surfaced as tooltips in the
# server/skill toggle panel (the WebSocket server's `toggles_list`). Keyed by SERVERS name.
SERVER_DESCRIPTIONS: dict[str, str] = {
    "math": "Safe math expression evaluation (evaluate).",
    "strings": "String operations: case, strip, replace, split, search (string_op).",
    "datetime": "Date/time arithmetic, formatting, timezones (date_op).",
    "memory": "Persistent, timestamped agent memory with tags and search.",
    "files": "Read, write, edit, insert, replace, and delete workspace files.",
    "search": "Workspace reads, directory listing, and tree summaries.",
    "web": "Safe HTTP GET/POST and JSON utilities (SSRF-guarded).",
    "github": "Read-only GitHub access via the public REST API.",
    "hpc": "HPC helpers for Slurm scheduling and batch job submission.",
    "platform": "Probes this host on demand, and searches the site's "
                "environment-module catalogue.",
    "env": "Install packages and create/delete Python environments (approval-gated).",
    "system": "Read-only OS metrics and environment inspection.",
    "code_intel": "Code navigation: definitions, references, symbol outline (LSP/ctags).",
    "bash": "Shell command execution: search, compile, run, validate, test.",
    "todo": "Live task checklist and prose plan for the agent.",
    "mimir_api": "MIMIR's own extension API: how to author a .mimir skill, server, "
                 "policy, hook or nudge.",
    "symbolic_math": "Symbolic math (SymPy): algebra, calculus, matrices (symbolic).",
    "proxy": "Proxy/surrogate registry, benchmarks, runs, and optimization loop.",
    "agent": "Fan work out to fresh sub-agents: read-only exploration, or a separable sub-task.",
    "interaction": "Ask the user structured questions via the elicitation channel.",
}
