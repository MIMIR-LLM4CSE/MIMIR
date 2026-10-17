# MIMIR Architecture — Dependency Graph

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The layering of `mimir/client/`, derived from the actual imports. Read top-to-bottom:
higher layers depend on lower ones. Solid edges are import-time; **dotted edges are
lazy** — imported inside a function, so they never run at load time.

The graph shows the layering, not every edge: every unit also imports the foundational
ones (`config`, `protocol`, `event_sink`, `human_pause`) and
`mimir/servers/_shared` freely, and those edges are omitted to keep it readable.

```mermaid
graph TD
    subgraph FE["🖥️ Front-ends"]
        cli["ui/cli/<br/><small>REPL · CliHost · text rendering</small>"]
        ws["ui/ws/<br/><small>serving · ws_session + handlers/ · event_router<br/>event_bus · ws_pool · wake_router<br/>ws_worker · human_bridge · job_watcher</small>"]
    end
    subgraph SES["💬 Conversations"]
        session["session/<br/><small>conversation · store · commands/</small>"]
    end
    subgraph ORCH["🎛️ Orchestrator"]
        agent["agent_core.py<br/><small>MimirAgent</small>"]
    end
    subgraph ENG["⚙️ Engine"]
        qe["query_engine/<br/><small>turn · loop_base · agent_loop · plan_loop<br/>dispatch/ · history · streaming · background<br/>context_budget · backends/</small>"]
    end
    subgraph GOV["🛡️ Governance &amp; services"]
        guard["guardrails/<br/><small>policy · nudges · workflow · observations</small>"]
        prompt["prompt/<br/><small>system_prompt</small>"]
        ext["extensions/<br/><small>.mimir/ servers · skills · plugins</small>"]
        integ["integration/<br/><small>MCP connect</small>"]
        te["tool_execution/<br/><small>executor · validation · direct</small>"]
    end
    subgraph CON["📜 Contracts"]
        core["core/<br/><small>ports (AgentPort, HumanIO) · settings · human</small>"]
    end
    subgraph FND["🧱 Foundational"]
        ctx["context/<br/><small>execution_context · capabilities · signals</small>"]
        cfg["config/ · protocol/<br/><small>paths · limits · settings · models<br/>events · messages · errors</small>"]
    end

    cli --> session
    cli --> qe
    cli -.->|lazy| agent
    ws --> session
    ws --> agent
    ws --> qe
    ws --> core

    session --> core
    session --> qe

    agent --> qe
    agent --> guard
    agent --> prompt
    agent --> ext
    agent --> integ
    agent --> te
    agent --> core

    qe --> guard
    qe --> te
    qe --> core
    qe --> ctx

    guard --> core
    guard --> ctx
    te --> guard
    te --> core
    te --> ctx
    prompt --> ext
    ext --> guard
    ext --> te
    integ --> core
    integ --> ctx

    core --> ctx
    core -.->|lazy| qe
    ctx --> cfg

    classDef fnd fill:#e6faf1,stroke:#1f9d6a,color:#0f3d29;
    classDef con fill:#eef7ee,stroke:#4a9d5b,color:#1e3a24;
    classDef gov fill:#f3ecff,stroke:#8257d1,color:#2e1a52;
    classDef eng fill:#fff7e6,stroke:#d99a2b,color:#5a3d0a;
    classDef orch fill:#e8f0ff,stroke:#3b6fd4,color:#1a2a4a;
    classDef ses fill:#e8f6ff,stroke:#3a8fc4,color:#123a52;
    classDef fe fill:#fde9ec,stroke:#d1526b,color:#521a29;

    class cli,ws fe;
    class session ses;
    class agent orch;
    class qe eng;
    class guard,prompt,ext,integ,te gov;
    class core con;
    class ctx,cfg fnd;
```

## How to read it

- **Foundational** (`config`, `protocol`, `context`, `event_sink`) depend on
  nothing above them. `config` is the vocabulary (paths, limits, the closed sets of each
  setting, the model catalogue); `protocol` the shapes that cross a boundary (event
  types, chat messages and their invariants, error payloads); `context` the
  `ExecutionContext` dataclass, the query/tool signals and the capability vocabulary;
  `event_sink` the injectable `emit()` seam that keeps the engine ignorant of front-ends.
- **`core`** holds the contracts the layers above agree on: `AgentPort` (everything the
  engine and the guardrails may call on an agent), `HumanIO` (how the agent reaches a
  person — the terminal, a WebSocket card, a sub-agent's caller, nobody), and the
  settings table (`apply_setting` / `read_setting`, one validation for every front-end).
- **`guardrails`** (hard `policy` gates as a pipeline, soft `nudges` as one cascade,
  the `workflow` state and the `observations` it records) sits below the engine.
- **`query_engine`** drives a turn: `turn.run_loops` switches between the `AgentLoop`
  and `PlanLoop` strategies (both `StepLoop`s), `dispatch/` admits and runs tool calls,
  `backends/` share one `LLMBackend.chat` template.
- **`agent_core.py`** (`MimirAgent`) is what every front-end, the runner and the
  sub-agent server instantiate; it implements `AgentPort`.
- **`session`** is what a conversation is, independent of the front-end showing it:
  `Conversation` (one in-memory object per conversation, handed to every writer by the
  `Conversations` registry), the `SessionStore` on disk, and the session commands
  (`Command`, `CommandRegistry`) both front-ends share.
- **`ui/`** is only the front-ends. The CLI drives an agent in-process; the WebSocket
  server runs one worker per conversation in a pool, whose pump commits answers and
  routes finished runs before any socket renders them.

### Back-edges

Two places where a lower unit reaches a higher one, each a **lazy import inside a
function**:

| Edge | Where | Why |
|---|---|---|
| `core → query_engine` | `core/settings._store_backend` | switching the backend needs the factory, and the engine reads the mode through `core.settings` at load |
| `ui/cli → agent_core` | `ui/cli/main.py` | the script-mode bootstrap (`python main.py`) has to fix `sys.path` first |

The proxy store (`guardrails/proxy_state`, `guardrails/policy/gates`) is lazy too: it
is the proxy server's heavy library, needed only by a conversation running an
optimisation. The WebSocket entry
(`ui/ws/ws_server.main`) imports `serving` only after applying `--cwd`, because
`config.paths` resolves the workspace root as it is imported.

### Cycles

None between units. `tool_execution` runs a call through the `guardrails` (policy
before, observations after), and nothing in `guardrails`, `core` or `context` imports
`tool_execution`: the path rules live in `context/paths.py` and the display vocabulary
in `context/tool_display.py`.

Everything else is a straight downward dependency.
