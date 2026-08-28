# MIMIR Architecture — Dependency Graph

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The internal dependency graph of `mimir/client/`, derived from the actual imports
(who imports whom). Read top-to-bottom: higher layers depend on lower ones. Solid
edges are import-time dependencies; **dotted edges are lazy** (imported inside a
function) — they are what keep the graph acyclic at load time even where two units
reference each other.

```mermaid
graph TD
    subgraph FE["🖥️ Frontends"]
        ui["ui/<br/><small>cli · ws</small>"]
    end
    subgraph ORCH["🎛️ Orchestrator"]
        core["agent_core.py<br/><small>MimirAgent</small>"]
    end
    subgraph ENG["⚙️ Engine"]
        qe["query_engine/<br/><small>agent_loop · plan_loop · dispatch<br/>history · streaming · background<br/>finalize · toollist · backends</small>"]
    end
    subgraph GOV["🛡️ Governance &amp; services"]
        guard["guardrails/<br/><small>policy · nudges<br/>workflow · observations · state_machine</small>"]
        prompt["prompt/<br/><small>system_prompt</small>"]
        ext["extensions/<br/><small>.mimir/ servers · skills · plugins</small>"]
        integ["integration/<br/><small>MCP connect</small>"]
    end
    subgraph SUP["🔧 Support"]
        te["tool_execution/<br/><small>executor · validation · normalizer</small>"]
    end
    subgraph FND["🧱 Foundational"]
        ctx["context/<br/><small>execution_context · signals · capabilities</small>"]
        cfg["config/<br/><small>constants · models</small>"]
        es["event_sink.py"]
    end
    srv["mimir/servers/_shared<br/><small>embed · capabilities · trusted_read_roots</small>"]

    ui --> core
    ui --> qe
    ui --> prompt
    ui --> ext
    ui --> ctx
    ui --> cfg
    core -.->|lazy| ui

    core --> qe
    core --> guard
    core --> prompt
    core --> ext
    core --> integ
    core --> te
    core --> ctx
    core --> cfg
    core --> es

    qe --> guard
    qe --> prompt
    qe --> te
    qe --> ctx
    qe --> cfg
    qe --> es
    qe --> srv
    cfg -.->|lazy| qe

    te --> guard
    te --> ctx
    guard -.->|lazy| te

    guard --> ctx
    guard --> cfg
    guard --> srv

    prompt --> ext
    prompt --> cfg
    ext --> guard
    ext --> cfg
    integ --> ctx
    integ --> cfg
    ctx --> te

    classDef fnd fill:#e6faf1,stroke:#1f9d6a,color:#0f3d29;
    classDef sup fill:#eef7ee,stroke:#4a9d5b,color:#1e3a24;
    classDef gov fill:#f3ecff,stroke:#8257d1,color:#2e1a52;
    classDef eng fill:#fff7e6,stroke:#d99a2b,color:#5a3d0a;
    classDef orch fill:#e8f0ff,stroke:#3b6fd4,color:#1a2a4a;
    classDef fe fill:#fde9ec,stroke:#d1526b,color:#521a29;
    classDef ext2 fill:#eef1f5,stroke:#8b95a5,color:#333b47;

    class ui fe;
    class core orch;
    class qe eng;
    class guard,prompt,ext,integ gov;
    class te sup;
    class ctx,cfg,es fnd;
    class srv ext2;
```

## How to read it

- **Foundational** (`config`, `context`, `event_sink`) depend on nothing above them —
  they are the shared data substrate + IO seam. `context` holds the `ExecutionContext`
  blackboard, the query/tool `signals`, and the capability vocabulary; `event_sink` is
  the injectable `emit()` bus that decouples the engine from any frontend.
- **`guardrails`** (the two guardrail systems — hard `policy` gates and soft `nudges` —
  plus the shared `workflow` state model and the `observations` blackboard writer) sits
  above the foundation and below the engine. Both subsystems read `context`; the
  dependency is one-directional (guardrails → context/config, never the reverse).
- **`query_engine`** drives the per-query loop and pulls in everything below it.
- **`agent_core.py`** (the `MimirAgent` core) is the orchestrator every frontend, the
  runner, and the sub-agent server instantiate — it is at the client root, **not** in
  `ui/`, because it is the engine the UIs pilot.
- **`ui/`** is only the frontends (CLI + WebSocket/VS Code); it drives `agent_core`.

### The three lazy back-edges (dotted)

These are the only places a lower/earlier unit references a higher one; each is a
**lazy import inside a function**, so it never runs at module-load time and the graph
stays acyclic on import:

| Edge | Where | Why lazy |
|---|---|---|
| `agent_core -.-> ui` | `MimirAgent.chat_loop()` lazily imports `ui.cli.chat_session` | the core exposes a convenience "start the CLI chat" method without depending on a frontend at load time |
| `config -.-> query_engine` | a token-budget helper in `config/constants.py` lazily imports the backend factory | keeps `config` foundational (imported everywhere) while still reaching the backend for token counting |
| `guardrails -.-> tool_execution` | `guardrails/policy/gates.py` lazily imports `tool_execution.validation` | the out-of-workspace gate needs path resolution; `tool_execution.executor` already imports guardrails top-level, so this side is deferred |

Everything else is a straight downward dependency — the layering holds.
