# MIMIR Architecture — Dependency Graph

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The layering of `mimir/client/` and its relation to `mimir/servers/`. Every rule on
this page is checked by `mimir/tests/test_architecture.py`; a change that breaks one
fails the suite.

## The client's layers

Read bottom to top: a unit imports, at load time, only units below it. Solid edges are
import-time; the **dotted edge is lazy** (imported inside a function). Every unit also
imports the foundations (`config`, `protocol`, `event_sink`, `human_pause`) and
`mimir/servers/_shared`; those edges are omitted.

```mermaid
graph TD
    subgraph FE["🖥️ Front-ends · ui/"]
        cli["ui/cli/<br/><small>REPL · CliHost</small>"]
        ws["ui/ws/<br/><small>server/ · connection/ · pool/ · worker/</small>"]
    end
    subgraph SES["💬 Conversations"]
        session["session/<br/><small>conversation · store · journal<br/>jobs_on_disk · summary · commands/</small>"]
    end
    subgraph ORCH["🎛️ Orchestrator"]
        agent["agent_core.py<br/><small>MimirAgent</small>"]
    end
    subgraph ENG["⚙️ Engine"]
        qe["query_engine/<br/><small>turn · loops · dispatch/ · jobs/<br/>history · streaming · backends/</small>"]
    end
    subgraph SVC["🧩 Services"]
        prompt["prompt/<br/><small>system_prompt</small>"]
        ext["extensions/<br/><small>.mimir/ servers · skills · plugins</small>"]
        integ["integration/<br/><small>MCP connect</small>"]
    end
    subgraph EXE["🔧 Execution"]
        te["tool_execution/<br/><small>executor · validation · direct</small>"]
    end
    subgraph GOV["🛡️ Governance"]
        guard["guardrails/<br/><small>policy/ · nudges/ · workflow/<br/>observations/ · ladder · verdict</small>"]
    end
    subgraph CON["📜 Contracts"]
        core["core/<br/><small>ports · settings · human<br/>turn_state · extension_set</small>"]
    end
    subgraph FND["🧱 Foundations"]
        ctx["context/<br/><small>execution_context · capabilities · carry<br/>paths · tool_display · signals</small>"]
        cfg["config/ · protocol/<br/><small>paths · limits · settings · toggles<br/>models · events · messages</small>"]
    end

    cli --> session
    ws --> session
    ws --> agent
    session --> qe
    agent --> qe
    agent --> prompt
    agent --> integ
    agent --> ext
    prompt --> ext
    qe --> te
    qe --> guard
    ext --> te
    ext --> guard
    te --> guard
    guard --> core
    integ --> core
    core --> ctx
    ctx --> cfg
    core -.->|lazy| qe

    classDef fnd fill:#e6faf1,stroke:#1f9d6a,color:#0f3d29;
    classDef con fill:#eef7ee,stroke:#4a9d5b,color:#1e3a24;
    classDef gov fill:#f3ecff,stroke:#8257d1,color:#2e1a52;
    classDef exe fill:#fdf3e7,stroke:#c98a2b,color:#4a300a;
    classDef svc fill:#eef3fb,stroke:#5a7fbf,color:#1c2c48;
    classDef eng fill:#fff7e6,stroke:#d99a2b,color:#5a3d0a;
    classDef orch fill:#e8f0ff,stroke:#3b6fd4,color:#1a2a4a;
    classDef ses fill:#e8f6ff,stroke:#3a8fc4,color:#123a52;
    classDef fe fill:#fde9ec,stroke:#d1526b,color:#521a29;

    class cli,ws fe;
    class session ses;
    class agent orch;
    class qe eng;
    class prompt,ext,integ svc;
    class te exe;
    class guard gov;
    class core con;
    class ctx,cfg fnd;
```

| Rank | Units | What they are |
|---|---|---|
| 0 | `config`, `protocol`, `event_sink`, `human_pause` | the vocabulary (paths, limits, the closed sets of each setting, the models, the toggles), the shapes that cross a boundary (events, chat messages, error payloads), the `emit()` seam |
| 1 | `context` | the `ExecutionContext` blackboard, the capability vocabulary, the carry context, the path rules, how a call reads on screen |
| 2 | `core` | the contracts the layers above agree on: `AgentPort`, `HumanIO`, `TurnState`, `ExtensionSet`, the settings table |
| 3 | `guardrails` | hard `policy` gates, soft `nudges`, the `workflow` state, the `observations` that write the blackboard, the repair `ladder` |
| 4 | `tool_execution` | a call run through the guardrails: policy before, observations after |
| 5 | `extensions`, `integration` | the user's `.mimir/` extensions; connecting MCP servers |
| 6 | `prompt` | the system prompt |
| 7 | `query_engine` | a turn: the loops, the dispatcher, detached jobs, the context budget, the backends |
| 8 | `agent_core` | `MimirAgent`, which every front-end, the runner and the sub-agent server instantiate; it satisfies `AgentPort` |
| 9 | `session` | a conversation, independent of the front-end showing it, and what it keeps on disk |
| 10 | `ui` | the front-ends: the CLI drives an agent in-process; the WebSocket server runs one worker per conversation in a pool |

The one lazy back-edge: `core.settings` reaches the backend factory in `query_engine`
when the backend is switched, because the engine reads the mode through
`core.settings` at load. The WebSocket entry (`ui/ws/server/main.py`) imports
`serving` only after applying `--cwd`, because `config.paths` resolves the workspace
root as it is imported.

## The servers

`mimir/servers/` is a package: each domain (`workspace`, `agent_state`, `hpc`, `proxy`,
…) imports `_shared` and its siblings by package. A bundled server runs as a module
through `servers/_launch.py` (`_shared/launcher.launch_argv`), which keeps the user's
workspace off `sys.path`; a server of `.mimir/servers/` still starts by its path.

- `_shared` never imports the client.
- Only `agent_state` (the sub-agent server builds a `MimirAgent`) and `interaction`
  import the client, and only its public names.

## Rules for every module

- **One home per concept.** The on-disk layout (`_shared/state_paths`), the atomic write
  (`_shared/atomic`), the shape of a tool answer (`_shared/responses`), the clock, the
  pid liveness check, the SKILL.md format: each has one module.
- **No private name crosses a module.** A name another module imports is public; a
  server's private packages (the proxy's `_lib`, `_ops`) are internal to that server.
- **Comments say what the code does now.** History belongs in the commits and
  [CHANGELOG.md](CHANGELOG.md).
- **Tests go through public construction and behaviour.** No test reads the source or
  builds an object around its constructor; shared fixtures live in `mimir/tests/support/`.
