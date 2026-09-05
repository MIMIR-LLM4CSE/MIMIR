# MIMIR Client Detailed Reference

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative reference for the client's internal architecture and execution flow —
where each responsibility lives under `mimir/client/`, the main entry points, and the
headless run engine. The VS Code extension frontend is documented separately in
[`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md).

## Canonical Layout

The client code is organized by responsibility under `mimir/client/`.

### Canonical packages
- `mimir/client/config/`: static configuration, model/mode constants, and the toggle preferences store
- `mimir/client/context/`: foundational data substrate — execution-context schema, signals, capabilities, and `@`-mention resource attach
- `mimir/client/prompt/`: system-prompt builders
- `mimir/client/integration/`: server lifecycle (spawn, tool discovery)
- `mimir/client/extensions/`: the single home for workspace-`.mimir/` user extensions — `servers.py` (`.mimir/servers/`), `skills.py` (`.mimir/skills/`), `plugins.py` (`.mimir/plugins/` policy/nudge packs) — plus the plugin-pack authoring surface. Path primitives stay in `config/constants`.
- `mimir/client/guardrails/`: agent-behavior governance — the shared `workflow.py` (state model + predicates + agent-loop copy) and `observations.py` (the execution_context writer), plus two subpackages: `policy/` (hard/blocking preconditions, the state-machine guard + approval) and `nudges/` (soft/advisory reminders + their message copy)
- `mimir/client/query_engine/`: the per-query run loop, decomposed into cohesive modules — `agent_loop.py` (orchestrator; also runs ask mode), `plan_loop.py` (plan mode), `readonly_guard.py` (the call-time write/exec guard shared by the read-only modes), `dispatch.py` (tool-call dispatch + spin/dedup guards), `history.py` (context budgeting), `streaming.py` (model round-trip), `background.py` (detached jobs), `finalize.py` (end-of-query), `toollist.py` (tool-list construction), `backends/`
- `mimir/client/tool_execution/`: tool argument normalization, path rewriting, and post-write validation
- `mimir/client/agent_core.py`: the **`MimirAgent` core** — the central class every frontend, the runner, and the sub-agent server drive (it is *not* UI; it's the engine the UIs pilot)
- `mimir/client/human_pause.py`: the shared blocking-prompt seam (approvals, continue-the-run, plan approval, elicitation), so a frontend wires one hook and a headless run neutralises them all at once
- `mimir/client/ui/`: the **frontends** that drive `MimirAgent`, split into two independent subpackages — `ui/cli/` (`main.py` entrypoint + `chat_session.py` + `chat_commands.py`) and `ui/ws/` (the WebSocket/VS Code bridge: `ws_server.py` / `ws_worker.py` / `ws_session.py` / `_ws_runtime.py` / `session_store.py` / `session_summary.py` (the one-sentence session description shown in the history panel — regenerated after every answered turn, in an executor so it never blocks the event loop nor competes with the query the user is waiting on; a failed generation stores the first user query as a *provisional* description, shown in the panel but always regenerated at the next turn) + the `file_preview.py` render helper). The two share nothing.
- `mimir/client/event_sink.py`: injectable structured-event sink (`emit()`) that decouples the engine from stdout — events route to a bound callback (WS) or print as JSON (CLI / default)
- `mimir/runner/`: headless run engine — the agent's "batch mode" (sibling of `client/`, not under it) — see **Headless Run Engine** below

---

## Main Entry Points

### Canonical class
- `mimir/client/agent_core.py`: concrete `MimirAgent` implementation

### Canonical CLI entrypoint
- `mimir/client/ui/cli/main.py`: async `main()` that builds the agent, connects all servers, starts the chat loop, and cleans up

---

## Purpose

The client orchestrates the local MCP stack:
1. starts MCP servers as child processes over stdio
2. discovers tools and JSON schemas dynamically
3. exposes tools to the selected LLM backend (Ollama or vLLM)
4. routes each tool call to the right server
5. enforces sensitive-tool approval
6. enforces repository discovery and write safety policies
7. enforces workflow-state progression for code edits
8. supports both `agent` and `plan` modes

Behavioral rules remain defined in `POLICY.md`. This file describes structure and execution flow.

---

## Package Responsibilities

Each sub-package under `mimir/client/` owns one responsibility. Behavioral rules stay in
[`POLICY.md`](POLICY.md); this section maps structure and the key functions per file.

### config

Static configuration and tuning knobs — no logic, only constants and per-model / per-mode resolution.

#### `constants.py`

Tunable knobs and the server registry.

- Exports `DEFAULT_MODEL`, `LLM_BACKEND`, `VLLM_BASE_URL`, `VLLM_API_KEY`, `VLLM_MODEL_PROFILES`, `SERVERS`, `SERVER_DESCRIPTIONS`, `VALID_MODES`, `READONLY_MODES`, `SERVER_BASE`, `STATE_DIR`.
- Holds the step / timeout / history / context-budget constants consumed by the agent loop, plus nudge knobs such as `TODO_NUDGE_OP_THRESHOLD` (min successful substantive actions before the todo nudge's op-count trigger fires).
- **Thinking-depth ladder** — `THINKING_DEPTH_LABELS` = `off` · `auto` · `quick` · `medium` · `deep` · `max`, with `THINKING_DEPTH_BUDGETS` = `(-1, -1, 512, 4096, 16384, -1)` (`-1` = no token budget). The default is **`auto`** (`DEFAULT_THINKING_DEPTH = THINKING_DEPTH_AUTO`): thinking is on but *uncapped and self-calibrated* — a prompt directive asks the model to keep the block short on trivial turns and spend a long chain only where the task is genuinely uncertain, rather than a classifier deciding for it. The fixed rungs impose a token budget instead, which the agent loop's per-phase scaling then modulates; `max` is thinking-on and unbudgeted **without** the calibration directive. `clamp_thinking_depth()` / `thinking_depth_from_label()` resolve a level (`on` stays accepted as an alias for `auto`, so the legacy `/think on` keeps working and now means "let the model calibrate"). The depth is **live**: it travels in the backend payload and is picked up per step, so `/think` mid-run lands on the very next one — and `agent_loop._sync_thinking_directive()` rebuilds `messages[0]` only when the run *enters or leaves* the `auto` rung, since that is the only rung whose calibration directive lives in the system prompt (one deliberate prefix-cache miss on an explicit user action, a single comparison in steady state). Sub-agents call `set_thinking_depth(0)`.

#### `models.py`

Per-model and per-mode resolution, read from the matched vLLM profile via `profile_for_model(model)`.

- `max_tools_for()` — max tools advertised (constant defined in `constants.py`).
- `enforcement_level(model)` → `"strict"` | `"light"` (**default**) | `"off"` — governs **only** the guidance nudge layer (validation, env_resolution, env_cleanup, discovery, doc, state, blast-radius, creation, todo) and the plan-mode explore phase; never the verification nudges or the safety/approval/state guards. `light` is the default because its membership rule is a criterion rather than a list (costly, hard to detect, non-self-correcting); `strict` is the per-model opt-in for models observed to need the rails. Resolved **once** at construction into `self.enforcement` (the model is immutable per agent), read via `resolve_enforcement(agent)`, overridable per model (`"enforcement"` in `vllm_model_profiles.json`) or at runtime via `/enforcement` (`MimirAgent.set_enforcement`). The surviving categories per `(enforcement, mode)` live in the `_GUIDANCE_BY_LEVEL_MODE` table in `guardrails/nudges/engine.py`.

#### `preferences.py`

- Load/save of the soft-hide toggles — the disabled server / skill / nudge names, persisted (sorted, atomic) to `<STATE_DIR>/preferences.json`. Only *disabled* names are written, so a new server or skill is visible by default. It is agent **state**, not a user extension, which is why it lives under the central state dir and not in the workspace `.mimir/`.

> **Workspace-`.mimir/` user extensions** (servers / skills / plugins) are **not** in `config/` — they moved to `mimir/client/extensions/` (see below). `config/constants.py` keeps only the path primitives (`MIMIR_DIR`, `resolve_extension_dir`, the `*_DIRNAME`/`*_DIR_ENV` constants).

### extensions

The single home for everything the user drops into the workspace `.mimir/` — env-overridable and fail-open, one module per extension type. Consumed via `from ..extensions import …` (the connect sites `cli.py` / `ws_worker.py` / `runner/engine.py` / `server_spawn_agent.py`, and `agent_core`).

#### `servers.py`

- `all_servers()` — bundled `SERVERS` merged with `discover_user_servers()` (scans `.mimir/servers/server_<name>.py|.js`, env `MIMIR_SERVERS_DIR`; a name colliding with a bundled server is skipped so the core is protected).
- `all_server_descriptions()` — same, for the toggle panel. Kept import-light (only `os` + config constants).

#### `skills.py`

- `resolve_skills_dir()` — `.mimir/skills/` (env `MIMIR_SKILLS_DIR`); the loading itself is `MimirAgent.load_skills(..., merge=True)` (a same-named user skill overrides the bundled one).

#### `plugins.py`

- `load_plugins()` / `resolve_plugins_dir()` — dir-scans `.mimir/plugins/` (env `MIMIR_PLUGINS_DIR`) and imports each pack; a pack self-registers via `register_policy_check` / `register_nudge` as an import side effect. `__init__.py` re-exports the authoring surface (`PolicyCheck` / `NudgeRule` / `register_*`, from `guardrails/`).

Nothing is auto-created in the workspace `.mimir/` — it is the user's alone. Copy-to-customize examples for every extension type live only in [`mimir/examples/`](mimir/examples/) (one `README.md` per type). Tests: `test_user_registry.py`, `test_extensions.py`.

### context

The per-query state schema plus the vocabularies that classify tools and queries — the shared facts every policy and nudge reads.

#### `execution_context.py`

The `ExecutionContext` schema and its per-query lifecycle. Single source of truth: the module-level `_FIELD_SPECS` registry of `(name, default_factory, accepted_types)` tuples generates both the template and the validator; the TypedDict is the static-typing surface (contract test `test_field_specs_match_typeddict_annotations` asserts the two key sets match).

- `ExecutionContext` — the TypedDict (44 fields).
- `execution_context_template()` / `build_execution_context()` / `validate_execution_context()` / `ensure_execution_context()` — create and validate a context.
- `loop_control(ctx)` — lazily attaches a `LoopControlState` dataclass (private `_loop_control` key) holding the five tool-dispatch dedup/spin fields (`write_calls`, `call_fails`, `repeat_warned`, `call_results`, `repeat_noted`), kept out of the schema contract so it stays about *semantic* discovery/edit/validation state, not loop plumbing.
- `VALIDATION_TIERS` + `validation_tier()` / `raise_validation_tier()` / `weakest_validation_tier()` / `files_below_tier()` — the **strength** of a check, on the ladder `structural` < `syntax` < `static` < `compiled` < `measured`. `validated_files` answers "was this file checked?"; `validation_tier_by_file` answers "with what?". Both are about **checkers** only — a parse, a structural scan, an import resolution, a lint, a compile, whose output *is* a list of problems. `structural` is the bottom rung and the one every checked file starts at: the built-in floor (`guardrails/builtin_check.py`) established that the file parses — or, for a language with no stdlib parser, that it is not truncated — with nothing installed, which is why the mandatory axis can no longer be waived for want of a binary. `compiled` is the top rung and the one nothing demands: it needs a toolchain the environment may not have, and a compiler asked for `-fsyntax-only` drops back to `syntax`. Everything above `structural` comes from a checker the model chose to run. Running the code is almost never on this ladder: an execution is a run, judged on its own axis. The one exception is the top rung, `measured`, which a server reaches by running the file itself *and recording which file it ran* — attribution is what normally keeps runs off this axis, and a proxy optimisation session is the one place where it is not a guess. It credits evidence, never correctness. Raised monotonically, retracted wherever `validated_files` is (re-edit, failed check). **Report-only** — it gates nothing and fires no nudge. Only a file that is not readable as text (binary, not UTF-8) is recorded in `unverifiable_files` instead and reported, never demanded. See `POLICY.md` → Validation Policy.
- `VERDICTS` + `record_run()` / `unsettled_runs()` / `failed_runs()` — the other axis. One `runs` entry per execution, holding `completed` (the machine's half: did it reach its end), `verdict` + `reason` (the model's half: what the output showed), and `failures` + `attempts` (the repair history the budget and the hand-back read). A run credits no file and a file's check says nothing about a run, which is the whole point: "it compiles" and "it is right" are different claims, and one word for both is what let a green `pytest` be reported as a verified solver. See `guardrails/verdict.py`.
- `_FIELD_SPECS` rows are `(name, factory, types, traits)`. The `traits` frozenset (`CARRY` / `FILE_PATH` / `KNOWN_FILE` / `DISCOVERY`) declares what a field *is*, next to the field; `fields_with(*traits)` derives every list that used to be hand-maintained (carry merge, session serialisation, delete purge, discovery signals, `known_existing_files`). `backfill_execution_context()` is the single spec-derived seeder that replaced four `bootstrap_*` helpers. `was_read()` / `is_known_to_exist()` / `was_checked_for()` name the distinctions POLICY.md states, so a call site cannot reach for the wrong set. `was_read()` answers "was it read" and deliberately **not** "was it read whole": reads are capped and targeted, so a window that stopped at the cap is the normal case, and a gate demanding the whole file would ask for the one thing the read policy tells the model not to do.
- `has_discovery_evidence(ctx, *, min_distinct)` / `discovery_signal_count()` — the **single owner of "what counts as discovery evidence"**, backed by `DISCOVERY_EVIDENCE_SIGNALS` (derived from the `DISCOVERY` trait: `searched` / `read_files` / `checked_paths` / `inspected_dirs` / `delegated_read_files`). Presence is the whole test: nothing seeds these fields, so a fresh context carries zero evidence and every gate measures the model's own work. (There used to be a discount subtracting the dirs a structural snapshot pre-filled; removing the seeding removed the need for it, and with it the risk that a consumer read the field raw and never applied it — which is exactly what the external-fetch gate did.) The discovery nudge, `plan_evidence_ready()` (the plan-mode explore phase, which adds a floor on `read_files`: locating files is not reading them), and `engine._missing_evidence` all read this one definition, at the same `DISCOVERY_EVIDENCE_MIN_DISTINCT` bar.

A module-level **state producer → consumer map** documents each field group's single writer and its readers. Notable fields: `steps_since_last_edit` (reset to 0 on each successful edit, incremented every step) and `declared_edit_set` (paths declared via the plan/todo tool, used to defer validation nudges until the plan is fully written). Two fields feed guidance from the substantive-action stream: `action_op_count` (count of successful `PLAN_BLOCKED` calls — writes/exec/mutations — the alternate op-count trigger for the todo nudge, so a many-operation/few-files task like an optimization loop is still recognised as multi-step) and `edit_fail_streak_by_file` (per-path consecutive edit failures *regardless of patch*, which re-arms the `error_recovery` reminder budget once the streak clears; it deliberately does **not** evict the file from the read sets — a wrong anchor is not missing content). A field-usage census removed four dead fields (`query_id`, `inspected_dir_siblings`, `analysis_announced`, `search_tools_used`).

#### `capabilities.py`

The **single source of truth for tool *semantics***: **no hardcoded classification lists** — each server declares its tools' caps via `@mcp.tool(**tool_caps(...))` and the client builds the per-agent live registry `agent.tool_caps` in `connect_server`.

- **25 capability flags**, grouped by the policy that consumes each: `READ`, `SEARCH`, `SEARCH_WITH_PATH`, `CANDIDATE_SEARCH`, `INSPECT_DIR`, `CHECK_EXISTENCE`, `CODE_NAV`, `ENV_DISCOVERY`, `CACHEABLE`, `EDIT`, `CONTENT_WRITE`, `OVERWRITE`, `REMOVE`, `REPLACEMENT_TRACK`, `VALIDATE`, `TASK_PLANNING`, `EXTERNAL_FETCH`, `CLUSTER_SUBMIT`, `ENV_MUTATE`, `CODE_EXEC`, `BACKGROUNDABLE`, `SENSITIVE`, `NON_BATCH`, `PLAN_BLOCKED`, `PLAN_READONLY` (each is described in the [capability table in `PLUGINS_DETAILED.md`](PLUGINS_DETAILED.md#tool-capabilities), which is the authoritative list — do not re-enumerate it elsewhere). `VALIDATE` stays declarable for plugin validators, but the first-party stack validates through the `bash` server (no first-party tool carries it). The three reversibility **levels** (`REVERSIBLE` / `RECOVERABLE` / `IRREVERSIBLE`) are a separate vocabulary, not flags: `SENSITIVE` is derived from them.
- `ToolCaps` — the descriptor: capabilities + `arg_roles` + `fallbacks` + status `label`.
- `is_write()` (= `EDIT`∪`CONTENT_WRITE`∪`REMOVE`) / `clears_edit_loop()` (= `READ`∪`VALIDATE`) — derived **helpers, not declared caps**, so a server can't declare write/read caps yet forget the umbrella.
- `infer_tool_caps(tool)` — resolves caps with **3-layer precedence**: `tool.meta["mimir"]` (our descriptor) › `tool.annotations` (standard `readOnlyHint`/`destructiveHint`, the coarse path for foreign servers) › conservative default (empty caps + path-arg inference from the input schema).
- Query helpers — `names_with_cap`, `has_cap`, `path_args`, `arg_role`, `fallbacks`, `validate_tools_ordered`, `label_for`, `unannotated_live_tools`, `readonly_servers()` — take the per-agent registry; **with no registry they resolve to empty (no static fallback)**. `unannotated_live_tools(registry)` powers the connect-time consistency report in `agent_core.py`.

Tests: `_golden_caps.py` (golden sets + `build_declared_registry()` which AST-parses the server decorators), `test_phase_b_servers.py` (declared registry reproduces the golden), `test_capabilities.py` (`infer_tool_caps` precedence + `tool_caps` round-trip), and `test_capability_consumers.py` (drift guard — fails if a declared capability has no live consumer). NB: `config/constants.py` no longer imports this module (avoids a circular import).

#### `signals.py`

Query-intent vocabularies used to route discovery / edit / HPC / science behaviour.

- `QUERY_EDIT_SIGNALS`, `QUERY_CREATE_SIGNALS`, `QUERY_HPC_SIGNALS`, `QUERY_SCIENCE_SIGNALS`, `QUERY_DISCOVERY_SIGNALS`.
- `SOURCE_FILE_EXTENSIONS` — every spelling of every language MIMIR may write (Python, C/C++, CUDA, Fortran incl. `.f03`/`.f08`/`.for`, JVM, JS/TS, Go/Rust/Swift, the shells, Julia/R/MATLAB, HDL). Independent of what this environment has installed: the mandatory check runs in-process (`guardrails/builtin_check.py`), so every extension here is checkable, and this tuple decides only whether an edit is *recorded as produced work* at all. It also carries the structured-data extensions the floor holds a real parser for (`.json`, `.toml`, `.ini`, `.cfg`, `.xml` and dialects) — an exact check that costs nothing — and deliberately not YAML, which has no stdlib parser. It used to be paired with a per-language table of external checker commands — `.f03` was in that table and missing from this tuple, so a Fortran 2003 edit was never even recorded as modified; both the table and the `shutil.which` probe over it are gone.

`QUERY_DISCOVERY_SIGNALS` is **composed** from the edit/create/HPC sets plus discovery-only terms, deliberately excluding the pure-theory `QUERY_SCIENCE_SIGNALS` (derive/prove/integrate/cite/theorem) so a math/bibliography query doesn't trigger repository discovery. Signal sets include French tokens alongside English (`améliore`, `modifie`, `fichier`, `arbo`, `conseil`). (The former `FILE_SEARCH_TOOLS`/`CODE_EDIT_TOOLS`/`CODE_VALIDATION_TOOLS` sets are gone — consumers query the live registry.)

#### `resource_context.py`

User-attached context via `@`-mention (Claude/Copilot-style) — **context, not model-invokable tools**. Frontend-agnostic: both the CLI and the WebSocket server call `augment_query_with_resources` to expand a raw user message into an effective query with the referenced content prepended.

- Two attach kinds: **MCP resources** (read-only, URI-addressed data a server exposes — `@memory://all` or the `@memory` name shorthand; registry lives on `agent.resources`, populated at connect time by `integration/server_manager`, reads dispatch via `agent.read_resource`) and **workspace files** (`@src/foo.py`, or a slice `@src/foo.py:10-20` / `@src/foo.py:10`, read locally against the workspace root — no server needed).
- A mention is `@` + a run of non-whitespace (`_MENTION_RE = (?<!\S)@(\S+)`), resolved against the registry then the filesystem; **unknown `@x` tokens are left untouched** so ordinary prose uses of `@` (emails, handles) are never swallowed.
- Whole-file attaches are soft-capped (`_WHOLE_FILE_CHAR_CAP = 20_000`) to protect the context window; an explicit line range is never capped.

The webview mirrors this with an `@` autocomplete dropdown (see [EXTENSION_DETAILED.md](EXTENSION_DETAILED.md)).

### prompt

System-prompt construction. (The package is `mimir/client/prompt/`; there is no `discovery/` package — the deterministic pre-plan discovery pipeline that gave it that name was removed.)

#### `system_prompt.py`

Builds the system prompt and the dynamic blocks appended to it (paths, memory, todo, plan, mode).

- `build_base_system_content()` — the system prompt, assembled as **doctrine + core**. A resolved `.mimir/system_prompt.md` replaces `_DEFAULT_DOCTRINE_CONTENT` (identity, style, scope, workflow, reasoning) and nothing else; `_CORE_SYSTEM_CONTENT` (non-negotiables, latitude, tool results, discovery, editing, validation, running, planning) is appended after it either way, with no opt-out — a section is core when it states a mechanical fact about MIMIR's own tools, or an obligation the loop checks at runtime, which is why `CoreNudgeCoverageTests` maps every verification-layer nudge to a phrase inside it. It carries one exemption, `unfinished_plan`: the rule it used to map to (*never mark a step done before its output exists*) was cut as too strict for a checklist that tracks progress rather than binding a contract, and the nudge that remains offers both endings — finish the step, or say you are not going to — so it asks for nothing the model must have been told in advance, which is the only thing the invariant protects. `## Planning & todo` sits in core for exactly that reason: `needs_incomplete_finalization` refuses to conclude while a non-optional checklist step is open, so an application prompt that dropped the section would leave the loop blocking on a contract the model was never given. The override goes first so its identity opens the prompt and the hard rules keep the recency slot. Default persona is a **scientific-computing / research engineer** ("From Math, to HPC") with a correctness-then-performance validation hierarchy. `## Validation` splits Tier 1 into **1a — executability** (CHECK, settled by the loop itself and named as such so the model never claims to have run it; then BUILD and RUN, both optional and both stated by capability rather than by binary) and **1b — correctness**, required when an edit changes what the code computes: compare against something independent of the code under test, assert the property that defines the requirement rather than a weaker proxy, report results as `key=value` lines so they are recorded rather than claimed, and — the required escape hatch — say so plainly when no oracle is available instead of inventing one. Deliberately kept general; the per-domain technique lives in the on-demand `write-tests` skill so the permanent prompt stays short (`test_context_file.py` guards both the length ceiling and the one-instruction-per-line shape).
- `build_system_content(...)` — assembles the memory/todo/plan sections (via `_section()`; `_render_checklist()` renders todo lines) on top of the base prompt. The one **unconditional** block is a pair of absolute paths, and nothing else foundational: the **workspace root** (`_workspace_root_for_prompt`, `SEARCH_ROOT` or the cwd) and the **scratchpad** (`_scratch_dir_for_prompt` → the session-scoped `scratch_dir`, not the home the standing grant covers), followed by the two rules that follow from scratch not counting as produced work — nothing throwaway in the workspace, and what runs once does not become a file at all. Neither path touches the disk to resolve, so the prefix stays byte-stable and cacheable.

  The root is a **prerequisite**, not a safeguard: file tools reject relative paths (`server_files._require_abs`), so the model needs it to construct any in-workspace destination — it is there to be *joined*, not reasoned about. It began as a safeguard and failed twice in that role, which is worth recording. A run asked to create files "outside the codes directory" wrote them into the workspace root and reported the constraint satisfied; adding the `Workspace root (absolute):` line to the repo-structure block left the tree rendered as `codes/ (6 dirs, 18 files)`, and the next run failed the same way — its plan reading "a new directory at the workspace root … outside the existing `codes/` directory", because a bare-name root is indistinguishable from a subdirectory. The lesson recorded in `SERVERS_DETAILED.md` is the general one: no prompt phrasing makes an inferred root reliable, so the inference was removed instead. What remains of that history is the line itself, now unconditional — the block that used to carry it was built only for a query classified as repo-touching, leaving the root unstated on exactly the greenfield runs where a misplaced file is least visible.
- `build_checklist_pin_block(execution_context)` — builds the compact `[Task checklist — auto-updated]` block (the **live todo checklist**, re-read from disk) injected as a transient tail message before every model call (`_PIN_MARKER` locates it for removal). Returns `""` when there is no checklist. Since the copy in the static prompt is a build-time snapshot rebuilt only on a mode/thinking switch, the pin is the sole live channel for it — and every mechanism that holds the model to its checklist depends on the model being able to see it. The pin used to also carry discovery evidence (`read_files`, `existing_paths`, `planned_edit_targets`, `dirty_written_files`, and the previous query's writes). That was removed: the paths are already in the transcript, and repeating a bare list of them at the tail of every prompt is a pattern the model **copies** rather than uses — a DeepSeek run looped on the file list until the step budget ran out. Its removal also retired `prev_query_written_files` (the pin was its only reader) and `_pin_path`.
- `auto_store_memory()`, `build_tool_catalog_for_planning()`, `summarize_search_matches()`.

The deterministic pre-plan `plan_discovery.py` pipeline (and the `/plan-depth` knob) were **removed**; plan mode now drives its own exploration.

### integration

The bridge to the MCP servers — spawning them and discovering their tools.

#### `server_manager.py`

- `connect_server()` — spawns the MCP server subprocess, initializes `ClientSession`, discovers tools, registers them in `agent.tool_owner` / `agent.tools`, and builds `agent.tool_caps[name] = infer_tool_caps(tool)` (the per-session capability registry the policy/approval/execution layers consult).

### query_engine.backends

The pluggable LLM backends (Ollama, vLLM) behind one common interface, plus token counting.

#### `base.py`

Shared `LLMBackend` interface (`chat(...) -> dict`) used by the agent loop, plus token counting.

- `count_text_tokens()`, `message_token_counts()`, `count_messages_tokens()` — per-content cache + an `allow_network` flag (so async loops can avoid blocking tokenize calls). The default `_tokenize_text()` is the chars-per-token heuristic (`chars_per_token_for()`), which subclasses may override with an exact tokenizer.

#### `factory.py`

- `get_backend()` — backend selector singleton: reads `LLM_BACKEND` (`ollama`/`vllm`) and returns the adapter. Shared process-wide, so token counts cached in the worker thread are reused by front-end budget checks.

#### `ollama_backend.py`

Ollama adapter (`ollama.chat(...)`) with streaming text/thinking/tool-call collection. Uses the base heuristic for token counts (Ollama exposes no tokenize endpoint); tune per model via `CHARS_PER_TOKEN_BY_MODEL`.

#### `vllm_backend.py`

vLLM OpenAI-compatible adapter: strict OpenAI message normalization for replayed tool-call history, per-model `extra_body` from `VLLM_MODEL_PROFILES` (including `top_k`), and streaming tool-call delta merge by index. Overrides `_tokenize_text()` with an exact count from vLLM's `/tokenize` endpoint (sibling of `/v1`), falling back to the heuristic on any error.

### guardrails

Agent-behavior governance: the shared workflow state model and the execution-context writer at the package root, plus two subpackages — `policy/` (hard, blocking preconditions + the approval gate) and `nudges/` (soft, advisory reminders + their message copy). File paths below are given relative to `mimir/client/guardrails/`.

#### `workflow.py` (at the `guardrails/` root)

Workflow-state constants, transitions, and completion/validation messaging — shared by policy **and** nudges, which is why it sits at the root rather than inside either subpackage.

- `WORKFLOW_STATES`, `VALIDATION_RETRY_BUDGET`.
- `set_workflow_state()`, `pending_validation_paths()`, `has_pending_validation()`, `has_blocking_denials()`.
- the **denial ladder**: `denial_stage(ctx, scope)` / `worst_denial_stage(ctx)` / `handback_required(ctx)` / `handback_scopes(ctx)` / `approval_is_settled(ctx, scope)`. A refusal carries one of three meanings (wrong means → another route; unnecessary step → drop it and continue; stop → hand back), and these stages take the earlier readings off the table as refusals accumulate on one approval scope, so a wrong first guess cannot become a loop. Counted from `denial_history`, which is append-only *precisely because* `denied_tool_calls` gets cleared when an action later succeeds. `approval_is_settled` is the one predicate `policy/engine.py` consults to decline re-prompting: from the second refusal of a scope, or once the query-wide hand-back is reached. Thresholds: `DENIAL_SCOPE_DROP_AFTER` / `DENIAL_SCOPE_HANDBACK_AFTER` / `DENIAL_QUERY_HANDBACK_TOTAL` in `config/constants.py`. See POLICY.md → *If approval is refused*.
- `unchecked_checklist_items(ctx)` — the single reader of live checklist state outside the prompt builder, shared by the completion issues, `needs_incomplete_finalization`, and the unfinished-plan nudge. **Fails closed to `[]`** on a missing/unreadable `todo_file_path`, so a run without a checklist — the majority — behaves exactly as before rather than having an obligation invented for it. Optional steps are tagged, not filtered; callers that must not block on them filter on `item["optional"]`.
- the agent-loop / plan-loop copy, including the loop-control correctives (`repeat_corrective_message()` / `handback_corrective_message()`) whose *firing* decision lives in `agent_loop.py` / `dispatch.py`.
- `finalize_incomplete_answer(answer, ctx, termination)` picks one of three headlines — `Stopped at your request.` (hand-back, risk high), `Task complete, except for what you refused.` (refusal absorbed, everything else done; risk medium, skipped actions listed under *Not performed*), or `Task is incomplete.` (some other blocker). `is_incomplete_answer()` is the predicate for the first two, used by the CLI re-plan offer and the sub-agent `completed` flag; a refusal alone no longer forces the incomplete verdict, since "that step was unnecessary" is a finished task with a named omission, not a failure. `_collect_completion_issues()` splits pending validation into three buckets: budget-exhausted, failing-but-unresolved, and fresh-unvalidated, and adds two plan-adherence issues: unchecked checklist steps, and paths declared in the plan but never written. What it deliberately does **not** collect is a missing verdict: a run nobody judged and a run judged `unknown` go to `unjudged_run_lines()` and print under their own heading, *Ran, with no verdict on record*, exactly as `blocked_run_lines()` does — reported, never counted. They were issue lines once, which let a recommended axis set the `Task is incomplete.` headline and taught the model to produce a label for every command it ran. Residual risk also reads the **run ledger** (`failed_runs()`): a run left red lifts it to medium, while a `blocked` run — a prerequisite this box lacks — does not, since that is a limitation of the environment and never a defect of the change. Its "all validated" line is **tier-qualified** (`All modified files validated (weakest evidence: executed)`, governed by the weakest tier across the change) — the bare form was what a model read back as licence to report the work as verified, and the label used to say "highest" while printing the floor.
- **The report speaks only in the past tense.** `TERMINATION_ANSWERED` / `TERMINATION_STEP_LIMIT` / `TERMINATION_USER_STOPPED` is computed where the loop exits and passed in, instead of being inferred downstream from a retry budget: budget-with-room-left means "the loop would try again" only while the loop runs, and read from the final report it became a promise nobody was going to keep (`Checks failing (will retry)` on the last line of a finished run). It also stops a user stop at the step checkpoint being reported as a step limit.
- `evidence_handback_message(ctx)` — once per query (`evidence_handback_used`), before the report is assembled, the ledger is injected as a user turn so the model rewrites its summary having *seen* it. "Successfully implemented, complete and correct" printed above "Modified files never checked" is a missing fact at the moment the prose is written, not a rhetoric problem to police afterwards; the answer is then quoted under *What the model claims*, subordinate to the machine record above it.

#### `policy/engine.py`

- `evaluate_tool_preconditions()` — the single call-time entry / orchestrator: registry → cluster-submit guard → proxy-exec guard → state guard → write policy → out-of-workspace guard → approval. Wires the module-level guards directly and pulls application `PolicyCheck`s from `PolicyRegistry.active_checks()` (no dependency-injection seam — that plus the former thin `manager.py` facade were folded into this one function).
- built-in gates live in `gates.py`: `_check_cluster_submit()`, `_check_proxy_exec()`, `_check_out_of_workspace_access()` (capability-driven, no hardcoded names).
- `_trusted_read_roots()` — the client mirror of the read roots the servers admit silently (proxy/HPC caches + the central state dir); reads under them never prompt. It shares `servers._shared.trusted_read_roots` with the servers, **plus `constants.STATE_DIR` appended explicitly**: the shared helper resolves the state dir from `MIMIR_STATE_DIR`, and `server_manager` only ever places that variable in the *server subprocesses'* env, so the client process does not carry it. Without the explicit append the agent could not read back its own plans/sessions without a prompt, while the servers — which do see the variable — would have allowed it.
- `_enrich_violation_payload()` — adds `policy_stage`, `state`, `missing_evidence`, `suggested_next_tool_class`. (Per-query tool-list filtering — `tools_for_context`/`cap_tools_by_relevance` — is **not** here; it moved to `query_engine/toollist.py`.)

#### `policy/write.py`

The hard write-policy gate (authoritative rule list in [`POLICY.md`](POLICY.md)).

- `check_write_policy()`, `has_delete_context()`, `write_policy_violation()` — read-before-overwrite, delete evidence, and the anti-thrashing limit on repeated identical failed edits. Query-intent classifiers it used to host (`query_prefers_existing_file_edits()` etc.) now live in `context/signals.py` (shared with nudges + toollist).

#### `observations.py` (at the `guardrails/` root)

The writer of the `execution_context` blackboard, shared by policy **and** nudges — hoisted out of the old `runtime.py` to the guardrails root because it is not itself a gate.

- `record_tool_observation()` — decomposed into ordered `_observe_*` handlers dispatched in a fixed, load-bearing order (pinned by `test_observations.py`). `_observe_edit_outcome` merges edit success + repeated-failure tracking; `_observe_command` classifies each bash segment (`bash_classify`) and credits the blackboard (read/search/inspect/write/env) on success; `_observe_bash_validation` runs status-agnostically and drives **two axes from one command, never mixed** — neither of them the mandatory one, which no command performs any more (`guardrails/builtin_check.py`). A *checker* (`py_compile` → syntax; `ruff`/`mypy`/a compiler → static; but not a *reformat* — `ruff format`, a bare `black` — which rewrites the file and exits 0 regardless) on a dirty file it names marks it **validated** on exit 0 and charges its retry budget on a non-zero one: the tool's output is a list of problems and an empty one is the finding, so nothing is left for anyone to read. An *execution* (`pytest`, `python solver.py`, `./solver`, `python -c …`) validates **no file at all** — `_record_run_outcome` registers it in `runs`, where a non-zero exit is a failure the machine already judged (straight onto the repair ladder via `_register_run_failure`) and a green one owes the model a reading of what it printed. Which file a run exercised is never asked: `python main.py` exercises `mesh.py` without naming it, and every rule for guessing that was a guess. A green run whose stdout declares its own failing verdict (`check=fail`, `_shared/numerics.observed_failure_verdict`) is treated as the non-zero case. A leading `cd` rebases the relative operands of later segments in the chain, and test files are recorded in `tests_run` either way. `_observe_tool_run` is the counterpart for execution tools that are *not* bash (`CODE_EXEC` without a `command_prefix` scope): the call *is* the execution, so it registers a run and nothing else.

#### `verdict.py` (at the `guardrails/` root)

The model's reading of what a run's output showed — the only place a model-authored claim enters the blackboard, and it is labelled as one everywhere it surfaces.

- `apply_verdict()` — the single entry point, called by `_observe_verdict_tool` when the model calls the tool carrying the `judge` capability. The verdict, its reason and the run it names are read through that tool's **declared arg-roles**, so neither the tool's name nor its parameter names appear client-side. Mimir never reads the *program's* output for a pass/fail — output is unbounded and belongs to whoever wrote the code — and the model's own statement now arrives structured, so nothing is parsed at all. Routes through the *existing* entry points: `pass` → `_mark_file_validated`, `fail` → `_register_validation_failure(..., arms_red_green=False)`, `unknown` → recorded, still pending, and the run **stays outstanding** carrying the verdict so the ledger reports it as unresolved rather than as never judged. There is no second repair ladder, and the red→green opt-out is what stops a self-declared `fail`→`pass` from forging discrimination. An `unknown` verdict closes the advisory axis (`exercise_advice_closed`); no other verdict touches a nudge budget, because nothing asks for a verdict any more (see below) and the budget it used to hand back belongs to the recommendation to *run* something. Returns the runs it settled so their rows can be badged in the UI (each run carries the `call_id` it was displayed under). Scope resolution is `_latest`: an unnamed `pass` credits the most recent run, full stop; the rest stay outstanding and are asked about on their own.
- The model is *told when* a verdict is due in two places, neither of them a hardcoded tool name in the prompt: the tool's own docstring (it says when to call it), and the `VERDICT_DUE` line the executor appends to the result of a run that just opened one (`_build_verdict_due_hint`, tool name resolved from the registry). There is deliberately **no turn-end nudge** for it — see *nudges/engine.py* below.

#### `policy/state_machine.py`

- `check_state_machine_guard()` — one guard: in `edit`, a file that exhausted its validation retry budget is refused further broad edits. The `validate`-phase branch was removed (see POLICY.md → *Workflow State Machine*); the states still drive nudges and the conclude gate, they just no longer gate edits.
- the validation-retry-budget accounting. The nudge **message builders** it used to host (`validation_nudge_message()`, `denial_nudge_message()`) now live in `nudges/messages.py`, and `finalize_incomplete_answer()` in `guardrails/workflow.py`.

#### `policy/approval.py`

- `ApprovalManager` — `is_sensitive()`, `request()`, `flush_pending_review()`, `record_snapshot()`, `render_prompt()`; supports `batch_mode` (auto-approve + defer to end-of-turn review) and `approval_mode` / `auto_tools()` / `auto_paths()` (`manual` | `auto` | `auto_all`, session-scoped — see POLICY.md "Approval mode").

#### `policy/plugins.py`

**Pluggable policy registry.**

- `PolicyCheck(name, check, stage, order)` descriptor + process-global `PolicyRegistry` + `register_policy_check()`. Application checks run at a fixed slot (`pre_mutation` after registry / `pre_approval` after write policy) via `engine._run_extra_checks()`; they can only ADD constraints (a `None` return never relaxes a core gate) and are **locked** (no toggle). See `mimir/client/extensions/`, the examples in `mimir/examples/`, and the authoring guide [`PLUGINS_DETAILED.md`](PLUGINS_DETAILED.md).

#### `nudges/engine.py`

At most one reminder per step. `maybe_append_nudge()` walks the built-in table `_CORE_NUDGES` via the generic runner `_append_core_nudge()` (packs add more via `_append_custom_nudge()`), each fired through the shared `_fire_nudge()` helper (increments the counter, appends the message). Every row is `(name, layer, should_fire, render, budget_key)`; `budget_key` defaults to the name and is what lets several rows ration one counter.

- **Verification layer** (runs at every enforcement level): denial (2×, but **uncapped at the `handback` stage** — a reminder to stop that is itself rationed leaves the model going), error_recovery (2×), **validation** (2× — a modified file that was never checked; the one axis `needs_incomplete_finalization` blocks on, hence its place here and ahead of the advisory rows), then the advisory axis sharing **one** budget (`EXERCISE_BUDGET`, `NUDGE_MAX_EXERCISE = 1`): regression (edited source whose `test_<stem>.py`/`<stem>_test.py` is in the discovered paths but absent from `tests_run`), unexercised (everything checked, nothing ever run). unfinished_plan (1×) keeps its own budget. `test_nudge_table.py` asserts this set is disjoint from `_ALL_GUIDANCE`, so a verification row can never be silently switched off by enforcement.
- **The advisory axis is recommended, never required.** Building and running are the two things the environment can refuse — no toolchain, no queue, no dataset, no GPU — so the two rows above ask **once between them**, stay silent when running is visibly out of reach (`_exercise_route`: an unresolved import, no `CODE_EXEC` tool, or no direct command to be found), and go quiet for good once `exercise_advice_closed` is set, which an `unknown` **or `blocked`** verdict does. Separate budgets used to turn one conclusion into several re-prompts. **Silent is not invisible**: the gate records why in `exercise_blocked_reason`, and `build_ledger` prints it beside "nothing here was built or run" — suppressing the ask and suppressing the fact are different things, and only the first was ever wanted.
- **The gate names the route, and the gate is now symmetric.** `_exercise_route` returns the one direct command it can find, ordered by what it proves — a Python test that already covers the edit, a file this box starts directly (runner asked of PATH, never assumed), a suite **already registered** (`CTestTestfile.cmake` + `ctest`, which is what "the test already exists" means for a compiled language: pairing `test_solver.f90` to `solver.f90` would name something that still has to be built), or a build **already configured** (`Makefile`/`CMakeCache.txt` seen this session; `CMakeLists.txt` alone is not a route, because configuring is a step of its own). The suite outranks the build for the reason the tier ladder gives: a build says the code is well formed, only a run produces a result. That last branch used to be refused by category via a `.py`/`.sh` suffix test, which left every compiled change with no recommendation at all — the "judges badly whether to build" half of the problem. Outbound, the gate used to be hard where it was soft inbound: any red exit from a run the model attempted anyway charged `VALIDATION_RETRY_BUDGET`, forced `workflow_state="edit"` and surfaced through `_collect_completion_issues` as `Build failing, unresolved: …`, so trying a *recommended* step and hitting `gcc: command not found` turned a finished task into `Task is incomplete.` — precisely the incentive to force a green run at any cost. A red exit now owes an **imputation**: `blocked` (claimed by the model, or set by the machine for a command that is not installed) charges nothing, steers nothing, and is reported by `blocked_run_lines` as a named limitation.
- **No row asks for a verdict.** `output_verdict` was one and was withdrawn. Its condition — a completed run nobody judged — holds on the ordinary *successful* session, so it fired after the final answer had streamed, discarded it (the webview drops the draft on `nudge_injected`), and typically sent the model back to re-run the command to recover output it no longer had in context, all for a label the ledger already prints. A recommendation must not be able to reject a finished answer; the demand lives in-band on the run's result instead, and the gap is reported by the ledger and the completion report.
- **Guidance layer** (skipped entirely when `enforcement_level == "off"`): env_resolution, env_cleanup, discovery (max 3×), doc, state, blast_radius, creation, todo (each max 1× unless noted).
- **`env_resolution` fires mid-loop, not at the end.** Every other row answers *"is the work done?"*, which is worth asking once the model stops calling tools; this one answers *"why did that just fail?"*, and a step ceiling away from the failure the answer is worth much less — the model has already spent its steps retrying against the interpreter that could never resolve the module, and the generic repeat guard only catches that when the retries are byte-identical. `maybe_inject_env_resolution()` is called from `_post_dispatch_inject()` right after the failing dispatch, with the **same gate and the same single budget** as the table row that still backs it up: whichever fires first spends `nudge_counts["env_resolution"]` and the other stays silent. Only the moment changed, not the policy.
- **Order per step**: core verification → pack verification → (stop if `off`) → core guidance → pack guidance; the first row whose `should_fire` predicate holds wins.
- `_GUIDANCE_BY_LEVEL_MODE` — the declarative `(enforcement, mode)` table (via `_guidance_enabled`): `strict` permits every category, `light` only `{blast_radius, env_cleanup}` in agent mode and nothing in plan mode. **Ask mode permits nothing at any level**: it neither plans nor edits, so no guidance category has anything to guard. The same table is reproduced in `POLICY.md` → Enforcement Levels, which is the authority for *why* each level has that membership.
- `needs_incomplete_finalization()` — blocks on **the check axis and denials only**. **Open non-optional checklist steps are checked first**, ahead of the budget-exhausted shortcut: that shortcut concludes from validation alone, which is no evidence about steps the model never started (validating the two files it wrote says nothing about the three it did not). It reads `workflow_state` nowhere: that condition made the *recommended* axes mandatory, since a failed run or a `fail` verdict sends the state machine back to `edit` and every answer then came back "Task is incomplete" until the run had failed `VALIDATION_RETRY_BUDGET` times. Validation/state nudges are deferred while `steps_since_last_edit < 2` or while `declared_edit_set` isn't yet covered by `dirty_written_files`.

Nudge/prompt text refers to tools by **capability/category**, never by literal MCP tool name (the plan/todo tool generically). Validation names nothing either, since the mandatory check no longer runs on the machine: its nudge reports what `guardrails/builtin_check.py` found.

#### `nudges/messages.py`

The nudge **message copy** — the `render` text for every built-in nudge plus the stateful `validation_nudge_message()` / `denial_nudge_message()` builders. It lives inside the nudges subpackage because only it consumes them.

#### `nudges/plugins.py`

**Pluggable nudge registry.**

- `NudgeRule(name, layer, predicate, render, priority, tiers)` descriptor + process-global `NudgeRegistry` + `register_nudge()`. Consulted via `_append_custom_nudge()` after each core layer (order: core-verification → custom-verification → core-guidance → custom-guidance), preserving at-most-one-per-call. Guidance rules are tier-gated by `rule_tier_enabled()` (default strict-both), suppressed when their name is in `agent.disabled_nudges` (**toggleable** via `/nudges`, persisted in `<STATE_DIR>/preferences.json`), and capped per query. Fires through the shared `_fire_nudge()`.

### query_engine

The heart of a query — the step loop that calls the model, dispatches tools, trims history, and injects nudges.

The per-query loop was split from one ~1750-line module into an orchestrator plus six focused siblings; the behavioral description below is unchanged, only relocated:

| Module | Owns |
|---|---|
| `agent_loop.py` | orchestrator: `run_agent_query`, `_run_agent_loop`, `_advertised_tools`, `_drain_steer`, `_inject_pin`/`_remove_pin`, `_checkpoint_summary` |
| `plan_loop.py` | `_run_plan_mode`, `_request_plan_decision`, `_PLAN_*` labels — tail-calls `_run_agent_loop` (lazy import; the only agent_loop↔plan_loop cycle point) |
| `dispatch.py` | `_dispatch_tool_calls`, `_post_dispatch_inject`, the spin/dedup guards + thresholds; the per-call wall comes from `capabilities.timeout_for` (the tool's declared `timeout_secs`, else the global default) rather than one flat constant |
| `history.py` | context-window budgeting (trim → compact → force-fit), `served_compaction_instruction` |
| `streaming.py` | `_stream_chat` (retry/backoff), `_process_response`, `_to_dict`, and the single `get_backend` handle |
| `background.py` | detached-job detect / register / await + `open_editor` |
| `finalize.py` | `_finalize_answer` / `_persist_answer` / `_annotate_answer_with_changes` (the **verification ledger** — see below) |
| `verification.py` | the ledger itself: `build_ledger` (structured rows + `status`/`files`/`summary`) / `render_ledger` (marker + markdown rows) / `split_answer_ledger` + `parse_ledger_block` (the front-end seam) |

#### `agent_loop.py`

The per-query loop. `run_agent_query()` is a thin orchestrator: shared setup, then **dispatch** to `_run_plan_mode()` (in `plan_loop.py`) or `_run_agent_loop()`; every exit path routes end-of-query bookkeeping through `_finalize_answer()` (in `finalize.py`: annotate answer → persist memory → save carry context → stash full messages).

Completion itself is `if not tool_calls:` — the model emitted no tool call. There is no goal check, so the honesty surface is the **verification ledger** `_annotate_answer_with_changes` appends to every answer: two kinds of row kept apart — files with the check they passed (`checked: static`, or `**not checked**`), and runs with what happened when the code was executed and what the model read in the output — plus a domain-neutral line saying a checker proves nothing about the answer when nothing was ever run, declared-but-unwritten paths, and unchecked checklist steps. It is machine-recorded and lands *after* the model stops acting, so it cannot loop and cannot be argued with — previously the closing prose and the recorded evidence sat side by side with nothing reconciling them. The block (built in `verification.py`) opens with a `<!--mimir:ledger status=… files=… summary=…-->` marker so a front-end can lift it off the answer and show it **collapsed** — a status line in the webview's `VerificationLedger` panel, a one-liner plus `/ledger` in the CLI — while history keeps the full text for the model. Full format in `POLICY.md` → Final Answer Gating.

- **Setup** — resets `_tool_cache`; fresh `ExecutionContext` per query; `_apply_carry_context()`; system-prompt assembly.
- `_stream_chat()` — iterative streaming backend calls (bounded by the step budget below: `MAX_AGENT_STEPS=100` for non-interactive callers, `AGENT_STEP_SOFT_BUDGET=50` before an interactive front-end asks to continue), retrying transient failures with exponential backoff + jitter (`LLM_RETRY_ATTEMPTS`, cancel-aware). Thinking blocks stream for live display but are **excluded from history** (reasoning is never re-fed; the vLLM non-streaming path routes assistant `content` through the same `<think>` parser). UI events (`status` / `thinking` / `tool_call` / `tool_result` / `diff`) go through `emit()` (see `event_sink.py`).
- `_dispatch_tool_calls()` — dedups `(name, args)` within a step (and across steps for writes), runs reads concurrently via `asyncio.gather` but serializes writes so two edits to the same file (or a read racing a write) can't interleave, wraps each call in `asyncio.wait_for(_TOOL_TIMEOUT_SECS=120)`, and records file targets in `execution_context['tool_msg_files']`.
- **Three cross-step repeat mechanisms** (all keyed on `(name, _make_hashable(args))`): (a) **write dedup** collapses an identical write; (b) the **failing-call guard** corrects an identical *failed* non-write call after `SOFT_REPEAT_THRESHOLD=2` and hard-blocks it after `HARD_REPEAT_LIMIT=3` (`LoopControlState.call_fails`), returning a synthetic error so the model gets feedback instead of spinning to the step ceiling. The corrective is staged (`_repeat_alert`) and injected next turn by `_post_dispatch_inject()` via `repeat_corrective_message()`; the loop keeps running after a block. (c) the **identical-success annotation** — see below.
- **`_post_dispatch_inject()` is the mid-tool-loop channel**, and carries the four reminders the end-of-turn nudge table cannot reach because it only fires once the model *stops* calling tools: the todo-completion tick after a successful edit, the repeat corrective above, the `handback` stop once refusals ran the denial ladder out, and `maybe_inject_env_resolution()` — the environment cascade, fired at the call that failed on a missing module rather than a step ceiling later. A model retrying against the wrong interpreter, like one that has been told to hand back, is by definition still calling tools.
- **A repeated *successful* call is annotated, never guarded.** There was a redundant-success guard — result hashing, a soft corrective, a hard block, and `_strip_redundant_history()` rewriting the conversation to keep one copy — and it is gone, along with the cross-query `_persistent_call_fails` counter. A repeated read is now answered by the per-query cache: no round trip, no refusal, no history surgery. The cost it does not spare is context, which the tool-history budget reclaims. What replaced the guard is upstream: a read says what it served and where to resume, so the second identical read has less reason to happen. What the guard could never do without blocking, `IDENTICAL_REPEAT` does: once a call has come back with the *same digest* `IDENTICAL_REPEAT_THRESHOLD` (3) times, the result carries a line saying so (`LoopControlState.call_results` / `repeat_noted`, counted in `_dispatch_tool_calls` beside the failure counter that skips successes). Said once per call key, appended to the real result — the objection that retired the guard was that refusing content sends the model to fetch the same thing another way, and nothing is refused here. It exists because the spin it catches is invisible to everything else in the loop: the failing-call guard counts only failures, and a nudge fires only once the model stops calling tools, which a spinning model never does.
- **Step budget** — a graceful checkpoint nudge 2 steps before the boundary; interactive front-ends (which set `allow_continue_prompt`) run to `AGENT_STEP_SOFT_BUDGET` then call `agent._request_continue(summary)` to extend by `AGENT_STEP_EXTENSION` (up to `AGENT_STEP_HARD_CEILING`), stopping gracefully on decline; non-interactive callers (sub-agents, tests) keep a fixed budget == `max_steps`.
- **Plan-mode "is a plan recorded"** — `_run_plan_mode` reads `plan_written` (prose document) straight from the execution context, where `observations._observe_todo_flags` sets it by telling the two `TASK_PLANNING` forms apart via the `plan_steps` arg-role. The prose document is the **only** form plan mode produces: the ordered checklist is written after the user approves, at the start of the execution. Which plan-writing tools a read-only mode exposes is decided once, by capability, in `toollist.hidden_planning_tools()` — **ask** hides both `TASK_PLANNING` writers (a question records nothing), **plan** hides the `plan_steps`-carrying checklist tool, plus the `plan_title`-carrying document tool while `exploring` — and the same set is re-applied at call time by `filter_readonly_tool_calls`, so a hallucinated call is answered rather than executed. Hiding the tool is what lets both prompts drop the matching "do not write a plan / a checklist" prohibitions: an absent tool needs no rule and no prompt tokens. `PLAN_APPROVED_EXECUTE` carries the instruction to record the checklist at the hand-off; the generic `todo` guidance nudge remains the backstop if the model starts working without one. The loop used to re-derive "is a plan recorded" locally from tool names, counting the checklist alone — a plan recorded in prose was invisible, so it kept telling a model whose plan was on disk that it had "not yet recorded a plan", the model answered by rewriting that document, and the run spun to `max_steps` delivering nothing. `_clear_recorded_plan()` resets both flags on *Rework* / *Other* so a discarded plan cannot be counted as its own replacement.
- **Plan-mode explore phase** — plan mode runs in two phases, and the plan-document tool does not exist during the first. On a repo-touching query (`query_requires_repo_discovery`, skipped when `enforcement_level` is `off`) `tools_for_plan_mode(..., exploring=True)` also hides the `plan_title`-carrying document tool, and the no-plan nudge is `PLAN_EXPLORE_FIRST` — asking for the exploration, not for the plan. The phase flips the moment `plan_evidence_ready(ctx)` holds (`PLAN_EVIDENCE_MIN_FILES_READ` files actually **read**, plus `DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN` distinct signal kinds): the tool list is rebuilt once — the same sanctioned prefix-cache break as a domain re-arm — and the normal record → approve path resumes. This replaces the old after-the-fact advisory gate, which fired *after* the plan was written, needed one distinct signal (a lone `find` cleared it), and only appended `PLAN_EVIDENCE_NUDGE`: it never stood between the model and a plan written over file names. A plan mode that offers the document from turn 1, under a nudge calling the plan mandatory, makes a plan *to explore* the cheapest way out — so the fix withholds the tool rather than policing the plan's wording. Because the arming signal is a broad exit filter that fires for greenfield work no exploration could ground, `PLAN_EXPLORE_MAX_TURNS` unlocks the tool regardless and `PLAN_EXPLORE_BUDGET_SPENT` tells the model to state its gaps: plan mode always reaches a plan. Where a `DELEGATE` capability is connected, phase 1 is a **fan-out**: `PLAN_EXPLORE_DELEGATE` is appended to the nudge and `_DELEGATION_CLAUSE` to the mode's prompt block, both asking for one to three read-only sub-agents issued in a *single* response (issued one per turn they do not run in parallel — the dispatcher gathers what one step emits). What they read comes back as `files_read` and is credited to `delegated_read_files`, which `plan_evidence_ready` counts: otherwise the phase would punish the fan-out it just asked for.
- **Plan-mode anti-parroting guard** — a turn that calls tools never reaches the delivery/approval branch, so a model that keeps re-reading or re-writing the recorded plan (and echoing its text back) would loop to `max_steps` and the user would never be asked to approve. After the plan is recorded the model gets `_PLAN_POST_RECORD_TOOL_TURNS` (2) further tool-calling turns; past that its calls are dropped by `_reject_stalled_calls()` (one `role="tool"` reply each, same convention as `filter_readonly_tool_calls`) and the turn falls through to delivery + approval using the prose gathered so far. The drop is **not** conditioned on prose having been emitted — a model stuck in this loop typically emits tool calls and nothing else, which is exactly the shape the guard exists for. The repeated deliver nudge also escalates (`PLAN_DELIVER_ANSWER` → `PLAN_DELIVER_ANSWER_FIRM`) rather than being re-sent verbatim, since the verbatim repeat is part of what the model echoes. Both counters reset on *Rework* / *Other*.
- **Plan approval → agent hand-off** — once a plan is recorded and presented, `_run_plan_mode` calls `_request_plan_decision()` (which reuses the interactive `_request_user_question` prompt: *Accept & start* / *Reject* / *Rework*, plus the front-end's always-present free-text *Other*). **Accept** rewrites `messages[0]` with the agent-mode system prompt, appends `PLAN_APPROVED_EXECUTE`, and hands off to `_run_agent_loop()` in the *same* query so the approved plan runs to completion. **Reject** is a hard stop: `PLAN_REJECTED_STOP` is recorded in history, nothing is executed, and the query returns `PLAN_REJECTED_ANSWER`. **Rework** re-plans from scratch (`PLAN_REWORK_NUDGE`); **Other** folds the free-text feedback in via `plan_revision_nudge()` and re-presents — both loop back to step 1. With no interactive front-end (default shim / sub-agents / tests) the prompt returns an empty selection and plan mode simply delivers the plan as before.
- **Mid-run mode switching** — the mode is a live setting, re-read at the top of every step by `_live_mode()` (in both loops), not a per-query constant. What triggers a switch is a *change* to `agent.mode` since the last observation (tracked in `execution_context['_observed_agent_mode']`, seeded in `run_agent_query`), so an explicit per-query `mode=` override — as passed by sub-agents and the runner — never reads as one. On a change, `_apply_mode_switch()` rebuilds `messages[0]` for the new mode and emits a `status` + `mode` event (the front-end toggle follows), and `_mode_tools()` rebuilds the tool list so a read-only mode's write/exec surface is revoked — or restored — from that step on. This costs the prefix cache for the rest of the query: a deliberate, user-triggered break, reported like the domain re-arm. Because plan mode is a different loop shape, switching **into** plan from the agent loop tail-calls `_run_plan_mode()` and switching **out of** plan tail-calls `_run_agent_loop()`, both carrying the conversation and the evidence gathered so far.
- **Background jobs** — a result from a `BACKGROUNDABLE` tool that carries a `background_job` descriptor is detected by `_detect_background_job()`. On a front-end with a persistent worker (`agent._register_background_job` set by the WS worker), `_maybe_register_background_job()` hands the descriptor to a completion watcher that polls the run's `status_op` off the critical path, then notifies the user and auto-resumes the agent with the `summary_op` result — the model is told to end its turn instead of polling. The CLI, with no worker loop, instead awaits it in-turn via `_await_background_job()`. Both paths are best-effort and name no tool literally (the descriptor carries the read-only ops the watcher calls generically).
- **After each dispatch** — `_trim_tool_history()` evicts the oldest tool results once over the **token** budget (`TOOL_HISTORY_TOKEN_BUDGET`, char fallback; never drops system/user/assistant; protects files in `dirty_written_files | declared_edit_set`; evicting a read invalidates its `read_files` entry so the policy forces a re-read); `_maybe_compact_intra_query()` summarises the middle over `INTRA_QUERY_COMPACT_TOKENS`. The checklist pin is a transient tail message (`_inject_pin` / `_remove_pin`), so `messages[0]` stays byte-stable across the whole query for prefix caching.

### tool_execution

Everything around a single tool call — argument normalization, the execution pipeline, per-query caching, post-write validation, and UI status text.

#### `executor.py`

- `execute_tool_call()` — the full pipeline: precondition check → per-query cache lookup (read-only tools) → snapshot → MCP call → observation update → cache store → write-invalidates cached reads for the written path → continuation/outline hints → verdict/fork hints → shell-effect report → auto-validation. Cache-eligibility (`has_cap(name, CACHEABLE, agent.tool_caps)`) and the read-display event query the per-agent live registry.
- **Nothing tracks which lines are held.** There was a line-coverage ledger that narrowed a partly-covered read down to the stretch it was missing, credited `grep` hits and lookup excerpts as lines served, and renumbered itself after every edit. It is gone. It duplicated the redundant-success guard in a second, harsher form; it short-circuited that guard, since a read the client answered never reached the result-hash counter; it depended on a shell-command scope parser to know which lines a search had printed; and the one thing it bought — a slid window costing only the lines it added — did not pay for four context fields, an mtime stamp, a diff re-indexer and a crediting path per tool. (The guard it duplicated was itself removed afterwards.) What replaces it is stated at the source instead of inferred at the client: the read tool caps and reports its own window, so the model is told what it got.
- **Continuation and orientation hints** — `_build_continuation_hint()` reads the server's own payload (`truncated` / `total_lines` / `next_start_line` / `line_cap`), not the arguments: reads are clamped by a default window and a per-call cap, so the range asked for is not the range served, and a caller not told the difference cannot tell "this is the file" from "this is its first page". `_build_outline_hint()` adds an `OUTLINE:` symbol map (`name:start-end`) for a truncated read of a code file, obtained through a `CODE_NAV` tool once per file per query and **without** an `execution_context` — the machine's call must not clear a discovery gate on the model's behalf. The **end** of each span is the load-bearing half: with start lines alone the model has no way to ask for "the block around line N" and crawls toward its end a few lines at a time.
- **What a shell command changed** (`tool_execution/bash_effect.py`) — an edit through the file tools returns a diff and the prompt says to check it; a `sed -i` returns an empty line, so the one actor that could catch a bad edit has nothing to look at. `capture()` runs before the dispatch and `report()` after, appending `BASH_EFFECT` with the per-file `+N/-M` and a capped diff body. The trigger is `bash_command_is_readonly` being **false**, not the classified kind: `git checkout -- f.py` and `patch -p1` classify as `unknown` with no operands, and `python fix.py` classifies as `exec` crediting the script rather than what it rewrites. Detection is a `git status`/`git diff --numstat` delta, or a bounded `os.scandir` outside a repo — never a parse of the command, which is the guess the module exists to avoid. `DUPLICATION_SUSPECTED` tests the added lines for a **period**; `created_paths()` feeds the existing `FORK_SUSPECTED`/`PROBE_PLACEMENT` rules, so a `cp x.py x.py.bak` reaches them without a new rule. See POLICY.md → *What a Shell Command Changed*.
- `_path_stamp()` — the `(mtime_ns, size)` every cache entry is stamped with, so a hit can tell "unchanged" from "a `sed -i` moved it under us". The only place a file's identity is checked outside the edit tools.

#### `formatter.py`

- `normalize_arguments()`, `normalize_tool_content()`, `truncate_text()`, `json_error_payload()`, `parse_tool_payload()`.

#### `normalizer.py`

- `normalize_tool_arguments()` — path normalization for all known path args.
- `rewrite_tool_for_context()` — heals the `read_file` alias to `read_file_lines`, and fills in the line range a bare read left out, so every read says what it asks for (the coverage ledger, the repeat guards and the cache key all read the arguments).
- `normalize_workspace_path()`.

#### `validation.py`

- `scratch_roots()` / `is_scratch_path(path)` — the client's view of the agent scratchpad (`servers/_shared/state_paths.standing_roots`, i.e. the scratchpad **home** under `<TMPDIR or /tmp>`, taken from `MIMIR_SCRATCH_DIR`; `constants.STATE_DIR` is still passed but only to resolve the active-session subdirectory, because `MIMIR_STATE_DIR` reaches only the server subprocesses). One definition, read by the out-of-workspace gate (scratch never prompts) and by `observations._record_code_edit` (scratch writes never enter `dirty_written_files`). Scratch files are working material, not deliverables — without the second exclusion the scratchpad would trade workspace clutter for ledger clutter and spurious validation obligations.
- `auto_validate_written_file()` — the post-write hook. The deterministic syntax→imports→lint→typecheck→tests **validator ladder was removed**, and the mandatory check that replaced it does not live here either: it is one sweep at the conclusion gate (`guardrails/builtin_check.sweep_builtin_checks`), so a file edited back and forth is read once rather than once per write. What remains are the two *completeness* checks with no bash equivalent: the replacement-completeness grep (leftover `old_text` after a replace) and the cross-file reference check (stale callers after a workspace-wide rename).

#### `tool_status_messages.py`

- `tool_status_message()` — a human-readable status derived generically from the tool *name* (e.g. "Reading file", "Running proxy benchmark", not "Performing agent action…"). No per-tool table: `_humanize_tool_name` locates an action verb from the reusable `_VERBS` lexicon, renders its gerund via `_gerund` (English `-ing` rules + a small irregular map), and appends the remaining name tokens; names with no known verb are plain-humanized.
- `tool_arg_preview` — surfaces the salient argument (the UI `detail` field). Servers wanting exact wording declare a `tool_caps(label=…)` template that `label_for` renders ahead of this fallback.
- `shorten_display_args(name, args, tool_caps)` — a copy of *args* with declared path arguments reduced to their **file name**, for display. Capability-driven off the `path` arg-role, so it needs no tool-name list. Applied to the activity row (`dispatch.py`) and to approval-card headers; the original arguments sent to the tool are never mutated.

  Why: tools carry absolute paths now, which is right for the model and unreadable for a person — a row reading `Reading file: /shared/data1/Projects/.../guardrails/observations.py` buries the one token the user is scanning for. It becomes `Reading file: observations.py`.

  **Never applied where the path is the decision.** An out-of-workspace approval asks the user to authorise *locations*, so the card carries `oow_paths` verbatim (the webview renders one "outside workspace" row per path) and the CLI prompt prints each absolute path on its own line under the shortened header. Readability wins in the activity log; precision wins in a consent prompt. `test_tool_row_display.py` pins both halves.

### client root (`agent_core.py`, `human_pause.py`, `event_sink.py`)

The orchestrator and the two seams it hands to a frontend. `agent_core.py` is deliberately **not** under `ui/`: it is the engine the frontends pilot, not a frontend.

#### `agent_core.py`

The `MimirAgent` central class: server lifecycle, mode/settings management, static helper wrappers, `run()`, `cleanup()`, and the per-session capability registry `tool_caps: dict[str, ToolCaps]` (populated by `connect_server`).

- `seed_classification_from_caps()` — called once after all servers connect (from `cli.py`, `ws_server.py`, `server_spawn_agent.py`): re-seeds the `ApprovalManager`'s `sensitive_tools` / `non_batch_tools` / `fallback_tools` **in place** from `self.tool_caps` (the manager is empty pre-connect; in-place mutation preserves the `session_approved_scopes` alias), then `report_capability_consistency()` warns (`unannotated_live_tools`) about connected tools that declared no caps.
- `_is_write_tool()` / `get_tool_file_targets()` — consult the registry.
- `_apply_carry_context()` / `_update_carry_context()` — merge prior-session discovery sets into each new `ExecutionContext` (evicting stale `read_files` via mtime) and save fields back after each query, recording per-file read mtimes (both iterate the shared `_CARRY_SET_FIELDS`).
- `_discard_carry_path()` — removes a deleted path from all carry sets. `_tool_cache` holds per-query read-only results (reset at query start).

#### `human_pause.py` (client root)

The blocking-prompt seam every "ask the human and wait" path shares (approval prompts, the continue-the-run question, plan approval, tool elicitation), so a frontend wires one hook instead of four and a headless run can neutralise all of them at once.

### ui

The **frontends** that drive `MimirAgent` — two independent subpackages, `ui/cli/` and `ui/ws/`, which share nothing.

#### `ui/cli/main.py`

- `main()` — reads `MIMIR_DEFAULT_MODEL` / `--model`, creates `MimirAgent`, connects all servers, runs the chat session, cleans up. `main_sync()` wraps it for the `mimir` console script; the module also carries an `if __name__ == "__main__"` guard, so `python -m mimir.client.ui.cli.main` works from the repo root without installing the package.

#### `ui/cli/chat_session.py`

- `run_chat_session()` — async REPL, slash-command dispatch, `history` maintenance. Nothing is probed or scanned at startup: the REPL is ready as soon as the servers are connected.
- `format_ledger_summary()` / `format_ledger_full()` — the verification ledger in a terminal: the answer's ledger block is split off (it stays in `history` for the model) and printed as one status line, expanded on `/ledger`. The same split also drives the write-triggered auto-compact, which used to test for a marker the ledger stopped emitting.

#### `ui/cli/chat_commands.py`

- `handle_chat_command()` — the slash-command table: `/help`, `/status` (shows session-trusted tools), `/mode`, `/think <depth>`, `/batch`, `/stream`, `/context compact|full`, `/compact`, `/enforcement strict|light|off`, `/nudges`, `/servers`, `/skills`, `/resources`, `/ledger` (expand the last answer's verification ledger), `/undo`, `/trust <tool>` (`approvals.trust_tool()` — session-wide trust), `/untrust <tool>` (`approvals.untrust_tool()` — revoke). (`/plan-depth` was removed with the deterministic plan-discovery pipeline.)

#### `ui/ws/`

The WebSocket / VS Code bridge — `ws_server.py`, `ws_worker.py`, `ws_session.py`, `_ws_runtime.py`, `session_store.py`, `session_summary.py`, `file_preview.py`. The message protocol and the React frontend it serves are documented in [`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md).

---

## Headless Run Engine

`mimir/runner/` is the agent's **batch mode**: a library that drives the **non-interactive**
`MimirAgent.run` path (`allow_continue_prompt=False` → fixed `max_steps`, no human prompt) over a list of
tasks and returns a JSON summary. It does **no scoring of its own** — a benchmark supplies the tasks
*and* grades the result. It is architecture-agnostic: servers are spawned with `sys.executable` via
`connect_server`, so it runs under whichever interpreter/arch launches it (ARM or x86).

It is a **library, not a CLI**, and ships **no benchmarks or adapters**. An external integration package
(outside this repo) imports `mimir.runner`, implements the `BenchmarkAdapter` contract over its
benchmark, and calls `run_benchmark(...)`. The two seams an adapter implements: `BenchTask.setup(workspace)`
materialises the task workspace (seed files, or a repo checkout), and `BenchmarkAdapter.score(task, answer,
ctx)` grades the run by inspecting `ctx.workspace` (e.g. compiling/running the solution, or running the
instance's test suite) — never `ctx.agent`. So **`mimir` imports neither the benchmark nor the
integration**; only the integration knows both, keeping the agent decoupled from any benchmark.

#### `types.py`

`BenchTask` (`id`, `query`, `mode`, `max_steps`, `servers`, `setup`, `requires`, `meta`), `CheckContext` (workspace + live agent), and the `BenchmarkAdapter` Protocol (`name`, `load_tasks(limit)`, async `score(task, answer, ctx)`).

#### `engine.py`

The engine:
- `run_one(task, adapter, model, get_backend_override, enforcement)`: skips up front if any `task.requires` executable is missing (`shutil.which`); else `tempfile.mkdtemp()`s a workspace, runs `task.setup`, `chdir`s into it (the `files`/`search` servers root at `os.getcwd()` at spawn — see `integration/server_manager.connect_server`), spins up a **fresh** `MimirAgent` (so `_carry_context`/`_tool_cache`/history never leak between tasks); when `enforcement` is passed it overrides the model-profile default via `agent.set_enforcement(...)` so the whole run is graded at one fixed nudge level (`strict`/`light`/`off`) regardless of which model is served, else each model keeps its profile default. Then installs `_install_auto_approve`, connects the task's server set, runs `agent.run(...)` capturing structured events, scores via the adapter, then always `cleanup()` + restores cwd. A per-task crash is recorded as a failed `RunResult` (with traceback) rather than aborting the suite.
- `run_benchmark(adapter, *, model, backend="vllm", report_path, limit, get_backend_override, enforcement)`: sets `LLM_BACKEND` and clears the factory singleton, runs every task the adapter yields, and builds the JSON summary (`benchmark`, `model`, `backend`, `enforcement`, `total`, `passed`, `skipped`, `pass_rate`, `elapsed_secs`, `results`). `pass_rate` is over **non-skipped** tasks. `get_backend_override` injects a backend factory (e.g. `ScriptedBackend`) for a model-free CI mode. `enforcement` (`strict`/`light`/`off`, default `None`) pins every task to one nudge level so a whole run is graded at a fixed level — the summary records it as `enforcement` (or `"model-default"` when `None`).
- **Unattended approval:** `_install_auto_approve(agent)` replaces `agent._request_tool_approval` with an always-approve shim and sets `allow_continue_prompt=False`. Batch mode already auto-approves write/edit tools, but `non_batch_tools` (code execution / shell) would otherwise block on `input()`; the shim approves *every* tool. **This runs every tool without confirmation — only point the engine at trusted workloads in a sandbox.**
- **Skip support:** `BenchTask.requires: tuple[str,...]` lets a benchmark gate a task on its toolchain; a missing prereq is recorded `RunResult(skipped=True)` without spending an agent run, and `pass_rate` ignores skips — so a behavioral suite is arch-portable (e.g. CUDA tasks skip where `nvcc` is absent).

---

## Test Coverage

Every test is written with **`unittest` + `asyncio.run`** — no pytest-only constructs, no
`conftest.py` — so either runner works on the same files. `pytest` is the documented
command (it is what `pyproject.toml` configures via `[tool.pytest.ini_options]`,
`testpaths = ["mimir/tests"]`, and what [`CONTRIBUTING.md`](CONTRIBUTING.md) asks for
before a PR); `python -m unittest discover mimir/tests` remains available when the `dev`
extra is not installed. Key modules:
- `mimir/tests/_fake_backend.py` — `ScriptedBackend(LLMBackend)`: a deterministic backend that replays canned `chat()` response dicts (one per call), drives the streaming callbacks (thinking → content tokens), records per-call inputs, and supports a custom tokenizer. Shared by the agent-loop tests and the run-engine tests. Pure-Python / dependency-light, so it (and the loop tests below) run on both ARM and x86.
- `mimir/tests/test_agent_loop.py` — direct coverage for the previously-untested loop functions: `_maybe_compact_intra_query` (compacts the middle when over budget; no-ops under budget / too few messages / no compact_fn), `_post_dispatch_inject` (todo-completion reminder after a successful edit), `_finalize_answer` (file-change annotation + memory persist + carry save + `_last_full_messages`), `_run_plan_mode` (todo_write → deliver-answer flow; `test_replaying_the_plan_forever_is_cut_short` covers the post-record parroting cut-off), and the non-interactive `run_agent_query` path (driven through the real `_stream_chat` → `get_backend()` wrapper with a `ScriptedBackend`, asserting the final answer and that the continue-prompt is never invoked). `RepeatedFailingCallGuardTests` covers the failing-call guard: soft-warn on the second identical failure, hard block on the third, and the synthetic error payload the blocked call returns; `IdenticalSuccessRepeatTests` covers its counterpart on the success side — annotated on the third identical digest, said once, silent when the result varies or the arguments differ, and never withholding the call.
- `mimir/tests/test_runner.py` — covers the headless run engine (`mimir.runner`) model-free via `ScriptedBackend` with a server-less test adapter: `run_benchmark` report shape / counts / `limit`, the `_install_auto_approve` hook (approves any tool incl. non-batch, disables the continue prompt), per-task isolation (distinct workspace + fresh agent per task, no seed-file leakage, cwd restored), and the skip path (a `BenchTask` with an unavailable `requires` is recorded `skipped` without running and never scored).
- `mimir/tests/test_completion_honesty.py` — the end-of-run honesty surface: the verification ledger (per-file check tier, the four run states plus the one for a run that never completed, the caveat line plus the guard that it stays domain-neutral, the conditions that suppress it, declared-but-unwritten, unchecked and optional steps, and that the ledger never replaces the model's own answer), the marker contract the front-ends collapse it on (`LedgerMarkerTests`: prose survives the split intact, header fields round-trip, `status` separates clean runs from soft caveats from gaps, and bold marks exactly the rows needing action), the tier-qualified completion sentence and weakest-tier rule, `needs_incomplete_finalization` with and without a checklist, the `unfinished_plan` nudge (firing conditions, cap, both valid exits in the copy), and the checklist reader's fail-closed behaviour + optional-prefix recognition (`optionally sneaky` must **not** parse as optional).
- `mimir/tests/test_absolute_paths.py` — the absolute-path precondition on file tools: every mutating tool rejects a relative path, the rejection **names the workspace-resolved candidate** so it is self-correcting, nothing is written on rejection, absolute paths still round-trip through every tool, the check does not weaken the sandbox (outside paths still refused, scratchpad still writable), and the internal `list_files` helper is unaffected.
- `mimir/tests/test_scratchpad.py` — home resolution (`MIMIR_SCRATCH_DIR` wins, else under `TMPDIR` scoped by uid + workspace id, no directory creation from a sandbox check), the session subdirectory vs the fallback, the standing grant being the home (so a session switch cannot revoke a path), `ensure_scratch_home` on a world-writable `/tmp` (creates `0700`, tightens loose modes, idempotent, refuses a symlink / non-directory / foreign owner / uncreatable parent), the sandbox grant (scratch admitted, workspace admitted, arbitrary outside paths and `<scratch>_evil` siblings still refused, relative paths still workspace-relative), and that scratch writes stay out of `dirty_written_files` and out of the ledger.
- Extended: `test_observations.py` (`ValidationTierTests` — per-validator tiers, red→green promotion, prose/placeholder rejection, monotonicity, retraction on re-edit and on failure, whole-project stamping; `RedGreenDiscriminationTests` — promotion on the whole-suite repair loop, no retry budget charged for an unattributable failure, no promotion when green on the first run or at the syntax tier, and the record surviving the very edit that earns it), `test_bash_coverage.py` (corpus-measured credit rate of the bash→blackboard pipeline plus the frozen blind surface), `test_bash_classify.py` (`NestedCommandParsingTests` — `find -exec` segmentation, terminator handling, the derived `READONLY_NESTED_COMMANDS`, and a **tokenization-invariance guard** over a corpus of `-exec`-free commands, since `parse_segments` is shared by the bash server, the classifier and the out-of-workspace gate), `test_server_contracts.py` (`-exec` policy: read-only nested commands allowed, writes/execs/`-ok`/`-delete`/`-fprint` still refused, nested operands still confined), `test_prefix_cache.py` (the checklist is pinnable alone and still nets to zero), `test_out_of_workspace.py` (scratch never prompts; the grant does not widen to its parent), `test_nudge_table.py` (verification set disjoint from `_ALL_GUIDANCE`), `test_env_resolution.py` (`MidLoopEnvResolutionTests` — the cascade fires at the failing call, spends the budget the end-of-turn row shares, and respects enforcement; `ResolvedEnvironmentRearmsExerciseTests` — a successful execution retracts `unresolved_modules`, so one transient `ModuleNotFoundError` no longer buries the run/verdict advice for the rest of the query).
- Existing suites: `test_capabilities.py` / `test_phase_b_servers.py` (`_golden_caps`), `test_policy_manager.py`, `test_approval.py`, `test_client_helpers.py` (now sources `ScriptedBackend` for its token-counting tests), `test_server_contracts.py`, `test_proxy_helpers.py`.

---

## VS Code Extension Frontend

The VS Code extension frontend (its layers, the WebSocket message contract, the React
file map, and recipes for extending the UI) now lives in its own reference:
[`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md). On the Python side, the WebSocket
emission paths are `emit()` / `event_sink.py` and the `out_q` drain loop in `ws_server.py`
(see **Package Responsibilities → ui** above).
