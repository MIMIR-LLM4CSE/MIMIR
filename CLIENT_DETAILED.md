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
- `mimir/client/prompt/`: repository baseline, hardware probe, and system-prompt builders
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

- `resolve_pin_role()` — discovery-pin attachment role.
- `max_tools_for()` — max tools advertised (constant defined in `constants.py`).
- `enforcement_level(model)` → `"strict"` | `"light"` (**default**) | `"off"` — governs **only** the guidance nudge layer (validation, env_resolution, env_cleanup, discovery, doc, state, blast-radius, creation, todo) and the plan-mode evidence gate; never the verification nudges or the safety/approval/state guards. `light` is the default because its membership rule is a criterion rather than a list (costly, hard to detect, non-self-correcting); `strict` is the per-model opt-in for models observed to need the rails. Resolved **once** at construction into `self.enforcement` (the model is immutable per agent), read via `resolve_enforcement(agent)`, overridable per model (`"enforcement"` in `vllm_model_profiles.json`) or at runtime via `/enforcement` (`MimirAgent.set_enforcement`). The surviving categories per `(enforcement, mode)` live in the `_GUIDANCE_BY_LEVEL_MODE` table in `guardrails/nudges/engine.py`.

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

- `ExecutionContext` — the TypedDict (47 fields).
- `execution_context_template()` / `build_execution_context()` / `validate_execution_context()` / `ensure_execution_context()` — create and validate a context.
- `loop_control(ctx)` — lazily attaches a `LoopControlState` dataclass (private `_loop_control` key) holding the six tool-dispatch dedup fields (`write_calls`, `call_fails`, `repeat_warned`, `call_results`, `redundant_warned`, `redundant_call_ids`), kept out of the schema contract so it stays about *semantic* discovery/edit/validation state, not loop plumbing.
- `VALIDATION_TIERS` + `validation_tier()` / `raise_validation_tier()` / `weakest_validation_tier()` / `files_below_tier()` — the **strength** of a validation, on the ladder `syntax` < `static` < `executed` < `oracle`. `validated_files` answers "was it checked?"; `validation_tier_by_file` answers "with what?", because exit code 0 is all bash reports and a vacuous test is indistinguishable from a rigorous one from outside the process. Raised monotonically, retracted wherever `validated_files` is (re-edit, failed check). **Report-only** — it gates nothing and fires no nudge; the completion ledger reads it so the *answer* can state what was actually established. See `POLICY.md` → Validation Policy.
- `VERDICTS` + `unjudged_run_paths()` / `verdict_for()` — the model's reading of what a run's *output* showed (`pass` / `fail` / `unknown`), which the tier ladder cannot express because it grades machine-observed evidence and this is a claim. An execution parks in `unjudged_runs` until that claim arrives; `verdict_by_file` holds the current one (retracted on re-edit, like the tier) and `verdict_attempts_by_file` is the append-only log of failing ones, which deliberately **survives** re-edits since "what was tried and why it did not work" is exactly what a re-edit would erase. See `guardrails/verdict.py`.
- `_FIELD_SPECS` rows are `(name, factory, types, traits)`. The `traits` frozenset (`CARRY` / `FILE_PATH` / `KNOWN_FILE` / `DISCOVERY`) declares what a field *is*, next to the field; `fields_with(*traits)` derives every list that used to be hand-maintained (carry merge, session serialisation, delete purge, discovery signals, `known_existing_files`). `backfill_execution_context()` is the single spec-derived seeder that replaced four `bootstrap_*` helpers. `was_fully_read()` / `is_known_to_exist()` / `was_checked_for()` name the distinctions POLICY.md states, so a call site cannot reach for the wrong set.
- `has_discovery_evidence(ctx, *, min_distinct)` / `discovery_signal_count()` — the **single owner of "what counts as discovery evidence"**, backed by `DISCOVERY_EVIDENCE_SIGNALS` (derived from the `DISCOVERY` trait: `searched` / `read_files` / `snippet_read_files` / `checked_paths` / `inspected_dirs`). `inspected_dirs` counts only **beyond `BASELINE_SEEDED_DIRS`** (`{"."}`, the set `repo_baseline` itself seeds from) so a snapshot can never pre-satisfy a gate, while a subtree the model inspected itself does count — excluding the field outright made structural discovery worth nothing, leaving a grep or a file read as the only way to clear any gate. The discovery nudge, plan-mode evidence gate, and `engine._missing_evidence` all read this one definition.

A module-level **state producer → consumer map** documents each field group's single writer and its readers. Notable fields: `steps_since_last_edit` (reset to 0 on each successful edit, incremented every step) and `declared_edit_set` (paths declared via the plan/todo tool, used to defer validation nudges until the plan is fully written). Two fields feed guidance from the substantive-action stream: `action_op_count` (count of successful `PLAN_BLOCKED` calls — writes/exec/mutations — the alternate op-count trigger for the todo nudge, so a many-operation/few-files task like an optimization loop is still recognised as multi-step) and `edit_fail_streak_by_file` (per-path consecutive edit failures *regardless of patch*, which forces a re-read at 2 and re-arms the `error_recovery` reminder budget once the streak clears). A field-usage census removed four dead fields (`query_id`, `inspected_dir_siblings`, `analysis_announced`, `search_tools_used`).

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
- `SOURCE_FILE_EXTENSIONS` — includes `.cu`/`.cuh` so CUDA is first-class code.

`QUERY_DISCOVERY_SIGNALS` is **composed** from the edit/create/HPC sets plus discovery-only terms, deliberately excluding the pure-theory `QUERY_SCIENCE_SIGNALS` (derive/prove/integrate/cite/theorem) so a math/bibliography query doesn't trigger repository discovery. Signal sets include French tokens alongside English (`améliore`, `modifie`, `fichier`, `arbo`, `conseil`). (The former `FILE_SEARCH_TOOLS`/`CODE_EDIT_TOOLS`/`CODE_VALIDATION_TOOLS` sets are gone — consumers query the live registry.)

#### `resource_context.py`

User-attached context via `@`-mention (Claude/Copilot-style) — **context, not model-invokable tools**. Frontend-agnostic: both the CLI and the WebSocket server call `augment_query_with_resources` to expand a raw user message into an effective query with the referenced content prepended.

- Two attach kinds: **MCP resources** (read-only, URI-addressed data a server exposes — `@memory://all` or the `@memory` name shorthand; registry lives on `agent.resources`, populated at connect time by `integration/server_manager`, reads dispatch via `agent.read_resource`) and **workspace files** (`@src/foo.py`, or a slice `@src/foo.py:10-20` / `@src/foo.py:10`, read locally against the workspace root — no server needed).
- A mention is `@` + a run of non-whitespace (`_MENTION_RE = (?<!\S)@(\S+)`), resolved against the registry then the filesystem; **unknown `@x` tokens are left untouched** so ordinary prose uses of `@` (emails, handles) are never swallowed.
- Whole-file attaches are soft-capped (`_WHOLE_FILE_CHAR_CAP = 20_000`) to protect the context window; an explicit line range is never capped.

The webview mirrors this with an `@` autocomplete dropdown (see [EXTENSION_DETAILED.md](EXTENSION_DETAILED.md)).

### prompt

Self-contained, server-independent context — the system prompt, the repository baseline, and the hardware probe, injected as orientation. (The package is `mimir/client/prompt/`; there is no `discovery/` package — the deterministic pre-plan discovery pipeline that gave it that name was removed.)

#### `system_prompt.py`

Builds the system prompt and the always-injected foundational context.

- `build_base_system_content()` — the system prompt; persona is a **scientific-computing / research engineer** ("From Math, to HPC") with a correctness-then-performance validation hierarchy. `## Validation` splits Tier 1 into **1a — executability** (`py_compile` → lint/types → tests: what those four rungs always actually tested) and **1b — correctness**, required when an edit changes what the code computes: compare against something independent of the code under test, assert the property that defines the requirement rather than a weaker proxy, report results as `key=value` lines so they are recorded rather than claimed, and — the required escape hatch — say so plainly when no oracle is available instead of inventing one. Deliberately kept general; the per-domain technique lives in the on-demand `write-tests` skill so the permanent prompt stays short (`test_context_file.py` guards both the length ceiling and the one-instruction-per-line shape).
- `build_system_content(...)` — assembles the foundational context + memory/todo/plan sections (via `_section()`; `_render_checklist()` renders todo lines). Takes `repo_baseline_context` and `platform_profile_summary` and injects them right after the base prompt, labeled **"orientation only — NOT a substitute for locating exact edit sites."** The scratchpad block follows, **unconditionally**: it states the location (`_scratch_dir_for_prompt` → the session-scoped `scratch_dir`, not the home the standing grant covers) and the two rules that follow from scratch not counting as produced work — nothing throwaway in the workspace, and what runs once does not become a file at all. It used to be appended inside the baseline block, so it disappeared exactly when the workspace was empty or unreadable — which is where a stray probe script is most visible. It no longer argues about how relative paths resolve: file tools reject them outright (`server_files._require_abs`), so the root is there to be *joined*, not reasoned about — see `repo_baseline.py`.
- `build_discovery_pin_block(execution_context, max_files, max_queries)` — builds the compact `[Discovery pin — auto-updated]` block (pinned `read_files`, `search_queries_used`, `planned_edit_targets`, `dirty_written_files`, plus the **live todo checklist** re-read from disk) appended to the system message after every tool dispatch; returns `""` only when *nothing* — including the checklist — is available (`_PIN_MARKER` locates/replaces it in `messages[0]`). Paths are rendered **absolute** via `_pin_path` — display only, `execution_context` keeps storing workspace-relative, so no gate shifts. The pin tells the model to use these paths *directly* and file tools require absolute ones, so showing the stored relative form would hand it a path its next call rejects and force it to rebuild the root — the inference the absolute-path rule removed, reintroduced where the model is most likely to copy blindly. The checklist is built **before** the emptiness test: it is pin-worthy on its own, and it used to be appended after an early `if not lines: return ""`, so a run whose only live state was the checklist (exactly the state just after a plan is approved) silently lost it. Since the copy in the static prompt is a build-time snapshot rebuilt only on a mode/thinking switch, that left no live channel at all — and every mechanism that holds the model to its checklist depends on the model being able to see it.
- `summarize_platform_profile()`, `auto_store_memory()`, `build_tool_catalog_for_planning()`, `summarize_search_matches()`.

The deterministic pre-plan `plan_discovery.py` pipeline (and the `/plan-depth` knob) were **removed**; plan mode now drives its own exploration.

#### `repo_baseline.py`

The in-memory repository baseline (never written to disk).

- `build_repo_baseline_snapshot(root=None, ...)` — a **self-contained `os.walk`** of the workspace (own ignore set `BASELINE_SKIP_DIRS`, mirroring the search server's `_SKIP_DIRS`), so it works whether or not the search server is registered; returns `{"context", "inspected_dirs", "searched", "root"}`. The rendered context opens with `Workspace root (absolute): <root>`, **and the tree's own root line is the absolute path** rather than the basename.

This is a **prerequisite**, not a safeguard: file tools require absolute paths, so the model needs the root to construct any in-workspace destination.

It began as a safeguard and failed twice in that role, which is worth recording. A run asked to create files "outside the codes directory" wrote them into the workspace root and reported the constraint satisfied. Adding only the `Workspace root (absolute):` line left the tree rendered as `codes/ (6 dirs, 18 files)`, and the next run failed the same way — its plan reading "a new directory at the workspace root … outside the existing `codes/` directory", because a bare-name root is indistinguishable from a subdirectory. Rendering the tree root absolutely fixed *that* reading, but the general lesson is the one in `SERVERS_DETAILED.md`: no prompt phrasing makes an inferred root reliable, so the inference was removed instead.
- `seed_execution_context_from_baseline(*, execution_context, baseline)` — seeds only `inspected_dirs` (from the shared `BASELINE_SEEDED_DIRS`, `{"."}`, which the discovery-evidence counter discounts by the same constant so the two cannot drift); does **not** set the global `searched` flag.

Held on `self._repo_baseline` and built **lazily** on the first repo-touching query (`MimirAgent._ensure_repo_baseline`), re-walked only on `/rescan` — so it can drift mid-session; the discovery pin is the live-truth layer that compensates.

#### `platform_probe.py`

- `probe_platform()` — a **self-contained** client-side hardware probe (hostname, CPU via `lscpu`, memory via `/proc/meminfo`, GPU via `nvidia-smi`, toolchains, Slurm); **no benchmarks, no disk**. Cached once per session on `self._platform_profile` (`_ensure_platform_profile`) and rendered by `summarize_platform_profile()`. The HPC platform server stays available for deeper, on-demand queries.

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
- the **denial ladder**: `denial_stage(ctx, scope)` / `worst_denial_stage(ctx)` / `handback_required(ctx)` / `handback_scopes(ctx)`. A refusal carries one of three meanings (wrong means → another route; unnecessary step → drop it and continue; stop → hand back), and these stages take the earlier readings off the table as refusals accumulate on one approval scope, so a wrong first guess cannot become a loop. Counted from `denial_history`, which is append-only *precisely because* `denied_tool_calls` gets cleared when an action later succeeds. Thresholds: `DENIAL_SCOPE_DROP_AFTER` / `DENIAL_SCOPE_HANDBACK_AFTER` / `DENIAL_QUERY_HANDBACK_TOTAL` in `config/constants.py`. See POLICY.md → *If approval is refused*.
- `unchecked_checklist_items(ctx)` — the single reader of live checklist state outside the prompt builder, shared by the completion issues, `needs_incomplete_finalization`, and the unfinished-plan nudge. **Fails closed to `[]`** on a missing/unreadable `todo_file_path`, so a run without a checklist — the majority — behaves exactly as before rather than having an obligation invented for it. Optional steps are tagged, not filtered; callers that must not block on them filter on `item["optional"]`.
- the agent-loop / plan-loop copy, including the loop-control correctives (`repeat_corrective_message()` / `redundant_corrective_message()` / `handback_corrective_message()`) whose *firing* decision lives in `agent_loop.py` / `dispatch.py`.
- `finalize_incomplete_answer()` picks one of three headlines — `Stopped at your request.` (hand-back, risk high), `Task complete, except for what you refused.` (refusal absorbed, everything else done; risk medium, skipped actions listed under *Not performed*), or `Task is incomplete.` (some other blocker). `is_incomplete_answer()` is the predicate for the first two, used by the CLI re-plan offer and the sub-agent `completed` flag; a refusal alone no longer forces the incomplete verdict, since "that step was unnecessary" is a finished task with a named omission, not a failure. `_collect_completion_issues()` splits pending validation into three buckets: budget-exhausted, failing-but-retryable, and fresh-unvalidated, and adds two plan-adherence issues: unchecked checklist steps, and paths declared in the plan but never written. Its "all validated" line is **tier-qualified** (`All modified files validated (weakest evidence: executed)`, governed by the weakest tier across the change) — the bare form was what a model read back as licence to report the work as verified, and the label used to say "highest" while printing the floor.

#### `policy/engine.py`

- `evaluate_tool_preconditions()` — the single call-time entry / orchestrator: registry → external-fetch guard → cluster-submit guard → proxy-exec guard → state guard → write policy → out-of-workspace guard → approval. Wires the module-level guards directly and pulls application `PolicyCheck`s from `PolicyRegistry.active_checks()` (no dependency-injection seam — that plus the former thin `manager.py` facade were folded into this one function).
- built-in gates live in `gates.py`: `_check_external_fetch()`, `_check_cluster_submit()`, `_check_proxy_exec()`, `_check_out_of_workspace_access()` (capability-driven, no hardcoded names).
- `_trusted_read_roots()` — the client mirror of the read roots the servers admit silently (proxy/HPC caches + the central state dir); reads under them never prompt. It shares `servers._shared.trusted_read_roots` with the servers, **plus `constants.STATE_DIR` appended explicitly**: the shared helper resolves the state dir from `MIMIR_STATE_DIR`, and `server_manager` only ever places that variable in the *server subprocesses'* env, so the client process does not carry it. Without the explicit append the agent could not read back its own plans/sessions without a prompt, while the servers — which do see the variable — would have allowed it.
- `_enrich_violation_payload()` — adds `policy_stage`, `state`, `missing_evidence`, `suggested_next_tool_class`. (Per-query tool-list filtering — `tools_for_context`/`cap_tools_by_relevance` — is **not** here; it moved to `query_engine/toollist.py`.)

#### `policy/write.py`

The hard write-policy gate (authoritative rule list in [`POLICY.md`](POLICY.md)).

- `check_write_policy()`, `has_delete_context()`, `write_policy_violation()` — read-before-overwrite, delete evidence, and the anti-thrashing limit on repeated identical failed edits. Query-intent classifiers it used to host (`query_prefers_existing_file_edits()` etc.) now live in `context/signals.py` (shared with nudges + toollist).

#### `observations.py` (at the `guardrails/` root)

The writer of the `execution_context` blackboard, shared by policy **and** nudges — hoisted out of the old `runtime.py` to the guardrails root because it is not itself a gate.

- `record_tool_observation()` — decomposed into ordered `_observe_*` handlers dispatched in a fixed, load-bearing order (pinned by `test_observations.py`). `_observe_edit_outcome` merges edit success + repeated-failure tracking; `_observe_command` classifies each bash segment (`bash_classify`) and credits the blackboard (read/search/inspect/write/env) on success; `_observe_bash_validation` runs status-agnostically and drives validation state, with **three** outcomes: a *checker* (`py_compile`/`ruff`/`mypy`/a compiler) on a dirty file marks it **validated** on exit 0, since its output is a list of problems and an empty one is the finding; an *execution* (`pytest`, `python solver.py`, `./solver`) on exit 0 proves only that the program ended, so the file stays pending and the run is parked in `unjudged_runs` by `_register_unjudged_run` until the model states a verdict; a non-zero exit increments the failure count (returning to `edit`, escaping to `conclude` once the retry budget is exhausted). It records test runs either way, and a leading `cd` rebases the relative operands of later segments in the chain. A green run whose stdout declares its own failing verdict (`check=fail`, `_shared/numerics.observed_failure_verdict`) is treated as the non-zero case and pre-fills a `fail` verdict — never the reverse. It also records **how strongly** the file was checked (`_VALIDATOR_TIER`, keyed on the command head), promoting to the `oracle` tier two ways: when the stdout reports a numerical invariant (`_shared/numerics.observed_invariant_metrics`), or — domain-agnostically — when an `executed`-tier check was seen **failing** on that file earlier in the query and now passes (`executed_failures`, recorded by `_record_executed_failure`; a red whole-project run records every pending file without charging any retry budget). Red→green is what lets non-numerical code reach `oracle` at all. `_observe_tool_run` is the counterpart for execution tools that are *not* bash (`CODE_EXEC` without a `command_prefix` scope: `proxy_exec`, `benchmark_*`) — split by surface, since a shell tool's calls differ in kind call by call while a structured tool's call *is* the execution, which keeps the two mutually exclusive and spares a third parse of the same command. The shared per-path edit slice + post-edit transition live in `_record_code_edit()` / `_enter_post_edit_state()` (reused by the single-edit and `apply_edits` batch paths); a re-edit retracts the tier **and** the verdict, but never the attempt log. On a successful `todo_write` it extracts source-file paths (regex derived from `SOURCE_FILE_EXTENSIONS`, so CUDA/Fortran are covered) into `declared_edit_set`.

#### `verdict.py` (at the `guardrails/` root)

The model's reading of what a run's output showed — the only place a model-authored claim enters the blackboard, and it is labelled as one everywhere it surfaces.

- `parse_verdict()` — the sole parser of a `verdict: pass|fail|unknown — <reason>` line. Generous about what a model writes around it (bullet, bold, any dash or colon, any case), strict about the line shape so prose mentioning the word never fires; the **last** match wins, because a model that reasons its way to a different conclusion meant the later one. Mimir never parses the *program's* output for a pass/fail — output is unbounded and belongs to whoever wrote the code — so a fixed grammar is legitimate only here, on the model's own statement.
- `record_verdict()` — called from `_run_agent_loop` after every assistant message (before the turn is judged terminal, so a verdict stated only in the final answer still counts; from `content` only, never the thinking block). Routes through the *existing* entry points: `pass` → `_mark_file_validated`, `fail` → `_register_validation_failure(..., arms_red_green=False)`, `unknown` → recorded and still pending. There is no second repair ladder, and the red→green opt-out is what stops a self-declared `fail`→`pass` from forging discrimination. Re-arms the `output_verdict` reminder budget once a verdict actually settles something.

#### `policy/state_machine.py`

- `check_state_machine_guard()` — one guard: in `edit`, a file that exhausted its validation retry budget is refused further broad edits. The `validate`-phase branch was removed (see POLICY.md → *Workflow State Machine*); the states still drive nudges and the conclude gate, they just no longer gate edits.
- the validation-retry-budget accounting. The nudge **message builders** it used to host (`validation_nudge_message()`, `denial_nudge_message()`) now live in `nudges/messages.py`, and `finalize_incomplete_answer()` in `guardrails/workflow.py`.

#### `policy/approval.py`

- `ApprovalManager` — `is_sensitive()`, `request()`, `flush_pending_review()`, `record_snapshot()`, `render_prompt()`; supports `batch_mode` (auto-approve + defer to end-of-turn review).

#### `policy/plugins.py`

**Pluggable policy registry.**

- `PolicyCheck(name, check, stage, order)` descriptor + process-global `PolicyRegistry` + `register_policy_check()`. Application checks run at a fixed slot (`pre_mutation` after registry / `pre_approval` after write policy) via `engine._run_extra_checks()`; they can only ADD constraints (a `None` return never relaxes a core gate) and are **locked** (no toggle). See `mimir/client/extensions/`, the examples in `mimir/examples/`, and the authoring guide [`PLUGINS_DETAILED.md`](PLUGINS_DETAILED.md).

#### `nudges/engine.py`

At most one reminder per step. `maybe_append_nudge()` walks the built-in table `_CORE_NUDGES` via the generic runner `_append_core_nudge()` (packs add more via `_append_custom_nudge()`), each fired through the shared `_fire_nudge()` helper (increments the per-category counter, appends the message). Every row is `(name, layer, should_fire, render)`.

- **Verification layer** (runs at every enforcement level): denial (2×, but **uncapped at the `handback` stage** — a reminder to stop that is itself rationed leaves the model going), error_recovery (2×), regression (1× — edited source whose `test_<stem>.py`/`<stem>_test.py` is in the discovered paths but absent from `tests_run`, which `observations._observe_command` fills when a test file is run through bash), unfinished_plan (1× — code was written while the model's own checklist still has open non-optional steps; the message offers two valid exits, do them or say explicitly that a step is out of scope, since a nudge with one acceptable answer is a loop), output_verdict (2× — a run's output was never judged, or a stated `unknown` is still standing; re-armed once a verdict settles the pending runs, so the cap is anti-spam rather than a mute). `test_nudge_table.py` asserts this set is disjoint from `_ALL_GUIDANCE`, so a verification row can never be silently switched off by enforcement.
- **Guidance layer** (skipped entirely when `enforcement_level == "off"`): validation (max 2×), env_resolution, env_cleanup, discovery (max 3×), doc, state, blast_radius, creation, todo (each max 1× unless noted). Validation moved here from the verification layer so it fires at `strict`+`light` but not `off`; it also defers to `output_verdict` (`_pending_needing_a_check`), since a file whose check already ran green is pending for want of a judgement, not another check.
- **Order per step**: core verification → pack verification → (stop if `off`) → core guidance → pack guidance; the first row whose `should_fire` predicate holds wins.
- `_GUIDANCE_BY_LEVEL_MODE` — the declarative `(enforcement, mode)` table (via `_guidance_enabled`): `strict` permits every category in agent mode and every category **except `validation`** in plan mode; `light` permits only `{blast_radius, env_cleanup, validation}` in agent mode and nothing in plan mode. `validation` is agent-only — plan mode is read-only/discovery, so there is no edited code to validate. **Ask mode permits nothing at any level**: it neither plans nor edits, so no guidance category has anything to guard. The same table is reproduced in `POLICY.md` → Enforcement Levels, which is the authority for *why* each level has that membership.
- `needs_incomplete_finalization()` — returns early when all dirty files are already validated (avoids spurious incompleteness flags after clean refactors). **Open non-optional checklist steps are checked first**, ahead of both that shortcut and the budget-exhausted one: each concludes from validation alone, and neither is evidence about steps the model never started (validating the two files it wrote says nothing about the three it did not). Validation/state nudges are deferred while `steps_since_last_edit < 2` or while `declared_edit_set` isn't yet covered by `dirty_written_files`.

Nudge/prompt text refers to tools by **capability/category**, never by literal MCP tool name (the plan/todo tool generically). Validation is the one place shell **commands** are named explicitly (`python -m py_compile` / `pytest` / `ruff` / `mypy`), since those are the bash surface, not MCP tools.

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
| `dispatch.py` | `_dispatch_tool_calls`, `_post_dispatch_inject`, the spin/dedup guards + thresholds |
| `history.py` | context-window budgeting (trim → compact → force-fit), `served_compaction_instruction` |
| `streaming.py` | `_stream_chat` (retry/backoff), `_process_response`, `_to_dict`, and the single `get_backend` handle |
| `background.py` | detached-job detect / register / await + `open_editor` |
| `finalize.py` | `_finalize_answer` / `_persist_answer` / `_annotate_answer_with_changes` (the **verification ledger** — see below) |
| `verification.py` | the ledger itself: `build_ledger` (structured rows + `status`/`files`/`summary`) / `render_ledger` (marker + markdown rows) / `split_answer_ledger` + `parse_ledger_block` (the front-end seam) |

#### `agent_loop.py`

The per-query loop. `run_agent_query()` is a thin orchestrator: shared setup, then **dispatch** to `_run_plan_mode()` (in `plan_loop.py`) or `_run_agent_loop()`; every exit path routes end-of-query bookkeeping through `_finalize_answer()` (in `finalize.py`: annotate answer → persist memory → save carry context → stash full messages).

Completion itself is `if not tool_calls:` — the model emitted no tool call. There is no goal check, so the honesty surface is the **verification ledger** `_annotate_answer_with_changes` appends to every answer: files written with their validation tier (and, at `oracle`, which basis earned it), a domain-neutral "no discrimination observed" line when code was executed but no check was ever seen failing, declared-but-unwritten paths, and unchecked checklist steps. It is machine-recorded and lands *after* the model stops acting, so it cannot loop and cannot be argued with — previously the closing prose and the recorded evidence sat side by side with nothing reconciling them. The block (built in `verification.py`) opens with a `<!--mimir:ledger status=… files=… summary=…-->` marker so a front-end can lift it off the answer and show it **collapsed** — a status line in the webview's `VerificationLedger` panel, a one-liner plus `/ledger` in the CLI — while history keeps the full text for the model. Full format in `POLICY.md` → Final Answer Gating.

- **Setup** — resets `_tool_cache`; fresh `ExecutionContext` per query; lazy baseline (`_ensure_repo_baseline`) then `_apply_carry_context()`; context seeding; system-prompt assembly.
- `_stream_chat()` — iterative streaming backend calls (bounded by the step budget below: `MAX_AGENT_STEPS=100` for non-interactive callers, `AGENT_STEP_SOFT_BUDGET=50` before an interactive front-end asks to continue), retrying transient failures with exponential backoff + jitter (`LLM_RETRY_ATTEMPTS`, cancel-aware). Thinking blocks stream for live display but are **excluded from history** (reasoning is never re-fed; the vLLM non-streaming path routes assistant `content` through the same `<think>` parser). UI events (`status` / `thinking` / `tool_call` / `tool_result` / `diff`) go through `emit()` (see `event_sink.py`).
- `_dispatch_tool_calls()` — dedups `(name, args)` within a step (and across steps for writes), runs reads concurrently via `asyncio.gather` but serializes writes so two edits to the same file (or a read racing a write) can't interleave, wraps each call in `asyncio.wait_for(_TOOL_TIMEOUT_SECS=120)`, and records file targets in `execution_context['tool_msg_files']`.
- **Three cross-step repeat guards** (keyed on `(name, _make_hashable(args))`): (a) **write dedup** collapses an identical write; (b) the **failing-call guard** corrects an identical *failed* non-write call after `SOFT_REPEAT_THRESHOLD=2` and hard-blocks it after `HARD_REPEAT_LIMIT=3` (`LoopControlState.call_fails`); (c) the **redundant-success guard** (`call_results`, keyed also on a SHA-1 of the result) catches a non-write call returning **byte-identical** content — the count resets when content changes — soft at `REDUNDANT_SOFT_THRESHOLD=1`, hard block at `REDUNDANT_HARD_LIMIT=2`, returning a neutral `status:"skipped"` and calling `_strip_redundant_history()` (collapses the identical exchanges down to the first occurrence, preserving tool_call/result pairing so the backend never sees orphans). Correctives are staged (`_repeat_alert` / `_redundant_alert`) and injected next turn by `_post_dispatch_inject()` via `repeat_corrective_message()` / `redundant_corrective_message()`; the loop keeps running after a block.
- **Step budget** — a graceful checkpoint nudge 2 steps before the boundary; interactive front-ends (which set `allow_continue_prompt`) run to `AGENT_STEP_SOFT_BUDGET` then call `agent._request_continue(summary)` to extend by `AGENT_STEP_EXTENSION` (up to `AGENT_STEP_HARD_CEILING`), stopping gracefully on decline; non-interactive callers (sub-agents, tests) keep a fixed budget == `max_steps`.
- **Plan-mode "is a plan recorded"** — `_run_plan_mode` reads `plan_written` (prose document) straight from the execution context, where `observations._observe_todo_flags` sets it by telling the two `TASK_PLANNING` forms apart via the `plan_steps` arg-role. The prose document is the **only** form plan mode produces: the ordered checklist is written after the user approves, at the start of the execution. Which plan-writing tools a read-only mode exposes is decided once, by capability, in `toollist.hidden_planning_tools()` — **ask** hides both `TASK_PLANNING` writers (a question records nothing), **plan** hides only the `plan_steps`-carrying checklist tool — and the same set is re-applied at call time by `filter_readonly_tool_calls`, so a hallucinated call is answered rather than executed. Hiding the tool is what lets both prompts drop the matching "do not write a plan / a checklist" prohibitions: an absent tool needs no rule and no prompt tokens. `PLAN_APPROVED_EXECUTE` carries the instruction to record the checklist at the hand-off; the generic `todo` guidance nudge remains the backstop if the model starts working without one. The loop used to re-derive "is a plan recorded" locally from tool names, counting the checklist alone — a plan recorded in prose was invisible, so it kept telling a model whose plan was on disk that it had "not yet recorded a plan", the model answered by rewriting that document, and the run spun to `max_steps` delivering nothing. `_clear_recorded_plan()` resets both flags on *Rework* / *Other* so a discarded plan cannot be counted as its own replacement.
- **Plan-mode evidence gate** — a plan recorded with zero model-initiated discovery on a repo-touching query is **flagged, not rejected**: a one-shot `PLAN_EVIDENCE_NUDGE` (skipped when `enforcement_level` is `off`) telling the model to either ground the plan or say in its answer that it rests on assumptions. The predicate is the shared `has_discovery_evidence`. It used to reject the plan, which discarded what the model had just recorded with nothing guaranteeing it would submit that form again; the trigger, `query_requires_repo_discovery`, is also a deliberately broad exit filter (see `context/signals.py`) that fires for greenfield work outside the repo, where no exploration could ever satisfy it — too coarse a signal to block on.
- **Plan-mode anti-parroting guard** — a turn that calls tools never reaches the delivery/approval branch, so a model that keeps re-reading or re-writing the recorded plan (and echoing its text back) would loop to `max_steps` and the user would never be asked to approve. After the plan is recorded the model gets `_PLAN_POST_RECORD_TOOL_TURNS` (2) further tool-calling turns; past that its calls are dropped by `_reject_stalled_calls()` (one `role="tool"` reply each, same convention as `filter_readonly_tool_calls`) and the turn falls through to delivery + approval using the prose gathered so far. The drop is **not** conditioned on prose having been emitted — a model stuck in this loop typically emits tool calls and nothing else, which is exactly the shape the guard exists for. The repeated deliver nudge also escalates (`PLAN_DELIVER_ANSWER` → `PLAN_DELIVER_ANSWER_FIRM`) rather than being re-sent verbatim, since the verbatim repeat is part of what the model echoes. Both counters reset on *Rework* / *Other*.
- **Plan approval → agent hand-off** — once a plan is recorded and presented, `_run_plan_mode` calls `_request_plan_decision()` (which reuses the interactive `_request_user_question` prompt: *Accept & start* / *Reject* / *Rework*, plus the front-end's always-present free-text *Other*). **Accept** rewrites `messages[0]` with the agent-mode system prompt, appends `PLAN_APPROVED_EXECUTE`, and hands off to `_run_agent_loop()` in the *same* query so the approved plan runs to completion. **Reject** is a hard stop: `PLAN_REJECTED_STOP` is recorded in history, nothing is executed, and the query returns `PLAN_REJECTED_ANSWER`. **Rework** re-plans from scratch (`PLAN_REWORK_NUDGE`); **Other** folds the free-text feedback in via `plan_revision_nudge()` and re-presents — both loop back to step 1. With no interactive front-end (default shim / sub-agents / tests) the prompt returns an empty selection and plan mode simply delivers the plan as before.
- **Mid-run mode switching** — the mode is a live setting, re-read at the top of every step by `_live_mode()` (in both loops), not a per-query constant. What triggers a switch is a *change* to `agent.mode` since the last observation (tracked in `execution_context['_observed_agent_mode']`, seeded in `run_agent_query`), so an explicit per-query `mode=` override — as passed by sub-agents and the runner — never reads as one. On a change, `_apply_mode_switch()` rebuilds `messages[0]` for the new mode and emits a `status` + `mode` event (the front-end toggle follows), and `_mode_tools()` rebuilds the tool list so a read-only mode's write/exec surface is revoked — or restored — from that step on. This costs the prefix cache for the rest of the query: a deliberate, user-triggered break, reported like the domain re-arm. Because plan mode is a different loop shape, switching **into** plan from the agent loop tail-calls `_run_plan_mode()` and switching **out of** plan tail-calls `_run_agent_loop()`, both carrying the conversation and the evidence gathered so far.
- **Background jobs** — a result from a `BACKGROUNDABLE` tool that carries a `background_job` descriptor is detected by `_detect_background_job()`. On a front-end with a persistent worker (`agent._register_background_job` set by the WS worker), `_maybe_register_background_job()` hands the descriptor to a completion watcher that polls the run's `status_op` off the critical path, then notifies the user and auto-resumes the agent with the `summary_op` result — the model is told to end its turn instead of polling. The CLI, with no worker loop, instead awaits it in-turn via `_await_background_job()`. Both paths are best-effort and name no tool literally (the descriptor carries the read-only ops the watcher calls generically).
- **After each dispatch** — `_trim_tool_history()` evicts the oldest tool results once over the **token** budget (`TOOL_HISTORY_TOKEN_BUDGET`, char fallback; never drops system/user/assistant; protects files in `dirty_written_files | declared_edit_set`; evicting a read invalidates its `read_files` entry so the policy forces a re-read); `_maybe_compact_intra_query()` summarises the middle over `INTRA_QUERY_COMPACT_TOKENS`; `_refresh_system_pin()` rewrites `messages[0]` with `base_system_content + build_discovery_pin_block(...)`.

### tool_execution

Everything around a single tool call — argument normalization, the execution pipeline, per-query caching, post-write validation, and UI status text.

#### `executor.py`

- `execute_tool_call()` — the full pipeline: precondition check → per-query cache lookup (read-only tools) → snapshot → MCP call → observation update → cache store → write-invalidates cached reads for the written path → read-hint injection → continuation-hint injection → auto-validation. Includes smart Python function-boundary range detection for search-result hints; cache-eligibility (`has_cap(name, CACHEABLE, agent.tool_caps)`), search-with-path read hints, and the read-display event all query the per-agent live registry.

#### `formatter.py`

- `normalize_arguments()`, `normalize_tool_content()`, `truncate_text()`, `json_error_payload()`, `parse_tool_payload()`.

#### `normalizer.py`

- `normalize_tool_arguments()` — path normalization for all known path args.
- `rewrite_tool_for_context()` — `read_file` → `read_file_lines` for code/text files.
- `normalize_workspace_path()`.

#### `validation.py`

- `scratch_roots()` / `is_scratch_path(path)` — the client's view of the agent scratchpad (`servers/_shared/state_paths.standing_roots`, i.e. the scratchpad **home** under `<TMPDIR or /tmp>`, taken from `MIMIR_SCRATCH_DIR`; `constants.STATE_DIR` is still passed but only to resolve the active-session subdirectory, because `MIMIR_STATE_DIR` reaches only the server subprocesses). One definition, read by the out-of-workspace gate (scratch never prompts) and by `observations._record_code_edit` (scratch writes never enter `dirty_written_files`). Scratch files are working material, not deliverables — without the second exclusion the scratchpad would trade workspace clutter for ledger clutter and spurious validation obligations.
- `auto_validate_written_file()` — the post-write hook. The deterministic syntax→imports→lint→typecheck→tests **validator ladder was removed**: validating written code now runs through the `bash` server (`python -m py_compile` / `pytest` / `ruff` / `mypy`), steered by the validation guidance nudge rather than run automatically here. What remains are the two *completeness* checks with no bash equivalent: the replacement-completeness grep (leftover `old_text` after a replace) and the cross-file reference check (stale callers after a workspace-wide rename).

#### `tool_status_messages.py`

- `tool_status_message()` — a human-readable status derived generically from the tool *name* (e.g. "Reading file", "Running proxy benchmark", not "Performing agent action…"). No per-tool table: `_humanize_tool_name` locates an action verb from the reusable `_VERBS` lexicon, renders its gerund via `_gerund` (English `-ing` rules + a small irregular map), and appends the remaining name tokens; names with no known verb are plain-humanized.
- `tool_arg_preview` — surfaces the salient argument (the UI `detail` field). Servers wanting exact wording declare a `tool_caps(label=…)` template that `label_for` renders ahead of this fallback.
- `shorten_display_args(name, args, tool_caps)` — a copy of *args* with declared path arguments reduced to their **file name**, for display. Capability-driven off the `path` arg-role, so it needs no tool-name list. Applied to the activity row (`dispatch.py`) and to approval-card headers; the original arguments sent to the tool are never mutated.

  Why: tools carry absolute paths now, which is right for the model and unreadable for a person — a row reading `Reading file: /shared/data1/Projects/.../guardrails/observations.py` buries the one token the user is scanning for. It becomes `Reading file: observations.py`.

  **Never applied where the path is the decision.** An out-of-workspace approval asks the user to authorise a *location*, so the card carries `oow_path` verbatim (the webview renders it as an explicit "outside workspace" line) and the CLI prompt prints the absolute path on its own line under the shortened header. Readability wins in the activity log; precision wins in a consent prompt. `test_tool_row_display.py` pins both halves.

### client root (`agent_core.py`, `human_pause.py`, `event_sink.py`)

The orchestrator and the two seams it hands to a frontend. `agent_core.py` is deliberately **not** under `ui/`: it is the engine the frontends pilot, not a frontend.

#### `agent_core.py`

The `MimirAgent` central class: server lifecycle, mode/settings management, static helper wrappers, `run()`, `cleanup()`, and the per-session capability registry `tool_caps: dict[str, ToolCaps]` (populated by `connect_server`).

- `seed_classification_from_caps()` — called once after all servers connect (from `cli.py`, `ws_server.py`, `server_spawn_agent.py`): re-seeds the `ApprovalManager`'s `sensitive_tools` / `non_batch_tools` / `fallback_tools` **in place** from `self.tool_caps` (the manager is empty pre-connect; in-place mutation preserves the `session_approved_scopes` alias), then `report_capability_consistency()` warns (`unannotated_live_tools`) about connected tools that declared no caps.
- `_is_write_tool()` / `get_tool_file_targets()` — consult the registry.
- `_apply_carry_context()` / `_update_carry_context()` — merge prior-session discovery sets into each new `ExecutionContext` (evicting stale `read_files` via mtime) and save fields back after each query, recording per-file read mtimes (both iterate the shared `_CARRY_SET_FIELDS`).
- `_ensure_repo_baseline(query)` / `_ensure_platform_profile()` — lazy baseline (first repo-touching query) and once-per-session hardware probe.
- `_discard_carry_path()` — removes a deleted path from all carry sets. `_tool_cache` holds per-query read-only results (reset at query start).

#### `human_pause.py` (client root)

The blocking-prompt seam every "ask the human and wait" path shares (approval prompts, the continue-the-run question, plan approval, tool elicitation), so a frontend wires one hook instead of four and a headless run can neutralise all of them at once.

### ui

The **frontends** that drive `MimirAgent` — two independent subpackages, `ui/cli/` and `ui/ws/`, which share nothing.

#### `ui/cli/main.py`

- `main()` — reads `MIMIR_DEFAULT_MODEL` / `--model`, creates `MimirAgent`, connects all servers, runs the chat session, cleans up. `main_sync()` wraps it for the `mimir` console script; the module also carries an `if __name__ == "__main__"` guard, so `python -m mimir.client.ui.cli.main` works from the repo root without installing the package.

#### `ui/cli/chat_session.py`

- `run_chat_session()` — async REPL, slash-command dispatch, `history` maintenance. Refreshes only the (cheap, cached) platform profile at startup; the repo baseline is built lazily on first use, not eagerly scanned.
- `format_ledger_summary()` / `format_ledger_full()` — the verification ledger in a terminal: the answer's ledger block is split off (it stays in `history` for the model) and printed as one status line, expanded on `/ledger`. The same split also drives the write-triggered auto-compact, which used to test for a marker the ledger stopped emitting.

#### `ui/cli/chat_commands.py`

- `handle_chat_command()` — the slash-command table: `/help`, `/status` (shows session-trusted tools), `/mode`, `/rescan`, `/think <depth>`, `/batch`, `/stream`, `/context compact|full`, `/compact`, `/enforcement strict|light|off`, `/nudges`, `/servers`, `/skills`, `/resources`, `/ledger` (expand the last answer's verification ledger), `/undo`, `/trust <tool>` (`approvals.trust_tool()` — session-wide trust), `/untrust <tool>` (`approvals.untrust_tool()` — revoke). (`/plan-depth` was removed with the deterministic plan-discovery pipeline.)

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
- `run_one(task, adapter, model, get_backend_override, enforcement)`: skips up front if any `task.requires` executable is missing (`shutil.which`); else `tempfile.mkdtemp()`s a workspace, runs `task.setup`, `chdir`s into it (the `files`/`search` servers root at `os.getcwd()` at spawn — see `integration/server_manager.connect_server`), spins up a **fresh** `MimirAgent` (so `_repo_baseline`/`_carry_context`/`_tool_cache`/history never leak between tasks); when `enforcement` is passed it overrides the model-profile default via `agent.set_enforcement(...)` so the whole run is graded at one fixed nudge level (`strict`/`light`/`off`) regardless of which model is served, else each model keeps its profile default. Then installs `_install_auto_approve`, connects the task's server set, runs `agent.run(...)` capturing structured events, scores via the adapter, then always `cleanup()` + restores cwd. A per-task crash is recorded as a failed `RunResult` (with traceback) rather than aborting the suite.
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
- `mimir/tests/test_agent_loop.py` — direct coverage for the previously-untested loop functions: `_maybe_compact_intra_query` (compacts the middle when over budget; no-ops under budget / too few messages / no compact_fn), `_post_dispatch_inject` (todo-completion reminder after a successful edit), `_finalize_answer` (file-change annotation + memory persist + carry save + `_last_full_messages`), `_run_plan_mode` (todo_write → deliver-answer flow; `test_replaying_the_plan_forever_is_cut_short` covers the post-record parroting cut-off), and the non-interactive `run_agent_query` path (driven through the real `_stream_chat` → `get_backend()` wrapper with a `ScriptedBackend`, asserting the final answer and that the continue-prompt is never invoked). `RepeatedFailingCallGuardTests` covers all three repeat guards: failing-call soft-warn-then-hard-block, the redundant-success guard (`test_warns_then_blocks_redundant_successful_call` — soft on the 1st repeat, hard block on the 2nd), `test_successful_call_with_changing_content_never_blocks` (changing content resets the count), and `test_hard_block_strips_redundant_history_to_first` (after the block only the first occurrence's assistant+tool pair survives, with intact pairing).
- `mimir/tests/test_runner.py` — covers the headless run engine (`mimir.runner`) model-free via `ScriptedBackend` with a server-less test adapter: `run_benchmark` report shape / counts / `limit`, the `_install_auto_approve` hook (approves any tool incl. non-batch, disables the continue prompt), per-task isolation (distinct workspace + fresh agent per task, no seed-file leakage, cwd restored), and the skip path (a `BenchTask` with an unavailable `requires` is recorded `skipped` without running and never scored).
- `mimir/tests/test_completion_honesty.py` — the end-of-run honesty surface: the verification ledger (per-file tier, which basis earned an `oracle`, the "no discrimination" line plus the guard that it stays domain-neutral, the conditions that suppress it, declared-but-unwritten, unchecked and optional steps, and that the ledger never replaces the model's own answer), the marker contract the front-ends collapse it on (`LedgerMarkerTests`: prose survives the split intact, header fields round-trip, `status` separates clean runs from soft caveats from gaps, and bold marks exactly the rows needing action), the tier-qualified completion sentence and weakest-tier rule, `needs_incomplete_finalization` with and without a checklist, the `unfinished_plan` nudge (firing conditions, cap, both valid exits in the copy), and the checklist reader's fail-closed behaviour + optional-prefix recognition (`optionally sneaky` must **not** parse as optional).
- `mimir/tests/test_absolute_paths.py` — the absolute-path precondition on file tools: every mutating tool rejects a relative path (including each sub-edit of `apply_edits`), the rejection **names the workspace-resolved candidate** so it is self-correcting, nothing is written on rejection, absolute paths still round-trip through all seven tools, the check does not weaken the sandbox (outside paths still refused, scratchpad still writable), and the internal `list_files` helper is unaffected.
- `mimir/tests/test_scratchpad.py` — home resolution (`MIMIR_SCRATCH_DIR` wins, else under `TMPDIR` scoped by uid + workspace id, no directory creation from a sandbox check), the session subdirectory vs the fallback, the standing grant being the home (so a session switch cannot revoke a path), `ensure_scratch_home` on a world-writable `/tmp` (creates `0700`, tightens loose modes, idempotent, refuses a symlink / non-directory / foreign owner / uncreatable parent), the sandbox grant (scratch admitted, workspace admitted, arbitrary outside paths and `<scratch>_evil` siblings still refused, relative paths still workspace-relative), and that scratch writes stay out of `dirty_written_files` and out of the ledger.
- Extended: `test_observations.py` (`ValidationTierTests` — per-validator tiers, oracle promotion from reported invariants, prose/placeholder rejection, monotonicity, retraction on re-edit and on failure, whole-project stamping; `RedGreenDiscriminationTests` — promotion on the whole-suite repair loop, no retry budget charged for an unattributable failure, no promotion when green on the first run or at the syntax tier, and the record surviving the very edit that earns it), `test_bash_coverage.py` (corpus-measured credit rate of the bash→blackboard pipeline plus the frozen blind surface), `test_bash_classify.py` (`NestedCommandParsingTests` — `find -exec` segmentation, terminator handling, the derived `READONLY_NESTED_COMMANDS`, and a **tokenization-invariance guard** over a corpus of `-exec`-free commands, since `parse_segments` is shared by the bash server, the classifier and the out-of-workspace gate), `test_server_contracts.py` (`-exec` policy: read-only nested commands allowed, writes/execs/`-ok`/`-delete`/`-fprint` still refused, nested operands still confined), `test_prefix_cache.py` (the checklist is pinnable alone and still nets to zero), `test_out_of_workspace.py` (scratch never prompts; the grant does not widen to its parent), `test_nudge_table.py` (verification set disjoint from `_ALL_GUIDANCE`).
- Existing suites: `test_capabilities.py` / `test_phase_b_servers.py` (`_golden_caps`), `test_policy_manager.py`, `test_approval.py`, `test_client_helpers.py` (now sources `ScriptedBackend` for its token-counting tests), `test_server_contracts.py`, `test_proxy_helpers.py`, `test_localgit.py`.

---

## VS Code Extension Frontend

The VS Code extension frontend (its layers, the WebSocket message contract, the React
file map, and recipes for extending the UI) now lives in its own reference:
[`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md). On the Python side, the WebSocket
emission paths are `emit()` / `event_sink.py` and the `out_q` drain loop in `ws_server.py`
(see **Package Responsibilities → ui** above).
