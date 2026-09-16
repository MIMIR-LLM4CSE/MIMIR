# MIMIR Client Detailed Reference

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

Where each responsibility lives under `mimir/client/`, how a query runs, and what the
headless engine does. Behavioural *rules* live in [`POLICY.md`](POLICY.md); this file is
about structure and flow. The VS Code frontend has its own file,
[`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md).

---

## Quick reference

### Where things live

| Package | Owns |
|---|---|
| `config/` | constants, per-model and per-mode resolution, the toggle preferences store |
| `context/` | the execution-context schema, tool capabilities, query signals, `@`-mention attach |
| `prompt/` | system-prompt construction |
| `extensions/` | everything the user drops in the workspace `.mimir/` — servers, skills, plugin packs |
| `integration/` | server lifecycle: spawn, tool discovery |
| `guardrails/` | behaviour governance — `policy/` blocks, `nudges/` advises, the root holds what both read |
| `query_engine/` | the per-query loop, plus `backends/` for the LLM adapters |
| `tool_execution/` | argument normalization, path rewriting, post-write checks, result formatting |
| `ui/` | the frontends — `ui/cli/` and `ui/ws/`, which share nothing |
| `agent_core.py` | **`MimirAgent`**, the engine every frontend pilots. Not UI |
| `human_pause.py` | the one blocking-prompt seam (approvals, plan approval, elicitation) |
| `event_sink.py` | `emit()` — structured events to a bound callback (WS) or JSON on stdout (CLI) |
| `mimir/runner/` | the headless batch engine. A sibling of `client/`, not under it |

### The per-query loop

One orchestrator plus focused siblings, all under `query_engine/`:

| Module | Owns |
|---|---|
| `agent_loop.py` | orchestrator: `run_agent_query`, `_run_agent_loop`, `_advertised_tools`, `_drain_steer`, `_sync_checklist` |
| `plan_loop.py` | `_run_plan_mode`, `_request_plan_decision`, the `_PLAN_*` labels. Tail-calls `_run_agent_loop` (lazy import — the only cycle point) |
| `readonly_guard.py` | `filter_readonly_tool_calls` — the call-time write/exec guard the read-only modes share |
| `dispatch.py` | `_dispatch_tool_calls`, `_post_dispatch_inject`, the spin and dedup guards |
| `history.py` | context budgeting: trim → compact → force-fit → repair, plus the compaction marker |
| `streaming.py` | `_stream_chat` (retry/backoff), `_process_response`, the single `get_backend` handle |
| `toollist.py` | per-query tool-list construction |
| `background.py` | detached jobs: detect, register, await, `open_editor` |
| `finalize.py` | `_finalize_answer` / `_persist_answer` / `_annotate_answer_with_changes` |
| `verification.py` | the verification ledger: `build_ledger` / `render_ledger` / `split_answer_ledger` |

### Slash commands

The CLI table is `ui/cli/chat_commands.handle_chat_command`. The webview has its own,
separate handler.

| Command | Does |
|---|---|
| `/help`, `/status` | usage; current model, mode, depth, approvals, trusted tools |
| `/mode [agent\|plan\|ask]` | switch session mode |
| `/think <depth>` | `off` · `auto` · `quick` · `medium` · `deep` · `max` |
| `/enforcement strict\|light\|off` | the guidance-nudge dial ([POLICY.md](POLICY.md#nudges-by-enforcement-level)) |
| `/approvals manual\|auto\|all` | who answers the approval cards ([POLICY.md](POLICY.md#policy-gates-by-approval-mode)) |
| `/batch on\|off` | queue write approvals until the end of the turn |
| `/trust <tool>`, `/untrust <tool>` | session-wide trust for one tool |
| `/context compact\|full`, `/compact` | window policy, and compact now |
| `/ledger` | expand the last answer's verification ledger |
| `/undo` | revert the last write |
| `/memory` | the memory store |
| `/servers`, `/skills`, `/nudges` | list or toggle, persisted in `preferences.json` |
| `/resources` | what `@`-mention can attach |
| `/modules` | module-catalogue status, `refresh`, or a search term. Status never builds |
| `/proxy clean <name>` | delete a proxy's runs, state and snapshots, and say what it removed |
| `/stream` | toggle token streaming |

### Key constants

All in `config/constants.py` unless noted.

| Constant | Value | Bounds |
|---|---|---|
| `TOOL_CALL_TIMEOUT_SECS` | 120 | default per-call wall; a tool may declare its own |
| `TOOL_CALL_TIMEOUT_MAX_SECS` | 1200 | the clamp on a tool-declared wall |
| `VALIDATION_RETRY_BUDGET` | 5 | failures of one file, or one command, before release |
| `REPEATED_EDIT_FAILURE_LIMIT` | 2 | identical failed edits before the write gate refuses |
| `IDENTICAL_REPEAT_THRESHOLD` | 3 | identical successful results before `IDENTICAL_REPEAT` |
| `DISCOVERY_EVIDENCE_MIN_DISTINCT` | 2 | distinct signals clearing the `discover` gate |
| `PLAN_EVIDENCE_MIN_FILES_READ` | 1 | files read before plan mode offers the document tool |
| `PLAN_EXPLORE_MAX_TURNS` | 8 | after which the document tool unlocks regardless |
| `LLM_RETRY_ATTEMPTS` | 3 | transient backend failures retried with backoff |

---

## Purpose

The client orchestrates the local MCP stack:

1. starts MCP servers as child processes over stdio;
2. discovers their tools and JSON schemas at connect time;
3. exposes those tools to the selected backend (vLLM, Ray Serve, Ollama, Anthropic);
4. routes each call to the owning server;
5. applies the policy gates and the nudge cascade;
6. runs in `agent`, `plan` or `ask` mode.

---

## config

Constants and per-model resolution. No logic.

### `constants.py`

The tuning knobs and the bundled server registry. Exports `DEFAULT_MODEL`, `LLM_BACKEND`,
the vLLM endpoint settings, `SERVERS`, `SERVER_DESCRIPTIONS`, `VALID_MODES`,
`READONLY_MODES`, `SERVER_BASE`, `STATE_DIR`, plus the step, timeout, history and nudge
constants listed [above](#key-constants).

**The thinking-depth ladder.** `THINKING_DEPTH_LABELS` is `off · auto · quick · medium ·
deep · max`, with budgets `(-1, -1, 512, 4096, 16384, -1)` where `-1` means no token
budget. The default is **`auto`**: thinking is on, uncapped, and self-calibrated — a prompt
directive asks the model to keep the block short on a trivial turn and spend a long chain
only where the task is genuinely uncertain, rather than a classifier deciding for it. The
fixed rungs impose a budget instead, which the loop's per-phase scaling then modulates;
`max` is on and unbudgeted *without* the calibration directive.

The depth is **live**: it travels in the backend payload and is read per step, so `/think`
mid-run lands on the very next one. `agent_loop._sync_thinking_directive()` rebuilds
`messages[0]` only when the run enters or leaves the `auto` rung, since that is the only
rung whose directive lives in the system prompt — one deliberate prefix-cache miss on an
explicit user action, and a single comparison in steady state. `on` stays accepted as an
alias for `auto`, so a legacy `/think on` keeps working. Sub-agents run at depth 0.

### `models.py`

Per-model and per-mode resolution, read from the matched vLLM profile.

`enforcement_level(model)` → `strict` | `light` (**default**) | `off`. It governs **only**
the guidance nudge layer and the plan-mode explore phase — never the verification nudges,
and never the approval, write or state guards. Which categories survive at each
`(enforcement, mode)` is the `_GUIDANCE_BY_LEVEL_MODE` table in `guardrails/nudges/engine.py`;
[POLICY.md](POLICY.md#nudges-by-enforcement-level) reproduces it with the reasoning.

Resolved **once** at construction into `self.enforcement` — the model is immutable for an
agent's life — and read through `resolve_enforcement(agent)`. A profile opts up with
`"enforcement": "strict"`; `/enforcement` overrides at runtime.

### `preferences.py`

Load and save of the soft-hide toggles: the disabled server, skill and nudge names,
written sorted and atomically to `<STATE_DIR>/preferences.json`. Only *disabled* names are
stored, so a newly added server or skill is visible by default. It is agent **state**, not
a user extension, which is why it lives under the state dir rather than the workspace
`.mimir/`.

---

## extensions

The single home for everything the user drops into the workspace `.mimir/`. One module per
extension type, env-overridable, fail-open. `config/constants.py` keeps only the path
primitives (`MIMIR_DIR`, `resolve_extension_dir`, the `*_DIRNAME` / `*_DIR_ENV` constants).

| Module | Provides |
|---|---|
| `servers.py` | `all_servers()` — bundled `SERVERS` merged with `discover_user_servers()` (scans `.mimir/servers/server_<name>.py\|.js`, env `MIMIR_SERVERS_DIR`). A name colliding with a bundled server is skipped, so the core is protected. `all_server_descriptions()` does the same for the toggle panel |
| `skills.py` | `resolve_skills_dir()` — `.mimir/skills/`, env `MIMIR_SKILLS_DIR`. Loading itself is `MimirAgent.load_skills(..., merge=True)`, where a same-named user skill overrides the bundled one |
| `plugins.py` | `load_plugins()` / `resolve_plugins_dir()` — scans `.mimir/plugins/`, env `MIMIR_PLUGINS_DIR`, and imports each pack. A pack self-registers through `register_policy_check` / `register_nudge` as an import side effect |

Nothing is auto-created in `.mimir/` — it is the user's alone. Copy-to-customize examples
live in [`mimir/examples/`](mimir/examples/), one `README.md` per type.

---

## context

The per-query state schema, the tool vocabulary, and the `@`-mention attach.

### `execution_context.py`

The `ExecutionContext` schema and its lifecycle. One source of truth: the module-level
`_FIELD_SPECS` registry of `(name, factory, types, traits)` rows generates both the
template and the validator, and the TypedDict is the static-typing surface. A contract test
asserts the two key sets match. **47 fields** today.

- `execution_context_template()` / `build_execution_context()` / `validate_execution_context()`
  / `ensure_execution_context()` — create and validate.
- `backfill_execution_context()` — the single spec-derived seeder. It replaced four
  `bootstrap_*` helpers that each seeded their own module's subset, one of which defaulted
  `steps_since_last_edit` to `99` where everything else used `0`, so an absent field meant
  "maximally idle" to one reader and "just edited" to another.
- `loop_control(ctx)` — lazily attaches a `LoopControlState` dataclass under a private key,
  holding the five dispatch dedup/spin fields. Kept out of the schema so the contract stays
  about *semantic* state, not loop plumbing.

**Field traits.** The `traits` frozenset on each row declares what a field *is* — `CARRY`,
`FILE_PATH`, `KNOWN_FILE`, `DISCOVERY` — and `fields_with(*traits)` derives every list that
used to be hand-maintained: the carry merge, session serialisation, the delete purge, the
discovery signals, `known_existing_files`. Eight such lists lived across four modules and
had already drifted apart.

**Named predicates, not bare set membership.** `was_read()` / `is_known_to_exist()` /
`was_checked_for()` name the distinctions [POLICY.md](POLICY.md#what-each-field-means)
states, so a call site cannot reach for the wrong set. `was_read()` answers "was it read"
and deliberately *not* "was it read whole": reads are capped and targeted, so a window that
stopped at the cap is the normal case.

**Two axes, never mixed.**

- `validated_files` + `validation_tier_by_file` — the **check** axis. The tier ladder is
  `structural` < `syntax` < `static` < `compiled` < `measured`, raised monotonically and
  retracted wherever `validated_files` is, since evidence is about one revision. Every
  checked file starts at `structural`, which the in-process floor establishes with nothing
  installed — that is why the mandatory axis can no longer be waived for want of a binary.
  `compiled` needs a toolchain and is demanded nowhere; `measured` has one route, a server
  that ran the file *and recorded which file it ran*. Report-only: it gates nothing and
  fires no nudge. A file not readable as text lands in `unverifiable_files` instead.
- `runs` + `VERDICTS` — the **run** axis, with `record_run()` / `unsettled_runs()` /
  `failed_runs()`. One entry per execution holding `completed` (the machine's half),
  `verdict` + `reason` (the model's half) and `failures` + `attempts` (the repair history).
  A run credits no file, and a file's check says nothing about a run. See
  [POLICY.md → Verdicts](POLICY.md#verdicts) for what each verdict reaches.

**Discovery evidence.** `has_discovery_evidence(ctx, *, min_distinct)` /
`discovery_signal_count()` own the definition, backed by `DISCOVERY_EVIDENCE_SIGNALS`
(derived from the `DISCOVERY` trait: `searched`, `inspected_dirs`, `checked_paths`,
`read_files`, `delegated_read_files`). Presence is the whole test — nothing seeds these
fields, so a fresh context carries zero evidence.

Two consumers, two bars. `engine._missing_evidence` holds the `discover`-state gate to
`DISCOVERY_EVIDENCE_MIN_DISTINCT`. `plan_evidence_ready()` reads the same signal set
against `DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN` **plus** a floor on files actually read,
because distinct signal *kinds* do not express its bar: listing a directory and running a
find scores two while grounding nothing.

**Notable fields.** `steps_since_last_edit` (reset on each successful edit, incremented
every step). `declared_edit_set` (paths scraped from the checklist's own step text;
**replaced** by each new checklist, never accumulated, so a revised plan retracts what it
dropped). `action_op_count` (successful `PLAN_BLOCKED` calls — the op-count trigger that
lets a many-operations/few-files task still read as multi-step). `edit_fail_streak_by_file`
(per-path consecutive edit failures *regardless of patch*, which drives the
`error_recovery` reminder; it deliberately does not evict the file from the read sets — a
wrong anchor is not missing content).

A module-level **producer → consumer map** documents each field group's single writer and
its readers.

### `capabilities.py`

The single source of truth for tool *semantics*, and there are **no hardcoded
classification lists**: each server declares its tools' caps with
`@mcp.tool(**tool_caps(...))`, and `connect_server` builds the per-agent registry
`agent.tool_caps`.

**27 capability flags** today. The authoritative list with one line each is the
[capability table in `PLUGINS_DETAILED.md`](PLUGINS_DETAILED.md#tool-capabilities) — it is
deliberately not re-enumerated here, because two copies drift. Separate from the flags, the
three **reversibility levels** (`REVERSIBLE` / `RECOVERABLE` / `IRREVERSIBLE`) are their own
vocabulary; `SENSITIVE` is derived from them rather than declared beside them.

- `ToolCaps` — the descriptor. Besides `name` and `capabilities` it carries `arg_roles`,
  `fallbacks`, the status `label`, the approval `scope` spec, `risk_note`, `preview`,
  `reversibility`, `timeout_secs`, `readonly_when` and `run_outcome`. The last four are
  what let the loop read a per-tool decision instead of holding a name-keyed table.
- `is_write()` (= `EDIT` ∪ `CONTENT_WRITE` ∪ `REMOVE`) and `clears_edit_loop()` (= `READ` ∪
  `VALIDATE`) are derived **helpers, not declared caps**, so a server cannot declare the
  parts and forget the umbrella.
- `infer_tool_caps(tool)` resolves with three-layer precedence: our descriptor in
  `tool.meta["mimir"]`, then the standard `annotations` (`readOnlyHint` / `destructiveHint`,
  the coarse path for a foreign server), then a conservative default of empty caps plus
  path-arg inference from the input schema.
- Query helpers — `names_with_cap`, `has_cap`, `path_args`, `arg_role`, `fallbacks`,
  `label_for`, `timeout_for`, `scope_spec`, `name_for_cap`, `unannotated_live_tools` — all
  take the per-agent registry, and **with no registry they resolve to empty**. There is no
  static fallback, deliberately.

The registry is strictly per-agent, because `spawn_agent` runs sub-agents concurrently with
a subset of servers.

### `signals.py`

Query-signal vocabularies. **One predicate still reads them**:
`query_requires_repo_discovery`, the plan-mode explore phase's coarse exit filter. The
`query_prefers_*` and `query_is_informational` classifiers were removed with the nudge
conditions that consumed them — a keyword match over a natural-language request guesses at
intent, and a nudge has to rest on something checkable.

- `QUERY_EDIT_SIGNALS`, `QUERY_CREATE_SIGNALS`, `QUERY_HPC_SIGNALS` survive **only** as
  ingredients of `QUERY_DISCOVERY_SIGNALS`, which is composed from all three plus
  discovery-only terms. Pure-theory terms (`derive`, `prove`, `cite`, `theorem`) are absent
  by construction, so a maths or bibliography query is not forced to scan the repository.
  Signal sets carry French tokens alongside English (`améliore`, `fichier`, `arbo`).
- `SOURCE_FILE_EXTENSIONS` — every spelling of every language MIMIR may write, independent
  of what this machine has installed, because the mandatory check runs in-process. This
  tuple decides only whether an edit is *recorded as produced work*. It also carries the
  structured-data extensions the floor holds a real parser for (`.json`, `.toml`, `.ini`,
  `.cfg`, `.xml` and dialects), and deliberately not YAML, which has no stdlib parser. It
  used to be paired with a per-language table of external checker commands — `.f03` was in
  that table and missing from this tuple, so a Fortran 2003 edit was never even recorded as
  modified.

### `resource_context.py`

User-attached context via `@`-mention — **context, not model-invokable tools**. Both
frontends call `augment_query_with_resources` to expand a raw message into an effective
query with the referenced content prepended.

- Two attach kinds: **MCP resources** (read-only, URI-addressed data a server exposes —
  `@memory://all`, or the `@memory` shorthand; the registry is populated at connect time)
  and **workspace files** (`@src/foo.py`, or a slice `@src/foo.py:10-20`, read locally — no
  server needed).
- A mention is `@` plus a run of non-whitespace, resolved against the registry then the
  filesystem. **Unknown `@x` tokens are left untouched**, so ordinary prose uses of `@` are
  never swallowed.
- Whole-file attaches are soft-capped to protect the window; an explicit line range never is.

---

## prompt

### `system_prompt.py`

Builds the system prompt and the dynamic blocks appended to it.

**`build_base_system_content()` — doctrine + core.** A resolved `.mimir/system_prompt.md`
replaces `_DEFAULT_DOCTRINE_CONTENT` (identity, style, scope, workflow, reasoning) and
nothing else. `_CORE_SYSTEM_CONTENT` is appended after it either way, with no opt-out. A
section is core when it states a mechanical fact about MIMIR's own tools, or an obligation
the loop checks at runtime — which is why `CoreNudgeCoverageTests` maps every
verification-layer nudge to a phrase inside it.

The override goes first so its identity opens the prompt and the hard rules keep the
recency slot. `## Planning & todo` is core for a concrete reason: the loop refuses to
conclude while a non-optional checklist step is open, so an application prompt that dropped
that section would leave the loop blocking on a contract the model was never given.

The default persona is a scientific-computing / research engineer, with a
correctness-then-performance validation hierarchy. `## Validation` splits tier 1 into
**1a — executability** (the check, settled by the loop itself and named as such so the model
never claims to have run it; then build and run, both optional, both stated by capability
rather than by binary) and **1b — correctness**, required when an edit changes what the code
computes: compare against something independent of the code under test, assert the property
that defines the requirement rather than a weaker proxy, report results as `key=value` lines
so they are recorded rather than claimed, and — the required escape hatch — say so plainly
when no oracle is available instead of inventing one. Kept deliberately general; per-domain
technique lives in the on-demand `write-tests` skill so the permanent prompt stays short.
`test_context_file.py` guards both the length ceiling and the one-instruction-per-line shape.

**`build_system_content(...)`** assembles the memory, todo and plan sections on top of the
base. The one unconditional block is a pair of absolute paths — the **workspace root** and
the **scratchpad** — followed by the two rules that follow from scratch not counting as
produced work: nothing throwaway in the workspace, and what runs once does not become a file
at all. Neither path touches the disk to resolve, so the prefix stays byte-stable and
cacheable.

The root is a **prerequisite, not a safeguard**: file tools reject relative paths, so the
model needs it to construct any in-workspace destination. It is there to be *joined*, not
reasoned about. It began as a safeguard and failed twice in that role. A run asked to create
files "outside the codes directory" wrote them into the workspace root and reported the
constraint satisfied; adding a `Workspace root (absolute):` line left the tree still
rendered as `codes/`, and the next run failed the same way, its plan reading "a new
directory at the workspace root … outside the existing `codes/` directory" — a bare-name
root is indistinguishable from a subdirectory. The general lesson: no phrasing makes an
inferred root reliable, so the inference was removed instead.

**The live checklist, and where state may not sit.** `build_system_content` renders the
checklist into `messages[0]` (agent mode only) and `_sync_checklist` refreshes it there
whenever the todo file's mtime moves *and* the rebuilt prompt actually differs.

> **The invariant: a block of STATE never occupies the last position of the prompt.**

Every chat template appends its generation prompt after the last message, so whatever sits
there is what the model is asked to respond to or continue — and a checklist has nothing to
answer. The symptom is template-dependent and the failure is not: one template continued the
block's own text until the run was stopped, another emitted a short reasoning block then
EOS. Measured on one backend: **37 empty turns / 108 draws** with the block in the tail,
**0 / 84** without, **0 / 40** with the same text in `messages[0]`. The numbers say where it
was quantified, not where it applies.

The corollary is the sorting rule: **pilotage** — nudges, reminders, the empty-turn retry —
belongs last, because that is its function, and it measured harmless there. **State** does
not. The placement is unconditional; nothing tests the model or the template. It *removes* a
per-model workaround: the block was a tail `user` turn specifically because a tail `system`
turn broke one template's generation prompt, a distinction that measured irrelevant to the
real failure (7/40 vs 8/40).

The block used to also carry discovery evidence — read files, existing paths, planned
targets. That was removed: the paths are already in the transcript, and repeating a bare
list of them is a pattern the model **copies** rather than uses.

**`_rebuild_system_content(agent, active_mode, execution_context)`** is the single answer to
"what must `messages[0]` be after a rebuild": the mode's prompt plus the folded skill block.
Five sites rebuild it — mode switch, thinking-rung change, plan→agent handoff, checklist
refresh, initial build — and before this existed the skill block was silently dropped by
every one of them but the last.

---

## integration

### `server_manager.py`

`connect_server()` spawns the MCP server subprocess, initialises the `ClientSession`,
discovers its tools, registers them in `agent.tool_owner` / `agent.tools`, and builds
`agent.tool_caps[name] = infer_tool_caps(tool)` — the per-session registry every other layer
queries.

---

## query_engine.backends

The pluggable LLM backends behind one interface, plus token counting.

| Module | Is |
|---|---|
| `base.py` | the `LLMBackend` interface (`chat(...) -> dict`) and token counting |
| `factory.py` | `get_backend()` — the process-wide selector singleton |
| `ollama_backend.py` | Ollama adapter with streaming text / thinking / tool-call collection |
| `vllm_backend.py` | vLLM OpenAI-compatible adapter |
| `ray_backend.py` | `RayBackend(VllmBackend)` pointed at the Ray Serve router |

- **Token counting** (`base.py`): `count_text_tokens()`, `message_token_counts()`,
  `count_messages_tokens()`, with a per-content cache and an `allow_network` flag so an
  async loop can avoid a blocking tokenize call. The default `_tokenize_text()` is a
  chars-per-token heuristic that a subclass may override with an exact tokenizer.
- `served_models()` reports the model ids an endpoint exposes, `[]` where a backend cannot
  enumerate itself. It is what lets the WS server resolve an unspecified model without
  testing which backend is active.
- `get_backend()` reads `LLM_BACKEND` (`vllm` / `ray` / `anthropic` / `ollama`). Note the
  `else` arm is Ollama, so an unrecognised name resolves there rather than raising. Shared
  process-wide, so token counts cached in the worker thread are reused by front-end budget
  checks.
- **vLLM** does strict OpenAI message normalization for replayed tool-call history,
  per-model `extra_body` from the profiles, and streaming tool-call delta merge by index.
  It overrides `_tokenize_text()` with an exact count from vLLM's `/tokenize` endpoint,
  falling back to the heuristic on any error. The endpoint is read through one overridable
  seam, `_config() -> (base_url, api_key)`, and the model-length cache is keyed by
  `(endpoint, model)` so two servers offering the same model name do not share a window.
- **Ray** inherits the request shaping, reasoning profiles and tool-call handling unchanged —
  Ray Serve orchestrates GPUs and drives vLLM engines behind the same API. What it overrides
  is what the *router* may not serve: `_fetch_context_window()` honours
  `MIMIR_RAY_MAX_MODEL_LEN` first, since the plain `/v1/models` shape carries no
  `max_model_len`, and `_tokenize_text()` latches after the first failure so a router
  without `/tokenize` costs one round trip rather than one per count. The `ray` package is
  not a client dependency — it runs on the cluster.

---

## guardrails

Behaviour governance. The root holds what both halves read; `policy/` blocks and `nudges/`
advises. Rules live in [`POLICY.md`](POLICY.md) — this section says where the code is.
Paths below are relative to `mimir/client/guardrails/`.

### `workflow.py` (root)

The workflow state model and the completion report — shared by policy **and** nudges, which
is why it sits at the root.

- `WORKFLOW_STATES`, `VALIDATION_RETRY_BUDGET`, `set_workflow_state()`,
  `pending_validation_paths()`, `has_pending_validation()`, `has_blocking_denials()`.
- **The denial ladder**: `denial_stage(ctx, scope)` / `worst_denial_stage(ctx)` /
  `handback_required(ctx)` / `handback_scopes(ctx)` / `approval_is_settled(ctx, scope)`.
  Counted from `denial_history`, which is append-only *precisely because*
  `denied_tool_calls` is cleared when an action later succeeds. `approval_is_settled` is the
  one predicate the policy engine consults to decline re-prompting. Thresholds in
  `config/constants.py`. See [POLICY.md](POLICY.md#if-approval-is-refused).
- `turn_made_commitments(ctx)` / `unhonoured_commitments(ctx)` — did this turn commit to
  producing something, and is some of it still undone? Three signals, all per-query: an edit
  happened, a set of target files was declared, or a checklist was written. The completeness
  guards used to ask `code_mutation_started` instead, which read a turn that declared eight
  files and wrote none as pure discovery — the exact case they exist for.
- `unchecked_checklist_items(ctx)` — the single reader of live checklist state outside the
  prompt builder. **Fails closed to `[]`** on a missing or unreadable file, so a run without
  a checklist behaves exactly as before rather than having an obligation invented for it.
  Optional steps are tagged, not filtered.
- The agent-loop and plan-loop copy, including the loop-control correctives, whose *firing*
  decision lives in `agent_loop.py` / `dispatch.py`.
- `finalize_incomplete_answer(answer, ctx, termination)` returns the model's prose followed
  by the report as a **marked block**, the same contract the ledger uses: the front-ends lift
  it off and render it collapsed. It picks one of three headlines — see
  [POLICY.md](POLICY.md#if-approval-is-refused). `is_incomplete_answer()` is the predicate
  the CLI's re-plan offer and the sub-agent `completed` flag read, off the marker's `status`.
- `_collect_completion_issues()` splits pending validation into three buckets —
  budget-exhausted, failing-but-retryable, fresh-unvalidated — and adds open checklist steps.
  It deliberately does **not** collect a missing verdict, nor a file the plan named and never
  wrote: both print under their own headings, reported and charged at nothing. Its
  "all validated" line is tier-qualified and governed by the **weakest** tier across the
  change; the label once said "highest" while printing the floor.
- **The report speaks only in the past tense.** The termination reason is computed where the
  loop exits and passed in, rather than inferred downstream from a retry budget:
  budget-with-room-left means "the loop would try again" only while the loop runs, and read
  from a final report it became a promise nobody was going to keep.
- `evidence_handback_message(ctx)` — once per query, before the report is assembled, the
  ledger is injected as a user turn so the model rewrites its summary having *seen* it.
  "Successfully implemented, complete and correct" printed above "Modified files never
  checked" is a missing fact at the moment the prose is written, not a rhetoric problem to
  police afterwards.

### `observations.py` (root)

The writer of the `execution_context` blackboard, shared by policy and nudges.
`record_tool_observation()` is decomposed into ordered `_observe_*` handlers dispatched in a
fixed, load-bearing order pinned by `test_observations.py`.

- `_observe_edit_outcome` merges edit success and repeated-failure tracking.
- `_observe_command` classifies each bash segment and credits the blackboard on success.
- `_observe_bash_validation` runs status-agnostically and drives **two axes from one
  command, never mixed** — neither of them the mandatory one, which no command performs any
  more. A *checker* on a dirty file it names marks it validated on exit 0 and charges its
  retry budget on a non-zero one. A *reformat* credits nothing: it rewrites the file and
  exits 0 whether or not the code is correct. An *execution* validates no file at all.
- `_observe_declared_edit_set` scrapes source paths out of the checklist's step text and
  **replaces** the declared set, mirroring the checklist tool's own contract.
- `_observe_run_outcome` reads a server's declared `run_outcome` spec — the floor under a
  model's stated verdict, which withholds credit and never grants it.
- `_register_run_failure` charges a run's failure against the repair budget and steers back
  to `edit`, but only where code was actually mutated. Past the budget it releases to
  `conclude` rather than wedging.

### `verdict.py` (root)

`apply_verdict()` — the model's stated reading of a run's output, applied to the runs it
addresses. Five verdicts, two target sets, and a deliberate reach asymmetry; the full rules
are in [POLICY.md → Verdicts](POLICY.md#verdicts).

### `builtin_check.py` (root)

The in-process check every modified file owes: a stdlib parser where one exists, a
structural scan otherwise. `sweep_builtin_checks()` runs it as **one sweep** where the loop
asks whether it may conclude, never after each write, so a file edited ten times is read
once — on the revision it will ship at.

### `policy/`

| Module | Holds |
|---|---|
| `engine.py` | `evaluate_tool_preconditions()` — the gate order, the two pack slots, violation enrichment |
| `gates.py` | `_check_cluster_submit()`, `_check_proxy_exec()`, `_check_out_of_workspace_access()` |
| `write.py` | `check_write_policy()`, `has_delete_context()`, `write_policy_violation()` |
| `state_machine.py` | `check_state_machine_guard()` and the retry budget |
| `approval.py` | `ApprovalManager`: prompts, `always` grants, batch queue, snapshots, revert |
| `bash_classify.py` | `classify_bash_command()` / `bash_command_is_readonly()` |
| `readonly_exempt.py` | the read-only dual-use waiver, in any mode |
| `plugins.py` | `PolicyCheck` descriptor + `PolicyRegistry` + `register_policy_check()` |

Two details worth stating here:

- `_trusted_read_roots()` is the client mirror of the roots the servers admit silently. It
  shares `servers._shared.trusted_read_roots` with them, **plus `constants.STATE_DIR`
  appended explicitly**: the shared helper resolves the state dir from `MIMIR_STATE_DIR`,
  and `server_manager` places that variable only in the *server subprocesses'* env. Without
  the explicit append the agent could not read back its own plans without a prompt, while
  the servers — which do see the variable — would have allowed it.
- `_enrich_violation_payload()` always sets `policy_stage`, `tool`,
  `suggested_next_tool_class`, `state` and `status`. `missing_evidence` is **conditional**:
  attached only at the `write_policy` and `approval` stages, then dropped in the `discover`
  state, and dropped again once the denial nudge has fired twice.

Per-query tool-list construction is **not** here — it lives in `query_engine/toollist.py`,
and withholds nothing but what the mode forbids.

### `nudges/`

At most one reminder per step. `maybe_append_nudge()` walks the built-in table
`_CORE_NUDGES` through the generic runner `_append_core_nudge()`; packs add rows through
`_append_custom_nudge()`. Every row is `(name, layer, should_fire, render, budget_key)`,
where `budget_key` defaults to the name and is what lets several rows ration one counter.

The full per-nudge table, with what each fires on and which survive at each level, is in
[POLICY.md](POLICY.md#nudges-by-enforcement-level). What belongs here is the shape:

- **Verification layer** — runs at every enforcement level: `denial`, `error_recovery`,
  `stuck_repair`, `validation`, `regression`, `unexercised`, `unfinished_plan`.
  `test_nudge_table.py` asserts this set is disjoint from `_ALL_GUIDANCE`, so a verification
  row can never be silently switched off by enforcement.
- **Guidance layer** — skipped entirely at `off`: `env_resolution`, `env_cleanup`, `doc`,
  `state`, `blast_radius`, `creation`, `todo`.
- **Order per step**: core verification → pack verification → *(stop if `off`)* → core
  guidance → pack guidance. The first row whose predicate holds wins.
- `_GUIDANCE_BY_LEVEL_MODE` is the declarative `(enforcement, mode)` table, consulted
  through `_guidance_enabled` inside each guidance predicate.
- `needs_incomplete_finalization()` blocks on the check axis, denials, and open non-optional
  checklist steps — the last checked **first**, because the other two conclude from
  validation alone, which is no evidence about steps the model never started. It reads
  `workflow_state` nowhere: that condition made the *recommended* axes mandatory, since a
  failed run sends the state machine back to `edit`.

**Every predicate reads recorded state, never the wording of the request.** Four guidance
rows used to open on a keyword match over the query; that is a guess about intent dressed as
a test. What replaced the one thing the keyword earned — telling `blast_radius` from
`creation` — is whether the declared target already exists on disk.

`nudges/messages.py` holds the message copy for every built-in row. `nudges/plugins.py`
holds the `NudgeRule` descriptor, the process-global `NudgeRegistry` and `register_nudge()`;
pack guidance rules are tier-gated by `rule_tier_enabled()`, suppressed when their name is in
`agent.disabled_nudges` (toggleable via `/nudges`), and capped per query.

---

## query_engine

The step loop: call the model, dispatch tools, trim history, inject reminders. The module
split is [above](#the-per-query-loop).

### `agent_loop.py`

`run_agent_query()` is a thin orchestrator — shared setup, then dispatch to
`_run_plan_mode()` or `_run_agent_loop()`. Every exit path routes end-of-query bookkeeping
through `_finalize_answer()`: annotate the answer, persist memory, save carry context, stash
the full messages.

**Completion is `if not tool_calls:`** — the model emitted no tool call. There is no goal
check, so the honesty surface is the verification ledger appended to every answer. Full
format in [POLICY.md](POLICY.md#verification-ledger).

- **Setup** — reset the per-query tool cache, build a fresh `ExecutionContext`, apply the
  carry context, assemble the system prompt.
- **`_stream_chat()`** — iterative streaming backend calls, retrying transient failures with exponential backoff and jitter, cancel-aware. Thinking
  blocks stream for live display but are **excluded from history**: reasoning is never
  re-fed. UI events go through `emit()`.
- **`_dispatch_tool_calls()`** — dedups `(name, args)` within a step, and across steps for
  writes. Reads run concurrently through `asyncio.gather`; **writes are serialized**, so two
  edits to one file, or a read racing a write, cannot interleave. Each call is wrapped in
  `asyncio.wait_for` with the wall `capabilities.timeout_for` resolves — the tool's own
  `timeout_secs` if it declared one, else `TOOL_CALL_TIMEOUT_SECS`, clamped by
  `TOOL_CALL_TIMEOUT_MAX_SECS`.
- **Step budget** — there is none. A query runs for as many steps as the work takes: it
  ends when the model delivers an answer, when a guard stops it (empty turns, validation
  budget), or when the user interrupts. Only a caller that asks for a bound gets one —
  the runner and `spawn_agent` pass their own `max_steps` (30), and the loop nudges the
  model to summarise two steps before that boundary. `max_steps` 0, the default, means
  no ceiling.

**Three cross-step repeat mechanisms**, all keyed on the call and its arguments:

| Mechanism | On | Does |
|---|---|---|
| write dedup | an identical write | collapses it |
| failing-call guard | an identical **failed** non-write call | corrects it at `SOFT_REPEAT_THRESHOLD`, hard-blocks at `HARD_REPEAT_LIMIT`, returning a synthetic error so the model gets feedback instead of spinning to the ceiling |
| identical-success annotation | the same result digest `IDENTICAL_REPEAT_THRESHOLD` times | appends `IDENTICAL_REPEAT`. Nothing is withheld |

**A repeated *successful* call is annotated, never guarded.** There was a redundant-success
guard — result hashing, a soft corrective, a hard block, and history surgery to keep one
copy — and it is gone. A repeated read is answered by the per-query cache: no round trip, no
refusal, no rewriting. What replaced it is upstream: a read says what it served and where to
resume, so the second identical read has less reason to happen. The annotation exists
because the spin it catches is invisible to everything else — the failing-call guard counts
only failures, and a nudge fires only once the model stops calling tools, which a spinning
model never does.

**`_post_dispatch_inject()` is the mid-tool-loop channel**, carrying the five reminders the
end-of-turn table cannot reach, because that table only fires once the model *stops* calling
tools — and a model retrying against the wrong interpreter, chasing a moving test, or told
to hand back is by definition still calling tools. The five, and the fact that only
`env_resolution` carries an enforcement gate, are tabulated in
[POLICY.md](POLICY.md#loop-control-correctives).

### `plan_loop.py`

- **"Is a plan recorded"** is read from `plan_written` in the execution context, where
  `_observe_todo_flags` sets it by telling the two `TASK_PLANNING` forms apart via the
  `plan_steps` arg-role. The prose document is the **only** form plan mode produces: the
  ordered checklist is written after the user approves, at the start of execution.
  `toollist.hidden_planning_tools()` decides by capability which writers a read-only mode
  exposes — **ask** hides both, **plan** hides the checklist tool, plus the document tool
  while exploring — and `filter_readonly_tool_calls` re-applies the same set at call time,
  so a hallucinated call is answered rather than executed. Hiding a tool is what lets both
  prompts drop the matching prohibition: an absent tool needs no rule and no prompt tokens.

  The loop used to re-derive this locally from tool names, counting the checklist alone — so
  a plan recorded in prose was invisible, the loop kept telling a model whose plan was on
  disk that it had not recorded one, the model answered by rewriting the document, and the
  run spun to the ceiling delivering nothing.
- **The explore phase.** Plan mode runs in two phases and the plan-document tool does not
  exist during the first. On a repo-touching query the document tool is hidden and the nudge
  asks for the exploration rather than for the plan. The phase flips the moment
  `plan_evidence_ready(ctx)` holds — `PLAN_EVIDENCE_MIN_FILES_READ` files actually **read**,
  plus the distinct-signal bar — and the tool list is rebuilt once, a sanctioned
  prefix-cache break paid for by the tool it unlocks.

  This replaced an after-the-fact advisory gate that fired *after* the plan was written and
  only appended a nudge: it never stood between the model and a plan written over file
  names. A plan mode that offers the document from turn 1, under a nudge calling the plan
  mandatory, makes a plan *to explore* the cheapest way out — so the fix withholds the tool
  rather than policing the plan's wording. Because the arming signal is a broad filter that
  fires for greenfield work no exploration could ground, `PLAN_EXPLORE_MAX_TURNS` unlocks
  the tool regardless and the model is told to state its gaps: plan mode always reaches a
  plan.

  Where a `DELEGATE` tool is connected, phase 1 is a **fan-out**: the nudge and the mode's
  prompt block ask for one to three read-only sub-agents in a *single* response. What they
  read comes back and is credited to `delegated_read_files`, which `plan_evidence_ready`
  counts — otherwise the phase would punish the fan-out it just asked for.
- **The anti-parroting guard.** A turn that calls tools never reaches the delivery branch,
  so a model that keeps re-reading or rewriting the recorded plan would loop to the ceiling
  and the user would never be asked to approve. After the plan is recorded the model gets
  `_PLAN_POST_RECORD_TOOL_TURNS` (2) further tool-calling turns; past that its calls are
  dropped and the turn falls through to delivery and approval on the prose gathered so far.
  The drop is **not** conditioned on prose having been emitted — a model stuck in this loop
  typically emits tool calls and nothing else, which is exactly the shape it exists for.

### `history.py`

Context budgeting: trim → compact → force-fit → repair. `_enforce_context_budget` rewrites
`messages` **in place while the turn runs**, so a position the front-end measured before
submitting stops meaning what it meant. The loop records the message the turn opens on and
resolves it back to an index by **identity**, not equality, so two byte-identical job wakes
are not confused. A stale boundary re-archives whatever the rewrite shifted past, drops
whatever it shifted over, and cuts through an assistant↔tool pair on the way.

**The compaction marker's count is cumulative.** `compacted_exchanges()` reads the count out
of the summary a second pass is about to swallow and adds to it, rather than counting that
message as one exchange. It is what the model reads to judge how much of its own past it
can no longer see, and per-pass counting made every pass announce less than the one before.
A second compaction feeds the previous summary back through the summariser, so the window is
always `[system, task, exactly one summary, last two exchanges]` and never a stack of them.

---

## tool_execution

### `executor.py`

- **Continuation and orientation hints.** `_build_continuation_hint()` reads the server's
  own payload (`truncated`, `total_lines`, `next_start_line`, `line_cap`), **not** the
  arguments: reads are clamped by a default window and a per-call cap, so the range asked
  for is not the range served, and a caller not told the difference cannot tell "this is the
  file" from "this is its first page". `_build_outline_hint()` adds an `OUTLINE:` symbol map
  for a truncated read of a code file, obtained through a `CODE_NAV` tool once per file per
  query and **without** an execution context — the machine's own call must not clear a
  discovery gate on the model's behalf. The **end** of each span is the load-bearing half:
  with start lines alone the model cannot ask for "the block around line N" and crawls
  toward its end a few lines at a time.
- **Nothing tracks which lines are held.** A line-coverage ledger existed and is gone. It
  duplicated the redundant-success guard in a harsher form, short-circuited it (a read the
  client answered never reached the result-hash counter), depended on a shell parser to know
  which lines a search had printed, and bought only a slid window — for four context fields,
  an mtime stamp, a diff re-indexer and a crediting path per tool. What replaces it is
  stated at the source instead of inferred at the client.
- `_path_stamp()` — the `(mtime_ns, size)` every cache entry carries, so a hit can tell
  "unchanged" from "a `sed -i` moved it under us". The only place a file's identity is
  checked outside the edit tools.

### `bash_effect.py`

What a shell command changed, appended as `BASH_EFFECT`. `capture()` runs before the
dispatch and `report()` after. The trigger is `bash_command_is_readonly` being **false**,
never the classified kind. Detection is a git delta, or a bounded scan outside a repo —
never a parse of the command, which is the guess the module exists to avoid. See
[POLICY.md](POLICY.md#what-a-shell-command-changed).

### `validation.py`

- `scratch_roots()` / `is_scratch_path(path)` — the client's view of the scratchpad, one
  definition read by the out-of-workspace gate (scratch never prompts) and by
  `observations._record_code_edit` (scratch writes never enter `dirty_written_files`).
  Without the second exclusion the scratchpad would trade workspace clutter for ledger
  clutter and spurious validation obligations.
- `auto_validate_written_file()` — the post-write hook. The deterministic
  syntax→imports→lint→typecheck→tests ladder was removed, and the mandatory check that
  replaced it does not live here either. What remains are the two *completeness* checks with
  no bash equivalent: the replacement-completeness grep for leftover text after a replace,
  and the cross-file reference check for stale callers after a rename.

### `normalizer.py`, `formatter.py`, `tool_status_messages.py`

- `normalizer.py` — `normalize_tool_arguments()`, `normalize_workspace_path()`, and
  `rewrite_tool_for_context()`, which heals the `read_file` alias to `read_file_lines` and
  fills in the line range a bare read left out, so every read says what it asks for.
- `formatter.py` — `normalize_arguments()`, `normalize_tool_content()`, `truncate_text()`,
  `json_error_payload()`, `parse_tool_payload()`.
- `tool_status_messages.py` — `tool_status_message()` derives a human-readable status from
  the tool *name*, with no per-tool table: a verb is located in a reusable lexicon, rendered
  as a gerund, and the remaining tokens appended. `shorten_display_args()` reduces declared
  path arguments to their **file name** for display, capability-driven off the `path`
  arg-role. A row reading `Reading file: /shared/data1/Projects/.../observations.py` buries
  the one token the user is scanning for; it becomes `Reading file: observations.py`.

  **Never applied where the path is the decision.** An out-of-workspace approval asks the
  user to authorise *locations*, so the card carries the paths verbatim and the CLI prints
  each absolute path on its own line. Readability wins in the activity log; precision wins
  in a consent prompt.

---

## Client root

### `agent_core.py`

`MimirAgent`: server lifecycle, mode and settings management, `run()`, `cleanup()`, and the
per-session registry `tool_caps`. Deliberately **not** under `ui/` — it is the engine the
frontends pilot, not a frontend.

- `seed_classification_from_caps()` — called once after all servers connect. Re-seeds the
  approval manager's sensitive / non-batch / fallback sets **in place** from `tool_caps`
  (in-place mutation preserves an alias), then reports connected tools that declared no caps.
- `_apply_carry_context()` / `_update_carry_context()` — merge prior-session discovery sets
  into each new context, evicting stale reads by mtime, and save fields back after each
  query. Both iterate the trait-derived carry list.
- `compact_history()` / `compact_messages()` / `detect_skill_implicit()` — the model calls
  MIMIR makes on **its own behalf**. All three go through `get_backend()`; two used to call
  Ollama directly, which broke them under every other backend. Each passes a discarding
  token callback, because `chat` streams to stdout when given none — that default belongs to
  the CLI answer path, and without a sink a compaction summary is printed into the middle of
  the session. Each degrades quietly when the endpoint is down, so an unreachable backend
  costs a summary, not the turn.

### `human_pause.py`

The blocking-prompt seam every "ask the human and wait" path shares — approvals, plan
approval, tool elicitation — so a frontend wires one hook instead of three, and a headless
run neutralises them all at once.

---

## ui

Two independent frontends that share nothing.

### `ui/cli/`

`main.py` builds the agent, connects servers, runs the session, cleans up; `main_sync()`
wraps it for the `mimir` console script. `chat_session.py` is the async REPL — nothing is
probed or scanned at startup, so it is ready as soon as the servers connect. It also splits
the ledger block off the answer for a one-line summary, expanded on `/ledger`, while the
block stays in history for the model. `chat_commands.py` is the slash-command table
[above](#slash-commands).

### `ui/ws/`

The WebSocket / VS Code bridge: `ws_server.py`, `ws_worker.py`, `ws_session.py`,
`_ws_runtime.py`, `session_store.py`, `session_summary.py`, `transcript_log.py`,
`file_preview.py`. The message protocol and the React frontend are documented in
[`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md). What belongs here are the invariants that
are not obvious from the protocol:

**Three lists, three audiences.** `display_messages` is the chat: never trimmed, never
compacted, and sent back in full on load — the person reading always sees the whole
conversation. `llm_history_full` is the record: raw turns only, carrying **no** summary, so
it stays the one account nothing was inferred from. `llm_history` is the model's window, and
the only one a compaction touches.

**What a resume restores** follows from that split. A window that already carries a
compaction summary *is* the conversation — the handoff note stands in for everything cut, at
a fraction of the tokens — so that is what the model resumes on. Absent a summary, the
window is only the tail a front-trim left, and the record is then the faithful resume.
Reloading the record over a compacted window is what handed the model a context that was
full before the user had typed.

**Progress is pushed, never recorded.** A tool call that blocks the turn cannot report
on itself, so the session polls the blocking run's channel once a second
(`_tick_run_channels`, reading `tool_execution/run_channel.py`) and pushes a
`tool_progress` for the row; once the run is detached, the watcher's own status poll
sends `job_progress` instead. Neither reaches `transcript` or any of the three lists
above: they describe a moment, and a watcher ticking for an hour would otherwise fill
the record with "still building". How a run actually ended is `job_complete`'s job.

**A divert names a row, not a tool.** `divert_to_background` carries the call id; the
session resolves it through `_live_rows` (call id → tool name, kept only for rows the
registry marked divertible) and writes the request to that tool's channel. The webview
therefore never learns a tool name, and one tool's request cannot reach another's run.

**Compaction never blocks the event loop.** Summarising is an LLM call, so the session
schedules it on the worker and awaits the future. It keeps the opening user message and the
last two exchanges, replaces the middle with one summary, and re-runs the tool-pair
reconciliation because that slice can strand a tool call. Front-trimming the oldest turns is
the **fallback** — taken when the middle is too short to be worth a call, when summarisation
fails, or when the summary still does not fit. It used to be the only behaviour, so the
window was amputated where it could have been summarised.

**A background-job wake belongs to the session that launched it.** A two-hour build outlives
the conversation on screen, so the job records its session at launch and the completion event
carries it back. When it names the active session the wake lands in the live history as any
turn would; otherwise it is appended to *that* session's stored history and submitted there.

**A connection that drops does not end the turn.** One worker serves the whole server and
outlives every session, so a socket that dies mid-run leaves a turn working with nobody
reading it. The next connection therefore only drains the event queue when the worker is
**idle**: what is queued under a busy worker is that turn's own output, not debris. A turn
parked on a person records the card it is waiting on, which the next connection puts back —
without that, the wait had no timeout by design and no card left to end it, so every later
query queued behind a wait nobody could answer and the session read as hung.

**Who owns the rendered chat.** `display_messages` as the server assembles it is text bubbles
and nothing else; the tool rows, reasoning panels and diff cards are built by the webview's
reducer and exist nowhere else, so the client hands its rendered copy back under guard. That
handover used to happen once, when the turn ended — which made every long run a window in
which the work on screen existed in one place, a browser tab. Three things close it: the
client checkpoints mid-turn, a turn **parking on a question** counts as a fall of `busy`
like any other, and a reconnect keeps the **richer** copy rather than the stored one.

---

## Headless run engine

`mimir/runner/` is the agent's **batch mode**: a library that drives the non-interactive
`MimirAgent.run` path over a list of tasks and returns a JSON summary. It does **no scoring
of its own** — a benchmark supplies the tasks *and* grades the result. Servers are spawned
with `sys.executable`, so it runs under whichever interpreter launches it, ARM or x86.

It is a **library, not a CLI**, and ships no benchmarks or adapters. An external package
imports `mimir.runner`, implements the `BenchmarkAdapter` contract, and calls
`run_benchmark(...)`. Two seams: `BenchTask.setup(workspace)` materialises the task
workspace, and `BenchmarkAdapter.score(task, answer, ctx)` grades the run by inspecting
`ctx.workspace` — never `ctx.agent`. So **`mimir` imports neither the benchmark nor the
integration**; only the integration knows both.

- `types.py` — `BenchTask` (`id`, `query`, `mode`, `max_steps`, `servers`, `setup`,
  `requires`, `meta`), `CheckContext` (workspace + live agent), and the `BenchmarkAdapter`
  protocol (`name`, `load_tasks(limit)`, async `score(...)`).
- `run_one(...)` — skips up front if any `task.requires` executable is missing; else creates
  a temp workspace, runs `setup`, `chdir`s into it (the file and search servers root at the
  cwd at spawn), and spins up a **fresh** `MimirAgent` so carry context, tool cache and
  history never leak between tasks. An `enforcement` argument overrides the model-profile
  default so a whole run is graded at one fixed nudge level. A per-task crash is recorded as
  a failed result with its traceback rather than aborting the suite.
- `run_benchmark(...)` — sets the backend, clears the factory singleton, runs every task,
  and builds the JSON summary. `pass_rate` is over **non-skipped** tasks.
  `get_backend_override` injects a backend factory for a model-free CI mode.
- **Unattended approval**: `_install_auto_approve(agent)` replaces the approval hook with an
  always-approve shim. Batch mode already auto-approves
  writes, but non-batch tools (execution, shell) would otherwise block on input, so the shim
  approves *every* tool. **This runs every tool without confirmation — point the engine only
  at trusted workloads, in a sandbox.**

---

## Test coverage

Every test is **`unittest` + `asyncio.run`** — no pytest-only constructs, no `conftest.py` —
so either runner works on the same files. `pytest` is the documented command;
`python -m unittest discover mimir/tests` remains available when the `dev` extra is not
installed.

| Suite | Covers |
|---|---|
| `_fake_backend.py` | `ScriptedBackend`: a deterministic backend replaying canned responses, driving the streaming callbacks and recording per-call inputs. Shared by the loop and runner tests; dependency-light, so it runs on ARM and x86 |
| `test_agent_loop.py` | the loop functions — intra-query compaction, `_post_dispatch_inject`, `_finalize_answer` (including the turn boundary surviving an in-turn rewrite, and matching on identity so two byte-identical job wakes are not confused), plan mode, and the non-interactive path. Plus the failing-call guard and the identical-success annotation |
| `test_completion_honesty.py` | the end-of-run honesty surface: the ledger's rows and statuses, the marker contract, the tier-qualified completion sentence, `needs_incomplete_finalization`, the `unfinished_plan` nudge, and the checklist reader's fail-closed behaviour |
| `test_observations.py` | the observer dispatch order, bash classification and credit, run-ledger keying, verdict grammar, exit attribution, `ValidationTierTests` (per-checker tiers, an execution earning none however green, a printed invariant earning nothing, monotonicity, retraction), and `DeclaredEditSetTests` (a revised checklist retracts what it dropped) |
| `test_policy_manager.py` | the gates and the state guard |
| `test_client_helpers.py` | the nudge predicates, token counting, eviction and `ContextOverflowError` |
| `test_capabilities.py` / `test_phase_b_servers.py` | `infer_tool_caps` precedence and the golden declared registry (`_golden_caps.py` AST-parses the server decorators) |
| `test_capability_consumers.py` | drift guard — fails if a declared capability has no live consumer |
| `test_session_persistence.py` | the window/record split, WS compaction and its front-trim fallback, which one a resume restores, the turn boundary in both stale directions, and the cumulative marker count |
| `test_scratchpad.py` | home resolution, the standing grant being the home, `ensure_scratch_home` refusing a symlink / foreign owner, and scratch writes staying out of `dirty_written_files` |
| `test_absolute_paths.py` | every mutating tool rejects a relative path, names the resolved candidate, and writes nothing on rejection |
| `test_bash_classify.py` / `test_bash_coverage.py` / `test_server_contracts.py` | segmentation and the tokenization-invariance guard, the corpus-measured credit rate, and the `-exec` policy |
| `test_nudge_table.py` | the verification set is disjoint from the guidance set |
| `test_env_resolution.py` | the mid-loop cascade fires at the failing call, shares the row's budget, respects enforcement — and a successful execution retracts `unresolved_modules` |
| `test_internal_model_calls.py` | the two calls MIMIR makes on its own behalf go through the configured backend, supply a token sink, and degrade quietly |
| `test_runner.py` | the headless engine model-free: report shape, the auto-approve hook, per-task isolation, and the skip path |
| `test_builtin_check.py` | the in-process floor, run over every file this repository tracks |

---

## VS Code extension frontend

Documented in its own reference: [`EXTENSION_DETAILED.md`](EXTENSION_DETAILED.md). On the
Python side the emission paths are `emit()` / `event_sink.py` and the drain loop in
`ws_server.py`.
