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
| `/temperature <0-2>\|default` | sampling temperature for the current model; `default` sends none |
| `/enforcement strict\|light\|off` | the guidance-nudge dial ([POLICY.md](POLICY.md#nudges-by-enforcement-level)) |
| `/approvals manual\|auto\|all` | who answers the approval cards ([POLICY.md](POLICY.md#policy-gates-by-approval-mode)) |
| `/batch on\|off` | queue write approvals until the end of the turn |
| `/trust <tool>`, `/untrust <tool>` | session-wide trust for one tool |
| `/context compact\|full`, `/compact` | window policy, and compact now |
| `/ledger` | expand the last answer's verification ledger |
| `/undo` | revert the last write |
| `/memory` | the memory store |
| `/servers`, `/skills`, `/nudges` | list or toggle, persisted in `preferences.json`. A switched-off skill is hidden from both `/<name>` and the index the model loads from |
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

The same file holds `temperatures`: the sampling temperature the user chose, per served
model name. A model absent from it sends **no** temperature, and the server applies the
model's own `generation_config`. The agent loads the value at construction and again on
`set_model`, so each model keeps its own and a sub-agent of the same model starts from it.
Both loops read `agent.temperature` before each call, so a change lands on the next step.
Only vLLM, Ray and Ollama forward it. Anthropic's extended thinking requires 1, so the
webview hides the control there. Each save rewrites only its own keys and keeps the rest.

---

## extensions

The single home for everything the user drops into the workspace `.mimir/`. One module per
extension type, env-overridable, fail-open. `config/constants.py` keeps only the path
primitives (`MIMIR_DIR`, `resolve_extension_dir`, the `*_DIRNAME` / `*_DIR_ENV` constants).

| Module | Provides |
|---|---|
| `servers.py` | `all_servers()` — bundled `SERVERS` merged with `discover_user_servers()` (scans `.mimir/servers/server_<name>.py\|.js`, env `MIMIR_SERVERS_DIR`). A name colliding with a bundled server is skipped, so the core is protected. `all_server_descriptions()` does the same for the toggle panel |
| `skills.py` | `resolve_skills_dir()` — `.mimir/skills/`, env `MIMIR_SKILLS_DIR`. Loading itself is `MimirAgent.load_skills(..., merge=True)`, where a same-named user skill overrides the bundled one. Only names and one-line descriptions reach the prompt (`model_invocable_skills()`); a body is read on demand by the `load_skill` tool |
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
  `label_for`, `kind_for`, `timeout_for`, `scope_spec`, `name_for_cap`,
  `unannotated_live_tools` — all
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

Only a skill the **user** invoked is in that block. One the **model** loads mid-run with
`load_skill` deliberately does **not** go into `messages[0]`: rewriting the system message
mid-query voids the prompt prefix for every remaining step, and a pull happens at step 14 as
readily as at step 1. So the body stays where it landed — the tool result in the tail — and
is protected there instead.

### The pin

One set of ids, `execution_context["pinned_call_ids"]`, and three passes that honour it. It
exists because the tail is where the context budget does its work: `_trim_tool_history`
evicts whole tool results, `_maybe_compact_intra_query` summarises the middle of the list,
and `_force_fit_to_window` truncates largest-first. All three would carry off a skill body,
and the model has been told the method is in its context.

- `_trim_tool_history` skips pinned ids outright, by `tool_call_id` — the same structural key
  as `tool_msg_files`, never a substring scan.
- `_maybe_compact_intra_query` carves each pinned result out of the slice it is about to
  summarise, **together with the assistant turn that declared it**. The pairing is not
  optional: `reconcile_tool_pairs` drops any tool message not immediately preceded by that
  turn, so a body saved alone would be deleted by the very next repair pass — protection that
  looks like it works and does nothing.
- `_force_fit_to_window` sorts pinned messages to the **end** of the reduction order rather
  than protecting them. Protecting them there could only turn an over-window prompt into a
  hard `ContextOverflowError`; ordering them last means a body is cut only once everything
  else has been, and usually not at all. Doing nothing was not an option either — the order
  is largest-first, and a skill body is the largest single message in the list.

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
- **Calibration that outlives the process** (`token_calibration.py`): the two figures
  measured against the server's own `prompt_tokens` — the per-call prompt overhead and
  the history's chars-per-token — are cached in `<state>/token_calibration.json`, so a
  session reopened against a fresh server is accounted for exactly as soon as its agent
  is up, instead of only once its first answer lands. The ratio is keyed by model; the overhead by a
  fingerprint of everything it is made of (model, context mode, system prompt,
  advertised tools), so a changed fixed part is a miss and the honest estimate answers
  rather than a stale measurement.
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
  is what the *router* may not serve: `_fetch_context_window()` detects the router's
  `max_model_len` first and falls back to `MIMIR_RAY_MAX_MODEL_LEN` only when the plain
  `/v1/models` shape carries none, and `_tokenize_text()` latches after the first failure so a router
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
through `_finalize_answer()`: annotate the answer, save carry context, stash
the full messages. Nothing is written to memory there: that is the model's call, made during the run.

**Completion is `if not tool_calls:`** — the model emitted no tool call. There is no goal
check, so the honesty surface is the verification ledger appended to every answer. Full
format in [POLICY.md](POLICY.md#verification-ledger).

- **Setup** — reset the per-query tool cache, build a fresh `ExecutionContext`, apply the
  carry context, assemble the system prompt.
- **`_stream_chat()`** — iterative streaming backend calls, retrying transient failures with exponential backoff and jitter, cancel-aware. Thinking
  blocks stream for live display but are **excluded from history**: reasoning is never
  re-fed. UI events go through `emit()`.
- **`_dispatch_tool_calls()`** — takes the model's per-call `doing` description out of the
  arguments first (the client adds that parameter to every tool's schema; no server declares
  or receives it), *before* the dedup key, so two identical calls with reworded descriptions
  stay one call instead of walking past the repeat guards. Then dedups `(name, args)` within
  a step, and across steps for writes. Reads run concurrently through `asyncio.gather`; **writes are serialized**, so two
  edits to one file, or a read racing a write, cannot interleave. Each call is wrapped in
  `asyncio.wait_for` with the wall `capabilities.timeout_for` resolves — the tool's own
  `timeout_secs` if it declared one, else `TOOL_CALL_TIMEOUT_SECS`, clamped by
  `TOOL_CALL_TIMEOUT_MAX_SECS`.
- **A dead transport is the one failure not to retry.** Any exception from a call becomes an
  ordinary tool error rather than killing the turn, with "retry once" as the advice — except
  `ClosedResourceError` / `BrokenResourceError`, which mean that server's stdio transport is
  gone: its process died, or its stack was unwound under the turn. Nothing reconnects an MCP
  server, so every later call to any tool of that server fails identically. That failure
  therefore names the server, lists the other tools it has just taken down, and tells the
  model to stop and report what is unfinished instead of spending the turn on calls that
  cannot work.
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

**A step is bounded before it is appended, not after.** `bound_step_results()` clamps one
step's tool results — per result, and per step together — and `dispatch.py` calls it in the
loop that appends them. Everything else here runs too late to help: `_trim_tool_history`
deliberately exempts the current step from eviction (a sub-agent's whole answer is one of
those results), so the only thing that could touch them was the force-fit backstop, and only
once the window was already over. That is how four parallel fetches took a 200k window to
215k in a single step, one of them a 131k-token HTTP 403 body, with nothing given the chance
to object.

The shares are set so that a tool returning its **own documented maximum** is never
what this cuts. Measured against every ceiling MIMIR ships, the largest legitimate
single result is `github_get_file` at 256 KB — about 65k tokens — so half of a 160k
usable window clears it, where the quarter share this started at (40k) would have cut
a file fetch doing exactly what it is designed to do. A backstop that fires in normal
operation is not a backstop; it is a second, worse ceiling that degrades results
silently. A test holds the line against re-tightening. On a window too small to hold
those ceilings at all — a served 32k, where one 128 KB page cannot fit whatever anyone
does — something has to give, and these shares only decide it earlier and more fairly
than the force-fit pass would have.

Both budgets are fractions of the usable window rather than fixed numbers — a 20k ceiling is
generous on 200k and catastrophic on 32k, and the served window is not known until the
backend reports it. Allowances are shared by water-filling (`_water_fill`): smallest first,
each taking either its whole size or an equal share of what is left, so a small result is
never cut to pay for a large one and the budget the small ones release raises the ceiling for
the rest. Truncation reuses `_truncate_text_to_tokens`, which keeps a head and a tail and is
guaranteed to land under the cap, and the note it appends is counted inside the allowance it
explains — bounding the result must not be the thing that breaks the bound.

This is a **backstop**, deliberately loose, not the main mechanism. The server-side ceilings
remain the first line: they cut knowing the shape of their own payload, where a head-and-tail
cut here leaves a JSON document unparseable. What this catches is what no server ceiling
covers — a third-party MCP server, or one of ours with a path that forgot to bound itself.
It should almost never fire, and it says so in the status line when it does.

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
  as a gerund, and the remaining tokens appended. `shorten_display_args()` reduces
  **absolute** path arguments to their file name for display — a row carrying
  `/long/absolute/path/.../observations.py` buries the one token the reader is scanning
  for. Only absolute ones, since a tool that names a workspace file is given an absolute
  path, while a repository path on a remote-fetch tool is already short and its leading
  segments are what identify it: basenamed, the GitHub row said `ci.yml` for a call on
  `.github/workflows/ci.yml`.

  It also strips a url's `user:password@` from the display copy. The label interpolates
  the url verbatim ("Fetching {url}"), and that label is the row's tooltip, the approval
  card's header and a line of the stored transcript — so the credential reached all four.
  The scheme and the query stay: a consent prompt asks about a precise call, and only the
  credential is never part of one. It does not make the credential *confidential* — the
  model wrote it into the call and the call is in the history either way — it removes the
  incidental copy, on screen and in the transcript the user shares.

  **Never applied where the path is the decision.** An out-of-workspace approval asks the
  user to authorise *locations*, so the card carries the paths verbatim and the CLI prints
  each absolute path on its own line. Readability wins in the activity log; precision wins
  in a consent prompt.

  `tool_arg_preview()` finds the call's **salient argument** — the url, the file, the
  verdict, the job id, the queried name — from argument *names* in priority order, with
  `op` last, since an op selects an action and the model's own description already
  carries that half. Every one of these used to reach the row through the server's label
  template, and went off screen with it. A url keeps its host and path and sheds its
  userinfo and query on both branches, parsed or not: a row is read over a shoulder and
  then stored. `dedup_row_detail()` drops the preview when the description already says
  it, matched on **word boundaries** — as a substring, a verdict of `pass` was eaten by
  "recording the passing run" and an op of `now` by "knowing the time".

  `clip_doing()` is the row's other half: the model's own description of the call, cut to
  its first line and bounded. The 15-word limit is asked of the model and not enforced —
  a sentence cut mid-word reads worse than a long one — and this only stops a model that
  ignores it entirely from pushing the rest of the row off screen.

  The name-derived label is no longer what a row shows. The row shows the tool's declared
  work family (`kind_for`), that description and that preview; the label remains its
  tooltip, and is
  what approval cards and policy messages are written from. So all of the above is still
  live — just not in the activity row.

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
- `compact_history()` / `compact_messages()` — the model calls MIMIR makes on **its own
  behalf**. Both go through `get_backend()`; they used to call Ollama directly, which broke
  them under every other backend. A third used to live here — a classifier picking a skill
  from the query before the first step. It is gone: the model is given the index and loads a
  skill itself, so no round-trip is spent guessing from a request that does not yet know what
  the task is made of. Both pass a discarding token callback, because `chat` streams to
  stdout when given none — that default belongs to the CLI answer path, and without a sink a
  compaction summary is printed into the middle of the session. Both degrade quietly when the
  endpoint is down, so an unreachable backend costs a summary, not the turn.

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

The WebSocket / VS Code bridge: `ws_server.py`, `ws_worker.py`, `ws_pool.py`,
`ws_session.py`, `event_bus.py`, `turn_commit.py`, `job_scan.py`, `server_registry.py`,
`detach.py`, `_ws_runtime.py`, `session_store.py`, `session_summary.py`,
`transcript_log.py`, `file_preview.py`. The message protocol and the React frontend are documented in
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

**A resume is budgeted as the session it was.** The window is sized from the context mode,
which only the agent can answer for — and a resumed conversation has no agent until its
first query. So the mode is saved with the session (`context_mode`) and is what
`_ctx_budget` assumes until an agent answers for itself; the pre-query budget check runs
*after* `_ensure_worker`, never before. Read too early, it answered compact for a full-mode
session, called a 60k window an overflow, and compacted and trimmed to fit 32k.

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
The agent it runs on is resolved through the **pool, by that session** — never off
`self.worker`, which answers for the conversation being read. Aimed there, a wake either runs
behind another conversation's work with `is_busy` wrong about both, or reaches the detached
stand-in, whose queue no loop reads and whose own docstring says so.

**A wake is only off the pending map once a turn holds it.** The flush takes the wakes before
it can know whether it will succeed, and every way it fails — a session deleted under it, a
store that will not write, an agent released out from under it — is a finished run reported
to nobody. So each of those puts them back, `told` preserved, for the next flush to carry.

**A run that says nothing for an hour is a run going wrong unseen.** Check-ins break the
silence on a ramp — 30s, 2min, 10min, 30min from the first launch (`_CHECKIN_SCHEDULE`) —
and hourly from there (`_CHECKIN_INTERVAL`) for as long as the run lasts. The ramp widens
the way the watcher's own backoff does, and for the same reason: so does the cost of having
been wrong for that long. The hourly tail is the floor under it, because a schedule with a
last point goes quiet exactly where the stakes are highest — an overnight run is the one
with most to lose from four more unreported hours. Nothing but the run ending stops the
cycle. One cycle per conversation, not per job: a worker *is* a conversation, so jobs
launched in one step share a schedule and report together. It costs no status traffic —
each bulletin is built from `_Watch.status`, written by the poll the watcher is running
anyway.

Because that tail never runs out, a cycle can be asleep for an hour after its last job
ended, so **a registration that starts a wave replaces the cycle rather than joining it** —
otherwise a job launched into that gap would inherit the remainder of a schedule describing
nothing and wait out the hour for its first bulletin. Starting a wave means being the
conversation's only live run. The cycle clears its own slot on the way out only if the slot
still holds *it*: cancellation lands after the replacement is installed, and an
unconditional clear would erase the live cycle and let the next launch start a second one
beside it.

**A check-in never interrupts.** A wake carries a result the running turn needs; a bulletin
carries "nothing to report", and steering that in makes the agent answer about the job
instead of the question it was asked — while the silence it exists to break is not there at
all with the user watching it work. So an owner that is busy, or parked on a card a person
has to answer, gets nothing then: the bulletin waits in `_held_checkin`, one slot per
conversation so the next one replaces it, and goes out after the answer lands. It is dropped
rather than delivered if the runs have since finished — their wakes say more — or if a wake
of that conversation is itself pending.

**And a check-in leaves no bubble.** A wake's notice is transcript — a run ended, and the
reader wants that where it happened. A bulletin is plumbing: the client is sent the
`job_checkin` event and renders the news that runs are still there, while the text the
session layer builds is the instruction asking the model for its one line. Stored as a
display message it was replayed on every reload, one full block per wake the conversation
had ever had, each above the answer it had produced. So only a wake writes one, and
`_text_count` — the guard that keeps a freshly opened webview from blanking a stored
history — ignores the notices this layer wrote itself. It counted them, and since the
webview never received one, every transcript the client sent afterwards looked short by one
and was refused: the stored chat froze at the first notice and came back without a tool row,
a diff card or a reasoning panel from that point on.

**Asking where a run is at is what puts a watcher back on it.** A watcher lives on the agent
that made it and dies with it, which a window reload is enough to cause; the run carries on
in its own session directory, indifferent. Nothing re-arms at load — that would poll jobs
nobody is thinking about any more, in every conversation opened. Instead the status ops
answer with the handle as well as the state (`background_jobs`, plural, *these are still
going* — as against `background_job`, *this call launched it*, which makes the row a run and,
with no watcher, something to wait out in-turn), and `_maybe_rewatch_background_jobs` picks
them up. It reads the payload's shape and no capability: `BACKGROUNDABLE` says a tool
*launches* a detached run, and hanging it on a reader to make a key legible would say
something untrue about the reader. What is checked instead is that each descriptor names an
op to poll and a job to poll it for — the whole trust boundary, and enough of one, since what
registration buys is a read-only poll of a named tool. Loading a conversation whose jobs
nothing is watching says so in one line, and leaves the asking to the person.

**A watcher's own tool calls have a deadline, because nothing beneath them does.**
`session.call_tool` takes no timeout and the MCP sessions are built without a read timeout, so
a status op that stops answering — a wedged server, a scheduler that hangs, a queue command
that never returns — blocks the watcher for ever. That is the one failure that genuinely loses
a run, and the quietest: the job stays in `_bg_jobs`, so the agent is never released and never
woken, and the bulletins keep repeating a status frozen at the moment it stopped. A watcher
that *dies* at least says so. `_TOOL_CALL_TIMEOUT` (60s) is the line between "slow" and
"never", not a budget to work inside; a tick that crosses it counts as unreadable, which is a
case the loop already had, so `_UNREADABLE_POLL_LIMIT` of them in a row end the run as
`unknown` with the reason — an honest "I lost track of it" in minutes instead of silence. The
summary call is under the same deadline, and there it matters more: the run is already over, so
a summary op that never answers withholds the wake itself, and better a wake with nothing in
it than no wake.

**A connection that drops does not end the turn, and no longer costs its record either.**
An agent outlives the connections that read it, so a socket that dies mid-run leaves a turn
working with nobody watching. What used to be true as well is that nothing *recorded* that
work: the draining, the journal tee and the writing-back of a finished turn's answer all
lived in `_Session._drain_loop`, which is created per connection and cancelled when the
socket goes. A disconnected run wrote nothing, grew `out_q` for the length of the
disconnection, and lost its answer if it finished while away.

So the WebSocket is no longer the bus. `_EventBus` (`event_bus.py`), owned by the pool and
started by `serve()`, is the one pump for the process: it drains every worker through
`drain()` — already the single choke point for everything the engine emits — writes each
event to that session's `transcript.jsonl` under a monotonic `seq`, hands `answer` and
`error` to whatever commits a turn, and only then fans the event out to however many
subscriptions are attached, zero included.

```
worker.out_q ──(pump)──► seq ──► transcript.jsonl      (durable, authoritative)
                           │
                           ├──► turn committer         (the session file)
                           └──► 0..N subscriber queues ──► sockets
```

Three things follow. `out_q` is bounded by a pump tick rather than by how long the user was
away. The journal being authoritative means a subscription may *drop*: its queue is bounded,
overflow is reported as a gap, and the client closes that gap by replaying from its
watermark — the same path as a fresh attach, not a special case. And `seq` is assigned by
the pump and by nobody else, which is what makes it a watermark a reconnect can resume from;
anything else appending to a session's journal goes through `record_client_event` (a query,
a steer, a compaction record, a job wake).

There used to be a sweep on connect — `_drop_stale_events` — that emptied an idle
conversation's queue and spared a busy one's, with a carve-out for `job_complete` and
`job_checkin` because those answer to no turn. The carve-out was the tell: deciding what to
discard was the wrong question. Nothing is discarded now. The one thing not *recorded* is a
streamed delta (`token`, `thinking`, and the rest of `_SKIPPED_TYPES`): hundreds per turn, so
a replayed turn shows aggregates rather than keystrokes — which is what makes replay
affordable.

**That left a real hole, reported from a real absence**: the tool calls came back and the
prose did not. Correct for the deltas, but the consequence was that everything the agent said
*between* its tools was gone, and that is most of what makes a turn legible — only the rows
and the final answer survived. So each streamed prose block is aggregated into one
`assistant_text` event as it closes, flushed at every boundary so the record interleaves what
was said with what was called, in order. It is **replay-only**: a connected client has already
had every delta, and sending the aggregate as well would print the paragraph twice, so the
pump journals it and does not fan it out. A replay feeds it to the webview's reducer as a
`token`, which is what makes a turn read back after an absence lay out the way it would have
been watched — the same draft-and-boundary logic decides where the block sits relative to the
cards that followed. Reasoning is still not kept: it is the one thing a replay deliberately
sacrifices, and the one the aggregate is not worth paying for.

**Who writes a finished turn down.** The answer carries the turn's own transcript, and
unpacking it into the session file used to be `_Session._persist_detached_answer` — on the
drain loop, so a turn that finished unattended was never written. `turn_commit.py` holds that
logic now with no socket and no session object: it takes the store, the session id, the
answer, the private extras the pump took off it, where the turn began and the context mode.
The announcing stayed behind, because notifying the user, refreshing the session list and
flushing the wakes that piled up are things a *connection* does; a commit that insisted on
them would be back where it started.

Two of the arguments used to be read off the wrong place. The boundary — how long the history
was when the turn was submitted — lived on the session, per socket, which for a turn that
outlives its socket is a boundary nobody has; the worker records it when it takes the turn off
its queue and carries it on the answer. And `context_mode` was read off whichever worker was
on screen, which is the wrong agent whenever the turn belongs to another conversation; it
rides on the answer too.

The pump **offers** the answer and the committer takes it if nobody else does. A connected
`_Session` writes this itself — into `self.history` for the conversation on screen, through
`_persist_detached_answer` for any other — and two live writers of one file lose history
silently, so exactly one of them may write. What elects it is a claim rather than a count of
subscribers: see *Who is allowed to act* below, which is the same mechanism and the same
reason. A view that loses the claim re-reads the conversation the committer wrote instead of
appending its own copy of the turn. The belt to that braces is a
revision on the session file: `save_session` bumps `rev` and refuses a save whose base is
behind what is on disk, raising `StaleSessionWrite` rather than overwriting newer history with
older. A successful save carries the new revision back onto the object, so an owner that holds
one session and writes it as a turn progresses is never refused — only a second writer that
loaded earlier is, and the committer says so in the log instead of dropping the turn quietly.

**Who starts the turn a wake asks for.** The committer's twin, and the other half of
"nobody is looking". A watcher outlives every connection — it is a task on a worker's loop,
and `releasable` refuses to free a worker that holds one — so a detached run reports in
whether or not a panel is open, and the pump journals what it says. Acting on that report is
`_AgentPool.consume_durable_event`, reached for `job_complete` and `job_checkin` through the
claim below.

A wake only a connection can route is a conversation that waits out the night for a job that
finished in three minutes. So the pool does what the session does, minus the socket: loads the
conversation that launched the run, appends the wake to its stored history, writes the 🔔 the
user finds on their return, journals a `job_wake` where the turn begins, and queues the turn on
that conversation's own agent — then settles the run. All of it synchronous, inside the pump
tick that drained the event, because loading, saving and `submit_query` are plain calls. The
answer is written back by the committer off the same pump, against the boundary the worker
carried on it: a wake taken in this way is a complete turn, asked, answered and persisted,
with no connection anywhere in the path. A busy conversation gets the wake steered into its
running turn instead, and keeps it pending until that turn says it read it; a bulletin is
held rather than steered, because "nothing to report" must not make the agent answer about
the job instead of the work.

**Writing the answer down is only half of what a connection does when a turn lands.** The
other half is carrying on, and a detached process that did only the first half answers the
step that finished and stops there — which is the difference between a chain of steps that
runs overnight and one that stops at its first link. On every answer a socket puts the
steering the loop never read to a new turn, starts a turn for every run that finished while
it was busy, and delivers the bulletin it was holding. `_AgentPool._carry_on_after_turn` is
that, with no socket: it reads `_unconsumed_steer` off the answer — what the loop ended
without taking in, and the only place that is knowable — settles the wakes the turn did read,
and gives the rest one turn between them. A burst that finished together is one piece of news,
not three turns that answer the first and then re-answer it twice.

**Who is allowed to act.** Two consumers can now act on a finished run or a finished turn,
and only one of them may. The first rule was a test on the subscriber list — act only when
`bus.attached()` is zero — which answers a different question than the one that matters: a
subscription says a socket *exists*, not that it will route anything. A view whose drain loop
returned on a failed send, or that ended without closing its subscription, is a promise nobody
is keeping, and because nothing re-emits a wake the run it was holding then waits for someone
to open the panel and ask. One leaked subscription silenced the committer, every wake and
every check-in for the life of the process. (The leak itself is closed too: everything in
`_Session.run` from the subscription onward is under one `finally`, because every step of that
handshake sends on a socket that can close under it — a window shut mid-replay raises there —
which makes an unguarded exit the ordinary case rather than the exotic one.)

So the bus *offers* each such event and the pool acts on what nobody claimed. An attached
socket gets first refusal for `_CLAIM_GRACE`; the drain loop claims an event before doing
anything with it, and `_sweep_unclaimed` at the end of each tick claims whatever is past its
deadline and runs the pool's handler for it. With no subscription the deadline is now, so the
detached case keeps the latency it had. `claim()` is true for exactly one caller, and grants an
event this bus never offered — something that reached a consumer by another route reached only
that one. The grace is a latency bound on the handover, not a correctness condition: whichever
consumer claims first is the only one that acts, whatever the timing.

**And an agent is never released while anything of its is unresolved**, which is what makes a
wake for a closed agent impossible rather than merely handled. Idleness is counted from the
last time the pool was asked for the worker, so a conversation waiting on an overnight run has
been "idle" for hours throughout — the clock never protects it, and `releasable()` is the whole
of what does. Its clauses are the list of what *unresolved* means: on screen, a turn running or
queued, a job watched, a card up, a deferral owed, a wake owed, output undrained, an event still
to be claimed. The last three close the window a watcher opens by dropping its job the instant
it reports it finished — the one moment at which the agent is needed most. Every emitter of a
durable event runs on a worker's own loop, so with those clauses a wake cannot arrive for a
conversation whose agent is closed; `consume_durable_event` logs that invariant as broken rather
than rebuilding anything, and leaves the run unsettled so the next agent built announces it.

An owed wake also holds the process open. `idle_report` counts a finished run whose wake
nothing has taken in alongside one still going — the same debt one step later — so the server
cannot shut down owing a turn it never started. Only for sessions past their baseline: an
unbaselined one holds every run it ever finished with no marker on any of them, and reading
that as debt would leave a workspace permanently un-stoppable.

**Coming back is a window opening onto a run that never stopped**, and two things follow
from reading it that way. The conversation returns under the level it was left running on —
the registry entry carries it, and it is both applied and recorded pool-wide, because a
worker rebuilt during the absence would otherwise come up on the default and park at its next
sensitive call, silently dropping a run from `auto` to `manual` with nothing said. And a turn
still in flight keeps going: `session_loaded.turn_running` is what brings the chat back busy,
with a stop button that stops something, and — not cosmetically — knowing a turn is open so
the answer that ends it hands the finished transcript back.

Scoped to the run and not past it: the level is read from the registry entry, which a clean
stop clears and whose liveness is checked. That is what keeps this from becoming the thing the
approval mode is deliberately never persisted for — a mode that suppresses prompts inherited
by a later session that never asked for it.

**Coming back: replay, then live, by watermark.** The journal's `seq` is the watermark —
monotonic per session, resumed from disk, gap-free by construction — so nothing else needed
inventing. What a client has rendered is the client's own statement, because the rich
transcript is assembled in the webview and exists nowhere else: `transcript` carries
`through_seq` beside the messages, and it is stored on the session as `rendered_seq` rather
than kept per connection, so a fresh window or another machine can resume too. It only ever
moves forward, for the same reason the transcript itself is only ever taken when it is not
shorter than what we hold.

The no-duplicate / no-gap argument is an **ordering**, and `run()` is where it lives:

1. subscribe — *before* the handshake. From that instant nothing the agents produce can be
   missed; it accumulates behind the subscription while the rest of the setup runs.
2. greet, list the sessions, load the newest.
3. read the journal from the watermark and send it as `replay` frames, a few hundred events
   each: the frame cap is 32 MiB, which a long detached run's transcript can exceed whole,
   and chunking also lets the webview draw the first frame while the rest arrive. A cap on
   the total keeps the *end* of the run and says `truncated`.
4. **only then** raise the subscription's gate to where the replay ended. Events produced
   while step 3 was running are in both the file tail and the queue; the gate discards the
   copies at or below it, so the overlap collapses to exactly one. A gate raised before the
   frames were sent would have discarded those same events instead — which is why the order
   is not an implementation detail. A socket that dies mid-replay leaves the gate alone, so
   what it did not receive is still live.
5. resend the parked cards, last, so they land under a chat already on screen. A card is
   state, not a journal line — which is why `_resend_parked_prompt` exists beside the replay
   rather than being replaced by it.

A `replay` frame is a bundle of ordinary events and goes through the webview's ordinary
handler, which is what rebuilds the tool rows, diff cards and reasoning panels for a stretch
the window was not there for — one reducer for what happened live and what happened while
nobody was looking.

**One gate per conversation, because `seq` is counted per conversation.** Each chat has its
own journal and its own writer, so a number from one says nothing about another: a single
gate on the subscription, raised to the position of whichever chat the connection opened on,
discards every event of every chat whose journal is shorter — which on a socket that opens on
a long conversation is every new one, since its own `seq` starts again at 1. The symptom is
specific and worth recognising: the tool rows never appear **live** and all of them appear on
the next reconnect. A replay frame is written straight to the socket and passes no gate, while
the live copies were stamped and dropped — so the journal is complete, the chat is not, and
nothing about the connection looks wrong. The subscription therefore keeps a watermark per
session id, and `wants()` compares an event against its own conversation's.

**The gate is clamped to what the journal actually holds, and that is not a nicety.** The
watermark arrives from the client and is stored, so it can outlive the journal it counted — a
session whose log was removed, or a number inherited from another conversation. A claim to
have rendered more than exists cannot be true, and honouring it sets the gate above every
event that session will ever produce. The clamp is read against that conversation's own
journal, which is what makes it sufficient. What that looks like is worth knowing, because it is
not a dead connection: `token` and `thinking` are never journaled, so they carry no seq and
bypass the gate entirely, while everything stamped is filtered. The chat streams the answer's
opening text, reasons for ever, and shows no tool call, no diff and no answer. A filtered
event is counted and logged for the same reason — the failure is otherwise invisible from
outside.

**Re-making the promise to report a run, without being asked.** A background run survives
anything: its own process session, a trap that writes the exit code, a descriptor in its own
directory. The *watcher* does not — it is an `asyncio.Task` on a worker's loop, which
`shutdown()` cancels. The run carries on indifferent; only the promise is lost. That was
already recoverable through a status tool returning `background_jobs`, but it needed a turn
in which somebody asked where the job had got to.

`job_scan.py` removes the asking. It reads the descriptors directly — `meta.json`, the
`exit_code` the trap wrote, and the pid liveness contract — and sorts what it finds into two
piles. A run still going needs a watcher, which is put back when that session's worker is
built (on the loop, not in the constructor's executor: a task created off a running loop
polls nothing). A run that *ended* while nothing was listening needs its wake, and that is
handed to `_handle_job_complete` — deliberately the same routing, wake text and coalescing a
watcher's report goes through, so a run reported late is indistinguishable from one reported
on time. A conversation with no agent yet keeps the wake pending rather than losing it.

Two places look for the ended pile — a connection arriving (`_report_ended_jobs`) and a
worker being built (`rearm_detached_jobs`) — so a marker file in the run's own directory
records what has been settled, outliving whichever process settled it: a run delivered twice
is a conversation woken twice for one build.

**A Slurm submission has to be told it is over.** Its state lives in the controller, not in
a pid here, so inspection cannot conclude anything: what the directory holds is a job id, and
an id never stops existing. Read on its own that is "there is a job, ask Slurm" for ever —
which is how a workspace that had submitted one job ended up with a server that could never be
idle, since the same scan answers "does this workspace still have work running". So
`slurm_job_status` writes the state into the submission's own directory the first time Slurm
has let the job go, and that record is this path's equivalent of the exit-code trap: the first
observation stands, because sacct's retention window expires and a later poll of the same job
reads `unknown`. `unknown` is recorded too — the scheduler forgetting a job is an ending, and
the one thing that must not be written is nothing at all. A submission nothing will ever
settle — no conversation left to poll it, or a machine whose Slurm tools have gone — ages out
of the live pile after `MIMIR_SLURM_STALE_AFTER` (a week), deliberately far longer than any
wall-time: while a server is up its watcher settles a finished job within one poll, so the
horizon only ever bites a job nothing is watching, and erring short would stop a server that
still owes a conversation its wake.

**The marker records delivery, not emission.** It is written where the wake enters a turn —
`mark_wakes_reported`, called by whichever consumer submitted it — and never where the event
is queued. A `job_complete` put on the bus with nothing ready to read it is a wake still
owed; settling it at that moment makes the only record of the debt say it was already paid,
and the run is then filtered out of every later scan. That is a conversation waiting for ever
on a job that finished hours ago, which is precisely the failure this path exists to fix. So
an announcement repeats until a consumer has taken the wake in, and the consumers dedup
instead: a job already pending for a conversation is not added twice. A steered wake is
deliberately left unsettled — a steer is only known to have been read when the loop says so,
and the flush that eventually gives it a turn writes the marker. The marker is best-effort,
because a duplicate wake is a nuisance and a missing one is the bug.

**The first look at a session establishes a baseline rather than claiming a backlog.** A
detached job's directory is never swept, however old, so an existing workspace has every
build it ever ran sitting there unmarked. Read as wakes owed, that is a conversation woken for
a two-month-old build on the first attach, and unlike a missed wake it is unbounded. So a session with no
`.wake_baseline` has its finished history marked and nothing emitted; from then on a run that
ends is genuinely one nothing has spoken for. A run still *going* is re-armed either way: the
baseline is about what has ended. Slurm
jobs are never declared finished by inspection: their state is in the controller, not in a pid
here, so they come back as live and the watcher's first poll settles it.

The liveness check is a deliberate twin of `_bash_jobs._is_running` rather than a call into
it — the MCP servers resolve imports against their own directory, so nothing in
`mimir.client` can import that tree, the same reason `tool_execution/run_channel.py` twins
`servers/_shared/run_channel.py`. What must change in both together: a pid is alive only if
it exists, its start time matches the one recorded, and it is not a zombie.

**Being findable at all.** A spawned server's port is learned by regexing its stdout, which
works exactly as long as the extension host is the parent holding that pipe. One meant to
outlive that window has to leave its address on disk, so `server_registry.py` writes
`<STATE_DIR>/server.json` once the socket is actually bound — with `--port 0` the argument
was a placeholder and only the bound socket knows the answer. The choice of directory *is*
"one server per workspace": `STATE_DIR` is already `<state home>/<basename>-<sha1(realpath)[:8]>`,
so two checkouts sharing a basename do not collide and two windows on one workspace resolve
to the same file.

**One server per workspace is enforced, not assumed**, and this is the piece that matters
most. Two servers sharing a workspace share its sessions directory: both append to the same
`transcript.jsonl` and both derive `seq` from it, so the numbering collides, the watermark
built on it stops meaning anything, and a client attached to one sees nothing of the turn
running in the other. What that looks like from a chat window is a conversation whose tools
run and never appear — until a reconnect happens to land on the other server and replays
them. It was diagnosed from `/diag` reporting a running pump that had moved zero events, no
worker in the pool, and a journal already at seq 55.

So `serve()` takes an exclusive `flock` on `<STATE_DIR>/server.lock` **before binding**, and a
loser does not serve. `flock` rather than a file the process writes, because the kernel
releases it however the process dies — a crashed server must not keep a workspace locked, and
nothing the process writes can promise that. A loser that finds a live entry prints
`Listening on <the winner's url>` on the line the extension parses, so a caller that meant to
spawn connects to the server that already serves this workspace instead of adding a second;
a loser that finds no address refuses outright rather than serving unlocked, which is the
state that corrupts the journal. The extension asks the same question before spawning —
auto-connect already did, and the Connect button did not, which is how the second server came
to exist.

Liveness is three questions, and the third is the one the other two cannot answer: the pid
exists; its start time matches what was recorded, because a recycled pid wears the same
number; and the port actually accepts a connection, because a process can be alive with its
listener already gone. The first two come from `job_scan`, which holds that contract for
detached runs. The entry is advisory throughout — a stale one costs a connection attempt that
fails and is then replaced, whereas trusting it is how a window ends up waiting on an address
nothing is listening at. It is removed on a clean shutdown and deliberately left behind by a
crash, since a reader checks liveness rather than the file's existence; `clear()` refuses to
delete a stranger's entry, because that would make a perfectly good server undiscoverable.

The extension reads the same file through `src/serverRegistry.ts`, and `workspaceId` there
must stay byte-identical to `state_paths.workspace_id` — disagree and each end looks in a
different file while both conclude there is no server. `serverRegistry.test.ts` pins it
against captured fixtures rather than recomputing the formula, which would pass even if both
ends changed together in the same wrong way. With a live entry the extension takes the attach
path `mimir.wsUrl` has always taken: nothing is started, `serverProcess` stays undefined, and
the server is never torn down — it is not this window's to kill, which is the right
relationship with one that was deliberately left running.

**Going on without the window.** Detaching is something the server does to *itself*, on a
`detach` message, and the spawn is never touched — which is what lets the decision be made
when the user is leaving rather than when they connected. At connect time there is nothing to
decide about; the question is whether anything is worth leaving running, and that is only
answerable once something is.

Two halves, and they are not equal. **The pipe is the real killer**: when the extension host
dies its end closes, and the next write here takes a SIGPIPE — a server that survives the
window only to die the first time it logs something has not survived. So `detach.py` re-points
fds 1 and 2 at `<STATE_DIR>/logs/server-<pid>.log` with `os.dup2`, and Python's `sys.stdout`
follows, because the file object writes through the descriptor rather than around it. That is
also what makes a detached server's output readable afterwards instead of lost. **Leaving the
process group is insurance**: `os.setsid()` succeeds only for a process that is not already a
group leader, and a child of `cp.spawn()` without `detached: true` inherits its parent's
group, so it is not one — but the extension host has no controlling terminal, so no group-wide
SIGHUP is coming either way. A declined `setsid` is logged and stepped over; it must never
cancel a detachment whose essential half already succeeded. Nothing here can be undone: a
detached server has no pipe to go back to.

**Detaching does not disconnect**, and that is the point of the shape: the socket stays
open, the turn goes on in front of the user, and what changed is who owns the process.
Which makes it revocable — `enabled: false` clears the claim of the conversations it
names, or of all of them, and a server nothing claims any more is mortal again. The process-level work is not undone and does not need to be: fds pointing
at a log file are harmless either way, and what makes a server survive a window closing
is that nobody kills it. The claim travels in the reply as an explicit flag rather than
being implied by the message's arrival, so the two directions cannot be confused.

The autonomy level rides on the message and is applied through the same seam `/approvals`
uses, so a turn already in flight picks it up at its next gate. Naming conversations sets only
those; naming none means all of them and also records the level pool-wide — the only form that
outlives a worker being rebuilt, since the pool records UI settings per pool rather than per
session. The registry entry is updated rather than rewritten: the address was settled at bind
time and has not changed, so only the claim and the log path are added.

**The claim is per conversation; survival is per process, and it cannot be otherwise.** The
entry holds `detached_sessions: {session id: autonomy}`, so each conversation is left under
its own level and taking one back leaves the others running — one flag for the workspace both
restored a conversation left at `auto` under another's `auto_all`, and made a second
conversation's run mortal the moment the first was taken back. What stays indivisible is the
*process*: one server per workspace by construction (one journal, one `seq` writer per
conversation, one `flock`), `dup2` on fds 1 and 2 belongs to the process rather than to a
conversation, and each agent's ~19 MCP servers are its children. So the process lives for as
long as any conversation claims it, and `detached` is kept beside the map as "anybody at
all" — which is what the extension reads to decide whether to skip the kill, a decision about
the process that stays right.

The extension reads `detached` in the host, not the webview, because it is the host that does
the killing: a module-global flag beside `serverProcess`, and `deactivate()` and
`_teardownServer()` both skip the kill when it is set. Three servers are not this window's to
end, each for its own reason — one it only attached to, one that has detached, and one already
gone.

**Which leaves "disconnect" with one meaning, and that is the point.** A server exists
claimed — detached, asked to be left running — or owned by the window that started it. There
was a third state and it was the fragile one: a server a window had merely attached to,
nobody's to kill, still writing to a pipe whose reader had gone. It kept working, unobserved,
and nobody had asked it to. So a server no conversation claims stops once its last client has
been gone for the grace, whoever started it — `_AgentPool._nobody_asked_to_keep_it`, checked
ahead of the idle predicate because a server nobody wants has no reason to live whatever it is
in the middle of.

Server-side, deliberately: the only place that can decide this reliably is the process itself.
An extension host that is dying has no time to end a process it does not own and may be killed
before it tries. The grace is the one a parked card already uses (`MIMIR_DETACH_GRACE`, 30s),
because both are answering the same question — has the window actually gone — and a window
reload closes and reopens the socket. A server that has never had a client has lost nothing and
is left alone, which is what keeps a standalone run from stopping itself out from under the
window about to attach.

The run is not killed either way: it has its own process session and a trap that writes its
exit code, so what the shutdown ends is the *watching*, and the next agent built for its
conversation picks it up again. And the dialog can now say one true thing — disconnecting ends
the server and their turns with it — where describing two outcomes meant guessing which one
applied.

**A card nobody can answer is set aside, not waited out.** The wait behind an approval
passes no timeout, deliberately: nothing may proceed because the user was slow. That is right
while somebody is there and a deadlock once nobody is — a detached run meets its first
sensitive tool, holds the worker thread for ever, and the pool then reaps the agent out from
under it. So the condition added is *attached*, not *elapsed*, and `test_approval_wait` stays
green unmodified, which is the check that no timeout crept in.

The mechanism was already written and had no caller: `query_engine/deferral.py` parks a turn,
hands the call a placeholder result, and resumes it when the answer comes. `_await_response`
already polled in quarter-second slices and already had a `_deferring()` branch; one condition
joins it. The grace period (`MIMIR_DETACH_GRACE`, 30s) is not decoration — reloading a VS Code
window closes and reopens the socket, and parking every card on that blink would put the
user's own question away under their nose.

**The order inside the loop is load-bearing**, and a test found it: the queue is read before
the grace is consulted. Checked the other way round, a reply that crossed with the grace
elapsing was discarded and the user had answered into a void.

The worker that needs to know is a *thread*, blocked in a queue poll; it cannot await
anything and must not touch loop state. So the pump pushes a plain timestamp onto every worker
each tick — including the ticks that move no events, since what it reports is the absence of
events. And `releasable()` gains a clause: a deferred turn has cleared its pending card, so
the parked-on-a-card clause no longer sees it, while the user still owes it an answer and
releasing the agent would throw away the ~19 servers that answer is resumed against. (It has
since gained three more, for the same shape of debt one step later: see *Who is allowed to
act*.)

**When a detached server stops.** The unclaimed case is settled above and never reaches
here. What remains is a server conversations *did* ask to keep: it is no longer killed by
closing the window, so it needs its own answer to "am I still needed", and
`_AgentPool.idle_report()` is that answer in one interrogable place — read by the idle shutdown and by anyone asking whether this window
has anything worth leaving running.

The criterion is **positive**. "Not busy" is not "has finished": a worker is also not busy
between queries, after a turn that broke, and while it waits behind a card, and an idle test
built on the absence of noise stops a server whose turn merely paused. So a conversation
counts as finished only once it has *concluded* — delivered a final answer, or ended in an
error, which is also an ending and is tracked as one lest a failed session look busy for ever
— which the bus records as it hands that event to the committer. And a run launched but not
collected is work in progress with no turn running at all: the "I submitted a two-hour build
and left" case, so a live job on disk holds the process open. Read from disk rather than from
the pool, because a run outlives the agent that launched it and its session may have no worker
right now; an unreadable state dir counts as busy, since erring the other way stops a server
mid-build. The remaining clauses are not activity but prohibitions — a client attached, a card
parked, a deferral owed, a turn queued for a slot, an agent being built.

The TTL (`MIMIR_SERVER_IDLE_TTL`, two hours) is measured from when the process *first* looked
idle and is reset by any activity, so the answer is "idle throughout" and not "idle at some
point" — the difference between stopping a forgotten server and stopping one between two turns
of a conversation the user is coming back to.

Three doors, one exit: a signal, an explicit `shutdown` from the client, and the pool deciding
it is no longer needed all resolve the same future, because what has to happen on the way out
— unwinding each agent's exit stack — is the same in every case and is the only thing that
reaps the MCP servers. A `shutdown` is *refused* while anything is still working, with the
reasons sent back, unless forced: an arrest that silently discarded a two-hour build would be
the worst answer this path could give. The VS Code command signals rather than asks over the
socket, because the window that wants the server stopped may not be connected to it — which is
the whole situation it exists for — and it sends SIGTERM, never SIGKILL, for the same reason
the exit matters.

**Asking the chain what it did.** Every link of this was tested on its own — the pump, the
watermark, the committer, the approval wait — and that is exactly how three separate faults
reached a running install: each piece was right, and what broke were the joins. A chat window
cannot tell those apart. The pump never starting, the pump dying, a watermark swallowing what
it delivers, a turn belonging to a conversation that is not on screen, and the agent emitting
nothing all look like a conversation that went quiet.

So the bus counts what it does — ticks, events stamped, delivery attempts, pump errors, and
per subscription the watermarks (one per conversation), the filtered and the dropped — and `/diag` reports them beside
the active session, the stored watermark and each worker's queue depth. One answer, and the
candidates separate. `test_event_chain.py` is the other half of the same lesson: a coarse test
that builds a real session over a real bus and a real worker queue, runs the handshake, sends
a query and asserts that what the engine emitted arrived. It is the test that was missing.

What has not changed is that the drain loop *handles* a durable event before it sends it: the
event has already left the pump, nothing re-emits it, and a socket dying on the send must not
be able to take the handler with it. Nor has the rule about cards: every parked card comes
back, in every conversation — each wait has no timeout by design and no card left to end it,
so a card left out is a conversation stopped for ever with nothing on screen to say why. A
card is state, not a journal line, which is why `_resend_parked_prompt` still exists beside
the replay.

**Conversations run at the same time, one agent each.** One worker used to serve every
session, which is why leaving a conversation had to cancel its turn — or defer it when it was
parked on a person. That was never a policy: a single worker cannot stream two conversations
anywhere the user can see, and a card it was parked on would have been answered from the next
conversation's UI. `_AgentPool` gives each conversation its own agent, so leaving one leaves
its turn running; its output goes to its own transcript — written by the pump, so it is
written whether or not anyone is attached — which is what makes coming back show the whole
turn rather than only the answer that ended it.

What that costs is that "the running turn", "the parked card" and "the answer" stop being
questions with one answer, and each one got wrong fails silently rather than loudly. So
everything addresses a conversation: busy, cancel, steer, and the divertible tool rows —
two conversations can each have a blocking run, and one map between them would let a divert
click detach the other's command. **An answer to a card carries the conversation that asked**,
copied off the card; an answer naming none is dropped rather than handed to whoever is on
screen, which would settle a question another conversation asked with the user's approval
attached to a call they never saw.

Three things make an agent per conversation affordable. It is built on that conversation's
**first query**, not when the session is created — the backend wait plus some nineteen MCP
servers is a cost only a conversation that asks something should pay. The number alive is
**capped** (`MIMIR_MAX_LIVE_SESSIONS`, three): the binding constraint is processes, roughly
sixty interpreters at that cap, and past it a turn waits for a slot and is told so, because a
queue nobody can see reads as a hang. An idle agent is **released**, servers included — but
never one that is busy, parked on a card, watching a background job, or on screen. When
nothing may be released the turn queues rather than evicting: taking a slot from a
conversation mid-task trades a visible wait for silently lost work.

**A slot is admitted the moment a turn ends**, on the on-screen path and the detached one
alike — not only in the pool's thirty-second idle sweep, which left a conversation just told
it was next in line sitting out the rest of that interval after the slot it needed had
already freed.

**Releasing an agent closes its MCP servers, in the task that opened them.** `stdio_client`
anchors its anyio cancel scope to the entering task: exiting the exit stack from any other
task raises and leaves it half-unwound — the streams closed, the subprocess alive — so
nothing is reaped and a turn still running on that agent fails its next tool call on a dead
stream. `_AgentWorker` therefore holds setup, the query loop and the close in one task
(`_live`), and the close runs in that loop's `finally`, which is after any turn. `aclose`
only sends the sentinel and waits, cancelling a turn in flight first so shutdown does not
cost the rest of a turn per conversation. Eviction asks `has_work_pending` rather than
`is_busy` for the same reason: a turn already submitted has not set `_current_task` yet, and
releasing in that gap closes the servers under a turn about to start.

**Stopping the server stops the MCP servers.** Each is spawned with
`start_new_session=True` — its own process group, which survives this process dying — so
only closing the owning agent's exit stack terminates one. `serve()` installs SIGTERM/SIGINT
handlers and closes the pool on the way out, every agent at once rather than one after
another. The VS Code extension kills and respawns the server on every connect, which makes
that the ordinary path rather than an edge case.

**Deleting a conversation that is working says so first.** A turn in flight counts as live
work, and so does a turn parked on a card — work waiting to continue — alongside the job
directories. The delete is refused once, naming what is running; asking again goes through.

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
| `test_event_chain.py` | coming back to a running turn — the conversation restored under the level it was left on, a rebuilt worker coming up on it too, an ordinary connect deciding nothing about it, a turn in flight reported as running and an idle one not, and what the turn produced while away coming back — that an absence keeps the prose: the record interleaving what was said with what was called, the aggregate not being sent to a client that already had the deltas, a reconnect replaying both, and a blank block not recorded — plus the whole chain with the real objects: handshake, query, and the turn's `status` / `tool_call` / `tool_result` / `answer` arriving, nothing lost to the watermark, the journal holding the same turn the client saw, a second turn arriving, a brand-new conversation not silenced by an inherited watermark, the approval card reaching the client and its answer coming back — plus the ways a chat *can* go quiet, pinned so each is deliberate |
| `test_event_bus.py` | the pump: that it drains and journals with nobody attached, that `seq` has one writer per session and no gaps, that the private answer keys reach the committer but neither the journal nor the wire, that a streamed delta is delivered live and never recorded, that an overflowing subscriber is gapped rather than allowed to stall the pump, and what counts as a session having *concluded* — the private keys reaching the committer once its first refusal expires, since an attached view is given the chance to write the turn itself |
| `test_detached_commit.py` | the turn committer: an answer landing in its session file with nobody attached, a deferral stored as the card to put back, the answer-alone path for a non-full context mode, that the pump commits what no attached view claimed and never a cancelled turn, and the revision guard — a second writer that loaded earlier is refused, an owner saving repeatedly is not, a failed write does not consume a revision, and a file from before the field existed still writes |
| `test_reattach_replay.py` | that the gate cannot silence a stream — a watermark above the journal is not honoured, the stream still arrives after one, a filtered event is counted rather than vanishing, and a stream delta bypasses the gate (which is why the failure looks like a half-working chat) — and that a new conversation clears the watermark instead of inheriting it, and that one conversation's gate never silences another's live events while still holding for its own; then the watermark: everything replayed to a client that has seen nothing, only the tail to one that has seen some, nothing to one that is current; that the gate lands where the replay ended so an event is never both replayed and delivered live; that nothing produced between subscribing and reading is lost; framing, the cap keeping the end and saying so, a socket dying mid-replay leaving the gate alone; and that streamed deltas are absent while their aggregates are not |
| `test_job_rearm.py` | the baseline — a first scan reporting nothing while recording that it looked, a job ending *after* it reported, a live run re-armed either way, and a watcher's own report marking the run so a restart does not repeat it — then the scan: a live run reported live, a recorded exit code winning over whatever the pid looks like, a dead pid with no code reading `unknown` rather than `done`, a recycled pid not mistaken for the job, an ephemeral scratch buffer skipped, a Slurm job live until Slurm says otherwise and over once a poll has written down that it is — `done`, `crashed`, and the scheduler having forgotten it, which claims no outcome — an unreadable state word settling nothing, a submission nothing can settle ageing out, the status op naming a tool the HPC server really registers, and a marker landing in the submission's own directory while one written under the bare id still counts — then the re-arm itself, the wake going to the session that launched the run, a run already watched left alone, and the report-once marker |
| `test_server_registry.py` | one server per workspace, with real contending subprocesses because `flock` is a kernel object a mock would not exercise: a second process is refused while the first holds the claim, the lock is free again once the holder is *killed* rather than stopped, and two workspaces do not contend — then the registry: a published entry reading back with its pid and start time, an unreadable or wrong-protocol file reading as nothing, a failed write leaving no half file — and liveness over real sockets and real pids: a live pid whose listener has gone is *not* alive (while the process-only answer still says yes), a recycled pid is not mistaken for the server, `clear()` retires our own entry and leaves a stranger's |
| `test_hot_detach.py` | detaching, exercised against the real system calls because a fake `dup2` would prove nothing about the thing that breaks: output following the descriptors into the log while the parent's pipe sees only what preceded the redirect, a child **surviving two hundred writes after its reader is gone**, a second detachment appending rather than truncating, `setsid` succeeding for a non-leader and declining for a leader without cancelling the redirect — then the handler: per-session autonomy touching only the sessions named, naming none recording it pool-wide, an unknown level refused with nothing detached, the registry entry keeping the address it was serving on, and that there is no terminal whichever conversations are named — then the claim, per conversation: each one coming back under its own level, taking one back leaving the others running, taking the last one back making the server mortal again — and what coming back says: the level named, informatively rather than as a warning, with no explanation of a control already on screen, nothing at all under `manual`, and the notice not kept in the conversation |
| `test_unattended_park.py` | the parking: a card with somebody there still waiting for ever, one with nobody there deferred through the pre-built mechanism, the grace period leaving room for a window reload, a wait with its own deadline left alone — **an answer already in hand, or landing during the poll, winning over the grace** (the bug this file found) — the pump publishing attachment on ticks that move nothing and not restarting the clock each tick, and `releasable()` refusing a session that holds a deferral |
| `test_wake_claim.py` | who is allowed to act, and that somebody always does: a wake taken in within the tick that drained it with nothing attached, an attached view given first refusal and the pool standing down, a claim granted to exactly one caller and to one this bus never offered, a view that routes nothing not costing the run its turn (with the take-over counted and logged), the run settled by whoever ends up delivering it, bulletins going the same way — then what a turn landing carries on: a steered wake the turn never read getting its own turn, one the turn did read settled instead of told twice, a burst arriving as one turn, the 🔔 of a steered wake, and a held bulletin delivered after the answer or dropped once its runs have finished — then the release invariant: an agent watching a run, owed a wake, holding an event still to be claimed or output still unread is never released while one with nothing outstanding still gives up its slot, so a wake for a closed agent cannot arise and is logged as a broken invariant if it does — and the subscription's lifetime: released on a handshake that raises, on a greeting that cannot be sent, and on an ordinary end, with a run finishing afterwards still answered for |
| `test_idle_predicate.py` | what counts as finished: a concluded conversation with no jobs idle, one that never answered *not* idle, an error counting as an ending, a live run on disk holding the process open even for a session with no agent, an unreadable state dir counting as busy, and a Slurm submission holding it open until a poll has settled it, still holding it while its wake is owed, and letting go when nothing can ever settle it — plus each prohibition (attached client, parked card, owed answer, queued turn, agent being built, a worker that cannot be asked), and the clock: it starts rather than stopping at once, stops on the TTL, and is reset by activity rather than shortened — then the state that must not exist, a server nobody claimed and nobody is watching: a window still here keeping it whatever the claims, a reload not counting as a departure, gone past the grace with no claim ending it, one conversation's claim being enough for the whole process, a server never yet connected to having lost nothing, and claims that cannot be read not reading as nobody |
| `test_background_jobs.py` | the whole detached-run path: the server descriptor, the registration hook, `_watch_job`, the wake text, the detached resume and its coalescing — plus the check-in schedule and its never-interrupt rule, the three ways a finished run used to wake nobody (the wrong agent, a store that would not write, a socket dying on the send), that a reconnect throws nothing away, and re-arming a watcher from a status result — plus the fourth and quietest way, a tool call that never answers: a status op that hangs ending the run as `unknown` rather than polling for ever, a timed-out probe reported as `unreadable` rather than as still running, and a summary op that hangs not withholding the wake |
| `test_policy_manager.py` | the gates and the state guard |
| `test_client_helpers.py` | the nudge predicates, token counting, eviction and `ContextOverflowError` |
| `test_capabilities.py` / `test_phase_b_servers.py` | `infer_tool_caps` precedence and the golden declared registry (`_golden_caps.py` AST-parses the server decorators) |
| `test_capability_consumers.py` | drift guard — fails if a declared capability has no live consumer |
| `test_session_persistence.py` | the window/record split, WS compaction and its front-trim fallback, which one a resume restores, the turn boundary in both stale directions, and the cumulative marker count |
| `test_scratchpad.py` | home resolution, the standing grant being the home, `ensure_scratch_home` refusing a symlink / foreign owner, and scratch writes staying out of `dirty_written_files` |
| `test_absolute_paths.py` | every mutating tool rejects a relative path, names the resolved candidate, and writes nothing on rejection; every read server admits the scratchpad |
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
