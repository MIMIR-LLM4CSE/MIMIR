# MIMIR Servers Reference

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative per-server tool catalog — every server under `mimir/servers/` and the
tools it exposes via `@mcp.tool()`. For a one-line summary of each server, see the
[Registered Servers table in the README](README.md#registered-servers).

Servers are organized into domain-based subdirectories:

| Directory | Purpose | Servers |
|---|---|---|
| `_shared/` | Cross-group utilities | `responses.py`, `capabilities.py`, `root_paths.py`, `approved_roots.py`, `trusted_read_roots.py`, `text_tools.py`, `module_env.py`, `embed.py`, `vector_cache.py`, `slurm_nodes.py`, `slurm_script.py`, `cpu_facts.py`, `lsp_client.py`, `shell_paths.py`, `state_paths.py`, `numerics.py`, `proc_run.py` |
| `workspace/` | File & code interaction | `server_bash`, `server_files`, `server_search`, `server_code_intel` |
| `utilities/` | Stateless data helpers | `server_math`, `server_strings`, `server_datetime`, `server_symbolic_math` |
| `agent_state/` | Agent memory, planning & delegation | `server_memory`, `server_todo`, `server_spawn_agent` |
| `interaction/` | Asking the user structured questions | `server_interaction` |
| `external/` | Network & remote APIs | `server_github`, `server_web`, `server_system` |
| `hpc/` | HPC & platform profiling | `server_hpc`, `server_platform`, `server_env` |
| `proxy/` | Scientific proxy optimization | `server_proxy` (7 op-dispatched tools: registry, refs, runs, suites, eval loop, Slurm) |

## Tool catalogue

Every `@mcp.tool` the bundled servers expose — 70 tools
across 19 servers, all registered by default. Each server has its own section
below with the arguments and the behaviour.

| Server | Tools |
|---|---|
| `workspace/server_bash.py` | `bash_run`, `bash_job`, `bash_job_stop`, `report_verdict` |
| `workspace/server_code_intel.py` | `find_definition`, `find_references`, `symbol_outline`, `hover` |
| `workspace/server_files.py` | `write_file`, `append_file`, `delete_file`, `replace_in_file`, `replace_lines`, `replace_all_in_file` |
| `workspace/server_search.py` | `read_file_lines`, `list_directory`, `tree_summary` |
| `agent_state/server_memory.py` | `memory_add`, `memory_search`, `memory_list_all`, `memory_update`, `memory_delete`, `memory_clear` |
| `agent_state/server_spawn_agent.py` | `spawn_agent`, `subagent_job` |
| `agent_state/server_todo.py` | `todo_set_plan`, `todo_read_plan`, `todo_list_plans`, `todo_delete_plan`, `todo_write`, `todo_read`, `todo_read_ready`, `todo_update` |
| `interaction/server_interaction.py` | `ask_user_question` |
| `utilities/server_datetime.py` | `date_op` |
| `utilities/server_math.py` | `evaluate` |
| `utilities/server_strings.py` | `string_op` |
| `utilities/server_symbolic_math.py` | `symbolic` |
| `external/server_github.py` | `github_repo_info`, `github_list_branches`, `github_list_issues`, `github_get_file`, `github_search_repositories` |
| `external/server_system.py` | `system` |
| `external/server_web.py` | `http_get`, `http_post`, `parse_json`, `json_extract` |
| `hpc/server_env.py` | `env_pip_install`, `env_pip_uninstall`, `env_create`, `env_delete` |
| `hpc/server_hpc.py` | `slurm_partitions`, `slurm_nodes`, `slurm_queue`, `salloc_submit`, `slurm_job_status`, `sbatch_submit`, `slurm_probe_node`, `slurm_node_profile` |
| `hpc/server_platform.py` | `platform_probe`, `platform_get_profile`, `platform_search`, `platform_catalogue_status` |
| `proxy/server_proxy.py` | `proxy_get`, `proxy_runs`, `proxy_eval_status`, `proxy_manage`, `proxy_exec`, `proxy_eval`, `proxy_slurm` |

What a tool *may do* is not listed here: capabilities are declared per tool and the
authoritative table is in
[PLUGINS_DETAILED.md](PLUGINS_DETAILED.md#tool-capabilities).

Response contract:
- success payload: `{"status": "ok", ...}`
- error payload: `{"status": "error", "error": "...", "hint"?: "..."}`

### Timeouts that stop the work (`_shared/proc_run.py`)

`subprocess.run(..., timeout=…)` kills the process it launched and nothing that process
forked. The bash tool runs `bash -c "<preamble><command>"`, which is compound, so bash
**forks** rather than execs — the kill landed on bash and the fork ran on. Observed:
`find / -name …` outliving its 30s timeout by an hour and forty minutes, three at once,
while a ratchet next door was timing a solver on the same disk. Leaked processes consume
the machine, and being invisible to the tool that started them they silently corrupt
every measurement taken afterwards.

`proc_run.run()` is a drop-in for `subprocess.run` (same `CompletedProcess`, same
`TimeoutExpired`) that gives the child its own process group via `start_new_session=True`
and signals the **group** on timeout — SIGTERM, then SIGKILL after a short grace. The
session leadership is the safety property: the child's pgid equals its pid and can never
be the caller's own group, so killing it cannot reach MIMIR. Used where MIMIR runs code
it did not write: `server_bash._run` and both proxy execution paths in
`proxy/_lib/execute.py` (a solver may fork MPI ranks; left running past a timeout they
compete with the very next timing the ratchet takes). Fixed-argv callers
(`module avail`, `squeue`, the LSP) are unchanged.

Normalization is handled by `mimir/servers/_shared/responses.py`:
- `ok(...)` always preserves `status = ok`
- `err(...)` always preserves `status = error`
- reserved keys (`status`, `error`, `hint`) are protected from payload override
- error messages are whitespace-normalized
- contextual hints may be added automatically when no explicit hint is provided

### Shared embeddings (`_shared/embed.py`)

Backend-aware text embedding with a lexical fallback, used by both the memory server
(`server_memory.py`) and the client tool-ranking policy. It dispatches on the same
`LLM_BACKEND` as chat: vLLM (`POST <base>/v1/embeddings`, OpenAI-compatible, via `httpx`)
or Ollama (`ollama.embed`). Config via `MIMIR_EMBED_MODEL` / `MIMIR_EMBED_BASE_URL` /
`MIMIR_EMBED_TIMEOUT` (see [SETUP.md](SETUP.md)). Contract: `embed_texts()` returns
`None` on **any** failure (backend down, no model, timeout) so every caller falls back to
lexical scoring and the hermetic test suite stays green. Also exposes `embed_one`,
`is_available` (memoized probe), `cosine_rank`, `lexical_rank`, and `embed_model_id`. `_shared/vector_cache.py` sits on top of it with the *persistence* half of
the pattern — a `{key: {model, vec}}` JSON cache beside a corpus, vectors backfilled when
missing or embedded under a different model, `semantic_rank()` returning `None` to signal
the caller's fallback. The module catalogue uses it; `server_memory` still carries its own
copy and is the next caller to migrate. Note: the server loads it flat via `sys.path` (`import embed`) while the
client imports `mimir.servers._shared.embed` — two module identities, so patch the correct
one in tests.

Client-side read cache:
The following tools are cache-eligible within a single query (same `(tool, args)` returns the stored result instead of re-executing):
`read_file_lines`, `tree_summary`, `list_directory` (plus the code-intel nav tools).
Any successful write to a path invalidates cached reads for that path. Each entry is
stamped with the file's `(mtime_ns, size)`, so a write that went around the edit tools
(a `sed -i`, a script, the user's editor) drops the entry instead of being answered from
a copy that is now wrong.

## utilities/server_math.py

Purpose: safe math expression evaluation.

Tools:
- `evaluate` — safely evaluate a math expression string (`+ - * / ** % //` plus a curated
  NumPy function set). Subsumes the former one-shot wrappers (`add`/`subtract`/`multiply`/
  `divide`/`modulo`/`power`/`sqrt`), e.g. `evaluate("2 ** 10")`, `evaluate("sqrt(144)")`.

## utilities/server_symbolic_math.py

Purpose: symbolic mathematics using SymPy (algebraic manipulation, calculus, equation
solving, matrix operations).

Tools:
- `symbolic(op, ...)` — single dispatch tool; set `op` to one of: `simplify`, `expand`,
  `factor`, `differentiate`, `integrate`, `solve_equation`, `compute_limit`,
  `series_expansion`, `create_matrix`, `matrix_determinant`, `solve_system`. Relevant
  args per op: `expression`, `variable`, `point`, `n`, `equation`, `matrix_str`,
  `equations`, `variables`. Pruned from the toolset unless the query mentions symbolic /
  calculus / algebra keywords.

## utilities/server_strings.py

Purpose: text transformation and substring analysis.

Tools:
- `string_op(op, text, ...)` — single dispatch tool; set `op` to one of: `reverse`,
  `uppercase`, `lowercase`, `length`, `strip`, `replace`, `split`, `contains`,
  `count_occurrences`, `starts_with`, `ends_with`, `title_case`. Extra args per op:
  `old`/`new`/`count` (replace), `sep`/`maxsplit` (split), `chars` (strip),
  `substring`/`case_sensitive` (contains/count_occurrences), `prefix`, `suffix`.

## utilities/server_datetime.py

Purpose: date, time, timezone, and calendar utilities.

Tools:
- `date_op(op, ...)` — single dispatch tool; set `op` to one of: `current_datetime`,
  `days_between`, `add_days`, `day_of_week`, `unix_to_date`, `format_date`. Args per op:
  `tz`, `date1`/`date2`, `date_str`, `n`, `timestamp`, `input_fmt`/`output_fmt`.

## agent_state/server_memory.py

Purpose: persistent, timestamped memory — Claude-style, human-editable Markdown.

Storage lives under the central per-workspace state dir (`<STATE_DIR>/memory/`, shared
across every session of the workspace; see [state dir](#state-directory) below). Each
memory is its own `<slug>.md` file (frontmatter: name / description / date / tags + body),
indexed by `MEMORY.md` (one scannable line per memory, loaded into context each session).
Adds are deduplicated (Jaccard word-overlap over the recent window) and the store is
capped, pruning the oldest entries.

The **writing** tools (`memory_add`, `memory_update`, `memory_delete`, `memory_clear`)
declare `MAIN_ONLY`: the store is shared by every session of the workspace, and what is
worth keeping is decided in front of the user. `memory_search` and `memory_list_all` stay
grantable.

**Semantic search**: `memory_search` ranks memories by embedding similarity to the
query (so reworded, synonymous, or other-language queries still match), falling back to
case-insensitive substring matching when no embedding backend is reachable. Vectors are
cached in a parallel `embeddings.json` (`slug → {model, vec}`, kept out of the
human-readable `.md` files): written on add/update, pruned on delete/clear/aging, and
lazily backfilled for pre-existing memories on first search. A model change invalidates
stale vectors (the model id is stored alongside each vector). The embedding backend and
model are configured via the shared `_shared/embed.py` helper — see the `MIMIR_EMBED_*`
env vars in [SETUP.md](SETUP.md). The `memory://all` index injection is unchanged.

Tools:
- `memory_add` — store a fact as its own `.md` file (auto-slug, dedup)
- `memory_search` — semantic top-k retrieval (`limit`, `tag` filters; per-result
  `score`), with substring fallback
- `memory_update` — edit a memory in place by slug (re-embeds on change)
- `memory_list_all`
- `memory_delete` — remove one memory by slug
- `memory_clear` — wipe all memory (irreversible)

## agent_state/server_todo.py

Purpose: live task checklist + named prose plans, letting the model actively manage its
execution plan instead of relying on a static system-prompt injection.

Storage is **per session** under the central state dir
(`<STATE_DIR>/sessions/<sid>/`): the checklist as a Markdown checkbox list
(`todo_list.md`) and named plans as a history under `plans/` (one `<date>-<slug>.md` per
plan, indexed by `PLANS.md`, with a `.active` pointer). The session is resolved by
`state_paths.active_session_id()`: `MIMIR_SESSION_ID` when the server was started with one
— that is how a sub-agent gets a checklist of its own — else `<STATE_DIR>/active_session`
(written by `ws_server` on each switch). CLI/standalone runs fall back to the shared
legacy `todo_list.md`.

The **plan** tools declare `MAIN_ONLY` and are never granted to a sub-agent: the plan is
what the user approved, and an axis of it has no standing to rewrite it. The **checklist**
tools are grantable, because a child writes them in its own session.

Tools:
- `todo_set_plan` — write a named plan (approach/rationale) and make it active. Declares two arg-roles so client-side consumers find its arguments without knowing its name: `plan_title` (the plan loop pins it so a revision overwrites in place) and `plan_document` (the prose body). A client-side guard used to read that body and refuse a plan whose axes were exploration steps; it was removed — see [POLICY.md](POLICY.md#why-there-is-no-plan-shape-guard) for why a verb list is the wrong instrument for a refusal
- `todo_write` — replace the entire checklist with new steps. Takes an optional `depends_on` (one list of prerequisite indices per step), so a plan whose steps are genuinely partially ordered is recorded as the DAG it is instead of being flattened into a line
- `todo_read` — return the current checklist (index, text, done)
- `todo_read_ready` — split the open steps into `ready` and `blocked` against that DAG (a step is ready once every step it depends on is done)
- `todo_read_plan` — read the active (or a named) prose plan
- `todo_list_plans` — browse the session's plan history
- `todo_delete_plan` — delete one plan from the history
- `todo_update` — mark one item done/undone. No ordering is enforced here or anywhere else in the server: which step is next is the model's call, and `## Planning & todo` in the system prompt says so — independent steps have no order, a real one is declared through `depends_on`, and the list is rewritten when the work diverges from the plan

Notes:
- None of these tools is approval-gated: they manage a markdown checklist and prose
  plans, not source files, so they must stay available without a prompt in agent mode.
- Which of them a read-only mode sees is decided by `toollist.hidden_planning_tools()`:
  **ask** hides both `TASK_PLANNING` writers (`todo_set_plan`, `todo_write`) since a
  question records nothing; **plan** hides only `todo_write` (the `plan_steps` arg-role).
  Both are also rejected at call time by `readonly_guard.filter_readonly_tool_calls`.
  Withdrawing the tool is what lets the prompts drop the matching "do not write a plan /
  a checklist" prohibitions.
- `todo_update` and the read tools are **not** hidden anywhere: the model may consult or
  tick an existing checklist in any mode.
- In plan mode the model is expected to record the prose plan document (`todo_set_plan`)
  before the turn ends — but not before it has read the code: on a repo-touching query
  the client withholds the document tool until the exploration is done (see the plan-mode
  explore phase in `CLIENT_DETAILED.md`), so the plan cannot be a plan *to* explore.
  The ordered steps are recorded only after the user approves the
  plan, at the start of the execution — a checklist written earlier tracks work nobody
  agreed to yet.
- After the plan is recorded and presented, plan mode asks the user to approve it
  (*Accept & start* / *Reject* / *Rework*, plus free-text *Other*). Accept switches to
  agent mode, where the model is asked to record the checklist before starting and then
  executes it; reject stops without executing
  anything; rework/free-text loop back to re-planning. See
  [CLIENT_DETAILED.md](CLIENT_DETAILED.md) → `plan_loop.py`.

### State directory

Every `agent_state` server (and other persistent state: the module catalogue, sessions)
resolves paths off a single per-workspace **state dir**. The client computes
`~/.mimir/<workspace-id>/` and passes it to server subprocesses via the `MIMIR_STATE_DIR`
env var (`client/integration/server_manager.py`); servers read it through
`servers/_shared/state_paths.py`. When `MIMIR_STATE_DIR` is unset (standalone runs, the
hermetic test suite), they fall back to the legacy in-workspace `<workspace>/.mimir`.

The same module owns the agent **scratchpad**, which lives under the temp dir rather than
the state dir: `scratch_home()` → `MIMIR_SCRATCH_DIR` if set, else
`<TMPDIR or /tmp>/mimir-<uid>-<workspace-id>`; `scratch_dir()` appends the active session
id (from the `active_session` sidecar — the *only* thing it still needs the state dir for),
falling back to the home outside a session. `standing_roots()` exposes the **home** as a
standing sandbox root, one entry covering both. `server_files._safe`,
`server_bash._is_within_workspace` and the read servers (`server_search`,
`server_code_intel`) all pass it as `extra_roots` alongside `approved_roots()`, so the
agent can write there without a user prompt and read back what it wrote. A read server
that lacked it refused a file the agent had just written, and the agent concluded the
edit tools did not work outside the workspace — see `POLICY.md` → Out-of-Workspace
Access Approval for why it is separate from the user-approval sidecar, why scratch writes
are excluded from the change ledger, and how `ensure_scratch_home()` vets a world-writable
`/tmp`. Resolution never creates directories: a sandbox check runs on every call and must
not materialise state. `workspace_id()` also lives here, not in the client's constants:
both the state dir and the scratchpad name are built from it.

### Numerical invariants (`_shared/numerics.py`)

`NUMERICAL_INVARIANT_METRICS` (`l2_rel`, `linf_rel`, `l2_abs`, `linf_abs`,
`convergence_order`, `conservation_residual`, `finite`) plus `wall_time_s` form
`RESERVED_METRICS`, which the **proxy** server strips from anything the code under
optimization prints — a solver must not be able to satisfy its own acceptance
constraints. Hoisted here from `proxy/_lib/metrics.py`, which re-imports
`RESERVED_METRICS` so its behaviour is unchanged.

The client does **not** read these names as evidence. A scan that promoted a run to the
`oracle` tier for printing an `l2_rel=…` line existed and was removed: the value could
never be interpreted (a forged number is unfalsifiable from outside the process — the
very reason the proxy seals references server-side), so it rewarded a string. What a run
printed is for the model to read and report through `report_verdict`.

One client-side reading remains, and it only ever *withholds* credit:
`observed_failure_verdict(stdout)` reports a whole-line `check=fail` / `verdict=fail`,
which the client treats as a failure even on exit 0 — a check that computes its own
criteria, prints them unmet and returns 0 anyway is otherwise indistinguishable from a
clean run. Read in that direction only: a passing verdict never rescues a red exit.

This is an auxiliary carrier, not the mechanism. Judging a run's output is the model's job
(`client/guardrails/verdict.py`), since no parser generalises across fields, plots, tables
and logs; a `check=fail` line only spares the model being asked about output that already
states its own answer, by pre-filling the verdict.


## workspace/server_files.py

Purpose: workspace-scoped file CRUD and surgical edit helpers.

### Absolute paths only

Every model-facing tool here requires an **absolute** `path`; a relative one is rejected by `_require_abs` at the tool boundary. The check itself is `require_absolute()` in `servers/_shared/root_paths.py`, shared with the read-only servers (see below); `_require_abs` is the file server's thin wrapper over it.

This is the structural fix for misplacement. `write_file` previously accepted *"absolute or relative to server start directory"* — a directory the model has no way to learn — so a relative path was silently resolved against a root it had to **infer**. Asked twice to create a solver *outside* the `codes` directory, MIMIR twice wrote it inside and reported the constraint satisfied. Two prompt-level attempts to make that inference reliable both failed; requiring an absolute path removes the inference, and the destination is stated in the call itself.

The error is designed to be **self-correcting**: it names the path the relative form would have produced, so the model re-issues in one step, and it states the workspace root at the moment placement is being decided — which no static prompt section can do.

```
Relative path 'wave_solver_2d/solver.py' — this tool requires an absolute path.
Inside the workspace that is /…/codes/wave_solver_2d/solver.py. If you meant
somewhere else, give that absolute path instead. The workspace root is /…/codes.
```

The rule now also covers the read-only tools that name a **file**: `read_file_lines` (`server_search`), `symbol_outline` and `hover` (`server_code_intel`) call the same `require_absolute()`. A read that quietly accepts a relative path teaches the model a habit the next write will refuse, so the two sides are held to one rule. Directory and scan roots keep their tolerant default — `path="."` for `list_directory` / `tree_summary` means the workspace, which is not an inference anyone can get wrong.

Deliberately unaffected: `bash_run` (requiring absolute operands in a shell would be absurd — its cwd is pinned to the workspace root instead), the internal `list_files` helper (which is why the check sits at the tool boundary, not inside `_safe`), and `resolve_path_in_root` including its duplicate-root stripping — that becomes unreachable for file tools but still serves the read-only roots. The check runs *before* `_safe`, so the sandbox is unchanged: an absolute path outside the workspace is still refused.

Client-side storage is untouched: `normalize_workspace_path` already converts absolute → workspace-relative, so `dirty_written_files`, `validated_files` and every gate comparing them keep their existing form.

### The round-trip: paths are absolute wherever the model reads them

The rule above is only coherent if the model can **copy** paths rather than construct them. So discovery reports absolute paths too:

- `server_search` — `read_file_lines`, `list_directory` (including a `path` per entry, not just `name`) and `tree_summary` echo the *resolved* path, not the argument they were given.
- `server_code_intel` — `_out_path` (formerly `_rel`) returns absolute for every navigation result; the ctags `by_file` index is keyed the same way, so lookups stay consistent.

What matters just as much is the output side: hand back a relative path and the model must join the root itself, which is the inference the rule removed, reintroduced exactly where it is most likely to copy without thinking.

`tree_summary`'s root line is absolute for the same reason the system prompt states the workspace root absolutely: a bare basename root reads as a subdirectory of itself.

Tools:
- `write_file`
- `append_file`
- `delete_file`
- `replace_in_file`
- `replace_all_in_file`
- `replace_lines`

Note: directory listing is owned by the search server's `list_directory`; the
file server keeps an internal `list_files` helper only to back the `files://list`
resource (not exposed as a tool).

## workspace/server_search.py

Purpose: file reads, partial reads, directory listing, and cached tree summaries.
Text search (`grep`/`rg`) is done via the bash server — classified read-only, so it
needs no approval and feeds the same discovery signals a dedicated tool would.

Tools:
- `read_file_lines` — at most `_MAX_READ_LINES` (400) lines per call, whatever the range asks for; `end_line=0` means "to EOF, up to that cap". A window that stops short reports `truncated`, `total_lines` and `next_start_line`. Takes an **absolute** path, like the file tools. Every returned line carries its file line number as an `N: ` prefix — the same format as the search excerpts and the post-edit context window. The edit tools are addressed *by number*, so a bare block left that binding to be counted by hand across the window; the way out of a miscount the model cannot detect is to re-read ever narrower ranges until the requested range is one line and the number is asserted by the arguments rather than inferred. The prefix is not part of the file: `replace_in_file` strips it from an anchor that only matches once un-numbered, and `replace_lines` strips it from a `new_content` numbered consecutively from `start_line` (a paste-back of what was read), leaving anything else literal.
- `list_directory`
- `tree_summary` (pass `use_cache=False` to force a fresh scan)

## workspace/server_code_intel.py

Purpose: backend-abstracted code **navigation** — one tool per task, with the server
picking the best available backend internally so the model never has to choose between
redundant tools.

Backend ladder, per call, best first: **LSP language server → universal-ctags index →
whole-word text scan**. Every tier is optional: with neither a language server nor
`ctags` installed, these tools return a structured error and the agent falls back to the
bash `grep`/`rg` path — the same graceful-degrade contract the compile toolchain has when
`nvcc`/`gfortran` is absent (see [SETUP.md](SETUP.md) §2b for the binaries involved).

Tools (all read-only, cacheable, none approval-gated):
- `find_definition(name)` — where a symbol is **defined**; returns each site as
  `{path, line, kind, signature}`. Prefers the language server's workspace symbol index,
  else the ctags index. Use it instead of grepping for a name when the authoritative
  definition is what you want.
- `find_references(name)` — **use** sites across the workspace, as `{path, line, text}`.
  Prefers LSP reference resolution anchored at the definition; falls back to a whole-word
  scan. `context_lines` is **off by default**: the question is where a symbol is used, and
  an excerpt per hit costs more than the answer. Raise it when the sites must be edited.
- `symbol_outline(path)` — the ordered symbol tree of one file, each entry carrying `line` **and `end_line`**. The end is what makes "read the block around line N" a single call: without it the model widens its window a few lines at a time looking for where the construct closes. It is exact when the backend reports one (LSP ranges; universal-ctags `--fields=+e`) and otherwise **derived** — a symbol runs until the next one at the same or shallower depth, the last to EOF — with `inferred_end` marking that it is an upper bound. Deriving it matters because the common ctags on HPC systems is *Exuberant* 5.8, which has no end field at all. Both LSP reply shapes are read (nested `DocumentSymbol.range` and flat `SymbolInformation.location.range`); reading only the first put every symbol of a flat server at line 1.
  (`[{name, kind, line, depth}]`), from LSP document symbols or the file's ctags entries.
- `hover(path, line, symbol="")` — type / signature / doc info at a position (`line` is
  1-based). **LSP-only**: it errors when no language server serves that file's language.

The `CODE_NAV` capability exists so the client locates these navigation tools by
capability rather than by literal name. `_out_path` returns **absolute** paths for every
result (see *The round-trip* above), and the ctags `by_file` index is keyed the same way.

## external/server_web.py

Purpose: constrained HTTP access plus JSON parsing helpers.

Tools:
- `http_get`
- `http_post`
- `parse_json`
- `json_extract`

Three ceilings, because a fetch can go wrong in three different ways.

| ceiling | value | what it defends |
|---|---|---|
| `_MAX_BYTES` | 2 MB | the socket read: a hostile or runaway endpoint |
| `_MAX_TEXT_CHARS` | 128 KB | what a page *with prose* may spend of the model's window |
| `_MARKUP_FALLBACK_CHARS` | 8 KB | what a page with **no** prose may spend |
| `_ERROR_BODY_READ` / `_ERROR_BODY_CHARS` | 64 KB / 2 KB | what a **failed** request may spend |

HTML is turned into text before any of them applies, since markup is what the ceiling
would otherwise spend itself on. When that extraction comes back empty the page is a
script shell, or markup the parser choked on, or a body that never arrived — and the
fallback used to hand back the raw markup under the 128 KB ceiling, on the reasoning
that markup beats nothing.

Measured, that is backwards. A thesis record page is 754 KB, 92% of it a single inline
`<style>`. The socket read stops at 512 KB, entirely inside that block, so the body
never arrives and the parser correctly reports no text — and the fallback then spent
131 072 chars, about 34 000 tokens, on CSS. Empty extraction is the strongest evidence
there is that the markup holds no prose.

So the fallback now answers with what the page says about itself, under one 8 KB
budget, in order:

1. `<head>` metadata — `<title>`, `description`, `og:*`, the `citation_*` set. It is
   the first thing off the wire, so it survives a read cut long before the body.
2. Embedded structured data — `application/json` and `application/ld+json` blocks,
   largest first, each whole or absent. On a script shell this *is* the content:
   keeping only the metadata would answer a 34k-token page by dropping the one part
   of it that held the answer. Half a JSON document cannot be parsed, so a block that
   does not fit is left out rather than cut.
3. Failing both, 8 KB of markup labelled `markup_only` — a sample to show what kind of
   page this is, not a window.

The thesis page costs about 100 tokens instead of 34 000 and says what it is instead of
saying nothing in CSS; a 300 KB script shell carrying a record comes back as ~120 tokens
of that record. `raw=True` bypasses all of it and always has.

The read ceiling could then be raised, which is what actually recovered the content.
It was 512 KB while the read and the prompt were the same thing; now that what reaches
the model is bounded separately, a bigger read costs time and memory, not context. On
the thesis page `<meta name="description">` sits at byte 700 336, behind ~700 KB of
inline CSS — at 512 KB the abstract was not slow to reach, it was unreachable. At 2 MB
the whole body arrives, extraction works normally, and the fetch returns the title,
author, director, jury, date, keywords and full abstract for about 1 400 tokens. The
no-prose fallback is what covers the pages where even that is not enough.

### Reading part of a document

A document that does not fit is not thereby unusable: what the caller wanted is
nearly always one region of it, and `truncated` on its own left them with the first
128 KB and no way to ask for the rest.

`_TextExtractor` now keeps `h1`…`h6` as Markdown headings instead of collapsing them
to a newline, so a page comes back as named regions (`## Résumé`) rather than one
undifferentiated wall. That is what makes the rest possible.

- `contains=` — `_find_matches` runs two passes, because a document names its own
  parts. A query matching a **heading** returns that whole region: asking a thesis
  page for "résumé" should give the abstract, not six sentences containing the word.
  Only when no heading matches does it fall back to windows around the occurrences,
  merged when they overlap, ranked by `lexical_rank` (`servers/_shared/embed.py`,
  the same scorer the memory and platform searches use). Each excerpt says its offset
  and the heading it sits under — the heading is looked up at the *hit*, not at the
  window start, which opens half its width earlier and on a short document lands in
  the previous region.
- `offset=` — resumes where a truncated reply stopped, with the key names
  `read_file_lines` established (`truncated`, `total_chars`, `next_offset`), which
  the client already turns into a MORE_CONTENT hint on its own
  (`tool_execution/executor.py` `_build_continuation_hint`).

Order matters and is the whole point of `_read_body`: **extract, then target, then
fit**. Fitting before targeting cuts away the very part the caller asked for — the
first version did exactly that, cutting to 128 KB inside `_readable_body` before the
offset was applied, so page two of a long read came back empty and the resume this
ceiling exists to enable could never be taken. `next_offset` is absolute for the same
reason: it is handed straight back as the next call's `offset`.

Measured across twelve sites (arXiv, MDN, Python docs, GitHub, two JSON APIs,
a thesis registry, a 403, Nature, the WHATWG HTML spec, example.com, a raw file): every
one bounded, worst case 33 750 tokens for the 10 MB spec, and targeted reads of
127–283 tokens against pages costing 20 500 and 33 750 whole.

One related defect, found while measuring: `hint` is a reserved protocol key that
`responses.ok()` strips from success payloads. Every piece of guidance this module
wrote under it — "fetch a more specific URL", "pass raw=True" — had been addressed to
a model that never received it. Success notes now use `body_note`, and a test holds
the line. `replace_all_in_file` (`servers/workspace/server_files.py`) had the same
trap: its "call again with confirm=True" line was dropped the same way.

A **failed** request went through none of this. Both error branches read up to
`_MAX_BYTES` and returned it as `body`, raw: no extraction, no ceiling. A paper host
answering 403 with a block page therefore cost 131 164 tokens — for a request that
returned nothing. `_error_body` now gives an error the same treatment as a page, then
a much tighter ceiling: nothing downstream *uses* this text, it only has to say why
the request failed. "Access Denied" survives; the markup around it does not.

That single defect is what put a 200k window at 215k, because it did not arrive alone:
four fetches issued in one step land together, and the history budget is only checked
between steps. Capping each body is what makes the batch affordable — the same four
calls went from ~137 800 tokens to ~7 100. The aggregate remains unbounded by design:
four *legitimate* 128 KB pages would still be ~170k in one step. Worth revisiting if
it is ever seen.

## external/server_github.py

Purpose: read-only GitHub repository inspection.

Tools:
- `github_repo_info`
- `github_list_branches`
- `github_list_issues`
- `github_get_file`
- `github_search_repositories`

`github_get_file` reads in **line windows**, not whole files. It takes `start_line` /
`end_line`, returns at most 400 lines a call, and when it stops short says so with the
keys `read_file_lines` established — `truncated`, `total_lines`, `next_start_line`,
`line_cap` — which the client's `_build_continuation_hint` already turns into a
MORE_CONTENT hint. The first page of a 1 462-line source file costs ~4 100 tokens
instead of ~15 000 for the whole thing, and the whole thing still costs the same if
you actually want it.

A file over `_MAX_FILE_BYTES` used to be **refused**, which cost the caller the file
entirely in order to avoid a large result — a total loss of information to avoid a
large one. It now falls back to the raw URL (bounded by `_MAX_RAW_BYTES`) and serves
the same window, with `source` saying which path answered. A 1.18 MB LAPACK source
that was unreadable comes back as 400 of its 41 864 lines for ~3 600 tokens.

## hpc/server_hpc.py

Purpose: Slurm inspection and job-submission helpers. (Environment Modules / Lmod are
not here either: *discovery* is `platform_search` on `server_platform`, which searches an
index of the site's whole module tree, and *loading* is the bash server's `module` command,
because a load mutates one subprocess's environment.)

Tools:
- `slurm_partitions`
- `slurm_nodes(partition="", states="", node="", detail=False)` — **compute-node inventory**: architecture, CPU topology, memory, GPU type/count, node features, and live occupancy (allocated/free CPUs, free memory, load), read from `scontrol show node` — Slurm's own database, which is what actually governs placement. Read-only and instant: it allocates nothing, so it can be consulted *before* choosing where to submit, unlike running a probe on the node (that needs `srun`, i.e. a queued allocation billed against your hours). Aggregates nodes onto their hardware signature by default with a count per state, since a per-node listing of a large cluster is mostly noise; `detail=True` or `node="<name>"` gives individual nodes. Falls back to `sinfo -N` where `scontrol` is restricted, flagging in `degraded` that architecture and CPU occupancy are then unknown. The `scontrol` parsing and the signature aggregation live in `_shared/slurm_nodes.py`: `server_platform`'s digest needs the same view of the cluster's hardware, and two copies of a `scontrol` parser would drift.
  > **Architecture is the field that earns the tool.** Where a cluster mixes architectures, a binary built where the agent runs will not run on a node of a different one. What Slurm cannot report — CPU model, SIMD, `-march`, caches, OS, toolchains — is read on the node by `slurm_probe_node` and served by `slurm_node_profile`.
- `slurm_queue`
- `salloc_submit` — synchronous **interactive** allocation. Takes the resources as arguments (partition, nodes, ntasks, cpus, mem, time, gres, constraint, account/qos) and builds the `salloc` command itself, so the validated command is the one that runs; launched as argv, never through a shell. `confirm=False` returns the exact command as a preview instead of executing — the old two-step the server + free-form `salloc_submit(command=...)` is gone, because the validation lived entirely in the step nothing forced you to call
- `sbatch_submit` — non-blocking Slurm **batch** submission (unlike synchronous `salloc_submit`): returns a `job_id` immediately plus a `background_job` descriptor (`BACKGROUNDABLE`), so the run is watched off the critical path and auto-resumes the agent on completion. Writes the script/log under `state_dir()/hpc_jobs/<ts>/` (env `MIMIR_HPC_JOBS_DIR`). Takes `constraint`, `nodelist`, `nodes`, `ntasks` and `exclusive` to aim at one kind of node and to time without a neighbour. The job inherits the server's environment, not modules loaded by earlier shell commands. Its output is not recorded as a measurement.
- `slurm_probe_node(partition, constraint="", nodelist="", account="", confirm=False)` — submits a job of a few seconds that runs `server_platform.py --profile-json` **on the node**, so the node gets the same profile as the host. The job picks its own Python (`.venv-<os>-<arch>`); with none, a shell fallback still reads `lscpu`, `-march`, GPU and OS, and the profile is marked `partial`. Irreversible (approval prompt) but not `CLUSTER_SUBMIT`: it runs no user code, so the local-validation hold does not apply.
- `slurm_node_profile(partition="", node="")` — read-only. Harvests finished probes into `state_dir()/hpc/node_profiles/<node>.json`, then returns each node kind of the partition with its profile, or `profiled: false`. A profile is kept until Slurm's description of the node changes (no TTL). `matches_this_host` lists what differs from the host MIMIR runs on (arch, CPU model, SIMD, OS, glibc). Inside an allocation, the current node is profiled on the spot, with no job.
- `slurm_job_status(job_id)` — normalized per-job state (running|pending|done|crashed|unknown) via squeue (active) + sacct (terminal); the poll target the background-job watcher uses.

> **Three contexts.** `none` (no Slurm), `login` (Slurm, but this host is not an allocated node) and `in_allocation` (`SLURM_JOB_ID` set and this host in the nodelist). Computed by `_shared/cpu_facts.execution_context` and reported by `platform_probe` and `slurm_node_profile`. `local` stays a valid target from all three.
>
> **`srun` in the shell.** Outside an allocation it is a submission: it is never an approval-exempt read, and the cluster-submission hold applies to it (`shell_paths.allocates_cluster`). Inside an allocation it is a job step and is unchanged.
>
> `salloc_submit` / `sbatch_submit` declare the `CLUSTER_SUBMIT` capability (shared with `proxy_slurm`). The client's pre-submission guard holds the first such call each query until something has been validated locally, then lets the retry through (see `POLICY.md` → Cluster-Submission Guard). `sbatch_submit`, `proxy_eval(op='run')`, and `proxy_slurm(op='eval')` additionally declare `BACKGROUNDABLE` (see the background-jobs note under `proxy_eval`).

## hpc/server_platform.py

Purpose: report what this host actually is — hardware, scheduler, toolchains, Python environments — and search the site's environment-module catalogue.

Tools:
- `platform_probe` — collect and return a full profile of **this host** (CPU with caches, `-march`, OS and glibc, GPU, memory, Slurm, loaded modules, toolchains, Python environments), plus `execution_context` and `machine_signature`. Every fact is built on demand, so none of it can be stale; it reports the module catalogue's summary but never builds it. `server_platform.py --profile-json` prints the same profile without serving MCP: that is what a node probe runs. Probes run with `LC_ALL=C`, since `lscpu` translates its field names.
  > **Nothing here assumes an architecture or a vendor.** The reported ISA extensions come from a per-architecture table (`ISA_EXTENSIONS` in `_shared/cpu_facts.py`: AVX/FMA/AMX on x86_64, ASIMD/SVE/BF16 on aarch64, VSX on ppc64le) read from whichever key that host's `lscpu` uses — `Flags:` on x86, `Features:` on aarch64 — and an architecture with no entry says so rather than reporting a vector unit it never looked for. The accelerator probe detects NVIDIA, AMD and Intel tooling and only *enumerates* NVIDIA, reporting the others as present-but-unenumerated: answering "no GPU" on a host whose accelerator it cannot read would be a lie. The toolchain scan covers GNU, LLVM, Intel oneAPI, the NVIDIA HPC SDK, ROCm and Cray wrappers; whatever is absent simply does not appear.
- `platform_get_profile` — build and return a fresh profile for the current host plus a live `sinfo` partition/node table so the agent knows what Slurm resources are available without a separate command. Always built fresh for the current host (so it can never serve another node's hardware), no cache, no the catalogue guard arg. The collectors whose answer cannot change while the process lives (CPU, GPU, Slurm, modules, toolchains) are memoized, so a second probe costs a fraction of the first. The probe tools are the **only** source of live platform facts: the client used to carry a duplicate probe whose output was injected into every system prompt, which paid for a full hardware summary on every query to answer a question most of them never asked.

- `platform_search(query, limit=10, refresh=False)` — search the host's **environment-module catalogue** by name (`"cuda"`) or by capability (`"parallel hdf5"`). Each hit carries `load`, the exact string to pass to `module load`.
- `platform_catalogue_status()` — what the index currently holds, without building or refreshing it. Backs the CLI's `/modules`.

> **Why the catalogue is the one thing this server persists.** `_collect_modules` used to run `module -t avail | head -n 120` and return 80 names under a key called `sample`. An alphabetical slice of a module tree is worse than no list: a model that reads `abaqus … cmake` with no `cuda` in sight concludes CUDA is unavailable. A site's tree runs to thousands of entries and cannot be put in a context window, so it is indexed once under `<STATE_DIR>/platform/modules/` and *searched*. What `platform_probe` still reports about modules is only the volatile part — the ones loaded right now.
>
> **Freshness is a signal, never a clock.** The catalogue is rebuilt when a fingerprint changes: hostname, the *effective* `MODULEPATH` (after the site's init scripts), a depth-2 stat-hash of the modulefile trees, and the mtime of Lmod's own spider cache — which a site regenerates exactly when modules change. No TTL is consulted. The file is named per hostname, because `STATE_DIR` is per-workspace while a module tree belongs to a machine.
>
> **Nobody waits for the expensive tier.** Only the name-level pass (`module -t avail`, under two seconds) runs on the request path. Descriptions are filled in behind the user: Lmod's `spider` where it exists — it is Lmod-only, and Tcl Environment Modules is never asked for it — otherwise a single-process bulk `whatis`. While that runs, results carry `catalogue.enriching: true`, so an empty description reads as *not yet known* rather than absent. A background pass whose signal moved while it worked discards its own result.
>
> **The digest.** Riding in the same file is the small set of stable, high-impact facts that *do* fit in a context window: host architecture and ISA, the cluster's compute-node *types* (architecture, CPU topology, memory, GPUs — from `scontrol`, aggregated by hardware signature), its partitions, and the toolchains present. It is returned by `platform_catalogue_status` and inside `platform_probe`; nothing puts it in the system prompt. Occupancy is deliberately excluded: it is volatile, `slurm_nodes` owns it, and including it would make the fingerprint churn every few seconds.

## hpc/server_env.py

Purpose: **mutating** Python-environment management — the write-side counterpart to
the read-only environment *discovery* in `server_platform.py`. It exists to satisfy a
missing-module dependency when no existing environment already provides it (Tiers 2 & 3
of the **environment-resolution cascade**: Tier 1 = use an env that already has the
module; Tier 2 = install into an existing project/conda env; Tier 3 = create a fresh
env). Inputs are validated and shell-free (argv lists, no `shell=True`), and mutating
operations are restricted to paths that actually look like Python environments.

Tools (all declared `reversibility="recoverable"` → approval-gated, plus `non_batch` → always prompt, never batched; all plan-blocked):
- `env_pip_install(packages, python_executable="python3")` — install packages into an existing environment
- `env_pip_uninstall(packages, python_executable="python3")` — the cleanup for `env_pip_install`
- `env_create(name, kind="venv", packages=None, python_executable="python3")` — create a new venv/conda environment. New venvs land under `state_dir()/envs/`, not in the workspace (a venv is agent state, and one dropped in the repo shows up in `git status`); a new conda env's interpreter is resolved by asking `conda env list --json` where the env actually is, since `envs_dirs` is routinely configured away from `~/.conda/envs` on a cluster
- `env_delete(target, kind="venv")` — delete an environment created by `env_create`

> A bare `python`/`python3` resolves to the server's own interpreter; an absolute path
> must exist and be executable; anything else resolves via `PATH`. These tools declare
> registry-driven approval **scope** (package-set for the pip tools, env basename for
> create/delete) so an `always` grant narrows to those packages / that environment
> rather than the whole tool (see `POLICY.md` → Sensitive Tool Approval).

## external/server_system.py

Purpose: host-level runtime and resource inspection.

Tools:
- `system(op, ...)` — single read-only dispatch; set `op` to one of: `info`, `disk`
  (uses `path`), `cpu`, `memory`, `env` (uses `name`, allow-listed vars only), `uptime`.

> **Removed (2026-07): `server_code.py` (exec) and `server_code_quality.py` (static
> checks).** Compiling, running, validating, linting, type-checking, and testing code all
> go through the `bash` server now — the agent invokes `gcc`/`gfortran`/`nvcc`,
> `python`/`node`, `python -m py_compile`, `ruff`/`pyflakes`, `python -m mypy`, `black`,
> and `pytest` directly with the exact flags it needs. Exec/compile commands are
> approval-gated and workspace-confined by the bash server. The client marks a written
> file *validated* when a successful bash validation command names it (see
> `POLICY.md` → Validation Policy), and a validation **guidance** nudge (strict/light, not
> off) steers the model to run those checks. Symbol navigation lives in `code_intel`.

## workspace/server_bash.py

Purpose: workspace shell access with command validation. Any command runs except a
short denylist; what governs the rest is the user's approval prompt plus path
confinement, with the no-substitution model kept load-bearing. This is also the
**text-search path**
(`grep`/`rg`) — there is no dedicated grep tool; a leading `grep`/`rg` classifies
read-only, so it needs no approval and feeds the same discovery signals a tool would.

Tools:
- `bash_run` — takes `background=True` for a run that does not fit the 120 s cap (see
  *Detached runs* below).
- `bash_job(op, job_key)` — read-only handle on a detached run: `status` (its state),
  `output` (the tail of its log, with the state), `list` (every job this host knows).
- `bash_job_stop(job_key)` — signals the job's whole process group (TERM, then KILL
  after a grace period), so a build does not leave its compiler running.
- `report_verdict` — the model's reading of what a run's output showed (`judge`
  capability). It executes nothing: the client's observer settles the run it names on the
  blackboard (see POLICY.md → *Validation Policy*). It lives here because a verdict is
  owed by a run, and where nothing can be executed nothing needs judging.
  Four values: `pass`/`fail`/`unknown` read what a green run showed; `blocked` re-imputes a
  **red** one from the change to the environment (a build to configure, a package, a dataset,
  an allocation). It never makes the run green — it stays reported as not completed — it only
  stops it being charged as a defect. Unclaimed, a red exit drives the repair ladder as before.

### Detached runs (`_bash_jobs.py`)

`bash_run` blocks its turn and is capped at 120 s — the right shape for a search, a
test suite or a short build, and the wrong one for a dependency tree that takes two
hours to compile. `background=True` spawns the same **already-validated** command into
its own session with its output redirected to a log, and returns a handle immediately.

Every run goes through the launcher now, detached or not: a blocking call starts a job
and waits on it, polling the handle it got back. What that buys is output on disk
rather than in a pipe nobody is left to drain — so a run stopped at the cap still
reports what it printed (it is stopped exactly as before; only the bytes are kept), and
one the user detaches mid-flight keeps everything it has already done.

A long finished run comes back as its **beginning and its end**, cut on whole lines, with
the middle replaced by a marker that gives the bytes left out. The beginning holds the
first error; the end says how the run finished. With the beginning alone, the model piped
every build into `tail`, and a pipe holds the output back until the build ends, which hid
its progress from the user. A run still going returns its tail instead.

That last hazard is now handled rather than only discouraged. A job's log is the
pipeline's own output, so `make | tail -40` leaves it empty for the whole build and no
count can be read from it. When the command pipes a build tool into anything,
`build_progress.tee_command` splices a `tee` in ahead of the filter, writing a second
copy to `progress.log` in the job directory, and every progress read looks there
(`_bash_jobs.progress_path`). The filter stays, so the output and the exit status are
what the caller asked for; `meta.json` records the rewritten command as
`effective_command` beside the one that was approved. Deciding this needs the pipeline's
shape, which is why `ParsedSegment` carries the separator that follows it — without it
`make | tail` and `make && tail log` read the same. The splice is textual, at the offset
`unquoted_pipe_offsets` gives: reassembling the command from shlex tokens would quote a
glob or a `$VAR` into a literal. A blocking run's
job directory is a scratch buffer, marked `ephemeral` in its `meta.json` and deleted
the moment the call returns; a detached one's is the handle itself and is never swept.

**Diverting a run in flight.** The user can move a still-running blocking call to the
background from its tool row in the UI. It cannot travel through the model — the agent
is parked awaiting this very call, and a steer is only drained at a step boundary — nor
through MCP, since the call itself is what the server is busy with. So it goes through
the shared state dir, like the approved-paths allowlist.

`servers/_shared/run_channel.py` is that channel, and it is generic. A tool that blocks
publishes the run it is waiting on in `<state_dir>/runs/<channel>/current.json`;
`client/tool_execution/run_channel.py` names it back in a `divert` file the wait loop
consumes on its next tick. The channel is the blocking tool's own registered name, so
the client addresses it straight off the row being diverted — neither end spells the
other's name out, and one tool's request can never reach another's run. Two tools
publish today: `bash_run`, and `proxy_eval` for `op='run'`. Both declare the
`divertible` capability, which is what puts the control on the row; a tool that returns
as soon as it has submitted (`proxy_slurm`, `sbatch_submit`) declares it and gets no
control, because there is no wait of its own to divert.

The process is left running, the result carries the work done so far plus the ordinary
`background_job` descriptor, and the existing watcher takes it from there. A shell
run's result carries **no** `returncode`: it has not produced one. This is the only
promotion there is — an explicit human act, never the clock.

**Saying what a run is doing.** `current.json` also carries a `phase` and, when the
work counts itself, a `percent`; the wait loop republishes both on every tick and the
client relays them to the row. For `proxy_eval` the phase comes from `phase.json`,
which the detached runner writes at the boundaries only it knows (which build of how
many, which case of how many, which replicate), and the percentage is read straight out
of `build.log` — the compiler prints it, and the runner is blocked inside the build for
its whole duration and is in no position to count anything. `procs._run_state` merges
both, so the blocking wait, `proxy_eval_status` and the detached watcher all learn it
from one place.

Backgrounding is a parameter on a command that has passed the same path checks and
denylists, which is why the `&` operator stays refused: detaching is the server's job,
not a shell operator the caller supplies. The job directory lives under a fixed cache
root (`_shared/trusted_read_roots.py`), so the log is readable with the ordinary file
tools while the run is still going.

States — `running` (the process is alive), `done` (exited 0), `crashed` (exited
non-zero), `unknown`. The last is its own answer rather than a variant of the other
two: the process is gone and left no exit code, so it was killed from outside rather
than having returned, and reporting that as `done` would be the worst answer this
module could give. A signal death records `128+signum`.

**Who promises the resume.** The launch result says what was *started* — nothing more.
Whether anything is watching the job is the client's to know, and the client says so
itself, in the note it appends when it registers a watcher (see `CLIENT_DETAILED.md` →
*Background jobs*). This split is load-bearing: a server note that guaranteed a resume
it does not perform once produced five two-hour builds that finished into silence while
the model kept ending its turns on a promise nobody was holding.

### Scope of the sandbox (read this before trusting "confined")

Two mechanisms guard execution, and it matters which one does what.

**The approval prompt is the real control.** Every build/exec command is approval-gated —
not only the ones reaching outside the workspace. `python solver.py`, `make`, `./a.out`
and `pdflatex main.tex` all stop and ask, inside the workspace as much as outside. Nothing
runs that the user did not say yes to. That is a genuine checkpoint, and it is where the
user's control actually lives.

**The path validation is a narrower thing**: it governs the command the agent *writes* —
which program runs, and which paths it names. It says nothing about what that program does
once running. `python`, `make`, `gcc` and a workspace-local `./a.out` execute with the full
privileges of the account running MIMIR, so a script the agent wrote can read any file that
account can read, write anywhere it can write, and open a socket — none of which this
server sees. `make` runs a workspace Makefile, arbitrary shell by construction.
`python -c "open('/etc/passwd')"` passes every check documented below, because the path
never appears as an argument.

| Mechanism | Gives you | Does not give you |
|---|---|---|
| Approval prompt | A yes/no on **every** execution, in or out of the workspace | A review of what the code being run actually does |
| Path confinement | The checked command is the executed command; out-of-workspace paths surface for consent | Any constraint on a process once it has started |

Three bounds on the approval checkpoint, worth knowing before relying on it:

- **"Always" is scoped to the first two argv tokens** (`_scope_command_prefix`), so
  approving `python solver.py` forever does not cover `python other.py` — but approving
  `python -c "print(1)"` forever *does* cover any future `python -c <anything>`. That
  breadth is deliberate (an inline body should not spawn a new prompt per character), and
  it is the widest grant available from a single click.
- **Approving a command is not reviewing the code it runs.** `python solver.py` tells you
  the file name; the agent may have written `solver.py` seconds earlier.
- **The headless runner auto-approves everything** (`_install_auto_approve` in
  `runner/engine.py`), so in benchmark/eval runs neither mechanism asks anything. This is
  already flagged in the README's engine section — point it at trusted workloads only.

So: the user is in control of *whether* something runs, not of *what it does* once it
does. Constraining the latter needs isolation at the **process** level — namespaces,
seccomp, a container, bubblewrap — which is a different mechanism, not more rules in the
validator, and **is not implemented**. Run MIMIR under an account whose reach you are
comfortable with.

Approval / modes:
- **Plan mode** — only read-only discovery commands run. Build/exec commands, `module
  load`, and in-place writes (`sed -i`) are **blocked outright** (not approvable); they
  become available only after the plan is approved and the loop is in agent mode.
- **Agent mode** — read-only discovery commands run **unattended** (no approval prompt);
  build/exec and write commands are approval-gated.

The read-only vs write/exec decision is the client-side
`bash_classify.bash_command_is_readonly` (plan-mode gate in `plan_loop`, approval
exemption in `readonly_exempt`; see POLICY.md). The server still validates and confines
every accepted call independently.

### What is refused, and why

Anything not named below runs. These three groups are refused because no approval
prompt could make them reviewable, and each rejection names the route that replaces it
— a refusal with no route is a dead end the model retries variants of. The sets live in
`servers/_shared/shell_paths.py` (`DENIED_COMMANDS`) so both ends read the same list, and
`test_the_header_table_is_the_denylist` fails if the server's docstring table drifts
from it.

| Group | Commands | Why | Instead |
|---|---|---|---|
| interpreter | `bash` `sh` `zsh` `ksh` `dash` `csh` `tcsh` `fish` `eval` `exec` `.` `command` `sudo` `su` `doas` | runs a nested command the validator never sees, voiding segmentation and with it path confinement | write the command out; chain it; `./script.sh`; `source env.sh` |
| destructive | `shred` `dd` `mkfs` `fdisk` `parted` `swapoff` `mkswap` | destroys data outside anything a command line can review | `rm` is available and confined |
| cluster | `sbatch` `salloc` `scancel` | a second submission route would produce jobs nothing tracks | the typed HPC tools, which return a tracked job handle |

Refused in every spelling, including by absolute path and behind a wrapper
(`timeout 5 /bin/sh -c x` is refused on `sh`).

### Wrappers are unwrapped, not denied

A command whose argument list *is* another command — `timeout`, `nohup`, `env`,
`xargs`, `stdbuf`, `nice`, `ionice`, `srun`, `mpirun`, `mpiexec` — is stripped by
`shell_paths.unwrap_argv` before anything reads the head, recursively and depth-bounded.
So `timeout 60 pytest -q` is validated, classified and credited as the pytest run it is,
while `timeout 5 nohup bash -c x` is refused on `bash`. Two tests hold the line:
`test_a_wrapper_is_unwrapped_never_denied` (the sets are disjoint, and every
wrapper × interpreter pair unwraps to the interpreter) and
`test_a_wrapper_cannot_smuggle_a_refused_command` (the behaviour end to end).

### What a command's category still decides

The taxonomy in `servers/_shared/shell_paths.py` is no longer a gate — it is what the
client's classifier reads to decide approval and plan-mode availability.

| Category | Commands | Plan mode | Approval |
|---|---|---|---|
| `neutral` | `pwd` `echo` `printf` `which` `basename` `dirname` `realpath` `df` `true` `false` `:` `printenv` `export` | ✅ | ❌ |
| `read` | `cat` `head` `tail` `nl` `sed`◆ `wc` `cut` `sort`◆ `uniq` `comm` `tr` `fold` `column` `cksum` `md5sum` `sha256sum` `stat` `file` | ✅ | ❌ |
| `search` | `grep` `rg` | ✅ | ❌ |
| `inspect` | `ls` `find` `du` | ✅ | ❌ |
| `chdir` | `cd` | ✅ | ❌ |
| `env` ◆ | `module` `pip` `pip3` `conda` `conda3` `mamba` `mamba3` | query only | on mutation |
| `write` | `mv` `cp` `mkdir` `chmod` | ⛔ | ✅ |
| `exec` | the compilers (`gcc` `g++` `gfortran` `clang` `nvcc` `mpicc` …), the build drivers (`make` `cmake` `ninja` `meson` …), the runners (`python` `python3` `java` `node` `pytest` `ctest`), the checkers (`ruff` `mypy` `pyflakes` `black`), the TeX engines, and `source` | ⛔ | ✅ |
| **unplaced** | **everything else** (`git`, `rm`, `curl`, `tar`, `awk`, …) | ⛔ | ✅ |

The last row is the one that changed: a head no group places classifies as `UNKNOWN`,
which is read as a **run** — never plan-safe, never approval-exempt, owing a verdict, and
with its path operands still extracted so they are confined and prompted for. Widening
what bash accepts therefore widens what the user is *asked* about, not what slips
through. A parity test (`test_no_classified_command_is_one_the_server_refuses`) fails if
a group names a denied command, or if a category disagrees with how a call to it is
gated. (It also used to check the client-side platform probe's toolchain list against
`EXEC_EFFECTS`; that probe is gone — the platform servers below are now the only source
of hardware and toolchain facts, queried on demand rather than injected.)

`printenv` and `export` are `neutral` because their arguments are words (`VAR=value`), not
paths, and neither can run a command; an `export` holds only for the rest of the one chain,
which is fresh per call. `source` is filed `exec` because it *runs* the file it reads — see
the interpreter note below for why it is the one runner that is allowed.

A Makefile recipe is effectively arbitrary execution, but no shell-injection vector.
`chmod` is a write of a file's *mode* rather than its contents, and it is allowed for one
reason: running a workspace script by path (`./build.sh`) is already supported, but a
script arriving without the x bit — a fresh checkout, or a file the agent just wrote —
left no allowed command that could grant it, so the agent dead-ended on "Permission
denied". It grants no new *kind* of power: `python f.py`, `make` and a compiled `./a.out`
already execute workspace code. Note that `black` rewrites the file it formats yet is
filed `exec`: approval and plan-mode
gating are already right for it, and its operand is credited as *validated* by
`_bash_validation_scan` — a mixed write/exec kind for a single tool would buy nothing.

### ◆ What shifts a command out of its category, per call

| Trigger | Effect |
|---|---|
| `sed -i` / `--in-place` / `-i.bak` / `-ni`, `sort -o FILE` | `read` → **write** (approval, blocked in plan mode) |
| `> f` or `>> f` on any side-effect-free command | → **write**, operand = the target |
| `< f`, `2>&1`, `2>/dev/null` | nothing — kind unchanged, still plan-safe |
| `module load` / `add` vs `avail` / `list` / `show` / … | mutation vs query |
| pip/conda `list show freeze check inspect info help` vs anything else | query vs mutation (the network queries `search`/`index`/`download` count as mutation: a plan is drafted without reaching the network) |
| an unknown `module`/pip/conda sub-command | assumed to mutate (the server refuses it anyway) |
| **chain rule** | one non-read-only segment makes the *whole* call non-read-only |

### Refusals that are not re-categorisations

Refused whatever category the head belongs to:

| Refused | Because |
|---|---|
| shell interpreters — `bash`, `sh`, `eval`, `sudo`, `.` | they nest a command the validator never sees |
| unreachable write/exec flags — `find -ok/-delete/-fprint*`, `rg --pre/--search-zip/-f`, `grep -f` | the operand walk cannot reach the target. A write flag whose target *is* visible is confined instead |
| a nested `-exec CMD` whose head is not read-only | its operands include `{}`, whose expansion nobody can resolve |
| TeX shell-escape / `write18`, every spelling | it runs a command out of a document |
| env managers' `run` / `execute` | they nest an arbitrary command. Install, create and remove run under the approval prompt instead |
| a path, write-flag value or redirect target outside the workspace and unapproved | the sandbox boundary |
| `$VAR` in path or redirect position, heredocs, backgrounding, substitution, subshells | the path checked would not be the path used |

Each rejection **names the route that replaces it** and never points at a tool to call:
the payload answers a *shell* call, so any "call X instead" pointer reads as one more
shell command to try — which is how an agent ends up running the name of an MCP tool.
`test_every_denied_command_names_the_route_that_replaces_it` holds every denied name to a
non-empty hint.

The detail behind each row:

- flags whose write/exec target the operand walk cannot reach: `find
  -ok/-okdir/-delete/-fprint*/-fls`, `rg --pre/--pre-glob/--search-zip/-f`,
  `grep -f`. A write flag whose target *is* visible is confined instead of denied — see
  `WRITE_VALUE_FLAGS_BY_CMD` below.
- a **nested** command (`find … -exec CMD {} \;`) whose head is not read-only. `-exec`/
  `-execdir` are *not* on the flag denylist: `parse_segments` lifts the payload into its
  own `ParsedSegment` (`nested=True`), consuming the `;`/`+` as a terminator rather than
  mis-reading it as a chain separator, and the nested command is then judged on its own
  head against `READONLY_NESTED_COMMANDS` (derived from `READ_COMMANDS | SEARCH_COMMANDS |
  INSPECT_COMMANDS | NEUTRAL_COMMANDS`, never re-listed). So `find . -name '*.py' -exec
  grep -l PATTERN {} \;` runs, while `-exec rm/python/chmod/mv` stays refused — a nested
  command's operands include the `{}` placeholder, whose expansion no caller can resolve,
  so a nested *write* targets paths that cannot be checked. Because the split happens in
  the shared parser, the nested segment inherits the ordinary operand confinement **and**
  the client's out-of-workspace gate for free (`find /etc -exec cat {} \;` still refuses).
  This grants no new power: `python f.py` was already directly invocable. Previously the
  whole family was denied on the `-exec` token alone, leaving read-only fan-out no
  working spelling at all.
- TeX shell-escape/`write18` in every spelling (plus `_safe_env` pinning
  `shell_escape=f`, `openout_any=p`, `openin_any=p`).
- env managers: `run`/`execute` only — they nest an arbitrary command. Install,
  uninstall, create, remove and config all run under the approval prompt; every
  `module` sub-command does too.
- any path operand, write-flag value or redirection target outside the workspace and not
  approved; a `$VAR` in path or redirection-target position; a heredoc; backgrounding;
  substitution; a subshell; every shell interpreter and generic runner.

Approval nuances: `bash_run` is `SENSITIVE` + `NON_BATCH`, so it is never queued for
end-of-turn batch review; an "always" grant is scoped by `_scope_command_prefix` to the
**first two tokens** (`pip install`, `gcc solver.c`) for the session; the headless runner
auto-approves everything.
- **Shell interpreters** (`bash`, `sh`, `eval`, `sudo`, `.`) are permanently excluded —
  they nest a command the validator never sees. They get their own rejection message
  saying exactly that, so the agent stops looking for a wrapper instead of trying one
  spelling after another. Generic *wrappers* (`env`, `xargs`, `timeout`, `nohup`) are no
  longer among them: they are unwrapped, so what is judged is what they carry.
  `source` is the deliberate exception (filed under `exec`): its operand is a
  *file*, confined like any other path, so it nests no more than the already-allowed
  `./build.sh` — and unlike it, it sets the environment the rest of the chain runs in
  (`source venv/bin/activate && pytest`), which no other command can do. Only the
  readable spelling is offered: the `.` form stays refused.
- **No-ops** (`true`, `false`, `:`) are classified `neutral` so the capability probe
  `which pdflatex 2>/dev/null || true` is expressible; without them the chain is
  rejected on its last segment and the agent has no way to ask "is X available?".
- **A no-match is a result, not an error**: `grep`/`rg` exit 1 (no match) and `which`
  exit 1 (absent) with empty stdout are returned as `status: ok` with `matches: 0`.
  Reported as failures they read as broken commands and get re-run verbatim. A real
  failure (exit 2) stays an error. Only the *last* segment decides, since that is what
  bash reports — and a search piped into `head` exits 0 on empty output, so an empty
  stdout at exit 0 is treated as an empty search result too.
- **TeX is sandboxed twice**: every shell-escape/`write18` flag is denylisted at
  validation, and `_safe_env` pins kpathsea's `shell_escape=f` / `openout_any=p` /
  `openin_any=p` so `\write18` stays off even if the site's `texmf.cnf` enables it.
- **Command chaining is allowed** (`;`, `&&`, `||`, `|`) — each segment is tokenized and
  validated independently, so one denied command anywhere rejects the call, and one
  non-read-only segment makes the whole call non-read-only.
- **Multi-line commands are accepted**: an *unquoted* newline is a command separator, so
  it is normalized to `;` before tokenizing and every line is validated on its own
  (never folded into the argv of the line above — that was the reason newlines were once
  refused outright). A newline *inside quotes* is data, so a multi-line `python3 -c "…"`
  body runs as written. A heredoc (`<<EOF`) is still refused: its body is not an argv.
- **Redirection is allowed, and its target is confined** — `> out.log`, `>> out.log`,
  `< in.txt` and the fd forms (`2>&1`, `1>&2`, `2>/dev/null`, `&>/dev/null`). A
  redirection target is a path operand written with an operator instead of a flag, so it
  is judged exactly like one: resolved with `realpath` against the segment's cwd (a `cd`
  rebases it) and required to be inside the workspace or a directory the user approved —
  `ls > /tmp/out.txt` is refused with the same "not approved" message as
  `cat /etc/passwd`, and the client's gate offers that path for approval. A target built
  from an expansion (`> $HOME/x`) is refused: the shell that expands it is not the one
  checking it. Backgrounding (`&`) and command/process substitution (`$(...)`,
  backticks, `<(...)`) remain rejected. A single pipe and shell globbing (`*.py`) are
  permitted.
- **Segmentation lives in `servers/_shared/shell_paths.py`** (`parse_segments`), the same
  module the client's gate and classifier import — one tokenizer, so "where does this
  command end" and "is `out.txt` a write or an argument of `ls`" cannot be answered two
  different ways by the guard and the prompt. The server keeps only the *policy*: the
  denylist, the flag denylists, and confinement.
- **`module` (HPC Lmod) is supported**, every sub-command included: `load`, `unload`,
  `swap` and `purge` change this one subprocess's environment and nothing else, which
  the approval prompt covers. Since `module` is a shell *function*, the server sources Lmod init
  in the wrapper around the validated command, so `module load cuda && nvcc ...` works
  within one call; module args are gated by a strict regex and a curated `MODULE*`/`LMOD_*`
  env is passed through — that allowlist is `MODULE_ENV_PASSTHROUGH` in
  `_shared/module_env.py`, shared with the platform server's catalogue build so the two
  cannot resolve different module trees.
- **The environment managers (`pip`, `conda`, `mamba`) are available**, scoped in the same
  spirit as `module`: query (`list`, `show`, `freeze`, `info`, …) and add (`install`,
  `create`, `env create`), never remove (no `uninstall`/`remove`/`clean` — the write with
  no way back), never `conda run` (it nests an unvalidated command, exactly what
  `bash -c` is excluded for) and never `config` (it persists settings past the session).
  Their *file* operands are confined like any other (`pip install -r req.txt` reads from
  the workspace, `/etc/req.txt` needs approval), but what an install writes lands in the
  target interpreter's site-packages — outside the workspace by construction. That is
  the point of allowing them, and it is the approval prompt that governs it, not a path
  check. Note the target: `_safe_env` puts MIMIR's own interpreter first on PATH, so a
  bare `pip install` provisions *that* env. Client-side they classify as `ENV_DISCOVERY`
  (a local query — plan-safe) or `ENV_MUTATE` (install/create, and the network queries
  `search`/`index`/`download`), never `EXEC`.
- **Every command that takes a file operand has its paths confined** to the workspace
  root: the readers (`cat`/`grep`/`wc`/…), `sed` (whose `-i` writes in place, so an
  in-place edit cannot escape), both ends of `mv`/`cp`, and — equally — the whole
  build/execution group. An interpreter or compiler takes a file operand exactly as
  `cat` does, so `python /tmp/evil.py`, `gcc /tmp/x.c -o /tmp/x.out` and `pytest /etc/`
  are refused for the same reason `cat /etc/passwd` is; a workspace-local binary's own
  operands (`./solver.out /etc/secrets`) are checked too. `tr` is exempt: it reads stdin
  only and its args are character sets. The neutral words (`echo`, `printf`, `which`, …)
  are exempt too: their args are text. A script written line by line with `printf` was
  refused because a quoted line starting with `/` was judged as a path. A redirection
  target is still confined, whatever the command.
- **A flag value is confined when the flag *writes*, and only then** — `WRITE_VALUE_FLAGS_BY_CMD`
  in `shell_paths.py` (`gcc/g++/gfortran/nvcc -o`, `javac -d`, `sort -o/--output`,
  `ruff --output-file`, `pytest --junitxml`, `mypy --cache-dir`, `cmake -B`, the TeX
  `-output-directory`/`-jobname`, …), in **both spellings**: separated (`-o out.o`) and
  glued (`-oout.o`, `--output-file=out.txt`). This closed a real asymmetry — the
  separated form parsed as a positional and was refused, while `gcc a.c -o/tmp/x.out`,
  `ruff check --output-file=/tmp/r.txt` and `pytest --junitxml=/tmp/j.xml` wrote outside
  the workspace with no prompt. Read flags stay deliberately unconfined: `-I/usr/include`,
  `-L/usr/lib`, `-Wl,-rpath,…`, `-lm` and `-m pytest` are how any real build finds its
  system headers and libraries, and confining them would refuse every HPC compile.
  Writing outside the workspace is what needs the user's approval; reading a system
  header is not.
- **Operand extraction is command-family aware**, because "every non-flag argument is a
  path" is wrong for the families that take text first: `grep /etc/passwd notes.txt`
  searches for a *string* and opens nothing, so the leading pattern (and a `sed` script)
  is skipped. The files are not: `grep foo /etc/passwd` and `sed -n 1p /etc/passwd` stay
  refused, and `-f/--file` (patterns read *from* a file) is denylisted on both engines
  rather than merely confined — its operand sits in flag-value position, exactly where
  the pattern-skipping rule stops looking. (`stat`/`file` are reads for the same reason:
  they open the file they are given, unlike the metadata words `pwd`/`which`/`df`.)
- **A path built from a shell expansion is refused, not guessed at.** Expansion happens
  in the child shell, whose environment is not the one doing the checking — and a chain
  can change it mid-flight: `module load cuda && cat $CUDA_HOME/version.txt` checked as a
  harmless relative path (`CUDA_HOME` unset in the server) and then read from the module
  tree at runtime. The command that runs must be the command that was checked. Only
  *path position* is affected, so `gcc -I$CUDA_HOME/include` and `echo $HOME` still work.
- **Confinement is a prompt, not a wall.** A refused path names *approval* as the way
  forward, because that is an action the agent can take: the client's out-of-workspace
  gate surfaces every path a command names (see POLICY.md) and a grant reaches this
  guard through the shared sidecar, so an approved `cat /data/runs/log.txt` runs.
- **One source of truth for "which tokens are paths"**: `servers/_shared/shell_paths.py`
  holds `EXEC_COMMANDS`, `PATH_INSENSITIVE_COMMANDS`, `normalize_path_arg`,
  `segment_path_operands` and `cd_destination`, and is imported by this guard *and* by
  the client gate. This guard confines what that gate prompts for; two copies would
  fail silently in both directions — a path the gate misses cannot be granted, a path
  this guard misses is never gated. Confinement is the *default* — `takes_path_operands`
  is true for everything but the listed no-ops — so a command nobody classified has its
  operands confined too, which is exactly the one that most needs it (asserted in
  `test_server_contracts`). `EXEC_COMMANDS` is the union
  of four declared by *effect* — `VALIDATOR_COMMANDS`, `BUILD_COMMANDS`, `RUN_COMMANDS`,
  `ENV_SETUP_COMMANDS`, mapped by `EXEC_EFFECTS`. The sandbox treats all four alike; the
  split exists because what a green exit *proves* differs, and the client's evidence
  layer must not have to guess it (see POLICY.md → *Validation Policy*).
- **There is no `working_directory` argument.** Every call starts at the workspace root;
  `cd` moves within it and holds for the rest of that call (`cd sub && pytest t.py`),
  but each call is a fresh subprocess so it does not carry over. One way to say "where",
  not two — a second base was only ever a second thing to keep confined.
- `cd`'s target is confined like any other path, **and it rebases the confinement of
  every later segment** — the base directory is threaded through the segment loop,
  because validating each segment against the *initial* cwd made both halves of
  `cd /etc && cat passwd` look individually fine while the chain read outside the
  sandbox. A `cd` that stays inside the workspace prompts for nothing; the command that
  follows it decides.

## agent_state/server_spawn_agent.py

Purpose: delegate a self-contained sub-task to a **fresh** `MimirAgent` that runs to
completion and returns its answer, so the orchestrator can fan work out instead of
carrying every intermediate step in its own context.

Tools:
- `spawn_agent(task, context="", tools=[], max_steps=30, time_budget_secs=600, model="")`
  — spin up a child agent, run it, return its answer. `context` is prepended to the task
  prompt.
  - `model` — empty means the caller's model. The client sends it with every call, in
    the request `_meta` (`mimir/model`), so a `/model` switch mid-session reaches the
    child; the server's environment was frozen at spawn and is only a fallback.
  - The caller may name another model **the endpoint serves**. At start-up the server
    asks the backend once for its served models (`served_model_info()`, i.e.
    `/v1/models`) and renders them with the client's model catalog
    (`config/model_catalog.py`) into the description of `model`. It is rendered once,
    so the tool list stays prefix-stable for the session.
  - A model the endpoint does not list, or one the catalog marks `"delegable": false`,
    is refused with the list of usable ones. There is no fuzzy matching.
  - The result carries `model`, the model the child ran on.
  - **The user's setting decides what may be handed over at all** — `explore` or
    `parallel` (`config.constants.SUBAGENT_LEVELS`), sent per call in the request
    `_meta` (`mimir/subagent_level`). At `explore` the grantable table holds nothing
    that writes. There is deliberately no rung in between: sub-agents editing one
    shared tree overwrite each other in silence, so a child that writes gets a copy of
    the repository or it does not write.
  - **A child granted a writing or running tool works in a git worktree of its own**,
    created before it starts, on branch `mimir/<child>`, under `/tmp/mimir-worktrees-*`
    — the local disk, not the state dir (a home under quota cannot hold a checkout plus
    what a build writes), and deliberately not the scratchpad (writable without
    approval, which would take the child's edits out of the approval layer). On a clean
    run MIMIR commits the work to the branch and **keeps the copy**: the branch carries
    the code, the copy carries the build tree, which git never had and which the axis
    paid its compile for. Git refuses to check out a branch a copy holds, so the result
    says to merge or cherry-pick it rather than leave the caller to read a git refusal.

    **A branch that carries nothing is dropped**, and the test is the commit the copy
    was cut from, recorded at creation — not `git branch -d`, which asks whether the
    branch is merged into whatever HEAD is *now* and so keeps every empty branch as soon
    as the main tree moves on. The copy is detached first (git will not delete a branch a
    worktree holds) and the report only says the branch went if it went.

    **A copy goes one way only**: whatever it holds is committed first, then the
    directory is removed. The copy most worth reclaiming is the one a failed run left,
    and that is exactly the one whose work exists nowhere else.

    Two rules reclaim them, because one registry is not enough:
      - *Per session, at creation.* A session keeps `SUBAGENT_WORKTREES_KEPT` (3)
        finished copies plus the one being made, read off the sibling cards. Only copies
        still on disk take a place, and "still running" is read off the **pid** on the
        card — a server killed mid-run leaves a card saying "running" for ever, and
        taking that at face value would exempt its copy from reclamation.
      - *Orphan sweep, also at creation.* The cards live with the session, so deleting a
        conversation takes the registry and leaves the copies. Each copy therefore also
        carries a **marker** beside it (never inside: `git add -A` would commit it),
        written before the copy is handed over and naming its session and pid. A copy is
        an orphan when its owner is gone **and** its card is gone — either alone is a
        copy merely between states. One with no marker predates this and is judged by
        git: a directory `git worktree list` does not register is nobody's copy.
  - **What the child reports, it reports in the repository's names.** `files_read` and
    `files_written` are made relative to the copy before they go back, as
    `files_changed` always was — a caller quoting `/tmp/mimir-worktrees-…` as evidence
    cites a path that means nothing in the repository.
  - `tools` — the tools the child gets, named from the caller's own list. **The caller
    cannot grant what it does not have**: the client sends what it may hand over with the
    call, in the request `_meta` (`mimir/grantable`, a `{tool: owning server}` table built
    by `query_engine.toollist.grantable_tools`), together with its mode
    (`mimir/mode`). A name outside that table is refused with the list of what is in it;
    no table at all (an unknown caller) grants nothing.
  - **Empty `tools` is reconnaissance**: the servers in `capabilities.explorer_servers()`,
    in a **read-only mode**, which is what makes it read-only — the mode strips every
    `PLAN_BLOCKED` tool and gates the dual-use shell at call time. The child is briefed to
    answer with a conclusion citing files, symbols and line numbers rather than the
    contents it read; the caller delegates precisely to keep those out of its window.
  - **Named tools decide the rest.** The child connects only the servers those tools live
    in, then *prunes* its registry to the grant (`_prune_tools`): connecting a server
    brings its siblings, and an unpruned grant would be approximate. It runs in `agent`
    mode when at least one granted tool is `PLAN_BLOCKED` or `PLAN_READONLY` — without the
    second, a child given the shell to build with could not run the build — and is briefed
    to end with a HANDOFF rather than stopping mid-air.
  - `time_budget_secs` — the wall for this run, clamped to
    `[SUBAGENT_MIN_BUDGET_SECS, SUBAGENT_HARD_CAP_SECS]` (60–1140 s). The child is told
    how long it has, in the brief.
  - `background=True` **detaches the child**: the call returns a `background_job`
    descriptor at once, the caller carries on, and the client's watcher resumes it with
    the answer when the child lands — the same machinery a detached shell command uses
    (`query_engine/background.py`, `ws_worker._watch_job`), so nothing in the client
    knows this kind of job from that one. The descriptor's `status_op` and `summary_op`
    both point at `subagent_job`; the dispatch guard then refuses the *status* question
    (the wake answers it) while allowing the *result* one, which is progress.
  - Each child gets a **session of its own**, `sessions/<parent>/subagents/sub-xxxxxxxx`,
    passed to the servers it starts as `MIMIR_SESSION_ID` through `agent.server_env`
    (merged by `integration/server_manager.connect_server`). Its todo list and its
    scratchpad are therefore its own — which is what makes the checklist tools grantable —
    and several children working in parallel cannot collide. It leaves a `subagent.json`
    card next to them for the sub-agents panel, and the whole thing goes when the parent
    session is deleted.
  - **A child that outlives its caller is stopped.** The child runs on a thread of its
    own that nothing can kill, so a budget that expires used to return to the caller and
    leave the thread working: it went on spending model calls on an answer nobody would
    read, and wrote its card as a clean `finished` minutes later. The timeout branch now
    sets the agent's `_cancel_flag` — the same cooperative flag the WebSocket worker uses
    to interrupt a turn — and marks the card `abandoned`, which outranks `finished` even
    when the thread does reach a clean result. The card's four terminal states are
    therefore `finished`, `abandoned`, `failed` and `unknown`. The line handed back to
    the caller says not to spawn the same task again unchanged: told only to use a fresh
    sub-agent, it respawned the task verbatim and spent a second full budget on it.
  - **A silent gap is accounted for.** A step is not one model call: an empty turn is
    retried, a nudge is answered, a checklist refresh re-asks — each up to the child's
    8192-token per-step ceiling (`SUBAGENT_ANSWER_TOKENS`), none of them a tool call.
    The activity log recorded only tool calls, so a child spending four minutes that
    way was, from the panel, indistinguishable from one that had hung. `_compact_event`
    now forwards the loop's `status` lines as `st` rows — blank ones dropped, clipped
    to 160 chars like every other field — and the panel renders them recessed between
    the tool rows. The token and diff streams still do not travel: those are per-token,
    this is per step.

- **Reaching the user.** A child whose delegating call is still open gets its
  approvals and its questions routed to the person, through that call's MCP session
  (`_install_user_channel`). Two rules are enforced rather than hoped for: the cards
  are shown **one at a time** (several children asking at once is a pile nobody can
  answer in order), and each says **which sub-agent is asking**. A detached child has
  no session left to raise a card on: it runs unattended and reports what its mode
  refused, as before.
- `subagent_job(op="status"|"result"|"list", job_key="")` — read a detached child:
  its state (`running` | `done` | `crashed`, or `unknown` once it has outlived its
  budget), everything it returned, or the list of this session's children. Read-only and
  `MAIN_ONLY`: these are the caller's own handles. `result` on a child still working
  gives what it has touched so far, which is what makes reading progress possible
  without polling its state.

Capabilities and gating:
- Declares `DELEGATE` (the channel prompt, guidance and observation all address by
  capability rather than by name) and `BACKGROUNDABLE` (a detached child's handle). It is **no longer dual-use by argument**: delegation
  cannot be the way around a read-only mode, because the grantable table is computed under
  the caller's own mode and has nothing writing in it there. So it stays plainly visible in
  plan and ask — that is where a sweep is the whole of the work.
- What a child is never granted is declared by the tools themselves, and read by
  `context.capabilities.reserved_for_main`: `MAIN_ONLY` (the plan the user approved, the
  questions asked of them, writing the shared memory), `DELEGATE` (no recursion) and
  `CLUSTER_SUBMIT` (allocation hours are spent where the user can see it). `TASK_PLANNING`
  is deliberately *not* reserved — it covers the working checklist too, which a child keeps
  in its own session.
- Declared `reversibility="reversible"`, hence **not** approval-gated: a card in front of
  every exploration is a card in front of the behaviour the tool exists to make cheap. A
  writing child is gated by its own approval layer.

Execution model:
- The tool is synchronous from the model's point of view but runs the sub-agent in a
  dedicated thread with its own event loop, so it is safe to call from inside the parent's
  running asyncio loop. Several `spawn_agent` calls emitted in one model step are
  dispatched **concurrently** by the parent's `asyncio.gather` in `dispatch.py` — which is
  why the prompt asks for the fan-out in a single response.
- The per-call budget is enforced here, up to `SUBAGENT_HARD_CAP_SECS` (1140 s). The tool
  *declares* a larger wall to the dispatcher (`timeout_secs`, read by
  `capabilities.timeout_for`, and kept under the client ceiling
  `TOOL_CALL_TIMEOUT_MAX_SECS`), so the inner cap always fires first and hands back what
  the child had instead of the parent killing the call with nothing to show. Before this,
  the dispatcher's flat 120 s applied and no non-trivial delegation could finish.
- A child that overruns keeps running in its own thread and its answer never arrives. The
  timeout branch therefore reports what can be read off the agent itself
  (`_partial_handoff`): the files it had already touched, and its session — enough to
  carry the work on with a fresh child instead of starting it again.
- Success payload: `{"status": "ok", "answer", "completed", "files_read", "files_written",
  "blocked_by_mode", "model", "tools", "session"}`
  — `completed=False` means the sub-agent ran out of steps or reported the task incomplete
  (the answer is still informative); `files_read` is what the caller's observation layer
  credits as delegated discovery evidence; `files_written` lets a parent coordinating
  concurrent sub-agents detect overlapping edits; `blocked_by_mode` lists the actions the
  child skipped because the user's approval mode does not allow them
  (`{"action", "needs"}`), and a non-empty list makes `completed` False.
- The child runs in the caller's approval mode, read from the request `_meta`
  (`manual` when absent). It never prompts: its stdin is this server's JSON-RPC pipe.
  What the mode does not cover is refused (see POLICY.md).
- Sub-agent stdout is prefixed with the task name so its activity stays identifiable in
  the UI. The capability registry is strictly per-agent, which is what makes concurrent
  sub-agents on different server subsets safe.

## interaction/server_interaction.py

Purpose: let the agent pause mid-run and ask the **user** a clarifying question with
selectable choices, then resume with the answer.

Declares `MAIN_ONLY`: a sub-agent runs unattended, with this server's JSON-RPC pipe for
stdin. What it cannot settle it reports back, and the orchestrator — which does have the
user's attention — decides whether to ask.

Tools:
- `ask_user_question(question, header, options, multi_select=False)`

How it works — **MCP elicitation**:
- The tool calls `ctx.session.elicit_form(message, requestedSchema)`, which sends an
  `elicitation/create` request *back* to the client over the same session while the tool
  is still running. The client's `elicitation_callback`
  (`mimir/client/integration/server_manager.py`) surfaces the question in the active
  frontend — the terminal CLI (numbered menu) or the VSCode webview (modal) — and returns
  the user's selection so the tool can finish.
- The rich answer shape (per-option descriptions, single/multi-select, free-text "Other")
  is carried in an `x_mimir` extension on the requested JSON schema; the reply comes back
  as `{"selected": [...], "other_text": ...}`.
- **Escape-hatch options are stripped server-side.** Both frontends always render the
  free-text "Other" field, so an option that only restates it ("Other", "Something else",
  "Request changes", "Add precisions", "None of these") is dropped by `_normalize_options`
  before the question is sent. The match is on the whole normalized label — a parenthetical
  aside and trailing punctuation are ignored — so real choices that merely contain those
  words ("Request changes from the reviewer") survive. A question left with no substantive
  option is dropped; a call left with no question returns a structured error.
- Protocol outcomes: **accept** (answered), **decline** (user said no), **cancel**
  (dismissed). Decline/cancel/timeout — or no interactive frontend connected — return an
  empty selection, and the tool tells the model to *proceed with its best judgment* rather
  than hang.

Notes:
- The tool is read-only and **not** approval-gated — asking a question is harmless.
- Because the elicitation callback is generic, **any** server can elicit the same way
  (`ctx.session.elicit_form(...)`) for under-specified arguments — e.g. "which of these PRs?"
  or "which GPU partition?".

### Elicitation vs. approval — when to use which

These are two different "pause and involve the user" mechanisms; don't conflate them.

| Need | Mechanism | Examples |
|---|---|---|
| "Is the agent *allowed* to do this risky/irreversible action?" | **Approval system** (`mimir/client/guardrails/policy/` + the WS `_approval_shim`) — risk/scope classification, yes / no / **always**, pre-write diffs, batch revert | bash commands, file writes/deletes, proxy run/eval tools |
| "I don't know what the user *wants*; the answer changes what I do" | **Elicitation** (`ask_user_question` / `ctx.session.elicit_form`) — a structured form prompt | architectural forks ("Postgres or SQLite?"), ambiguous requirements, disambiguating an under-specified argument |

Guidance:
- **Do not** route command/write confirmation (e.g. bash) through elicitation — it has no
  concept of risk, scope, "always", or rollback, so it would be a downgrade. Keep risky
  actions on the approval system.
- Use elicitation only to **gather missing intent**, not to grant permission.
- A single tool action should not trigger *both* an approval prompt and an elicitation for the
  same decision — pick the one that matches the question being asked.

## proxy/_lib/ — shared helper library

Helper package (not an MCP server) imported by `proxy/server_proxy.py`, the `proxy/_ops/` modules, and `_proxy_runner.py`.  Dependency direction is `store ← procs/metrics/command/report ← execute/ratchet`.

| module | contents |
|---|---|
| `store.py` | Storage root `_CACHE_DIR` (env-overridable via `MIMIR_PROXY_BENCH_DIR`, default `<workspace>/proxy_bench` — with the project it belongs to, so deleting the project resets the experiment; it used to be `~/.cache/proxy_bench`, and a deleted project came back with its registry, an "in progress" optimisation and a run command pointing at a file that no longer existed); every path derives from it **at call time** via functions (`registry_path()`, `runs_dir()`, `refs_dir()`, `suites_dir()`, `scaffolds_dir()`, `opt_runs_dir()`, …), so a hermetic test repoints one attribute. Generic atomic IO (`_read_json`, `_write_json_atomic`, `_write_text_atomic`, `_atomic_symlink`); registry I/O (`_load_registry` with corrupt-JSON backup, `_save_registry`, `_registry_lock()` — exclusive `fcntl.flock`, thread-local re-entrancy depth; `_file_lock(path)` for any other exclusive section); suite persistence (`_load_suite`/`_save_suite`/`_latest_suite_results`); reference layout (`_ref_dir`, `_load_ref_metrics`, `_ref_output_path`); optimization-session layout (`_opt_config_file`, `_opt_session_runs_dir`, `_opt_ledger_file`, `_opt_best_file`, `_resolve_proxy_name`, `_write_active_session`). |
| `procs.py` | Process/run state: pid + `/proc` starttime bookkeeping (PID-recycling and zombie detection in `_is_running`), `_squeue_state`, `_run_state`; log access (`_log_path`, `_read_log` capped at `_MAX_LOG`); run lifecycle — `_new_run_dir`, `_write_run_config`, `_launch_detached`, `_submit_sbatch`, `_cancel_run` (`scancel` else SIGTERM → SIGKILL), `_validate_slurm_args`; active-run symlinks. |
| `metrics.py` | `_parse_metrics_block` (block-delimited; strict fallback scan), `_strip_reserved_metrics` (against `RESERVED_METRICS` from `_shared/numerics.py`) (server-side invariants the proxy cannot forge), `_normalize_time_metrics` (server-measured `wall_time_s` + `time_s` plausibility guard), field I/O + `_field_norms`, invariants (`_finite_check`, `_conservation_residual`, `_convergence_order`), `_evaluate_requirements`. |
| `command.py` | `_render_param_file`, `_expand_cmd_template` (single template-expansion path for local + sbatch), `_build_run_cmd`, `_sbatch_header`, `_build_sbatch` (injection-safe `repr()` postrun; captures solver wall time + exit code for the post-run guard). |
| `report.py` | Roofline (`_resolve_roofline` with platform-profile fallback, `_compute_roofline`), `_row_with_extra`, `_diff_run_pair`, `_diff_run_dirs`. |
| `execute.py` | The three run paths every launch funnels through: `_run_benchmark_case` (synchronous case), `_seal_reference` (immutable reference store), `_post_run_finalize` (settles detached local/Slurm runs). All three share `_settle_metrics` (reserved-metrics purge, time_s guard, returncode) and `_apply_invariants` (reference comparison + lifted error norms, `finite`, `conservation_residual`), so a detached run carries exactly the metrics a synchronous one does. |
| `ratchet.py` | Optimization ratchet: `_ratchet_verdict` / `_is_improvement` / `_run_primary_value` / `_select_best_case`, best-so-far persistence (`_load_best`/`_save_best` incl. `wall_value` for the timing audit), `_append_ledger`. |

---

## proxy/server_proxy.py

Purpose: registration, benchmarking, execution, and iterative optimization of proxy scientific-computing codes — behind **seven op-dispatched tools**.

Each tool takes an `op` parameter selecting the operation; the docstring carries the full ops table with per-op required args. Ops are grouped strictly by capability class, because capabilities are declared per tool: read-only tools carry no caps, mutating tools are `sensitive` + `PLAN_BLOCKED` + `non_batch` with a `confirm=True` gate on every op, and Slurm submission is isolated on its own `CLUSTER_SUBMIT` tool. Unknown ops and missing per-op args return `err()` with a corrective hint. **Every** `ok()` response carries a `next_step` field naming the exact next call to make — read-only listings included, so an empty registry points at the call that would fill it.

`proxy_eval`, `proxy_eval_status` and `proxy_exec` also declare a **`run_outcome`** spec (`_RUN_OUTCOME` in `server_proxy.py`), which tells the client what *this server* saw of a run it performed: `run_dir` identifies the run, `state="crashed"` means it did not complete, `feasible=False` means its result failed the session's requirements, and `state="done"` credits the client-side `measured` validation tier for the session's source file. `proxy_exec` extends it with a `rows` clause so each failing case of a suite run — which answers `ok` overall — fails its own run. This is a floor, never a credit: there is no way to declare a *passing* verdict, and a ratchet `reject` (measured fine, just no better than the incumbent) is deliberately not a failure. See `POLICY.md` → the run-outcome floor.

Tool bodies live in `proxy/_ops/` as plain functions (no `@mcp.tool` decorators there — the AST-based capability parity gate only sees the seven declarations in `server_proxy.py`):

| `_ops` module | Contents |
|---|---|
| `registry.py` | proxy list/inspect (registration + readme + recent runs), register/update/unregister; the ~16 optional registration fields are carried by one `metadata` dict — descriptive but for the build triple (`build_cmd`/`build_cwd`/`build_timeout_s`), which `_lib/build.py` executes before a run measures anything. A proxy declaring a build may be registered before its executable exists, since the build is what produces it |
| `references.py` | reference list, blocking reference sealing |
| `runs.py` | run history/logs/diff/compare/aggregate, local detached launch, cancel |
| `suites.py` | suite list/inspect/report, define/update/delete, blocking local suite runs, one-call `benchmark_create` |
| `eval_session.py` | ratchet session init/configure/run/stop/reset/reset_to_best/end + status/results/log/runs/diff/config |
| `slurm.py` | the three sbatch submission ops (single / per-case suite / eval) |
| `scaffold.py` | component-harness scaffolding + the four template generators |

Tools (read-only):
- `proxy_get(op, name, run_timestamp)` — ops: `proxies`, `proxy` (registration + readme + last-5-runs), `references`, `suites`, `suite` (definition + validation), `report` (suite results table)
- `proxy_runs(op, …)` — ops: `list`, `logs`, `diff`, `compare` (field norms vs a reference), `aggregate` (table across arbitrary completed runs, with roofline columns)
- `proxy_eval_status(op, …)` — ops: `status` (default — zero-arg polling call), `results` (per-case metrics + requirements pass/fail + ratchet `verdict`/`best`/`stall` + recommendation), `log`, `runs` (each entry flags `is_best`), `diff`, `config`

Tools (sensitive / approval-gated, `confirm=True` required):
- `proxy_manage(op, …)` — ops: `register`, `update`, `unregister`, `clean`, `suite_define`, `suite_update`, `suite_delete`, `scaffold` (generate reference + test harness files for a source component). `unregister` drops the registry entry and keeps everything else; **`clean`** removes a proxy's runs, optimisation state and tree snapshots, clears the active-session pointer, and deliberately does NOT cascade into sealed references and suites (a reference costs real compute and may be shared) — instead it **names what it left and how to remove it**, which is the answer to "I deleted everything and it still remembers". Also reachable as the `/proxy clean <name>` slash command, on both the CLI and the WebSocket UI, so housekeeping does not have to be asked of the model
- `proxy_exec(op, …)` — ops: `run` (non-blocking local), `reference` (blocking seal), `suite` (blocking run of all cases × sweeps), `benchmark_create` (reference + one-case suite in one call), `cancel`
- `proxy_eval(op, …)` — a monotone optimization ratchet. **The proxy at `proxy_source_path` is a HARNESS, never the subject**: it runs the code and prints metrics, and `optimize_paths` names the real sources it exercises — imported, linked or loaded — and the ratchet may edit. Nothing in the loop is language-specific: a compiled project lists its `.cpp`/`.f90` sources and declares `metadata={"build_cmd": …}` at registration, which the server runs **once per evaluation run, before any case is measured** (`_lib/build.py`). That is not a convenience: the ratchet fingerprints sources only, so a binary left unbuilt would be measured and *accepted* against code that never ran, and a `reset_to_best` would restore sources while leaving the rejected attempt's artifact in place. A failed build ends the run with no verdict and no ledger entry; `results` returns the build log tail instead of "may still be in progress". Build time sits outside the measurement budget and is paid once per run whatever `repeat` is, and incrementality is the project's build system's job — which is why `tree_snapshot` stamps restored files with a fresh mtime, or `make` would skip the rebuild it must do. `init` refuses an empty `optimize_paths`, a path outside the workspace, and `proxy_source_path` appearing among them — that last refusal is what makes a self-contained copy inexpressible. Twice running, with nothing forbidding it, a model answered "make a runnable proxy" by writing a standalone script mirroring the package it was meant to accelerate, so the accuracy constraints held for the duplicate and for nothing that shipped. Ops: `init` (snapshots the `optimize_paths` **tree** as the baseline — a re-init never moves an existing one, or it would promote already-optimised code to "the original"; takes `primary_metric`/`primary_goal`/`min_improvement`/`max_stall` + optional `convergence`), `configure`, `run` (non-blocking; errors if one is active), `stop`, `reset` (restore every tracked file to the baseline tree), `reset_to_best` (restore them to the tree of the best *accepted* run — undo a regression without discarding progress). Both restore the whole set or nothing: a per-file best would let a restore assemble file A from one run beside file B from another, a combination that was never measured and can still run, `end` (close the session: clears `active_session`, keeps source/snapshots/ledger; lifts the direct-execution guard — refuses while a run is active). While a session is active, the client blocks running the proxy source/executable directly through the exec surface (`bash_run`) — see `POLICY.md` → Proxy Direct-Execution Guard. `results` returns a `verdict`: **accept** (feasible + improves the objective → new best-so-far), **reject** (regression → revert with `reset_to_best`), or **converged** (feasible and no improvement for `max_stall` runs). Feasibility = all `requirements` pass, including numerical invariants (`l2_rel`/`linf_rel`/`finite`/`conservation_residual`/`convergence_order`) which the server computes and exposes as ordinary metrics. These names are **reserved**: values the proxy prints for them are stripped before evaluation (`reserved_metrics_ignored` records the drops), so the code under optimization cannot satisfy its own acceptance constraints; `init`/`configure` reject reference-dependent requirements when no sealed reference (or registry `conserved_metric`) backs them. Per-session state under `opt_runs/<proxy>/`: `best.json` pointing at a **tree snapshot** (`_lib/tree_snapshot.py`: a shadow git repository at `proxy_bench/opt.git` whose work tree is the workspace and which only ever adds `optimize_paths`, so the user's own `.git` is never touched and the workspace need not be a repository; a per-run directory copy is the fallback where git is unavailable, with the same atomicity). The snapshot is taken at LAUNCH (`tree_at_launch.json` in the run dir), so what is accepted is the code that produced the run and not whatever the files hold when it settles. The first run must be launched from the baseline tree — `_prepare_run` refuses otherwise — because without a measurement of the untouched code every later number is an assertion rather than a comparison, append-only `ledger.jsonl`, per-run frozen `ratchet.json`. The ratchet is settled **by the runner at run completion** (flock-serialized, idempotent) — ledger/best are the source of truth even if the agent never calls `results`; `results` replays the frozen verdict. `op='run'` accepts `background=True` (cap `BACKGROUNDABLE`): the result carries a `background_job` descriptor (status/summary ops) so the WebSocket worker watches the detached run off the critical path, notifies the UI (`job_complete` event), and auto-resumes the agent with the verdict/best — the agent ends its turn instead of polling. Degrades safely in the CLI (no watcher → normal polling).

Tools (cluster, `CLUSTER_SUBMIT`):
- `proxy_slurm(op, partition, …)` — ops: `run` (single job), `suite` (one job per case × sweep), `eval` (optimization run); carries a `risk_note` shown in the approval prompt
  - Aims at one kind of node with `constraint`/`nodelist`; `ntasks` for an MPI proxy. `exclusive` defaults to on for `eval` and `suite` (timings) and off for `run`. The batch header comes from `_shared/slurm_script.py`, shared with `sbatch_submit`.
  - Everything runs on the node, build included. The runner and the post-run step pick a Python the node can run (`.venv-<os>-<arch>`, then the server's), unless `python_executable` is set.
  - **Machine pinning.** Each eval run writes `machine.json` (host, context, `machine_signature`) where it runs. The baseline run pins the session to its signature. A later run with a timing metric on another signature gets the verdict `incomparable`: best and stall do not move, and the next step is to run where the baseline ran or to `rebaseline` on the target. `rebaseline` clears the pin. Accuracy metrics are never held back.

Param file system:
- Each registration optionally includes a `param_file_template` (inline text), a `param_file_path` (path relative to the executable), and a `param_file_format` (`text`/`json`/`yaml`/`fortran_namelist`/`ini`).
- The template supports `{executable}`, `{output_file}`, `{param_file}`, and `{extra_params}` placeholders inside `run_cmd_template`; any additional `{key}` token is filled from `param_overrides`.
- Per-run `param_overrides` replace `{key}` tokens in the rendered template AND patch `key = value` / `key=value` lines via regex, so they work with Fortran namelists, INI files, and free-text parameter files without format-specific parsing.

Proxy output protocol:
- The proxy writes a `PROXY_METRICS_BEGIN … PROXY_METRICS_END` block to stdout with `key=value` pairs.
- A special `output_file=<path>` key triggers the server to copy the named file into the run directory and include it in output comparisons.
- Optional roofline keys: `flops` (total FP operations), `bytes_moved` (bytes transferred), `time_s` (wall time). Roofline peaks come from the registration `metadata` (`peak_gflops_per_s`, `peak_bandwidth_gbytes_per_s`) only; unset means no roofline row rather than a guessed ceiling.
- If no block markers are found, the server falls back to scanning the last stdout lines for `KEY=VALUE` (single-token RHS only).

Suite case schema (for `proxy_manage(op='suite_define')`):
```json
{"case_id": "baseline_32", "description": "baseline at size 32",
 "scope": "full", "proxy_name": "myproxy_v2", "reference_name": "myproxy_ref",
 "param_sweeps": [{"size": 32}, {"size": 64}], "extra_params": "",
 "metrics": ["iterations", "memory_mb"]}
```
`scope` is `"full"` (whole proxy) or `"component"` (extracted harness).

Optimization ratchet (see also the `proxy-optimize` skill):
`proxy_eval(op='init', primary_metric=…)` → `proxy_eval(op='run')` → poll `proxy_eval_status()` →
`proxy_eval_status(op='results')` → follow the `verdict`: **accept** → edit source to
improve further → run; **reject** → `proxy_eval(op='reset_to_best')` → try a different
edit → run; **converged** → `proxy_eval(op='reset_to_best')` → summarize.
Companion script: `_proxy_runner.py` — standalone subprocess runner (not an MCP server). Reads `<run-dir>/config.json`, executes the full benchmark suite, evaluates requirements, and writes `<run-dir>/metrics.json`; its `[proxy_runner]` log lines are what `proxy_eval_status(op='results')` parses.

Storage layout (all under `<workspace>/proxy_bench/`):
- `registry.json` (+ `.lock`) — registered proxy metadata
- `references/<name>/` — sealed reference datasets (metrics + output files)
- `runs/<proxy_name>/<timestamp>/` — per-run outputs, metrics, param file, postrun.py; `runs/<proxy_name>/active` symlink to the most-recent run
- `suites/<name>/suite.json` + `suites/<name>/results/<timestamp>/summary.json` (+ per-case `run_dir` pointers)
- `opt_runs/active_session`, `opt_runs/canonical/<proxy>.<ext>` (one-time source snapshot, never overwritten), `opt_runs/<proxy>/opt_config.json` + timestamped run dirs + `active` symlink; ratchet state: `opt_runs/<proxy>/best.json` + `best_source<ext>` (best-so-far pointer + its source snapshot for `reset_to_best`), `opt_runs/<proxy>/ledger.jsonl` (append-only per-run verdicts), and a frozen `ratchet.json` inside each run dir
- `scaffolds/<component>/` — generated harness files

Safety model:
- Every mutating op requires `confirm=True`; unknown/missing-arg calls fail closed with a hint.
- **Registry concurrency**: register/update/unregister acquire `_registry_lock()` (exclusive `fcntl.flock`) around the read-modify-write sequence.
- Runs are non-blocking (detached subprocess or Slurm job) — `reference`, `suite`, and `benchmark_create` are the blocking exceptions.
- PID recycling guard: starttime from `/proc/<pid>/stat` is stored alongside the PID; `_is_running` rejects mismatching starttimes.
- Registry corruption: `json.JSONDecodeError` copies the file to `.corrupt` and raises `RuntimeError` before any write attempt.
- Slurm injection safety: `_build_sbatch` writes a `postrun.py` with all paths embedded as `repr()` literals; the sbatch script passes it via `shlex.quote`.
- `proxy_slurm` is the only `CLUSTER_SUBMIT` tool — the client's pre-submission guard holds the first submission each query until something has been validated locally.

---

When a server adds or removes a tool:
1. update this file
2. update `README.md` if the default-server capabilities changed materially
3. run `mimir/tests/test_server_contracts.py`
4. declare the tool's semantics so the client classifies it (see below)

If adding shared helper code used by multiple servers, put it in `_shared/`
(cross-group utilities), `proxy/_lib/` (proxy-group helpers),
or a new group-local `_shared_<domain>.py` module — never duplicate it.

If the client registration set changes, also update:
- `mimir/client/config/constants.py` (`SERVERS` **and** `SERVER_DESCRIPTIONS`)
- the *Registered Servers* table in `README.md` (and its server count)
- the *Registered-by-default servers* list above

### Declaring tool capabilities (`_shared/capabilities.py`)

The client derives every tool's *semantics* — which tools are sensitive,
plan-blocked, writes/edits, searches, validations, reads, cacheable, which args
carry paths, what approval fallbacks they have — from what the server **declares**,
not from hardcoded client-side lists. A tool declares this with the
`tool_caps(...)` helper, which returns the `meta`/`annotations` kwargs that ride
the MCP `Tool` model through `list_tools()`:

```python
from _shared.capabilities import tool_caps, READ, SEARCH, EDIT, SENSITIVE

@mcp.tool(**tool_caps(caps=[READ], path_args=["path"], label="Reading file: {path}"))
def read_file_lines(path: str, start_line: int = 1, end_line: int = 200) -> dict: ...

@mcp.tool(**tool_caps(caps=[EDIT], path_args=["path"], sensitive=False))
def write_file(path: str, content: str) -> dict: ...
```

The capability vocabulary is defined once in `_shared/capabilities.py` and mirrored
exactly in `client/context/capabilities.py` (parity guarded by
`test_capabilities.test_vocab_in_sync`); the per-flag reference table lives in
[`PLUGINS_DETAILED.md`](PLUGINS_DETAILED.md#tool-capabilities) — this file does not
re-list it. Declare only the kinds
that matter to a policy; umbrella flags the client can *derive* are not declarable
(`is_write` = `EDIT`∪`CONTENT_WRITE`∪`REMOVE`; `clears_edit_loop` = `READ`∪`VALIDATE`),
so declaring e.g. a writer's `EDIT` is enough — no separate "is-write" flag to forget.
No first-party tool carries `VALIDATE` (the stack validates through the `bash` server),
but it stays declarable so an extension-pack server can ship its own validator.
The client reads
`tool.meta["mimir"]` first (authoritative), then standard
`readOnlyHint`/`destructiveHint` annotations, then a conservative default —
**there is no static fallback table**. Classification is owned entirely by the
servers and lives in the per-agent live registry (`agent.tool_caps`). At startup
the client logs the connected tools that declared no caps, so a tool that forgot to
declare surfaces instead of silently losing its policy/approval/caching semantics.

**Status:** every classified first-party tool (across `files`, `search`, `code_intel`,
`bash`, `web`, `memory`, `todo`, `hpc`,
`proxy`) self-declares via
`@mcp.tool(**tool_caps(...))`. Pure tools (math, string, datetime ops, read-only
queries/advisors) declare nothing — they correctly carry no capability.
The expected classification is the golden oracle in `mimir/tests/_golden_caps.py`;
`test_phase_b_servers.py` AST-parses every decorator and asserts the resulting
registry reproduces it. A new server "just works": declare each tool's caps and the
client classifies it for policy/approval/caching/labels with **zero client edits**.
