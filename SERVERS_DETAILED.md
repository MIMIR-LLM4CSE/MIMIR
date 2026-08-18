# MIMIR Servers Reference

> **MIMIR docs** — [Overview](README.md) · [Architecture](ARCHITECTURE.md) · [Setup](SETUP.md) · [Policy](POLICY.md) · [Client internals](CLIENT_DETAILED.md) · [Servers](SERVERS_DETAILED.md) · [Extension](EXTENSION_DETAILED.md) · [Plugins](PLUGINS_DETAILED.md)

The authoritative per-server tool catalog — every server under `mimir/servers/` and the
tools it exposes via `@mcp.tool()`. For a one-line summary of each server, see the
[Registered Servers table in the README](README.md#registered-servers).

Servers are organized into domain-based subdirectories:

| Directory | Purpose | Servers |
|---|---|---|
| `_shared/` | Cross-group utilities | `responses.py`, `capabilities.py`, `root_paths.py`, `approved_roots.py`, `trusted_read_roots.py`, `text_tools.py`, `module_env.py`, `platform_profile_store.py`, `embed.py`, `lsp_client.py`, `shell_paths.py`, `state_paths.py`, `numerics.py` |
| `workspace/` | File & code interaction | `server_bash`, `server_localgit`, `server_files`, `server_search`, `server_code_intel` |
| `utilities/` | Stateless data helpers | `server_math`, `server_strings`, `server_datetime`, `server_symbolic_math` |
| `agent_state/` | Agent memory, planning & delegation | `server_memory`, `server_todo`, `server_spawn_agent` |
| `interaction/` | Asking the user structured questions | `server_interaction` |
| `external/` | Network & remote APIs | `server_github`, `server_web`, `server_system` |
| `hpc/` | HPC & platform profiling | `server_hpc`, `server_platform`, `server_env`, `server_benchmark` |
| `proxy/` | Scientific proxy optimization | `server_proxy` (7 op-dispatched tools: registry, refs, runs, suites, eval loop, Slurm) |
| `ml/` | Machine learning workflows | `server_finetune` |

Response contract:
- success payload: `{"status": "ok", ...}`
- error payload: `{"status": "error", "error": "...", "hint"?: "..."}`

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
`is_available` (memoized probe), `cosine_rank`, `lexical_rank`, and `embed_model_id`. The
HPC host-resolution `_resolve_host` lives here and is re-exported by the vLLM chat backend
(single source). Note: the server loads it flat via `sys.path` (`import embed`) while the
client imports `mimir.servers._shared.embed` — two module identities, so patch the correct
one in tests.

Client-side read cache:
The following tools are cache-eligible within a single query (same `(tool, args)` returns the stored result instead of re-executing):
`read_file_lines`, `read_files`, `tree_summary`, `list_directory` (plus the code-intel nav tools).
Any successful write to a path invalidates cached reads for that path.

## Registered-by-default servers

The client currently registers these servers by default:
- `workspace/server_bash.py`
- `workspace/server_localgit.py`
- `workspace/server_files.py`
- `workspace/server_search.py`
- `workspace/server_code_intel.py`
- `utilities/server_math.py`
- `utilities/server_strings.py`
- `utilities/server_datetime.py`
- `utilities/server_symbolic_math.py`
- `agent_state/server_memory.py`
- `agent_state/server_todo.py`
- `agent_state/server_spawn_agent.py`
- `interaction/server_interaction.py`
- `external/server_github.py`
- `external/server_web.py`
- `external/server_system.py`
- `hpc/server_hpc.py`
- `hpc/server_platform.py`
- `hpc/server_env.py`
- `hpc/server_benchmark.py`
- `ml/server_finetune.py`
- `proxy/server_proxy.py`

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
plan, indexed by `PLANS.md`, with a `.active` pointer). The active session is resolved via
`<STATE_DIR>/active_session` (written by `ws_server` on each switch); CLI/standalone runs
fall back to the shared legacy `todo_list.md`.

Tools:
- `todo_set_plan` — write a named plan (approach/rationale) and make it active
- `todo_write` — replace the entire checklist with new steps
- `todo_read` — return the current checklist (index, text, done)
- `todo_read_plan` — read the active (or a named) prose plan
- `todo_list_plans` — browse the session's plan history
- `todo_delete_plan` — delete one plan from the history
- `todo_update` — mark one item done/undone

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
  before the turn ends. The ordered steps are recorded only after the user approves the
  plan, at the start of the execution — a checklist written earlier tracks work nobody
  agreed to yet.
- After the plan is recorded and presented, plan mode asks the user to approve it
  (*Accept & start* / *Reject* / *Rework*, plus free-text *Other*). Accept switches to
  agent mode, where the model is asked to record the checklist before starting and then
  executes it; reject stops without executing
  anything; rework/free-text loop back to re-planning. See
  [CLIENT_DETAILED.md](CLIENT_DETAILED.md) → `plan_loop.py`.

### State directory

Every `agent_state` server (and other persistent state: platform profile, sessions)
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
standing sandbox root, one entry covering both. Both `server_files._safe` and
`server_bash._is_within_workspace` pass it as `extra_roots` alongside `approved_roots()`,
so the agent can write there without a user prompt — see `POLICY.md` → Out-of-Workspace
Access Approval for why it is separate from the user-approval sidecar, why scratch writes
are excluded from the change ledger, and how `ensure_scratch_home()` vets a world-writable
`/tmp`. Resolution never creates directories: a sandbox check runs on every call and must
not materialise state. `workspace_id()` also lives here, not in the client's constants:
both the state dir and the scratchpad name are built from it.

### Numerical invariants (`_shared/numerics.py`)

One name set with two readings. `NUMERICAL_INVARIANT_METRICS` (`l2_rel`, `linf_rel`,
`l2_abs`, `linf_abs`, `convergence_order`, `conservation_residual`, `finite`) plus
`wall_time_s` form `RESERVED_METRICS`, which the **proxy** server strips from anything the
code under optimization prints — a solver must not be able to satisfy its own acceptance
constraints. The **client's** validation observer reads the same names as *evidence*:
`observed_invariant_metrics(stdout)` scans for `key=value` lines, and a green validation
run that reports one is promoted to the `oracle` tier, because computing such a value
requires comparing against something the code does not itself define. The two readings do
not conflict — the proxy distrusts the *value*, the observer only trusts the *presence* of
the key. Hoisted here from `proxy/_lib/metrics.py`, which re-imports `RESERVED_METRICS` so
its behaviour is unchanged.

The same line grammar carries a run's verdict *about itself*: `observed_failure_verdict(stdout)`
reports a `check=fail` / `verdict=fail` line, which the client reads as a failure even on
exit 0 — a check that computes its own criteria, prints them unmet and returns 0 anyway is
otherwise indistinguishable from a clean run. Read in that direction only: a passing verdict
never rescues a red exit. Client-side only; the proxy has no use for it.

This is an auxiliary carrier, not the mechanism. Judging a run's output is the model's job
(`client/guardrails/verdict.py`), since no parser generalises across fields, plots, tables
and logs; a `check=fail` line only spares the model being asked about output that already
states its own answer, by pre-filling the verdict.


## workspace/server_files.py

Purpose: workspace-scoped file CRUD and surgical edit helpers.

### Absolute paths only

Every model-facing tool here requires an **absolute** `path`; a relative one is rejected by `_require_abs` at the tool boundary (`apply_edits` checks each sub-edit's `path` in its validation phase, so a batch is not a way around the rule).

This is the structural fix for misplacement. `write_file` previously accepted *"absolute or relative to server start directory"* — a directory the model has no way to learn — so a relative path was silently resolved against a root it had to **infer**. Asked twice to create a solver *outside* the `codes` directory, MIMIR twice wrote it inside and reported the constraint satisfied. Two prompt-level attempts to make that inference reliable both failed; requiring an absolute path removes the inference, and the destination is stated in the call itself.

The error is designed to be **self-correcting**: it names the path the relative form would have produced, so the model re-issues in one step, and it states the workspace root at the moment placement is being decided — which no static prompt section can do.

```
Relative path 'wave_solver_2d/solver.py' — file tools require an absolute path.
Inside the workspace that is /…/codes/wave_solver_2d/solver.py. If you meant
somewhere else, give that absolute path instead. The workspace root is /…/codes.
```

Deliberately unaffected: `bash_run` (requiring absolute operands in a shell would be absurd — its cwd is pinned to the workspace root instead), the read-only search/code-intel servers (a wrong relative path there returns wrong results, it does not misplace an artifact), the internal `list_files` helper (which is why the check sits at the tool boundary, not inside `_safe`), and `resolve_path_in_root` including its duplicate-root stripping — that becomes unreachable for file tools but still serves the read-only roots. The check runs *before* `_safe`, so the sandbox is unchanged: an absolute path outside the workspace is still refused.

Client-side storage is untouched: `normalize_workspace_path` already converts absolute → workspace-relative, so `dirty_written_files`, `validated_files` and every gate comparing them keep their existing form.

### The round-trip: paths are absolute wherever the model reads them

The rule above is only coherent if the model can **copy** paths rather than construct them. So discovery reports absolute paths too:

- `server_search` — `read_file_lines`, `read_files`, `list_directory` (including a `path` per entry, not just `name`) and `tree_summary` echo the *resolved* path, not the argument they were given.
- `server_code_intel` — `_out_path` (formerly `_rel`) returns absolute for every navigation result; the ctags `by_file` index is keyed the same way, so lookups stay consistent.
- the client's discovery pin renders absolute (`system_prompt._pin_path`) — display only, `execution_context` still stores workspace-relative.

Asymmetry is deliberate: these tools still **accept** relative input. Their failure mode is benign (wrong results, immediately visible) so rejecting relative would add friction without preventing anything irreversible — the justification for the write-side rule is the *silent, irreversible* misplacement of an artifact. What matters is the output side: hand back a relative path and the model must join the root itself, which is the inference the rule removed, reintroduced exactly where it is most likely to copy without thinking. The discovery pin makes this vivid — it says "use these paths directly", and for a while it said that above paths the next call would reject.

`tree_summary`'s root line is absolute for the same reason as `repo_baseline`'s: a bare basename root reads as a subdirectory of itself.

Tools:
- `write_file`
- `append_file`
- `delete_file`
- `replace_in_file`
- `replace_all_in_file`
- `replace_lines`
- `apply_edits`

Note: directory listing is owned by the search server's `list_directory`; the
file server keeps an internal `list_files` helper only to back the `files://list`
resource (not exposed as a tool).

## workspace/server_search.py

Purpose: file reads, partial reads, directory listing, and cached tree summaries.
Text search (`grep`/`rg`) is done via the bash server — classified read-only, so it
needs no approval and feeds the same discovery signals a dedicated tool would.

Tools:
- `read_file_lines` (pass `end_line=0` to read the whole file)
- `read_files`
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
  scan.
- `symbol_outline(path)` — the ordered symbol tree of one file
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

## external/server_github.py

Purpose: read-only GitHub repository inspection.

Tools:
- `github_repo_info`
- `github_list_branches`
- `github_list_issues`
- `github_get_file`
- `github_search_repositories`

## hpc/server_hpc.py

Purpose: Slurm inspection and job-submission helpers. (Environment Modules /
Lmod are handled directly via the bash server's allowlisted `module` command —
discovery + load — not here.)

Tools:
- `slurm_partitions`
- `slurm_nodes`
- `slurm_queue`
- `salloc_build_command`
- `salloc_submit`
- `sbatch_submit` — non-blocking Slurm **batch** submission (unlike synchronous `salloc_submit`): returns a `job_id` immediately plus a `background_job` descriptor (`BACKGROUNDABLE`), so the run is watched off the critical path and auto-resumes the agent on completion. Writes the script/log under `~/.cache/mimir_hpc/jobs/<ts>/` (env `MIMIR_HPC_JOBS_DIR`).
- `slurm_job_status(job_id)` — normalized per-job state (running|pending|done|crashed|unknown) via squeue (active) + sacct (terminal); the poll target the background-job watcher uses.

> `salloc_submit` / `sbatch_submit` declare the `CLUSTER_SUBMIT` capability (shared with `ft_run_slurm` and `proxy_slurm`). The client's pre-submission guard holds the first such call each query until something has been validated locally, then lets the retry through (see `POLICY.md` → Cluster-Submission Guard). `sbatch_submit`, `proxy_eval(op='run')`, and `proxy_slurm(op='eval')` additionally declare `BACKGROUNDABLE` (see the background-jobs note under `proxy_eval`).

## hpc/server_platform.py

Purpose: platform profiling and architecture-aware recommendations.

Tools:
- `platform_probe` — collect and return a full platform profile (CPU, GPU, memory, Slurm, modules, toolchains). **Stateless** — built on demand and returned; nothing is persisted
- `platform_get_profile` — build and return a fresh profile for the current host plus a live `sinfo` partition/node table so the agent knows what Slurm resources are available without a separate command. **Stateless** — always built fresh for the current host (so it can never serve another node's hardware), no cache, no `refresh_if_missing` arg. (The client also keeps its own self-contained probe for the always-on foundational context; these server tools are for deeper, on-demand queries.)
- `platform_compiler_recommendations` — recommend compiler flags based on detected SIMD and GPU
- `platform_scientific_plan` — return a workload strategy (libraries, parallelism model, tuning hints) for a given problem type and scale
- `platform_code_advisor` — analyse a code excerpt and return architecture-specific optimisation advice

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
- `env_create(name, kind="venv", packages=None, python_executable="python3")` — create a new venv/conda environment
- `env_delete(target, kind="venv")` — delete an environment created by `env_create`

> A bare `python`/`python3` resolves to the server's own interpreter; an absolute path
> must exist and be executable; anything else resolves via `PATH`. These tools declare
> registry-driven approval **scope** (package-set for the pip tools, env basename for
> create/delete) so an `always` grant narrows to those packages / that environment
> rather than the whole tool (see `POLICY.md` → Sensitive Tool Approval).

## hpc/server_benchmark.py

Purpose: lightweight benchmark helpers and summary generation.

Tools:
- `benchmark_python_compute`
- `benchmark_memory_copy`
- `benchmark_numpy_matmul`
- `benchmark_summary`

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

Purpose: controlled workspace shell access with command validation. Read-mostly, but
deliberately loosened so HPC/CUDA benchmark steps work end-to-end while keeping the
allowlist + no-substitution model load-bearing. This is also the **text-search path**
(`grep`/`rg`) — there is no dedicated grep tool; a leading `grep`/`rg` classifies
read-only, so it needs no approval and feeds the same discovery signals a tool would.

Tools:
- `bash_allowed_commands`
- `bash_run`

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

### The 74 allowed commands, by category

One taxonomy, declared once in `servers/_shared/shell_paths.py` and read by both ends:
the server builds `_ALLOWED_COMMANDS` from these groups, the client maps each to a
capability `Kind`, and the category is what decides approval and plan-mode availability.
`bash_allowed_commands()` returns it per command, so the agent can see the gating without
probing for it. A parity test (`test_every_classified_command_is_one_the_server_can_run`)
fails if a group names a command the server will not run, if an allowed command has no
category, or if a category disagrees with how a call to it is actually gated.

| Category | Commands | Plan mode | Approval |
|---|---|---|---|
| `neutral` (10) | `pwd` `echo` `which` `basename` `dirname` `realpath` `df` `true` `false` `:` | ✅ | ❌ |
| `read` (18) | `cat` `head` `tail` `nl` `sed`◆ `wc` `cut` `sort`◆ `uniq` `comm` `tr` `fold` `column` `cksum` `md5sum` `sha256sum` `stat` `file` | ✅ | ❌ |
| `search` (2) | `grep` `rg` | ✅ | ❌ |
| `inspect` (3) | `ls` `find` `du` | ✅ | ❌ |
| `chdir` (1) | `cd` | ✅ | ❌ |
| `env` (7) ◆ | `module` `pip` `pip3` `conda` `conda3` `mamba` `mamba3` | query only | on mutation |
| `write` (4) | `mv` `cp` `mkdir` `chmod` | ⛔ | ✅ |
| `exec` (29) | `gcc` `g++` `gfortran` `nvcc` `javac` `java` `node` `python` `python3` · `make` `cmake` `ctest` · `pytest` `ruff` `mypy` `pyflakes` `black` · `pdflatex` `latex` `xelatex` `lualatex` `pdftex` `tex` `bibtex` `biber` `makeindex` `latexmk` `dvips` `dvipdf` | ⛔ | ✅ |

`git` (use `localgit`) and deletion (`rm`/`rmdir`) remain intentionally excluded. A
Makefile recipe is effectively arbitrary execution, but no shell-injection vector.
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
| `module load|add` vs `avail|list|show|…` | mutation vs query |
| pip/conda `list show freeze check inspect info help` vs anything else | query vs mutation (the network queries `search`/`index`/`download` count as mutation: a plan is drafted without reaching the network) |
| an unknown `module`/pip/conda sub-command | assumed to mutate (the server refuses it anyway) |
| **chain rule** | one non-read-only segment makes the *whole* call non-read-only |

### Refusals that are not re-categorisations

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
  whole family was denied on the `-exec` token alone, and the rejection hint suggested
  chaining via `xargs` — which is permanently banned in `_SHELL_RUNNERS`, so read-only
  fan-out had no working spelling at all.
- TeX shell-escape/`write18` in every spelling (plus `_safe_env` pinning
  `shell_escape=f`, `openout_any=p`, `openin_any=p`).
- env managers: `uninstall`/`remove`/`clean`/`run`/`config`, `conda env` outside
  {`create`,`list`,`export`,`update`}, any unrecognised sub-command; `module` outside
  discovery+load.
- any path operand, write-flag value or redirection target outside the workspace and not
  approved; a `$VAR` in path or redirection-target position; a heredoc; backgrounding;
  substitution; a subshell; every shell interpreter and generic runner.

Approval nuances: `bash_run` is `SENSITIVE` + `NON_BATCH`, so it is never queued for
end-of-turn batch review; an "always" grant is scoped by `_scope_command_prefix` to the
**first two tokens** (`pip install`, `gcc solver.c`) for the session; the headless runner
auto-approves everything.
- **Shell interpreters and generic runners** (`bash`, `sh`, `eval`, `source`, `env`,
  `xargs`, `sudo`) are permanently excluded — they nest a command the validator never
  sees. They get their own rejection message saying exactly that, so the agent stops
  looking for a wrapper instead of trying one spelling after another.
- A rejection **inlines the full allowlist** in an `allowed_commands` field rather than
  pointing at `bash_allowed_commands`: the payload is the reply to a *shell* call, so
  any "call X to find out" pointer reads as one more shell command to try — which is
  how an agent ends up running the name of an MCP tool and failing again on it.
- **No-ops** (`true`, `false`, `:`) are allowlisted so the capability probe
  `which pdflatex 2>/dev/null || true` is expressible; without them the chain is
  rejected on its last segment and the agent has no way to ask "is X available?".
- **A no-match is a result, not an error**: `grep`/`rg` exit 1 (no match) and `which`
  exit 1 (absent) with empty stdout are returned as `status: ok` with `matches: 0`.
  Reported as failures they read as broken commands and get re-run verbatim. A real
  failure (exit 2) stays an error. Only the *last* segment decides, since that is what
  bash reports.
- **TeX is sandboxed twice**: every shell-escape/`write18` flag is denylisted at
  validation, and `_safe_env` pins kpathsea's `shell_escape=f` / `openout_any=p` /
  `openin_any=p` so `\write18` stays off even if the site's `texmf.cnf` enables it.
- **Command chaining is allowed** (`;`, `&&`, `||`, `|`) — each segment is tokenized and
  validated against the allowlist independently, so the allowlist stays load-bearing.
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
  allowlist, the flag denylists, and confinement.
- **`module` (HPC Lmod) is supported**, scoped to discovery + load only (`avail`,
  `spider`, `list`, `show`, `load` — no `unload`/`swap`/`purge`/`reset` or
  `save`/`restore`). Since `module` is a shell *function*, the server sources Lmod init
  in the wrapper around the validated command, so `module load cuda && nvcc ...` works
  within one call; module args are gated by a strict regex and a curated `MODULE*`/`LMOD_*`
  env is passed through.
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
  only and its args are character sets.
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
  holds `EXEC_COMMANDS`, `PATH_SENSITIVE_COMMANDS`, `normalize_path_arg`,
  `segment_path_operands` and `cd_destination`, and is imported by this guard *and* by
  the client gate. This guard confines what that gate prompts for; two copies would
  fail silently in both directions — a path the gate misses cannot be granted, a path
  this guard misses is never gated. `_ALLOWED_COMMANDS` is built from the same
  `EXEC_COMMANDS` group, so a command added to the toolchain cannot be forgotten in the
  confinement list (asserted in `test_server_contracts`).
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

## workspace/server_localgit.py

Purpose: read-only inspection of the **local working tree** (the checkout the agent
is operating on), complementing `external/server_github.py` (remote GitHub API).

Tools:
- `git(op, ...)` — single read-only dispatch; set `op` to one of: `status`, `log`
  (uses `max_count`, `path`), `diff` (uses `path`, `staged`), `show` (uses `ref`),
  `branches`, `blame` (uses `path`, `start_line`, `end_line`), `grep` (uses `pattern`, `path`).

Safety model:
- All tools are read-only, so none are approval-gated.
- A single private runner executes `git` with `shell=False`, `cwd` pinned to the
  workspace root, and a minimal environment; output is capped.
- Only an allowlist of read-only subcommands is accepted (`status`, `log`, `show`,
  `diff`, `branch`, `rev-parse`, `remote`, `ls-files`, `grep`, `describe`, `tag`,
  `blame`).
- The `-c` and `--exec-path` global options are refused outright (they are
  config-injection code-execution vectors).

## agent_state/server_spawn_agent.py

Purpose: delegate a self-contained sub-task to a **fresh** `MimirAgent` that runs to
completion and returns its answer, so the orchestrator can fan work out instead of
carrying every intermediate step in its own context.

Tools:
- `spawn_agent(task, context="", model="", max_steps=30, readonly=False)` — spin up a
  child agent, run it, return its answer. `context` is prepended to the task prompt;
  `model` defaults to `MIMIR_DEFAULT_MODEL` then the parent's active model; `readonly=True`
  connects only read/search servers, so the sub-agent can neither write files nor run code.

Execution model:
- The tool is synchronous from the model's point of view but runs the sub-agent in a
  dedicated thread with its own event loop, so it is safe to call from inside the parent's
  running asyncio loop. Several `spawn_agent` calls emitted in one model step are
  dispatched **concurrently** by the parent's `asyncio.gather` in `dispatch.py`.
- Hard cap of 600 s per sub-agent; a crash or timeout returns `status="error"` with the
  partial answer rather than an `ok` payload, so the orchestrator branches on `status`
  instead of string-matching the answer.
- Success payload: `{"status": "ok", "answer", "completed", "files_written"}` —
  `completed=False` means the sub-agent ran out of steps or reported the task incomplete
  (the answer is still informative), and `files_written` lets a parent coordinating
  concurrent sub-agents detect overlapping edits.
- Sub-agent stdout is prefixed with the task name so its activity stays identifiable in
  the UI. The capability registry is strictly per-agent, which is what makes concurrent
  sub-agents on different server subsets safe.

## interaction/server_interaction.py

Purpose: let the agent pause mid-run and ask the **user** a clarifying question with
selectable choices, then resume with the answer.

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
| "Is the agent *allowed* to do this risky/irreversible action?" | **Approval system** (`mimir/client/guardrails/policy/` + the WS `_approval_shim`) — risk/scope classification, yes / no / **always**, pre-write diffs, batch revert | bash commands, file writes/deletes, `ft_run`, proxy run/eval tools |
| "I don't know what the user *wants*; the answer changes what I do" | **Elicitation** (`ask_user_question` / `ctx.session.elicit_form`) — a structured form prompt | architectural forks ("Postgres or SQLite?"), ambiguous requirements, disambiguating an under-specified argument |

Guidance:
- **Do not** route command/write confirmation (e.g. bash) through elicitation — it has no
  concept of risk, scope, "always", or rollback, so it would be a downgrade. Keep risky
  actions on the approval system.
- Use elicitation only to **gather missing intent**, not to grant permission.
- A single tool action should not trigger *both* an approval prompt and an elicitation for the
  same decision — pick the one that matches the question being asked.

## ml/server_finetune.py

Purpose: manage LoRA fine-tuning runs for HuggingFace causal language models,
with built-in support for agent-driven iterative optimization.

All state is stored under `~/.cache/ft_llm/`: a shared `config.json` and one
timestamped run directory per launch (containing `config.json`, `run.log`,
`pid` or `slurm_job_id`, `metrics.json`, and `model/`). A `runs/active` symlink
tracks the most-recently-launched run. The heavy training work is delegated to
the companion `_ft_runner.py` script (same directory).

Two execution backends:
- **local** (`ft_run`): launches `_ft_runner.py` as a detached `Popen` subprocess.
- **Slurm** (`ft_run_slurm`): generates an `sbatch` script and submits it;
  requires Slurm on the host.

Both backends write to the same `run.log`, so `ft_log_read()`, `ft_metrics_parse()`,
and `ft_runs_list()` are backend-agnostic. `ft_status()` and `ft_stop()` detect
the backend from the presence of a `pid` or `slurm_job_id` file.

Safety model:
- Read-only tools have no side effects.
- `ft_config_set`, `ft_run`, `ft_run_slurm`, `ft_stop`, and `ft_runner_promote` are approval-gated. `ft_run_slurm` is declared **irreversible** (it spends allocation hours); the rest are `recoverable`.
- Both run tools are non-blocking and return immediately after launch/submission.
- Only one run may be active at a time; both tools reject if a live PID / active
  Slurm job is detected.

Session reset:
- At server startup (i.e. at the start of every new agent session),
  `_ft_runner.py` is automatically restored from the canonical copy in
  `~/.cache/ft_llm/_ft_runner.canonical.py`.
- Any modifications made by the agent during the previous session are discarded.
- The canonical lives **outside** `MCP_FILES_ROOT` so server_files write tools
  cannot overwrite it.
- On first launch, the canonical is seeded from the package-shipped
  `_ft_runner.canonical.py` (in `mimir/servers/`) and then stored in
  `~/.cache/ft_llm/` for all subsequent sessions.
- Use `ft_runner_promote` to intentionally update the canonical after the
  agent has verified improvements across multiple runs.

Agent optimization loop:
- The `ft_run` and `ft_metrics_parse` docstrings contain explicit guidance
  instructing the agent to use the observe→modify→re-run loop.
- After calling `ft_metrics_parse()`, if results do not meet user constraints,
  the agent may: (a) call `ft_config_set` for hyperparameter changes, or
  (b) read `_ft_runner.py` via `read_file_lines`/`search` tools and edit it freely
  via `replace_in_file` to change any aspect of the training implementation
  (architecture, optimizer, data preprocessing, dtype, etc.), then re-run.
- The session reset guarantees that these modifications never leak to the
  next session.

Tools:
- `ft_config_get` — return current configuration (file values merged with defaults)
- `ft_config_set` — write `~/.cache/ft_llm/config.json` (sensitive)
- `ft_data_inspect` — check training/validation file existence, line count, 3-line sample
- `ft_run` — launch `_ft_runner.py` locally as a detached subprocess (sensitive, confirm-gated)
- `ft_run_slurm` — generate an sbatch script and submit via Slurm (sensitive, confirm-gated)
- `ft_status` — PID / squeue alive-check, elapsed time, last 3 log lines
- `ft_stop` — SIGTERM (local) or `scancel` (Slurm), confirm-gated
- `ft_runner_promote` — persist current `_ft_runner.py` as the new canonical (sensitive, confirm-gated)
- `ft_log_read` — tail the last N lines of `run.log` from the active/last run
- `ft_metrics_parse` — extract per-step loss/epoch/lr, per-epoch memory snapshots,
  and a final structured summary (peak_vram_mb, cpu_ram_mb, throughput_sps,
  train_runtime_s, precision, trainable_params, total_params, train_loss, eval_loss)
- `ft_runs_list` — list all run directories, state, and final metrics
- `ft_runs_diff` — compare config and metrics between two runs (config diff + numeric metric deltas)

Companion script:
- `_ft_runner.py` — standalone script (no MCP); reads `--run-dir/<config.json>`,
  runs the full HuggingFace `Trainer` + `peft` LoRA loop, writes `metrics.json`
  and saves adapter weights to `model/`. Invoked by both `ft_run` and `ft_run_slurm`.
  The agent may edit this file freely between runs to experiment with the training
  implementation.
- `_ft_runner.canonical.py` (in `mimir/servers/ml/`) — seed copy shipped with the package.
  Copied to `~/.cache/ft_llm/_ft_runner.canonical.py` on first server startup.
- `~/.cache/ft_llm/_ft_runner.canonical.py` — live canonical; restored over
  `_ft_runner.py` at every session start. Updated only by `ft_runner_promote`.

Structured log lines emitted by `_ft_runner.py` (parseable by `ft_metrics_parse`):
```
[ft_runner] epoch=<n> peak_vram_mb=<n> cpu_ram_mb=<n>
[ft_runner] summary peak_vram_mb=<n> cpu_ram_mb=<n> throughput_sps=<n>
    train_runtime_s=<n> trainable_params=<n> total_params=<n>
    precision=<str> train_loss=<n> eval_loss=<n>
```

Required Python packages for `_ft_runner.py`: `torch`, `transformers`, `peft`, `datasets`.
Optional: `psutil` (CPU RAM tracking), `bitsandbytes` (required for `precision=int8`;
if missing and `precision=int8` is set, the runner exits immediately with a clear error message).

Configuration fields (set via `ft_config_set`):

| Field | Default | Description |
|---|---|---|
| `model_id` | `distilgpt2` | HuggingFace model identifier |
| `train_data` | `""` | Absolute path to training `.txt` file |
| `val_data` | `""` | Absolute path to validation `.txt` file (optional) |
| `lora_r` | `8` | LoRA rank |
| `lora_alpha` | `16` | LoRA alpha scaling factor |
| `target_modules` | `["c_attn", "c_proj"]` | Modules to inject LoRA into |
| `lora_dropout` | `0.05` | Dropout probability inside LoRA layers |
| `batch_size` | `8` | Per-device training batch size |
| `lr` | `2e-4` | Learning rate |
| `epochs` | `3` | Number of training epochs |
| `max_length` | `256` | Max token sequence length |
| `output_dir` | `<run_dir>/model` | Where to save adapter checkpoints |
| `precision` | `auto` | Floating-point precision: `auto`\|`fp16`\|`bf16`\|`fp32`\|`int8` |
| `python_executable` | `""` | Absolute path to Python interpreter for `_ft_runner.py`. Defaults to the server's own interpreter. Useful when Slurm compute nodes use a different environment. |

---

## proxy/_lib/ — shared helper library

Helper package (not an MCP server) imported by `proxy/server_proxy.py`, the `proxy/_ops/` modules, and `_proxy_runner.py`.  Split of the former `_shared_proxy.py` god-module into cohesive units; dependency direction is `store ← procs/metrics/command/report ← execute/ratchet`.

| module | contents |
|---|---|
| `store.py` | Storage root `_CACHE_DIR` (env-overridable via `MIMIR_PROXY_BENCH_DIR`, default `~/.cache/proxy_bench`); every path derives from it **at call time** via functions (`registry_path()`, `runs_dir()`, `refs_dir()`, `suites_dir()`, `scaffolds_dir()`, `opt_runs_dir()`, …), so a hermetic test repoints one attribute. Generic atomic IO (`_read_json`, `_write_json_atomic`, `_write_text_atomic`, `_atomic_symlink`); registry I/O (`_load_registry` with corrupt-JSON backup, `_save_registry`, `_registry_lock()` — exclusive `fcntl.flock`, thread-local re-entrancy depth); suite persistence (`_load_suite`/`_save_suite`/`_latest_suite_results`); reference layout (`_ref_dir`, `_load_ref_metrics`, `_ref_output_path`); optimization-session layout (`_opt_config_file`, `_opt_session_runs_dir`, `_opt_ledger_file`, `_opt_best_file`, `_resolve_proxy_name`, `_write_active_session`). |
| `procs.py` | Process/run state: pid + `/proc` starttime bookkeeping (PID-recycling and zombie detection in `_is_running`), `_squeue_state`, `_run_state`; log access (`_log_path`, `_read_log` capped at `_MAX_LOG`); run lifecycle — `_new_run_dir`, `_write_run_config`, `_launch_detached`, `_submit_sbatch`, `_cancel_run` (`scancel` else SIGTERM → SIGKILL), `_validate_slurm_args`; active-run symlinks. |
| `metrics.py` | `_parse_metrics_block` (block-delimited; strict fallback scan), `_RESERVED_METRICS` + `_strip_reserved_metrics` (server-side invariants the proxy cannot forge), `_normalize_time_metrics` (server-measured `wall_time_s` + `time_s` plausibility guard), field I/O + `_field_norms`, invariants (`_finite_check`, `_conservation_residual`, `_convergence_order`), `_evaluate_requirements`. |
| `command.py` | `_render_param_file`, `_expand_cmd_template` (single template-expansion path for local + sbatch), `_build_run_cmd`, `_sbatch_header`, `_build_sbatch` (injection-safe `repr()` postrun; captures solver wall time + exit code for the post-run guard). |
| `report.py` | Roofline (`_resolve_roofline` with platform-profile fallback, `_compute_roofline`, `_arch_label`), `_row_with_extra`, `_diff_run_dirs`. |
| `execute.py` | The three run paths every launch funnels through: `_run_benchmark_case` (synchronous case; reserved-metrics purge, time_s guard, non-zero-exit gate), `_seal_reference` (immutable reference store), `_post_run_finalize` (settles detached local/Slurm runs with the same invariants). |
| `ratchet.py` | Optimization ratchet: `_ratchet_verdict` / `_is_improvement` / `_run_primary_value` / `_select_best_case` / `_select_best_run`, best-so-far persistence (`_load_best`/`_save_best` incl. `wall_value` for the timing audit), `_append_ledger`. |

`proxy/_shared_proxy.py` remains as a ~20-line **legacy facade**: generated `postrun.py` scripts in old run directories (possibly still queued on Slurm) import `_post_run_finalize` from it by name, so the module must keep existing; new code imports from `_lib.*` directly.

---

## proxy/server_proxy.py

Purpose: registration, benchmarking, execution, and iterative optimization of proxy scientific-computing codes — behind **seven op-dispatched tools**.

Each tool takes an `op` parameter selecting the operation; the docstring carries the full ops table with per-op required args. Ops are grouped strictly by capability class, because capabilities are declared per tool: read-only tools carry no caps, mutating tools are `sensitive` + `PLAN_BLOCKED` + `non_batch` with a `confirm=True` gate on every op, and Slurm submission is isolated on its own `CLUSTER_SUBMIT` tool. Unknown ops and missing per-op args return `err()` with a corrective hint. Eval-loop responses include a `next_step` field naming the exact next call.

Tool bodies live in `proxy/_ops/` as plain functions (no `@mcp.tool` decorators there — the AST-based capability parity gate only sees the seven declarations in `server_proxy.py`):

| `_ops` module | Contents |
|---|---|
| `registry.py` | proxy list/inspect (registration + readme + recent runs), register/update/unregister; the ~12 descriptive registration fields are carried by one `metadata` dict |
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
- `proxy_manage(op, …)` — ops: `register`, `update`, `unregister`, `suite_define`, `suite_update`, `suite_delete`, `scaffold` (generate reference + test harness files for a source component)
- `proxy_exec(op, …)` — ops: `run` (non-blocking local), `reference` (blocking seal), `suite` (blocking run of all cases × sweeps), `benchmark_create` (reference + one-case suite in one call), `cancel`
- `proxy_eval(op, …)` — a monotone optimization ratchet. Ops: `init` (snapshots the source as the canonical baseline, never overwritten; takes `primary_metric`/`primary_goal`/`min_improvement`/`max_stall` + optional `convergence`), `configure`, `run` (non-blocking; errors if one is active), `stop`, `reset` (restore canonical), `reset_to_best` (restore the best *accepted* run — undo a regression without discarding progress), `end` (close the session: clears `active_session`, keeps source/snapshots/ledger; lifts the direct-execution guard — refuses while a run is active). While a session is active, the client blocks running the proxy source/executable directly through the exec surface (`bash_run`) — see `POLICY.md` → Proxy Direct-Execution Guard. `results` returns a `verdict`: **accept** (feasible + improves the objective → new best-so-far), **reject** (regression → revert with `reset_to_best`), or **converged** (feasible and no improvement for `max_stall` runs). Feasibility = all `requirements` pass, including numerical invariants (`l2_rel`/`linf_rel`/`finite`/`conservation_residual`/`convergence_order`) which the server computes and exposes as ordinary metrics. These names are **reserved**: values the proxy prints for them are stripped before evaluation (`reserved_metrics_ignored` records the drops), so the code under optimization cannot satisfy its own acceptance constraints; `init`/`configure` reject reference-dependent requirements when no sealed reference (or registry `conserved_metric`) backs them. Per-session state under `opt_runs/<proxy>/`: `best.json` + `best_source<ext>` (snapshotted from the source *as launched*, `source_at_launch<ext>` in the run dir), append-only `ledger.jsonl`, per-run frozen `ratchet.json`. The ratchet is settled **by the runner at run completion** (flock-serialized, idempotent) — ledger/best are the source of truth even if the agent never calls `results`; `results` replays the frozen verdict. `op='run'` accepts `background=True` (cap `BACKGROUNDABLE`): the result carries a `background_job` descriptor (status/summary ops) so the WebSocket worker watches the detached run off the critical path, notifies the UI (`job_complete` event), and auto-resumes the agent with the verdict/best — the agent ends its turn instead of polling. Degrades safely in the CLI (no watcher → normal polling).

Tools (cluster, `CLUSTER_SUBMIT`):
- `proxy_slurm(op, partition, …)` — ops: `run` (single job), `suite` (one job per case × sweep), `eval` (optimization run); carries a `risk_note` shown in the approval prompt

Param file system:
- Each registration optionally includes a `param_file_template` (inline text), a `param_file_path` (path relative to the executable), and a `param_file_format` (`text`/`json`/`yaml`/`fortran_namelist`/`ini`).
- The template supports `{executable}`, `{output_file}`, `{param_file}`, and `{extra_params}` placeholders inside `run_cmd_template`; any additional `{key}` token is filled from `param_overrides`.
- Per-run `param_overrides` replace `{key}` tokens in the rendered template AND patch `key = value` / `key=value` lines via regex, so they work with Fortran namelists, INI files, and free-text parameter files without format-specific parsing.

Proxy output protocol:
- The proxy writes a `PROXY_METRICS_BEGIN … PROXY_METRICS_END` block to stdout with `key=value` pairs.
- A special `output_file=<path>` key triggers the server to copy the named file into the run directory and include it in output comparisons.
- Optional roofline keys: `flops` (total FP operations), `bytes_moved` (bytes transferred), `time_s` (wall time). Roofline peaks come from the registration `metadata` (`peak_gflops_per_s`, `peak_bandwidth_gbytes_per_s`), falling back to the latest `benchmark_summary` in the platform profile.
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

Storage layout (all under `~/.cache/proxy_bench/` — unchanged from the two-server era, no migration needed):
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
`bash`, `web`, `memory`, `todo`, `benchmark`, `hpc`,
`finetune`, `proxy`) self-declares via
`@mcp.tool(**tool_caps(...))`. Pure tools (math, string, datetime ops, read-only
queries/advisors, `bash_allowed_commands`) declare nothing — they correctly carry no capability.
The expected classification is the golden oracle in `mimir/tests/_golden_caps.py`;
`test_phase_b_servers.py` AST-parses every decorator and asserts the resulting
registry reproduces it. A new server "just works": declare each tool's caps and the
client classifies it for policy/approval/caching/labels with **zero client edits**.
