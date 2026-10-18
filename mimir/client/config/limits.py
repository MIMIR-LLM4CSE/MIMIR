"""The numbers that bound what the agent does: timeouts, budgets, caps and ladders.

Central so the cadence of the whole loop can be read and tuned in one place, and so
both front-ends behave the same.
"""
from __future__ import annotations

# ── Agent runtime tuning ───────────────────────────────────────────────────────
# Central home for the agent-loop behavioural knobs. Tune here, not at the call
# site, so behaviour stays consistent across the CLI and WebSocket front-ends.

# Hard wall for a single MCP tool call (seconds). Keeps a hung bash_run / web_get
# from blocking the agent loop indefinitely.
TOOL_CALL_TIMEOUT_SECS: int = 120

# A tool may declare its own wall (``timeout_secs`` in its descriptor) when the global
# default, calibrated on a grep, cannot bound what it actually does — a sub-agent run,
# a package install, a build. The client keeps the last word via this ceiling, which is
# set above the longest inner cap a first-party server enforces (the env server's 900s
# pip budget) so that cap fires first and the caller gets the server's own diagnosis
# rather than a bare "timed out".
TOOL_CALL_TIMEOUT_MAX_SECS: int = 1200

# How long a clarification question waits before the model goes on with its own
# recommendation. Human waits are outside the tool budget, so this is their only wall.
# Not applied to approvals (expiring into "yes" would authorise unseen commands) nor to
# plan approvals (already resolved to "execute nothing").
USER_QUESTION_TIMEOUT_SECS: int = 300

# Separate budget for the post-write auto-validation ladder (syntax/imports/lint/
# typecheck/format + completeness + cross-file grep). It runs AFTER the write has
# already hit disk, so it is purely advisory: exceeding this budget drops the
# validation annotation but must never turn a successful edit into a failed tool
# row. Kept independent of TOOL_CALL_TIMEOUT_SECS so a slow validator (a heavy
# import subprocess, mypy, a repo-wide grep) cannot cancel the completed write.
AUTO_VALIDATION_TIMEOUT_SECS: int = 60

# Char budget for tool-result messages kept in the live context window. When the
# total chars of all tool messages exceed this, the oldest unprotected ones are
# evicted. Char-budget is safer than a count limit because tool results vary
# wildly in size. 600 000 chars ≈ 150 K tokens on most models.
TOOL_HISTORY_CHAR_BUDGET: int = 600_000

# Trigger intra-query compaction when total message chars exceed this.
# 480 000 chars ≈ 120 K tokens — a comfortable margin below the 200 K window
# before the hard tool-history budget kicks in.
INTRA_QUERY_COMPACT_CHARS: int = 480_000

# There is no step ceiling. A query runs for as many model ↔ tool round-trips as
# the work takes: it ends when the model delivers an answer, when a guard stops it
# (empty turns, validation budget), or when the user interrupts. A ceiling only ever
# cut runs off mid-work and then asked the user to re-authorise the same work in
# blocks; callers that do want a bound (the runner, sub-agents, tests) pass their own
# ``max_steps``, and 0 means unbounded.

# Consecutive turns that come back with neither prose nor a tool call before the
# loop gives up retrying and ends the run (see agent_loop's empty-turn guard).
AGENT_EMPTY_TURN_RETRIES: int = 3

# Maximum consecutive validation failures per file before edits are blocked and
# the session auto-escapes to conclude (incomplete).
VALIDATION_RETRY_BUDGET: int = 5

# The per-step tool list is not capped: every advertised tool is sent, whatever the
# model. A cap traded a constant, cacheable prompt prefix for a request-dependent one,
# and it decided what to drop from the request's wording — a guess whose cost, when
# wrong, is a capability the run then cannot use.


# Transient backend-call resilience. A single model call (one step of a run that
# has no step ceiling) should survive a flaky connection / 5xx / rate-limit rather
# than discarding the whole query. Total attempts = 1 + LLM_RETRY_ATTEMPTS.
# Backoff is exponential with jitter: LLM_RETRY_BASE_DELAY_SECS * 2**(n-1).
# ~2 min in total: gateway outages (503 "no endpoint candidates", ext_proc resets)
# outlast a few seconds, and losing a long run costs far more than waiting.
LLM_RETRY_ATTEMPTS: int = 8
LLM_RETRY_BASE_DELAY_SECS: float = 1.0
LLM_RETRY_MAX_DELAY_SECS: float = 30.0


# ── Nudge frequency caps & thresholds ──────────────────────────────────────────
# Central home for the "magic numbers" that govern how often a workflow nudge may fire
# and the situational gates that trigger it. Here rather than inline at each guard, so the
# whole nudge cadence can be read and tuned in one place instead of hunting through
# the guardrails.

# Per-category max fires per query (the guarding branch checks
# ``nudge_counts[cat] < NUDGE_MAX_<CAT>``). A cap of 1 means "one reminder only".
NUDGE_MAX_VALIDATION: int = 2
NUDGE_MAX_DENIAL: int = 2
NUDGE_MAX_ERROR_RECOVERY: int = 2
# One fire per rung of the stuck-repair ladder (see the section below).
NUDGE_MAX_STUCK_REPAIR: int = 2
NUDGE_MAX_UNFINISHED_PLAN: int = 1
# One shared budget for the whole advisory axis — "run the existing test"
# (regression) and "nothing has been exercised" (unexercised) are two phrasings of one
# question, and running is a recommendation, not a requirement: the toolchain or the
# data may simply not be here. One ask per query, and an answer of "not runnable here"
# closes it for good (``exercise_advice_closed``).
NUDGE_MAX_EXERCISE: int = 1
# The ``nudge_counts`` key those two rows share.
EXERCISE_BUDGET: str = "exercise"
NUDGE_MAX_ENV_RESOLUTION: int = 1
NUDGE_MAX_ENV_CLEANUP: int = 1
# A run nobody is watching. One reminder, like the cleanup it resembles: the fact is
# checkable and the remedy is one call, so a model that does not act on the first is
# not going to act on the third.
NUDGE_MAX_UNWATCHED_RUN: int = 1
NUDGE_MAX_DOC: int = 1
NUDGE_MAX_STATE: int = 1
NUDGE_MAX_BLAST_RADIUS: int = 1

# Application-pack (extension) nudges: max fires per query per registered rule.
CUSTOM_NUDGE_MAX_PER_QUERY: int = 3

# ── Stuck-repair escalation ladder ─────────────────────────────────────────────
# Counted per run command (``runs[cmd]["failures"]``, carried by ``record_run``):
# advice, then a constraint, then at VALIDATION_RETRY_BUDGET the workflow concludes.
# ADVISE < CONSTRAIN < VALIDATION_RETRY_BUDGET.
STUCK_REPAIR_ADVISE_AFTER: int = 2     # nth failure of one command that earns "look wider"
STUCK_REPAIR_CONSTRAIN_AFTER: int = 4  # nth failure that forbids re-editing the same target

# ── Denial escalation ladder ───────────────────────────────────────────────────
# A refusal means, in order: find another route, drop the step, stop and hand back.
# Counted per query and per approval scope, so refusing `pip install X` also escalates
# `pip install Y`.
DENIAL_SCOPE_DROP_AFTER: int = 2      # nth refusal of one scope that rules out "another route"
DENIAL_SCOPE_HANDBACK_AFTER: int = 3  # nth refusal of one scope that forces the hand back
DENIAL_QUERY_HANDBACK_TOTAL: int = 4  # total refusals in a query that force the hand back

# Consecutive no-op ("I'm done") turns that may still be nudged. A second bare turn in a
# row is ignoring the reminder; any tool call resets the count.
NUDGE_MAX_CONSECUTIVE_NOOP: int = 1

# How many skills the model may load into one conversation (load_skill). A methodology
# that stacks five applies none of them, and the real failure mode of on-demand loading
# is a model that keeps reaching for one more playbook instead of doing the work. The
# user's own /<name> counts towards it: it put a body in the context the same way.
SKILL_PULL_MAX: int = 3

# Idle-step gate: how many steps the agent must have paused editing before the
# nudge fires, so an in-progress refactor is not interrupted. Validation has no such
# gate — its nudge carries a finding the loop already made, not a request to pace.
NUDGE_STATE_IDLE_STEPS: int = 3

# Multi-file thresholds: at least this many touched/planned files before the
# todo-tracking nudge fires.
NUDGE_MAX_CREATION: int = 1
NUDGE_MAX_TODO: int = 1
# The mid-loop "you just wrote X — tick the step off" reminder, which fires from
# post_dispatch_inject rather than from the nudge table. Capped, or consecutive edits of
# one file mid-debug would repeat it at every call.
NUDGE_MAX_TODO_TICK: int = 2
TODO_NUDGE_MULTIFILE_THRESHOLD: int = 3
# Alternative trigger for the same nudge: at least this many successful substantive
# actions (writes/exec/mutations — PLAN_BLOCKED tools) before it fires, so a task
# that is many operations but few files (e.g. an optimisation loop editing one file
# while running many benchmarks) is still recognised as multi-step.
TODO_NUDGE_OP_THRESHOLD: int = 5

# Distinct discovery-evidence signals required before the agent-mode discovery
# gate is satisfied (local exploration). A single stray search/read is not enough.
DISCOVERY_EVIDENCE_MIN_DISTINCT: int = 2
# Same bar for plan mode, where it now gates the *draft phase* rather than flagging a
# plan already written: a plan rests on at least a search and some reads.
DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN: int = 2
# Files whose content the model must have read before the plan-document tool is
# offered. Distinct signal kinds alone are too weak a bar — `ls` plus `find` scores
# two without a line of code having been read — so the bar is about reading, not
# about volume: one real read plus the breadth term above, which keeps a one-file
# task cheap while still rejecting a plan written over file names.
PLAN_EVIDENCE_MIN_FILES_READ: int = 1
# Explore-phase turns after which the plan tool is unlocked whatever the evidence.
# Plan mode must always reach a plan: past this budget the model is told to record
# what it has and say plainly what it could not establish.
PLAN_EXPLORE_MAX_TURNS: int = 8

# Repeated identical failed edit attempts on one file before the write is blocked.
REPEATED_EDIT_FAILURE_LIMIT: int = 2

# Identical successful calls (same tool, same args, same result) before the result is
# annotated. Three, not two: a second identical call is often a legitimate re-read
# after an edit elsewhere, while a third has stopped being about the file.
IDENTICAL_REPEAT_THRESHOLD: int = 3

# Repeated-failing-call guard. Nudges only fire when the model stops calling tools,
# and cross-step dedup only blocks *writes* — so a non-write call that fails can be
# re-issued identically every step. These two thresholds turn that silent spin into
# (a) a one-time mid-loop corrective and (b) a hard backstop.
SOFT_REPEAT_THRESHOLD: int = 2  # identical FAILED dispatches before the corrective, once
HARD_REPEAT_LIMIT: int = 3      # identical failures after which the call is not run again

# ── What a bash command changed (BASH_EFFECT annotation) ──────────────────────
# A shell write returns nothing, so the model that just ran one is the only actor in
# the loop with no diff to check. These bound the report it gets back.
BASH_EFFECT_MAX_FILES: int = 8
BASH_EFFECT_DIFF_MAX_LINES: int = 40
# Shortest repeated block that counts as suspicious duplication. Below three lines a
# repetition is ordinary code (closing braces, blank lines, decorators).
BASH_EFFECT_DUP_MIN_BLOCK: int = 3
# Added-line ceiling for the periodicity scan, which is quadratic in this number.
BASH_EFFECT_DUP_MAX_LINES: int = 1_000
# Per-file ceiling for the no-git fallback, which snapshots content in memory.
BASH_EFFECT_SNAPSHOT_MAX_BYTES: int = 512_000
BASH_EFFECT_SNAPSHOT_MAX_FILES: int = 200


# ── Context-window budget ──────────────────────────────────────────────────────
# Shared by both front-ends (CLI chat_session + the WebSocket server) so the
# token-usage bar and history trimming stay in agreement. How the window is resolved
# against the live backend is the engine's business (query_engine.context_budget).

# Approximate chars-per-token for a rough token estimate. Used as the fallback
# when an exact tokenizer is unavailable (Ollama, or a vLLM /tokenize failure).
CHARS_PER_TOKEN: int = 4

# Default context-window sizes (tokens) by mode.
CTX_TOTAL_COMPACT: int = 32_000
CTX_TOTAL_FULL: int = 200_000
# Fraction of the window reserved for the model's answer.
CTX_RESERVED_RATIO: float = 0.20

# Token-denominated counterparts of the char budgets above, used by the agent
# loop's hot-path trim/compaction once an exact token count is available. They
# mirror the char budgets at the 4 chars/token assumption (600K chars ≈ 150K
# tokens; 480K chars ≈ 120K tokens) but are enforced against real token counts.
TOOL_HISTORY_TOKEN_BUDGET: int = 150_000
INTRA_QUERY_COMPACT_TOKENS: int = 120_000
