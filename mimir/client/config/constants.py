from __future__ import annotations

import hashlib
import os


SERVER_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "servers")) + os.sep
SKILL_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "skills")) + os.sep

# Directories the self-contained repo-baseline walk prunes. Mirrors the search
# server's _SKIP_DIRS (server_search.py) so the deterministic baseline matches what
# the search tools would show — kept as a client-local copy because the baseline is
# built independently of any server (it must work even if the search server is off).
BASELINE_SKIP_DIRS: frozenset[str] = frozenset({
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache",
    ".ruff_cache", ".tox", ".eggs", "node_modules",
    ".venv", "venv", "env", ".env",
    "dist", "build", "site-packages",
    ".mcp_backups", ".continue",
})

# Workspace root: the folder the agent operates on. Matches the convention the
# MCP servers use (MCP_FILES_ROOT, set by server_manager / the extension), so the
# client and the server subprocesses resolve .mimir to the same place. Falls back
# to the process cwd when MCP_FILES_ROOT is unset.
WORKSPACE_ROOT = os.path.abspath(os.environ.get("MCP_FILES_ROOT") or os.getcwd())

# Per-workspace *extensions* directory (<workspace>/.mimir). Holds ONLY the user
# extensions that belong with the repo: plugins/, skills/, servers/, and the
# system_prompt.md template. The agent's own STATE (memory, sessions, todos,
# plans, platform_profile.json, preferences.json, active_session) no longer lives
# here — see STATE_DIR below. Resolved the same way the UI front-ends and servers
# resolve it, so the extension path stays in agreement.
MIMIR_DIR = os.path.join(WORKSPACE_ROOT, ".mimir")

# Central per-workspace STATE directory, kept OUT of the workspace so the agent's
# runtime state never pollutes the repo. Layout: <STATE_HOME>/<workspace-id>/,
# where the id is a readable basename plus a short hash of the absolute workspace
# path (so two different checkouts that share a basename don't collide). Overrides:
#   MIMIR_STATE_DIR  — full path, wins over everything (tests / headless runs)
#   MIMIR_STATE_HOME — the root under which per-workspace dirs are created (~/.mimir)
# The same STATE_DIR is handed to the server subprocesses via MIMIR_STATE_DIR
# (server_manager), so client and servers resolve state to the same place.
STATE_HOME = os.path.abspath(
    os.environ.get("MIMIR_STATE_HOME")
    or os.path.join(os.path.expanduser("~"), ".mimir")
)


def workspace_id(root: str) -> str:
    """Readable, collision-free id for a workspace root: ``<basename>-<sha1[:8]>``."""
    real = os.path.realpath(root)
    digest = hashlib.sha1(real.encode("utf-8")).hexdigest()[:8]
    return f"{os.path.basename(real) or 'root'}-{digest}"


STATE_DIR = os.path.abspath(
    os.environ.get("MIMIR_STATE_DIR")
    or os.path.join(STATE_HOME, workspace_id(WORKSPACE_ROOT))
)

# Modular "general context" (the agent's base system prompt). Resolution order:
#   SYSTEM_PROMPT_ENV env var → MIMIR_DIR/SYSTEM_PROMPT_FILENAME → built-in
#   default. The env-var form needs no .mimir/ dir. Nothing is auto-created in the
#   workspace; a template lives in mimir/examples/system_prompt.md. See
#   discovery/context_builder.py.
SYSTEM_PROMPT_ENV = "MIMIR_SYSTEM_PROMPT_FILE"
SYSTEM_PROMPT_FILENAME = "system_prompt.md"

# Application extension packs (custom policies + nudges). Resolution order:
#   MIMIR_PLUGINS_DIR env var → MIMIR_DIR/plugins/. Each *.py module in the dir is
#   imported once at agent init and registers its descriptors on import. See
#   client/extensions/plugins.py.
PLUGINS_DIR_ENV = "MIMIR_PLUGINS_DIR"
PLUGINS_DIRNAME = "plugins"

# User-provided skills and MCP servers, discovered under .mimir/ the same way as
# plugins (env override → MIMIR_DIR/<name>). User skills merge over the bundled set
# (same name overrides); user servers whose name collides with a bundled server are
# skipped. See extensions/servers.py.
SKILLS_DIR_ENV = "MIMIR_SKILLS_DIR"
SKILLS_DIRNAME = "skills"
SERVERS_DIR_ENV = "MIMIR_SERVERS_DIR"
SERVERS_DIRNAME = "servers"


def resolve_extension_dir(env_var: str, dirname: str) -> str:
    """Resolve a user-extension directory: ``$env_var`` if set, else ``MIMIR_DIR/dirname``.

    Single source of truth for the skills/servers/plugins locations under ``.mimir/``.
    """
    return os.path.abspath(os.environ.get(env_var) or os.path.join(MIMIR_DIR, dirname))


# ── Agent runtime tuning ───────────────────────────────────────────────────────
# Central home for the agent-loop behavioural knobs. Tune here, not at the call
# site, so behaviour stays consistent across the CLI and WebSocket front-ends.

# Hard wall for a single MCP tool call (seconds). Keeps a hung bash_run / web_get
# from blocking the agent loop indefinitely.
TOOL_CALL_TIMEOUT_SECS: int = 120

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

# Maximum agent steps (model ↔ tool round-trips) per query before the loop stops.
# Kept as the default ceiling for non-interactive callers (sub-agents, tests)
# that never prompt to continue.
MAX_AGENT_STEPS: int = 100

# Adaptive step budgeting for interactive front-ends. The loop runs up to the
# SOFT budget, then asks the user whether to continue; each "yes" grants another
# EXTENSION block, never exceeding the HARD ceiling. Non-interactive callers
# ignore these and simply stop at their fixed MAX_AGENT_STEPS.
AGENT_STEP_SOFT_BUDGET: int = 50
AGENT_STEP_EXTENSION: int = 50
AGENT_STEP_HARD_CEILING: int = 200

# Maximum consecutive validation failures per file before edits are blocked and
# the session auto-escapes to conclude (incomplete).
VALIDATION_RETRY_BUDGET: int = 5

# Cap on the number of tool schemas exposed to the model on a single step. Small
# models (e.g. Devstral-Small-24B served in mistral mode) stop emitting native
# tool calls and degenerate into echoing the tool *schema* as plain text once
# given more than ~45 tools. When the per-step tool list exceeds this cap, it is
# ranked by per-query relevance — always retaining the agent's core discovery /
# edit / validate / todo plumbing — and the long tail is trimmed. Set to 0 to
# disable the cap (large models that handle the full toolkit fine). Override via
# the MIMIR_MAX_TOOLS environment variable.
def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw.lstrip("-").isdigit() else default

MAX_TOOLS_PER_QUERY: int = _env_int("MIMIR_MAX_TOOLS", 40)


def max_tools_for(model: str) -> int:
    """Per-model tool-count cap (B300-ready).

    The cap exists for small models that garble past ~45 tools (see the Devstral
    note); larger models tolerate the full surface. Resolution order:
      1. ``MIMIR_MAX_TOOLS`` env (explicit global override) — if set.
      2. the model's vLLM profile ``max_tools`` (``0`` = uncapped).
      3. the default (40).
    So a 400B-class model on the B300s can declare ``"max_tools": 0`` in its profile
    and see every tool, while Devstral keeps the safe default — same codebase.
    """
    if os.environ.get("MIMIR_MAX_TOOLS", "").strip():
        return MAX_TOOLS_PER_QUERY
    try:
        from .models import profile_for_model
        val = profile_for_model(model).get("max_tools")
    except Exception:
        val = None
    return val if isinstance(val, int) and val >= 0 else MAX_TOOLS_PER_QUERY


# Thinking-depth scale — the single source of truth for the reasoning-depth ladder
# exposed by the CLI (/think) and the WS command (/thinking-depth), and mirrored by
# the webview slider. Index = depth level.
#
# ``auto`` is the default: thinking is on with no imposed budget, and the system
# prompt carries a calibration directive telling the model to close the reasoning
# block immediately on trivial turns and spend a long chain only where the task is
# genuinely uncertain. The fixed rungs (quick/medium/deep) impose a token budget
# instead, which the per-phase scaling in the agent loop then modulates; ``max``
# is thinking-on, unbudgeted, without the calibration directive.
THINKING_DEPTH_LABELS: tuple[str, ...] = ("off", "auto", "quick", "medium", "deep", "max")
THINKING_DEPTH_BUDGETS: tuple[int, ...] = (-1, -1, 512, 4096, 16384, -1)
THINKING_DEPTH_AUTO: int = 1
DEFAULT_THINKING_DEPTH: int = THINKING_DEPTH_AUTO


def clamp_thinking_depth(level: int) -> int:
    """Clamp a depth level into the valid range of the ladder."""
    return max(0, min(len(THINKING_DEPTH_LABELS) - 1, int(level)))


def thinking_depth_from_label(label: str) -> int | None:
    """Resolve a depth label (``"auto"``, ``"deep"``, …) to its level, else None.

    ``on`` is accepted as an alias for ``auto`` so the legacy ``/think on`` keeps
    working and now means "let the model calibrate" rather than "always think hard".
    """
    key = label.strip().lower()
    if key in ("on", "true", "1", "yes"):
        return THINKING_DEPTH_AUTO
    if key in ("false", "0", "no"):
        return 0
    try:
        return THINKING_DEPTH_LABELS.index(key)
    except ValueError:
        return None


# Transient backend-call resilience. A single model call (one step of up to
# MAX_AGENT_STEPS) should survive a flaky connection / 5xx / rate-limit rather
# than discarding the whole query. Total attempts = 1 + LLM_RETRY_ATTEMPTS.
# Backoff is exponential with jitter: LLM_RETRY_BASE_DELAY_SECS * 2**(n-1).
LLM_RETRY_ATTEMPTS: int = 3
LLM_RETRY_BASE_DELAY_SECS: float = 1.0
LLM_RETRY_MAX_DELAY_SECS: float = 20.0


# ── Nudge frequency caps & thresholds ──────────────────────────────────────────
# Central home for the previously-inline "magic numbers" that govern how often a
# workflow nudge may fire and the situational gates that trigger it. Values are
# unchanged from the former literals — centralised here so the whole nudge cadence
# can be read and tuned in one place instead of hunting through nudge_logic.py.

# Per-category max fires per query (the guarding branch checks
# ``nudge_counts[cat] < NUDGE_MAX_<CAT>``). A cap of 1 means "one reminder only".
NUDGE_MAX_VALIDATION: int = 2
NUDGE_MAX_DENIAL: int = 2
NUDGE_MAX_ERROR_RECOVERY: int = 2
NUDGE_MAX_REGRESSION: int = 1
NUDGE_MAX_UNFINISHED_PLAN: int = 1
NUDGE_MAX_DISCOVERY: int = 3
NUDGE_MAX_ENV_RESOLUTION: int = 1
NUDGE_MAX_ENV_CLEANUP: int = 1
NUDGE_MAX_DOC: int = 1
NUDGE_MAX_STATE: int = 1
NUDGE_MAX_BLAST_RADIUS: int = 1

# Application-pack (extension) nudges: max fires per query per registered rule.
CUSTOM_NUDGE_MAX_PER_QUERY: int = 3

# Consecutive no-op ("I'm done") turns that may still be nudged. The model
# produces a bare final-answer turn (no tool call); the first one earns a useful
# reminder (e.g. "validate your changes"). But once the model answers that nudge
# with ANOTHER no-op turn — without acting on it (no intervening tool call) — it is
# ignoring the reminder, and re-nudging just echoes the same summary. Any tool
# dispatch resets the counter, so a model that keeps *acting* between reminders is
# still nudged normally; only stalled prose-only re-affirmations are cut off.
NUDGE_MAX_CONSECUTIVE_NOOP: int = 1

# Idle-step gates: how many steps the agent must have paused editing before the
# nudge fires, so an in-progress refactor is not interrupted.
NUDGE_VALIDATION_IDLE_STEPS: int = 2
NUDGE_STATE_IDLE_STEPS: int = 3

# Multi-file thresholds: at least this many touched/planned files before the
# todo-tracking nudge fires.
NUDGE_MAX_CREATION: int = 1
NUDGE_MAX_TODO: int = 1
TODO_NUDGE_MULTIFILE_THRESHOLD: int = 3
# Alternative trigger for the same nudge: at least this many successful substantive
# actions (writes/exec/mutations — PLAN_BLOCKED tools) before it fires, so a task
# that is many operations but few files (e.g. an optimisation loop editing one file
# while running many benchmarks) is still recognised as multi-step.
TODO_NUDGE_OP_THRESHOLD: int = 5

# How many times a single query may un-prune a domain tool group that its wording
# never signaled (see query_engine.toollist.domains_signaled_by_text). Each re-arm
# rebuilds the tool list and therefore breaks the vLLM prefix cache once, so the
# budget is deliberately tight: 0 disables the mechanism entirely.
DOMAIN_REARM_MAX_PER_QUERY: int = 1

# Distinct discovery-evidence signals required before the agent-mode discovery
# gate is satisfied (local exploration). A single stray search/read is not enough.
DISCOVERY_EVIDENCE_MIN_DISTINCT: int = 2
# Weaker bar for the plan-mode evidence gate (a plan needs only some grounding).
DISCOVERY_EVIDENCE_MIN_DISTINCT_PLAN: int = 1

# Repeated identical failed edit attempts on one file before the write is blocked.
REPEATED_EDIT_FAILURE_LIMIT: int = 2


# ── Context-window budget ──────────────────────────────────────────────────────
# Shared by both front-ends (CLI chat_session + WebSocket ws_server) so the
# token-usage bar and history trimming stay in agreement.

# Approximate chars-per-token for a rough token estimate. Used as the fallback
# when an exact tokenizer is unavailable (Ollama, or a vLLM /tokenize failure).
CHARS_PER_TOKEN: int = 4
# Per-model-family overrides for the chars-per-token heuristic, matched as a
# substring of the lowercased model name. Tune here for models whose tokenizer
# diverges from the ~4 chars/token rule of thumb (code/CJK pack denser). Empty
# by default — every model falls back to CHARS_PER_TOKEN until measured.
CHARS_PER_TOKEN_BY_MODEL: dict[str, float] = {}


def chars_per_token_for(model: str | None) -> float:
    """Return the chars-per-token ratio for *model* (longest substring match)."""
    name = (model or "").lower()
    best_key = ""
    for key in CHARS_PER_TOKEN_BY_MODEL:
        if key in name and len(key) > len(best_key):
            best_key = key
    return CHARS_PER_TOKEN_BY_MODEL[best_key] if best_key else float(CHARS_PER_TOKEN)


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


def context_budget_for(model: str | None, mode: str = "full") -> tuple[int, int, int, int]:
    """Return (total, reserved, tool_history_budget, intra_compact_budget) tokens.

    The context window is resolved dynamically from the active backend — vLLM's
    reported ``max_model_len`` or Ollama's model context length — so the token
    bar and history trim/compaction match the real window instead of the static
    200K assumption. The trim/compaction thresholds scale to sit just under the
    usable window so they actually engage on small windows.

    Falls back to the static per-mode defaults whenever the window can't be
    determined, so behavior is unchanged in that case.
    """
    default_total = CTX_TOTAL_FULL if mode == "full" else CTX_TOTAL_COMPACT
    window: int | None = None
    try:
        from ..query_engine.backends.factory import get_backend
        window = get_backend().context_window(model or "")
    except Exception:
        window = None
    if not window:
        return default_total, int(default_total * CTX_RESERVED_RATIO), \
            TOOL_HISTORY_TOKEN_BUDGET, INTRA_QUERY_COMPACT_TOKENS
    # Compact mode stays a deliberately small budget even on a large window.
    total = window if mode == "full" else min(window, CTX_TOTAL_COMPACT)
    reserved = int(total * CTX_RESERVED_RATIO)
    usable = max(1, total - reserved)
    return total, reserved, usable, int(usable * 0.8)

# Tool classification (sensitive / plan-blocked / path args / …) is no longer a
# static list here: each server declares its tools' capabilities and the client
# reads them from the per-agent live registry (agent.tool_caps). See
# context/capabilities.py and POLICY.md.

SERVERS: dict[str, str] = {
    "math": SERVER_BASE + "utilities/server_math.py",
    "strings": SERVER_BASE + "utilities/server_strings.py",
    "datetime": SERVER_BASE + "utilities/server_datetime.py",
    "memory": SERVER_BASE + "agent_state/server_memory.py",
    "files": SERVER_BASE + "workspace/server_files.py",
    "search": SERVER_BASE + "workspace/server_search.py",
    "web": SERVER_BASE + "external/server_web.py",
    "github": SERVER_BASE + "external/server_github.py",
    "hpc": SERVER_BASE + "hpc/server_hpc.py",
    "platform": SERVER_BASE + "hpc/server_platform.py",
    "env": SERVER_BASE + "hpc/server_env.py",
    "benchmark": SERVER_BASE + "hpc/server_benchmark.py",
    "system": SERVER_BASE + "external/server_system.py",
    "code_intel": SERVER_BASE + "workspace/server_code_intel.py",
    "bash": SERVER_BASE + "workspace/server_bash.py",
    "localgit": SERVER_BASE + "workspace/server_localgit.py",
    "todo": SERVER_BASE + "agent_state/server_todo.py",
    "symbolic_math": SERVER_BASE + "utilities/server_symbolic_math.py",
    "finetune":    SERVER_BASE + "ml/server_finetune.py",
    "proxy": SERVER_BASE + "proxy/server_proxy.py",
    "agent": SERVER_BASE + "agent_state/server_spawn_agent.py",
    "interaction": SERVER_BASE + "interaction/server_interaction.py",
}

# One-line, human-readable purpose per server. Surfaced as tooltips in the
# server/skill toggle panel (ws_server `toggles_list`). Keyed by SERVERS name.
SERVER_DESCRIPTIONS: dict[str, str] = {
    "math": "Safe math expression evaluation (evaluate).",
    "strings": "String operations: case, strip, replace, split, search (string_op).",
    "datetime": "Date/time arithmetic, formatting, timezones (date_op).",
    "memory": "Persistent, timestamped agent memory with tags and search.",
    "files": "Read, write, edit, insert, replace, and delete workspace files.",
    "search": "Workspace reads, directory listing, and tree summaries.",
    "web": "Safe HTTP GET/POST and JSON utilities (SSRF-guarded).",
    "github": "Read-only GitHub access via the public REST API.",
    "hpc": "HPC helpers for Slurm scheduling and batch job submission.",
    "platform": "Builds a hardware/software platform profile on demand.",
    "env": "Install packages and create/delete Python environments (approval-gated).",
    "benchmark": "Lightweight micro-benchmarks for architecture calibration.",
    "system": "Read-only OS metrics and environment inspection.",
    "code_intel": "Code navigation: definitions, references, symbol outline (LSP/ctags).",
    "bash": "Controlled read-only bash command execution (strict allowlist).",
    "localgit": "Read-only git inspection of the local repository.",
    "todo": "Live task checklist and prose plan for the agent.",
    "symbolic_math": "Symbolic math (SymPy): algebra, calculus, matrices (symbolic).",
    "finetune": "Manage LoRA fine-tuning runs (local + Slurm backends).",
    "proxy": "Proxy/surrogate registry, benchmarks, runs, and optimization loop.",
    "agent": "Spawn fresh sub-agents to handle self-contained sub-tasks.",
    "interaction": "Ask the user structured questions via the elicitation channel.",
}
