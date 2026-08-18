from __future__ import annotations

import logging
import os
import re
from typing import Any, Awaitable, Callable

# The user system-prompt override lives in ``.mimir/`` — its resolution belongs with
# the other user-extension resolvers (servers/skills/plugins); building lives here.
from ..extensions.system_prompt import resolve_system_prompt_file
from ..config.constants import THINKING_DEPTH_AUTO
from ..context.execution_context import recent_first

logger = logging.getLogger(__name__)

# Phrases that signal the user explicitly wants something persisted. Kept
# specific (multi-word where possible) so ordinary task verbs like "save the
# figure" or "keep the loop" do NOT trip auto-storage. Memory is written only
# when one of these appears in the user's query — see auto_store_memory.
_MEMORY_RECALL_SIGNALS = {
    "remember", "remember this", "remember that",
    "don't forget", "do not forget",
    "keep in mind", "memorize",
    "make a note", "take a note",
    "for future reference", "note that",
}

# Canonical, version-controlled examples for every extension type live in
# `mimir/examples/` (the single source of truth). Nothing is written into the
# workspace `.mimir/` — that directory is the user's alone; see PLUGINS_DETAILED.md.


# The built-in default "general context" — the agent's base system prompt. Used
# verbatim whenever no override .md is resolved (see build_base_system_content).
#
# Written as one markdown section per constant, joined below. Two reasons the shape
# matters: (a) the rendered string is what a mid-size open-weight model actually
# parses — a flat wall of prose loses its middle, so every section carries a real
# ``##`` heading and one instruction per line; (b) the conditional sections (see
# _SECTION_SUBAGENTS) must be separable so they are injected only when the matching
# capability is connected. Keep sections short: what is stated once, in the right
# section, is followed; what is repeated three times is noise.
#
# INVARIANT — coverage vs. the nudge layer. Guidance nudges are enforcement-gated
# (``guardrails/nudges/engine._GUIDANCE_BY_LEVEL_MODE``): at "light" only a subset
# fires, at "off" none do. So every guidance nudge's *obligation* must also exist
# here, in one line, or it silently disappears for strong models — the prompt is the
# only carrier that survives every enforcement level. The nudge stays the detailed,
# situational reminder; this file states the rule once. Verification-layer nudges
# need no such mirroring (they run at every level), though the hard ones are restated
# in Non-negotiables anyway. ``test_context_file.py`` guards this mapping.

_SECTION_IDENTITY = (
    "You are MIMIR (Mathematical Intelligence: Modeling, Implementation, Runtime), "
    "an expert scientific-computing research engineer operating as an autonomous CLI agent "
    "with access to tools and the filesystem.\n"
    "Your philosophy is \"From Math, to HPC\": you are equally strong at theory and derivation, "
    "faithful bibliography recall, fast prototyping, and high-performance computing. "
    "Move fluently across that spectrum — ground claims in mathematics, prototype quickly, then "
    "make code correct and fast on the target hardware.\n"
    "Execute tasks with precision, determinism, and verifiable correctness."
)

# The hard floor. These are the rules whose violation is not self-correcting, hoisted
# out of the sections where they used to be restated. They are stated once, here, and
# the sections below no longer repeat them.
_SECTION_NON_NEGOTIABLES = (
    "## Non-negotiables\n"
    "These are absolute. Every other section is a default you may adapt; these you may not.\n"
    "- NEVER fabricate: files, commands, command output, references, or numbers. "
    "If you did not observe it, say so.\n"
    "- NEVER report success when a modified artifact was not validated, a required tool call was "
    "denied, or a step was skipped. Report what actually happened.\n"
    "- NEVER bypass an approval gate, and never reach for a general shell to do what an "
    "approval-gated capability does.\n"
    "- A refused approval is an instruction, not an error. Weigh which one it is, in order: "
    "(1) not this way — reach the goal another way; (2) unnecessary — drop the step, continue "
    "the rest, report it skipped at the user's request; (3) stop — end the turn saying what is "
    "done, what is blocked, and what you need. NEVER re-ask for something already refused.\n"
    "- NEVER fake validation: do not simulate, wrap, inline, or monkey-patch a check to force a pass.\n"
    "- NEVER display internal tool or server names to the user in chat. Describe what you did by its "
    "effect (\"ran the tests\", \"searched the repository\", \"installed the package\").\n"
    "- NEVER repeat a tool call that policy already rejected. Read the hint and adapt."
)

_SECTION_LATITUDE = (
    "## Latitude\n"
    "You are trusted to exercise engineering judgment. Everything below the Non-negotiables is a "
    "set of sensible defaults, not a rigid procedure: when a rule plainly does not fit the "
    "situation, do the reasonable thing and state briefly why.\n"
    "This latitude covers workflow, depth, and style. It never covers the Non-negotiables above."
)

_SECTION_STYLE = (
    "## Style\n"
    "- Be concise; skip preamble, caveats, and filler.\n"
    "- Use structured output only when it improves clarity.\n"
    "- Write math as LaTeX: $...$ inline, $$...$$ on its own lines for display — in replies and in "
    ".md files alike. Markdown eats \\( \\) and \\[ \\], so those reach the reader broken.\n"
    "- Assume a non-interactive, batch-oriented CLI environment unless told otherwise."
)

_SECTION_SCOPE = (
    "## Scope\n"
    "- Make the minimum change the task requires.\n"
    "- Refactor, reformat, rename, and clean only inside the direct scope; leave unrelated files alone."
)

_SECTION_WORKFLOW = (
    "## Workflow\n"
    "Phases, as the task warrants: discover (gather evidence) → edit (produce the artifact) → "
    "validate (verify) → conclude. Trivial tasks collapse to a single step.\n"
    "- Evidence may be repository context, symbolic/numeric computation, or bibliographic/web "
    "references — not only source files. Begin editing once it is sufficient.\n"
    "- Reach for the scientific toolkit instead of reasoning unaided: symbolic math for derivations "
    "and closed forms, benchmarking to measure performance, the platform and HPC advisors for "
    "architecture-aware and cluster decisions, the web tools for bibliography. Derive and verify with "
    "them rather than recalling a result or a reference from memory. Otherwise use tools only when "
    "reasoning alone is insufficient.\n"
    "- After every tool call, read all returned fields: stdout, stderr, status, error, hint. "
    "If the same error occurs twice, stop and report it.\n"
    "- A pure-theory or literature task may conclude straight from discovery, with no artifact and no "
    "validation — say so explicitly when that is the case.\n"
    "- Update documentation when a change affects behavior, usage, or guarantees.\n"
    "- If a task is ambiguous, unsafe, impossible, or out of scope, stop and report the blocker, "
    "requesting only the minimum missing information.\n"
    "- Conclude when the objectives are met and validation has passed, stating completion or failure "
    "clearly with a brief summary of the steps taken."
)

_SECTION_DISCOVERY = (
    "## Discovery\n"
    "- Grep first, read second: search for the symbol or keyword, then read only the section around "
    "the match (startLine/endLine around the hit).\n"
    "- Read a whole file when it is unfamiliar and you need structural orientation, or when it is "
    "short enough that a targeted search would save nothing.\n"
    "- Repeated fixed-size block reads (1-120, then 121-240, then 241-360, with no search in between) "
    "mean you are scanning without a target: stop, search for the relevant symbol, read that section.\n"
    "- Any repository-structure or platform summary in your context is orientation only; it does not "
    "tell you which files or symbols this task touches.\n"
    "- Identify exact files, symbols, and boundaries yourself before editing."
)

_SECTION_EDITING = (
    "## Editing\n"
    "- Read the relevant section, with surrounding context, before editing a file.\n"
    "- Pick the edit tool that fits: line-based edits are safer when exact line numbers are known; "
    "anchor-based edits work well for unique, structurally stable text.\n"
    "- Copy anchor text verbatim from your most recent read — never reconstruct it from memory. "
    "If an anchor is ambiguous, widen it or switch to a line-based edit.\n"
    "- New text must contain ONLY the replacement content. Context lines already in the file belong "
    "in the old text, to make the anchor unique; putting them in the new text duplicates them.\n"
    "- Check the returned diff after every edit and repair immediately if duplication is visible.\n"
    "- For new files, inspect the parent directory and read a peer file to match conventions.\n"
    "- File tools take absolute paths. Join the workspace root (given in the orientation block) "
    "for a file inside it, or give a different absolute path to place one elsewhere — so where a "
    "file goes is always stated, never inferred.\n"
    "- Before a global rename or wide replacement, search all references and summarise the impact.\n"
    "- Confirm exact targets before any destructive operation (delete, overwrite)."
)

_SECTION_VALIDATION = (
    "## Validation\n"
    "After modifying code, treat validation as a primary objective, not generic execution. "
    "Scale its depth to the size and risk of the edit: a docstring change needs a syntax check, "
    "a change to solver logic warrants the whole tier.\n"
    "\n"
    "TIER 1a — EXECUTABILITY. For any real code edit, validate through the shell, in order:\n"
    "1. Syntax first, always — e.g. `python -m py_compile <file>` (or the language's own syntax check).\n"
    "2. Static checks without execution: import resolution, types, lint — e.g. `ruff check <file>`, "
    "`python -m mypy <file>`.\n"
    "3. Project test execution, when tests exist — e.g. `pytest -q <file>`.\n"
    "If an import fails, the default python may be the wrong environment — point the command at the one "
    "the platform/environment query tool reports. If a validator is unavailable in every environment you "
    "can reach, do not loop: state which check could not run and why, record it as a residual risk, and "
    "move on.\n"
    "\n"
    "TIER 1b — CORRECTNESS, whenever the edit changes what the code computes rather than how it is "
    "arranged. Tier 1a proves the code RUNS, never that it is RIGHT: a check asserting only that "
    "nothing raised, that output is finite, or that it lies in a broad range passes for almost any "
    "wrong answer.\n"
    "- Compare against something independent of the code under test: a known-good reference, an "
    "analytically derived result, an invariant that must hold, or a measured trend under refinement.\n"
    "- Assert the property that actually defines the requirement, not a weaker proxy a broken "
    "implementation would also satisfy.\n"
    "- Report each measured result as `key=value` on its own line so it is recorded rather than claimed "
    "in prose, using the conventional invariant name where one applies (`l2_rel`, `linf_rel`, "
    "`convergence_order`, `conservation_residual`, `finite`).\n"
    "- With no oracle available, say so and record it as residual risk. Never call an artifact verified "
    "or correct on Tier 1a alone, and never report a check you did not run.\n"
    "\n"
    "TIER 2 — PERFORMANCE, only after correctness holds (Tier 1b, not merely 1a) and only when the "
    "task is about performance optimization. "
    "MEASURE, never assert: establish a correct baseline, then quantify the speedup with the "
    "benchmark tools, consulting the platform and HPC advisors for architecture-aware choices. "
    "An optimization without a measurement is a claim, and a faster wrong answer is a regression.\n"
    "\n"
    "A syntax check, a throwaway fragment and the project's own test suite are all the same surface — "
    "only the command differs. A fragment that reimplements or stubs what the project actually imports "
    "proves nothing about the project: run the real code, in its real layout.\n"
    "\n"
    "Validate the artifact as it stands. Re-running, inlining, wrapping, or simulating validation "
    "logic to force a pass, and monkey-patching or bypassing side effects, invalidate the result. "
    "When validation fails, analyse the error and revise the patch rather than repeating the same edit."
)

_SECTION_RUNNING = (
    "## Running code, and where working files go\n"
    "The controlled shell is the execution surface: compile, run, test and lint by invoking the "
    "toolchain directly, with the exact flags the task needs.\n"
    "- Prefer a one-shot command over a new file: a check you run once belongs inline — the code as a "
    "quoted `-c` argument, steps chained with `&&`.\n"
    "- A probe that genuinely needs a file (too long to pass inline, or re-run many times) goes in the "
    "scratchpad, by absolute path, with its outputs: logs, intermediate data, diagnostic figures.\n"
    "- A file in the workspace is a deliverable: what the user asked for, or what the artifact you "
    "produce needs. Never a throwaway check.\n"
    "- Each shell call is a fresh process at the workspace root with a short timeout ceiling. A long "
    "run needs the detached/background capability, not a bigger timeout.\n"
    "- When the default interpreter is the wrong environment, invoke the resolved one by absolute "
    "path.\n"
    "- A check that fails on a missing module is an environment problem, not a code defect: enumerate "
    "the available environments and retry with an interpreter that resolves it before concluding.\n"
    "- A package you install or an environment you create is a reversible obligation: before "
    "concluding, ask the user whether to undo it rather than deciding silently."
)

_SECTION_PLANNING = (
    "## Planning & todo\n"
    "For multi-step tasks, set a plan with your approach and reasoning (what you will do, why, and "
    "what risks exist), then record the concrete ordered steps as a todo list if that capability is "
    "available, updating each step as you complete it. This is optional for simple or single-step tasks.\n"
    "CRITICAL: mark a step done only once its output exists — never on intent, partial progress, or "
    "because you are about to start it. When unsure, verify first."
)

_SECTION_REASONING = (
    "## Reasoning\n"
    "- Match the depth to the stakes: an obvious next step deserves no deliberation — act. Reason "
    "where it is genuinely uncertain: which approach, an ambiguous requirement, the cause of a failure.\n"
    "- Keep it brief and decisive: identify the key unknowns, pick the most direct approach, stop. "
    "Narrating every possibility, restating the problem, or re-deriving known facts adds nothing."
)

# Conditional — injected by build_system_content only at the "auto" rung of the
# thinking ladder, where no token budget is imposed and the depth is the model's to
# choose. The fixed rungs (quick/medium/deep) carry their budget in the backend
# payload instead, and "max"/"off" need no directive, so they see the base prompt
# unchanged.
_THINKING_DIRECTIVE_AUTO = (
    "## Reasoning depth\n"
    "No reasoning budget is imposed on this turn — the depth is yours to set, and it should differ "
    "sharply from turn to turn. Open and close the reasoning block immediately when the answer or "
    "the next action is already clear; reserve a long chain for the part that is genuinely hard."
)

# Conditional — injected by build_system_content only when the sub-agent capability is
# actually connected. Describing a tool the model does not have invites it to hallucinate
# the call.
_SECTION_SUBAGENTS = (
    "## Sub-agents\n"
    "You can spawn agents (task, context?, readonly?) to run a focused child agent on a self-contained "
    "sub-task; it works in its own context and returns its answer as a string, and several agents "
    "spawned in one step run concurrently.\n"
    "- Default to doing the work yourself. Delegate only when it clearly pays off: a cleanly separable "
    "subtask, a parallel sweep over many files or cases, or a heavy isolated computation whose "
    "intermediate detail you do not need in your own context.\n"
    "- Keep trivial steps, and any work needing the full surrounding context, in your own loop.\n"
    "- Pass readonly=True unless the sub-task must write.\n"
    "- Before spawning, check whether the result already exists — prefer reuse over recomputation."
)

_DEFAULT_BASE_SYSTEM_CONTENT = "\n\n".join([
    _SECTION_IDENTITY,
    _SECTION_NON_NEGOTIABLES,
    _SECTION_LATITUDE,
    _SECTION_STYLE,
    _SECTION_SCOPE,
    _SECTION_WORKFLOW,
    _SECTION_DISCOVERY,
    _SECTION_EDITING,
    _SECTION_VALIDATION,
    _SECTION_RUNNING,
    _SECTION_PLANNING,
    _SECTION_REASONING,
])


def build_base_system_content(context_file: str = "") -> str:
    """The agent's base system prompt (its "general context").

    Loads the resolved override .md (``extensions.resolve_system_prompt_file``) and
    returns its content verbatim, replacing the built-in default. Falls back to
    ``_DEFAULT_BASE_SYSTEM_CONTENT`` when no file is resolved, the file is empty, or
    it cannot be read.
    """
    path = resolve_system_prompt_file(context_file)
    if path:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read().strip()
            if text:
                return text
            logger.warning("system-prompt file %s is empty; using built-in default", path)
        except OSError as exc:
            logger.warning("could not read system-prompt file %s: %s; using built-in default", path, exc)
    return _DEFAULT_BASE_SYSTEM_CONTENT


def _build_memory_summary(
    query: str,
    answer: str,
    execution_context: dict[str, Any] | None,
    truncate_text: Callable[[str, int], str],
) -> str:
    """Build a compact fact for persistent memory.

    The note records the durable facts of the exchange — what was asked, which
    files were produced, and a one-line outcome — not a transcript of the model's
    deliberation. Process noise (files read, search patterns) and long answer dumps
    are deliberately excluded.
    """
    parts: list[str] = []

    # Task
    parts.append(f"Task: {truncate_text(query.strip(), 300)}")

    if execution_context:
        # Files written — the durable artifact of the turn.
        written = sorted(execution_context.get("dirty_written_files", set()))
        if written:
            parts.append(f"Files written: {', '.join(written)}")

    # Outcome: the first sentence/line of the answer (what was decided), not the
    # full deliberation. The verification ledger we append at return time is dropped —
    # its file list is already captured above.
    # Imported here, not at module scope: query_engine.finalize imports this module,
    # so a top-level import would close the cycle.
    from ..query_engine.verification import split_answer_ledger

    answer_clean = split_answer_ledger(answer)[0].strip()
    first_line = next((ln.strip() for ln in answer_clean.splitlines() if ln.strip()), "")
    # Prefer the first sentence if the line runs long.
    first_sentence = re.split(r'(?<=[.!?])\s', first_line, maxsplit=1)[0] if first_line else ""
    outcome = truncate_text(first_sentence or first_line, 200)
    if outcome:
        parts.append(f"Outcome: {outcome}")

    return "\n".join(parts)


async def auto_store_memory(
    *,
    query: str,
    answer: str,
    tool_owner: dict[str, str],
    run_tool: Callable[[str, dict[str, Any], dict[str, Any] | None], Awaitable[str]],
    truncate_text: Callable[[str, int], str],
    execution_context: dict[str, Any] | None = None,
    logger: logging.Logger | None = None,
) -> None:
    """Persist a compact summary of the exchange.

    Triggers ONLY when the user explicitly asked to remember something (a recall
    signal in the query). Writing code files is the normal case for this agent and
    is intentionally NOT a trigger — persisting every code-writing turn turns memory
    into a transcript of low-value, near-duplicate entries.
    """
    if "memory_add" not in tool_owner:
        return

    query_lower = (query or "").lower()
    explicit_recall = any(signal in query_lower for signal in _MEMORY_RECALL_SIGNALS)
    if not explicit_recall:
        return

    summary = _build_memory_summary(query, answer, execution_context, truncate_text)
    try:
        await run_tool(
            "memory_add",
            {
                "text": summary,
                "tags": ["conversation", "auto"],
            },
            execution_context,
        )
    except Exception as exc:
        # Memory persistence is best-effort and should not break responses.
        if logger:
            logger.debug("memory_add failed: %s", exc)
        return


# Index lines look like: ``- [<description>](<slug>.md) — <YYYY-MM-DD>``
_INDEX_LINE_RE = re.compile(
    r'^-\s*\[(?P<desc>.*?)\]\((?P<slug>[^)]+?)\.md\)\s*(?:[—-]\s*(?P<date>\d{4}-\d{2}-\d{2}))?\s*$'
)


def _load_recent_memories(memory_file: str, max_entries: int = 10) -> list[dict]:
    """Read the most recent memories from the MEMORY.md index (newest first)."""
    try:
        with open(memory_file, "r", encoding="utf-8") as f:
            content = f.read()
    except OSError:
        return []
    entries = []
    for line in content.splitlines():
        m = _INDEX_LINE_RE.match(line.strip())
        if not m:
            continue
        entries.append({
            "slug": m.group("slug"),
            "text": m.group("desc").strip(),
            "timestamp": m.group("date") or "",
            "tags": [],
        })
    return entries[:max_entries]


def summarize_search_matches(matches: list[dict], max_items: int = 3) -> str:
    lines = []
    for item in matches[:max_items]:
        if not isinstance(item, dict):
            continue
        path = item.get("path") or item.get("file") or "<unknown>"
        score = item.get("score")
        score_suffix = f" (score={score})" if score is not None else ""
        lines.append(f"- {path}{score_suffix}")
    return "\n".join(lines)


def build_tool_catalog_for_planning(
    tool_owner: dict[str, str],
    sensitive_tools: set[str],
) -> str:
    by_server: dict[str, list[str]] = {}
    for tool_name, server_name in tool_owner.items():
        suffix = " [sensitive]" if tool_name in sensitive_tools else ""
        by_server.setdefault(server_name, []).append(f"{tool_name}{suffix}")

    if not by_server:
        return "No tools registered."

    lines = []
    for server in sorted(by_server):
        tools = sorted(by_server[server])
        lines.append(f"- {server}: {', '.join(tools)}")
    return "\n".join(lines)


# A step the plan itself marked aspirational ("(optional) convergence study").
# Without this, such a step is stored as a plain `- [ ]` like any other and, since
# nothing gates on unchecked items, vanishes silently — which is exactly how a
# planned convergence study disappeared between plan and delivery. Tagged here so
# the completion ledger can report it separately without it blocking anything.
_OPTIONAL_STEP_RE = re.compile(r"^\s*[\(\[]?optional[\)\]]?\s*[:\-–]?\s+", re.IGNORECASE)


def _scratch_dir_for_prompt() -> str:
    """The scratchpad path to advertise, or "" if it cannot resolve.

    The session-scoped directory, not ``standing_roots()[0]``: the standing grant is
    the scratchpad *home* (it has to cover a mid-run session switch), so advertising it
    would point the model one level above where its own working files belong.
    STATE_DIR is passed explicitly — see tool_execution.validation.scratch_roots.
    """
    try:
        from ...servers._shared.state_paths import scratch_dir
        from ..config.constants import STATE_DIR
        return scratch_dir(STATE_DIR)
    except Exception:
        return ""


def _load_todo_items(todo_file: str) -> list[dict]:
    """Read the current todo list from todo_list.md, return [] on any error.

    If a todo_deps.json sidecar exists in the same directory, each item is
    annotated with a ``ready`` boolean so the context injection can inform
    the model which steps are unblocked.
    """
    _checkbox = re.compile(r"^- \[([ x])\] (.+)$")
    try:
        with open(todo_file, "r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    items = []
    for line in lines:
        m = _checkbox.match(line.rstrip("\n"))
        if m:
            text = m.group(2)
            items.append({
                "index": len(items),
                "text": text,
                "done": m.group(1) == "x",
                "optional": bool(_OPTIONAL_STEP_RE.match(text)),
            })

    # Annotate with DAG ready status if deps sidecar exists.
    deps_file = os.path.join(os.path.dirname(todo_file), "todo_deps.json")
    if os.path.exists(deps_file):
        try:
            import json as _json
            with open(deps_file, "r", encoding="utf-8") as f:
                deps: list[list[int]] = _json.load(f)
            for item in items:
                if item["done"]:
                    item["ready"] = False  # already done, not actionable
                    continue
                idx = item["index"]
                step_deps = deps[idx] if idx < len(deps) else []
                item["ready"] = all(
                    items[d]["done"] for d in step_deps if d < len(items)
                )
        except Exception:
            pass  # deps annotation is best-effort

    return items


def _section(body: str) -> str:
    """Render one system-prompt section: a leading separator plus the body.

    Centralises the ``"\\n" + ...`` append boilerplate that every injected block
    shares, so ``build_system_content`` reads as a list of sections.
    """
    return "\n" + body


def _render_checklist(items: list[dict]) -> list[str]:
    """Render todo items as ``[x] text`` / ``[ ] text`` lines."""
    return [f"[{'x' if it.get('done') else ' '}] {it['text']}" for it in items]


def summarize_platform_profile(profile: dict[str, Any]) -> str:
    """Render a compact, plan-ready summary of the cached platform profile.

    Returns an empty string when there is nothing useful to show. Kept terse: this
    block is part of the agent's always-on foundational context so hardware-aware
    work is grounded in the real machine (SIMD width, cores, GPU, scheduler).
    """
    if not isinstance(profile, dict) or not profile:
        return ""
    parts: list[str] = []

    host = profile.get("hostname")
    if host:
        parts.append(f"Host: {host}")

    cpu = profile.get("cpu", {})
    if isinstance(cpu, dict) and cpu:
        bits: list[str] = []
        if cpu.get("model"):
            bits.append(str(cpu["model"]))
        if cpu.get("arch"):
            bits.append(str(cpu["arch"]))
        if cpu.get("logical_cpus"):
            bits.append(f"{cpu['logical_cpus']} logical CPUs")
        if cpu.get("numa_nodes"):
            bits.append(f"{cpu['numa_nodes']} NUMA nodes")
        simd = cpu.get("simd", {})
        if isinstance(simd, dict):
            flags = [k for k, v in simd.items() if v]
            bits.append("SIMD: " + (", ".join(flags) if flags else "none"))
        if bits:
            parts.append("CPU: " + "; ".join(bits))

    mem = profile.get("memory", {})
    if isinstance(mem, dict) and mem.get("total_gb"):
        parts.append(f"Memory: {mem['total_gb']} GB total")

    gpu = profile.get("gpu", {})
    if isinstance(gpu, dict) and gpu:
        parts.append(f"GPU: {gpu.get('count', 1)} available" if gpu.get("available") else "GPU: none")

    slurm = profile.get("slurm", {})
    if isinstance(slurm, dict) and slurm.get("available"):
        parts.append("Scheduler: Slurm available")

    toolchains = profile.get("toolchains", {})
    if isinstance(toolchains, dict):
        tc: list[str] = []
        for k in ("gcc", "g++", "nvcc", "python3"):
            val = toolchains.get(k)
            if not val:
                continue
            ver = re.search(r"\d+\.\d+(?:\.\d+)?", str(val))
            tc.append(f"{k} {ver.group(0)}" if ver else f"{k} {str(val)[:24]}")
        if tc:
            parts.append("Toolchains: " + "; ".join(tc))

    latest = profile.get("benchmarks", {}).get("latest") if isinstance(profile.get("benchmarks"), dict) else None
    if isinstance(latest, dict):
        gflops = latest.get("numpy_matmul", {}).get("gflops")
        gbps = latest.get("memory_copy", {}).get("throughput_gbps")
        bench = []
        if gflops is not None:
            bench.append(f"{gflops} GFLOP/s matmul")
        if gbps is not None:
            bench.append(f"{gbps} GB/s memory copy")
        if bench:
            parts.append("Last benchmark: " + ", ".join(bench))

    return "\n".join(parts)


def build_system_content(
    *,
    active_mode: str,
    tool_owner: dict[str, str],
    sensitive_tools: set[str],
    platform_profile_summary: str = "",
    repo_baseline_context: str = "",
    memory_context_file: str = "",
    todo_file: str = "",
    plan_todos: list[str] | None = None,
    context_file: str = "",
    thinking_depth: int = 0,
) -> str:
    system_content = build_base_system_content(context_file)

    # Depth-gated section: only the "auto" rung leaves the reasoning length to the
    # model, so only it needs the calibration directive. Placed right after the base,
    # ahead of the dynamic blocks — the rung only moves on an explicit user action, so
    # the prefix stays stable across a run of queries.
    if thinking_depth == THINKING_DEPTH_AUTO:
        system_content += _section("\n" + _THINKING_DIRECTIVE_AUTO)

    # Capability-gated section: describing the sub-agent contract when no sub-agent
    # tool is connected is a phantom instruction the model tries to act on. Placed
    # right after the base (and before the dynamic blocks) because tool_owner is
    # frozen once the servers are connected, so this stays prefix-stable per session.
    if "spawn_agent" in tool_owner:
        # Leading blank line: _section only prepends one newline, which would glue
        # the "## Sub-agents" heading to the previous section's last bullet.
        system_content += _section("\n" + _SECTION_SUBAGENTS)

    # Foundational context = agent identity: the repo structure and the target
    # hardware are gathered by self-contained core functions (repo walk + cheap
    # platform probe), not by the model choosing to call tools, and not via any
    # on-disk cache. Inject them on every query (both modes) right after the base
    # prompt so they stay byte-stable and prefix-cacheable within a query.
    # Task-specific evidence (which files/symbols matter) is the model's job via
    # its own searches/reads; deeper hardware detail is on demand via the platform
    # tools (which now just return their answer, nothing is persisted).
    if repo_baseline_context:
        system_content += _section(
            "Repository structure (baseline orientation — NOT a substitute for "
            "locating exact edit sites; search/read to find the specific code you change):\n"
            + repo_baseline_context
        )

    # Unconditional: where working files go is a rule about the user's tree, so it must
    # not depend on whether a repo baseline was built (an empty or unreadable workspace
    # is exactly where a stray probe script is most visible). Same position in the order
    # on every query, so the prefix stays cacheable.
    system_content += _section(
        f"Scratchpad (yours, outside the workspace, no approval needed): {_scratch_dir_for_prompt()}\n"
        "Use it by absolute path for throwaway scripts, probes, intermediate data and working files. "
        "Nothing there counts as produced work or is reported to the user. Two rules follow from that:\n"
        "- Nothing throwaway goes in the workspace. A verification script, a plot you drew to convince "
        "yourself, a log, a scratch dataset: the scratchpad, not the user's tree.\n"
        "- What runs once does not become a file at all — run it as a one-shot shell command.\n"
        "Deliverables go in the workspace, or wherever the user asked."
    )

    if platform_profile_summary:
        system_content += _section(
            "Target platform (optimize for this hardware):\n" + platform_profile_summary
        )

    if memory_context_file:
        entries = _load_recent_memories(memory_context_file, max_entries=10)
        if entries:
            lines = []
            for e in entries:
                ts = e.get("timestamp", "")[:10]  # date only
                text = e.get("text", "").strip()
                slug = e.get("slug", "")
                suffix = f"  ({slug}.md)" if slug else ""
                lines.append(f"[{ts}] {text}{suffix}")
            system_content += _section(
                "Memory index (one line per stored memory — historical context only; files listed "
                "may have changed or been deleted since, so always verify on disk before assuming "
                "something exists; if a task checklist is active, follow it — do not skip steps "
                "because memory says they were done before). Search or read the individual "
                "memory file for the full note:\n"
                + "\n".join(lines)
                + "\n(Index at: " + memory_context_file + ")"
            )
        else:
            system_content += _section(
                "Persistent memories are stored under: " + os.path.dirname(memory_context_file) + "\n"
                "Search them for relevant past context when useful."
            )

    if active_mode == "agent":
        if todo_file:
            # Live state from the todo server — takes priority over the static plan_todos.
            items = _load_todo_items(todo_file)
            if items:
                pending = sum(1 for it in items if not it.get("done"))
                system_content += _section(
                    f"Task checklist ({pending} pending — this is the authoritative plan for the current task; "
                    "execute every unchecked step even if memory suggests it was done before; "
                    "mark each step complete only after verifying the work exists on disk):\n"
                    + "\n".join(_render_checklist(items))
                )
            else:
                system_content += _section(
                    "No task checklist yet. For multi-step tasks it is strongly recommended "
                    "to write the plan before starting work, then mark each step complete as you go."
                )
        elif plan_todos:
            # Fallback: static list from plan output (todo server not registered).
            system_content += _section(
                "Task checklist (cross off each item as you complete it):\n"
                + "\n".join(f"[ ] {step}" for step in plan_todos)
            )

    if active_mode == "plan":
        tool_catalog = build_tool_catalog_for_planning(tool_owner, sensitive_tools)
      
        # Inject the existing draft so the model can refine it in subsequent turns.
        if todo_file:
            draft_items = _load_todo_items(todo_file)
            if draft_items:
                system_content += _section(
                    "Current draft plan (refine, extend, or replace as needed — "
                    "record the final ordered steps with the plan/todo tool from your available tools):\n"
                    + "\n".join(_render_checklist(draft_items))
                )

        system_content += _section(
            "Current mode: PLAN. Write, execution, and mutation tools are blocked, except the plan/todo tool. "
            "The repository structure and target hardware above are orientation only — you must call "
            "the read-only exploration tools (search, read, inspect, platform/memory queries) to locate the exact "
            "files, symbols, and boundaries this task touches before writing the plan. Gather that evidence "
            "first, then produce the plan grounded in what you found.\n"
            "\n"
            "Record the plan in TWO forms with the plan/todo tool:\n"
            "  1. A written plan document (free-form markdown prose) — this is the primary deliverable "
            "the user reads. Do NOT reduce it to a bare list of steps. Structure it like a short design "
            "note, using Markdown headings and a few sentences of prose under each:\n"
            "       • an Overview stating the goal and the outcome the change produces;\n"
            "       • an Approach broken into the main axes of the work (e.g. one subsection per component, "
            "layer, or concern), each explaining WHAT will change, WHERE (concrete files/symbols found "
            "during exploration), and WHY that approach over the alternatives;\n"
            "       • key design decisions, trade-offs, assumptions, and risks or edge cases;\n"
            "       • how the result will be validated (tests, checks, manual verification).\n"
            "  2. An ordered checklist of the concrete implementation/validation steps — the actionable "
            "spine that mirrors the written plan and is executed once approved.\n"
            "\n"
            "Write for a reader who wants to understand the reasoning, not just a task list: explain the "
            "rationale, reference the specific evidence you gathered, and keep it focused. Scale the depth "
            "to the task — a trivial change may need only a couple of paragraphs and 2-3 steps, a larger "
            "one warrants several axes with explanation. "
            "You will receive more context after the first step, so focus on the critical discovery and initial action here. "
            "\nAvailable tools by server:\n"
            + tool_catalog
        )

    if active_mode == "ask":
        system_content += _section(
            "Current mode: ASK. This is a read-only question-answering turn. Write, execution, and "
            "mutation tools are unavailable, and the dual-use execution tool is restricted to read-only "
            "discovery commands (search, list, read). "
            "The repository structure and target hardware above are orientation only — call the "
            "read-only exploration tools (search, read, inspect, platform/memory queries) to find the "
            "actual files, symbols, and behaviour the question is about before answering. Never answer "
            "a question about this codebase from assumption when you could have read the code.\n"
            "\n"
            "Answer the question directly, in prose, grounded in what you actually read: cite the "
            "concrete file paths (with line numbers where it helps) that back each claim. Scale the "
            "length to the question — a one-line question deserves a one-line answer, a design question "
            "warrants a few paragraphs. Do NOT write a plan, a task checklist, or an implementation "
            "proposal, and do not offer to make changes unless the user asked for them. If the question "
            "cannot be settled from the code, say what you found and what remains unknown."
        )

    return system_content


_PIN_MARKER = "\n[Discovery pin — auto-updated]\n"


def _pin_path(path: str) -> str:
    """Render a stored workspace-relative path as an absolute one, for display.

    The pin tells the model to use these paths *directly*, and file tools require
    absolute paths — so showing the stored relative form would hand it a path its
    next call rejects, and force it to reconstruct the root. That reconstruction is
    exactly the inference the absolute-path rule removed (see
    ``server_files._require_abs``); leaving it in the pin would reintroduce it at
    the point the model is most likely to copy blindly.

    Display only. ``execution_context`` keeps storing workspace-relative paths, so
    every gate, nudge and set-membership check is untouched.
    """
    if not path:
        return path
    from ..config.constants import WORKSPACE_ROOT
    return path if os.path.isabs(path) else os.path.join(WORKSPACE_ROOT, path)


def build_discovery_pin_block(
    execution_context: dict[str, Any],
    *,
    max_files: int = 10,
    max_queries: int = 5,
) -> str:
    """Build a compact pinned summary of in-session discovery evidence.

    Appended to the system message after every tool dispatch so key file
    paths and search patterns survive tool-history trimming.  Returns an
    empty string when the context holds no evidence worth pinning.
    """
    lines: list[str] = []

    def _pin_section(header: str, field: str, cap: int, render=_pin_path) -> None:
        """Append a capped, recency-ranked section.

        Every section is capped: the three write-side (planned targets, files
        written, files written last query).

        Ordering is recency-first (see ``recent_first``): the slice has to be the paths
        the model just touched.
        """
        values = recent_first(execution_context.get(field) or ())
        if not values:
            return
        lines.append(header)
        lines.extend(f"  {render(v)}" for v in values[:cap])
        if len(values) > cap:
            lines.append(f"  ... and {len(values) - cap} more")

    # Paths confirmed to exist on disk (carried across queries).  Showing
    # these prevents the model from trying to re-create files that already
    # exist (e.g. __init__.py, directories) from a previous query.
    _pin_section(
        "Known existing paths (already on disk — do not re-create):",
        "existing_paths", max_files,
    )
    _pin_section(
        "Files read this session (use these paths directly; avoid re-searching):",
        "read_files", max_files,
    )
    _pin_section(
        "Search patterns already executed (avoid re-running identical searches):",
        "search_queries_used", max_queries, render=repr,
    )
    _pin_section("Planned edit targets:", "planned_edit_targets", max_files)
    _pin_section(
        "Files written this session (validate before claiming done):",
        "dirty_written_files", max_files,
    )
    _pin_section(
        "Files written in the previous query — content has changed; "
        "search for the target symbol then read only that section before using any edit tool:",
        "prev_query_written_files", max_files,
    )

    # Live todo checklist — re-read from disk so the pin always reflects the latest
    # state, even after the model has marked a step complete mid-session. Built
    # BEFORE the emptiness test: the checklist is pin-worthy on its own. (It used to
    # be appended after an early `if not lines: return ""`, so a run whose only live
    # state was the checklist — no reads, no writes, no searches yet — silently lost
    # it; the copy in messages[0] is a build-time snapshot, so that left no live
    # channel at all.)
    todo_lines: list[str] = []
    todo_fp = execution_context.get("todo_file_path", "")
    if todo_fp:
        todo_items = _load_todo_items(todo_fp)
        if todo_items:
            pending = sum(1 for it in todo_items if not it.get("done"))
            todo_lines.append(f"\nTask checklist ({pending} pending):")
            todo_lines.extend(
                f"  [{'x' if it['done'] else ' '}] {it['text']}" for it in todo_items
            )

    if not lines and not todo_lines:
        return ""

    return _PIN_MARKER + "\n".join(lines + todo_lines) + "\n"
