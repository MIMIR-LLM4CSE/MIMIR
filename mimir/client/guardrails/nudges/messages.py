"""Nudge message copy — the text/renderers for the advisory nudges.

The built-in nudge messages (discovery, state, validation, denial, doc, env,
blast-radius, creation, todo, …) live here, in the nudges subsystem, rather than
in the shared ``workflow`` module: they are consumed only by ``nudges.engine``.
``validation_nudge_message`` is a stateful renderer (reads validation progress);
the rest are static text.
"""
from __future__ import annotations

from typing import Any

from ..workflow import pending_validation_paths, VALIDATION_RETRY_BUDGET, WORKFLOW_STATES
from ...context import bootstrap_state_context, declared_edit_set_complete


_DISCOVERY_NUDGE: str = (
	"Before proposing a final solution, gather evidence first. "
	"Evidence may be repository context, symbolic/numeric computation, or references. "
	"Do the relevant subset now: (1) use the repository structure already in your context for orientation "
	"only — it does not tell you which files this task touches, "
	"(2) run targeted searches to locate the specific relevant files, "
	"(3) read at least one relevant existing file to understand the established conventions, "
	"(4) for a mathematical claim, derive or verify it with the symbolic-math tools rather than asserting it, "
	"(5) for a factual/bibliographic claim, consult a reference instead of recalling it from memory; "
	"then proceed with the task, grounded in those findings."
)

_DENIAL_PREAMBLE: str = (
	"A required tool call was denied by the user. Do not claim full success. "
	"Either choose a safe alternative or explain clearly what remains incomplete because of the denial."
)


def discovery_nudge_message() -> str:
	return _DISCOVERY_NUDGE


def state_nudge_message(execution_context: dict) -> str:
	state = execution_context.get("workflow_state", "discover")
	if state == "discover":
		return (
			"Workflow state is DISCOVER (gather evidence). Collect enough material — repository context "
			"(list/search/read), symbolic/numeric results, or references — "
			"then, only when grounded, proceed to produce the artifact."
		)
	if state == "edit":
		return (
			"Workflow state is EDIT (produce the artifact). Read the exact target file first, "
			"then apply a focused change and move to validation. "
			"After any policy rejection, re-read the actual target before retrying — "
			"do not revise a previous draft from memory."
		)
	if state == "validate":
		return (
			"Workflow state is VALIDATE (verify). Validate the modified files through bash_run "
			"(`python -m py_compile` / `pytest -q` for tests / `ruff check` / `python -m mypy`) — "
			"and benchmarks when the task is about performance — and resolve all errors before more edits."
		)
	if state == "conclude":
		return "Workflow state is CONCLUDE. Summarize completion and residual risks."
	return f"Unknown workflow state '{state}'. Expected one of: {', '.join(WORKFLOW_STATES)}."


def _validation_nudge_message(pending_paths: list[str]) -> str:
	pending_text = ", ".join(pending_paths[:5]) if pending_paths else "modified files"
	return (
		"Do not conclude success yet. The following code file(s) were modified but not yet "
		f"validated: {pending_text}. Validate each one through bash_run before finalizing — e.g. "
		"`python -m py_compile <file>` (syntax), `pytest -q <file>` (tests), `ruff check <file>` "
		"(lint), `python -m mypy <file>` (types). If a check reports errors, fix the code and re-run it. "
		"If an import cannot be resolved, the default interpreter may be wrong — pass the right "
		"environment's python (the platform/environment query tool lists available environments). "
		"If a validator is unavailable in every environment, say so and why, then proceed — do not loop."
	)


def denial_nudge_message(execution_context: dict | None = None) -> str:
	denied_calls = (execution_context or {}).get("denied_tool_calls", [])
	if not denied_calls:
		return _DENIAL_PREAMBLE

	fragments: list[str] = []
	for item in denied_calls[:3]:
		tool_name = item.get("tool", "unknown")
		fallback_tools = item.get("fallback_tools", [])
		if fallback_tools:
			fragments.append(f"{tool_name}: try {', '.join(fallback_tools[:3])}")
		else:
			fragments.append(f"{tool_name}: state clearly what this action was needed for")

	return _DENIAL_PREAMBLE + " Suggested alternatives: " + "; ".join(fragments)


# --- Guidance-nudge copy ---------------------------------------------------
# These build the message body for the reasoning-babysitting nudges in
# guardrails.nudges.engine. The *when-to-fire* gating stays in nudge_logic; only
# the wording lives here so all nudge copy is in one place.

def error_recovery_nudge_message(failing_path: str) -> str:
	return (
		f"Repeated edit failures detected on '{failing_path}'. "
		"Re-read the relevant file section to refresh your context, then apply a "
		"smaller, differently-anchored patch instead of repeating the same edit."
	)


def regression_nudge_message(untested: list[tuple[str, str]]) -> str:
	preview = "\n".join(f"- {src}  →  {test}" for src, test in untested[:3])
	return (
		"You modified source file(s) that have existing tests you have not run "
		"this session:\n"
		+ preview
		+ "\n\nRun the associated test(s) through the dedicated test-validation tool "
		"before concluding, so an edit doesn't silently break existing behavior. "
		"If a test is intentionally out of scope, say so explicitly."
	)


def unfinished_plan_nudge_message(unchecked: list[dict]) -> str:
	preview = "\n".join(f"- {it['text']}" for it in unchecked[:5])
	more = f"\n(+{len(unchecked) - 5} more)" if len(unchecked) > 5 else ""
	return (
		"Your own task checklist still has unfinished step(s):\n"
		+ preview
		+ more
		+ "\n\nEither complete them now, or — if a step is no longer needed, turned out "
		"to be unnecessary, or is genuinely out of scope — say so explicitly in your "
		"final answer and leave it unchecked. Both are acceptable endings; silently "
		"concluding with open steps is not, because the checklist is what the user "
		"reads as the definition of done. Never tick a step whose output does not exist."
	)


def env_resolution_nudge_message(execution_context: dict) -> str:
	modules = sorted(m for m in execution_context.get("unresolved_modules", set()) if m)
	mod_hint = f" (missing: {', '.join(modules[:3])})" if modules else ""
	return (
		"A check or run failed because a required module is not importable in the "
		f"interpreter you used{mod_hint} — this is an environment problem, not a code defect. "
		"Do not give up and do not retry against the same interpreter. First enumerate the "
		"available environments (the platform/environment query tool lists conda envs and "
		"virtualenvs with their python paths). Then use the static import-resolution check "
		"(no execution) to find one whose interpreter resolves the module(s), and re-run the "
		"original check passing that interpreter as python_executable. "
		"If none resolves it, you may (with the user's approval) install the module into a suitable "
		"environment, or create a dedicated one, using the package-install / environment-creation "
		"capabilities — then offer to undo that at the end. If every option is unavailable or declined, "
		"conclude clearly: name the missing module(s), the environments you checked, and the remedy. "
		"Never report success without resolving it."
	)


def env_cleanup_nudge_message(execution_context: dict) -> str:
	mutations = execution_context.get("env_mutations", []) or []
	lines = []
	for m in mutations[:4]:
		if m.get("installed"):
			where = m.get("python") or "the target interpreter"
			lines.append(f"- installed {', '.join(m['installed'])} into {where}")
		elif m.get("path") or m.get("name"):
			lines.append(f"- created environment {m.get('name') or m.get('path')}")
	detail = ("\n" + "\n".join(lines)) if lines else ""
	return (
		"Before you conclude: you mutated the environment to complete this task." + detail
		+ "\n\nAsk the user (via the user-question/elicitation capability) whether to undo these "
		"changes — uninstall the package(s) or delete the created environment — and act on their "
		"answer. Leaving them in place is fine if the user wants to keep them; just don't decide silently."
	)


def documentation_nudge_message(execution_context: dict) -> str:
	written = sorted(execution_context.get("dirty_written_files", set()))
	file_hint = f" You modified: {', '.join(written[:3])}." if written else ""
	return (
		"Before concluding, quickly check whether any documentation or reference file "
		"(README, *.md, catalogs, server indexes) should be updated to reflect the code change."
		+ file_hint
		+ " If nothing relevant exists, you can ignore this."
	)


def blast_radius_nudge_message(execution_context: dict) -> str:
	read_files = sorted(execution_context.get("read_files", set()))
	file_hint = f" You have read: {', '.join(read_files[:3])}." if read_files else ""
	return (
		"Before changing a function/class definition or renaming a symbol, "
		"it may help to quickly search for callers or import sites first."
		+ file_hint
		+ " If the change is purely local and clearly contained, you can proceed."
	)


def creation_nudge_message(execution_context: dict) -> str:
	targets = sorted(execution_context.get("planned_edit_targets", set()))
	target_hint = f" Declared target(s): {', '.join(targets[:3])}." if targets else ""
	return (
		"A write target was declared earlier in this task but nothing has been "
		"written yet, and you have gathered context since."
		+ target_hint
		+ " If that work is still intended, write it now and validate it before "
		"concluding. If the request only called for an answer — or you have already "
		"given one — conclude without writing anything."
	)


def todo_nudge_message(execution_context: dict) -> str:
	files = sorted(
		execution_context.get("dirty_written_files", set())
		| execution_context.get("planned_edit_targets", set())
	)
	file_hint = f" Files touched so far: {', '.join(files[:4])}." if files else ""
	if execution_context.get("plan_written", False):
		guidance = (
			" Record the ordered remaining steps as a checklist using the plan/todo tool "
			"so progress is tracked."
		)
	else:
		guidance = (
			" First record your approach and reasoning, then record the ordered "
			"concrete steps as a checklist using the plan/todo tool from your available tools."
		)
	return (
		"You are editing multiple files without a task checklist."
		+ file_hint
		+ guidance
	)


def validation_nudge_message(agent: Any, execution_context: dict) -> str:
    execution_context = bootstrap_state_context(execution_context) or {}

    # If the model is still in the middle of a declared multi-file edit set,
    # do not push too hard toward validation yet.
    declared_complete = declared_edit_set_complete(execution_context)
    steps_since_last_edit = int(execution_context.get("steps_since_last_edit", 0))

    base_message = _validation_nudge_message(
        pending_validation_paths(execution_context),
    )

    if not declared_complete:
        remaining = sorted(
            (execution_context.get("declared_edit_set", set()) or set())
            - (execution_context.get("dirty_written_files", set()) or set())
        )
        if remaining:
            remaining_preview = ", ".join(remaining[:5])
            return (
                "You appear to still be completing a planned multi-file edit. "
                f"Finish the remaining declared targets first: {remaining_preview}. "
                "Then validate the modified files."
            )

    # If the model just edited very recently, keep the nudge softer.
    if steps_since_last_edit < 2:
        return (
            base_message
            + " If you are still finishing the current refactor, complete it first; "
              "otherwise start validation now."
        )

    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    if not fail_counts:
        return base_message

    hottest_path = max(fail_counts, key=fail_counts.get)
    hottest_count = int(fail_counts.get(hottest_path, 0))
    if hottest_count < 2:
        return base_message

    hint = ""
    last_replace_file = execution_context.get("last_replace_file", "")
    last_replace_old_text = execution_context.get("last_replace_old_text", "")
    read_line_counts = execution_context.get("read_file_line_counts", {})

    if last_replace_file == hottest_path and last_replace_old_text:
        hint = (
            " Read the local failing region around the last replacement anchor first, "
            "then apply a smaller targeted fix."
        )
    elif hottest_path in read_line_counts:
        lines_read = int(read_line_counts.get(hottest_path, 0))
        if lines_read < 50:
            hint = " Read more of the file before retrying, so the fix is grounded in broader context."

    suffix = f"Validation has failed {hottest_count} time(s) for {hottest_path}."
    if hottest_count >= VALIDATION_RETRY_BUDGET:
        suffix += (
            " The retry budget for this file is exhausted; avoid another broad rewrite unless the user explicitly wants it."
        )

    if hint:
        return base_message + " " + suffix + hint

    return (
        base_message
        + " "
        + suffix
        + " Prefer a smaller, localized repair and re-run validation."
    )
