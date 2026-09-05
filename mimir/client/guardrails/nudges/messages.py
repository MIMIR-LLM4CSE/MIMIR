"""Nudge message copy — the text/renderers for the advisory nudges.

The built-in nudge messages (discovery, state, validation, denial, doc, env,
blast-radius, creation, todo, …) live here, in the nudges subsystem, rather than
in the shared ``workflow`` module: they are consumed only by ``nudges.engine``.
``validation_nudge_message`` is a stateful renderer (reads validation progress);
the rest are static text.
"""
from __future__ import annotations

from typing import Any

from ..builtin_check import builtin_check_failures
from ..workflow import (
	pending_validation_paths,
	worst_denial_stage,
	STAGE_DROP_OR_STOP,
	STAGE_HANDBACK,
	VALIDATION_RETRY_BUDGET,
	WORKFLOW_STATES,
)
from ...context import (
    bootstrap_state_context,
)
from ...context.capabilities import scope_spec
from ...config.constants import STUCK_REPAIR_CONSTRAIN_AFTER


def shell_tool_name(agent: Any) -> str:
	"""The connected shell tool, resolved from the scope its server declares.

	Same identity ``observations._carries_shell_command`` reads, for the same reason:
	the tool that takes a raw command line is the one declaring a ``command_prefix``
	scope, and that declaration — not a name written here — is what says which one it
	is. Falls back to a description, so copy stays sensible with none connected.
	"""
	registry = getattr(agent, "tool_caps", None) or {}
	names = sorted(
		n for n in registry
		if (scope_spec(n, registry) or {}).get("kind") == "command_prefix"
	)
	return names[0] if names else "the shell"


_DISCOVERY_NUDGE: str = (
	"Ground this in evidence before answering. Do the relevant subset now: search for the "
	"files this task touches and read the section that matters; derive a mathematical claim "
	"with the symbolic-math tools rather than asserting it; consult a reference for a "
	"factual one. Then proceed from what you found."
)

# A refusal is an instruction, not an error to report and retry. The three readings
# below are stated in priority order and mirror the tool-result hint the model got
# mid-loop (agent_core._denied_tool_result) and the Non-negotiables line in the system
# prompt — this nudge is the copy that reaches it once it has stopped calling tools.
_DENIAL_RECONSIDER: str = (
	"A tool call was refused by the user. Do not claim full success, and do not re-issue it. "
	"Decide what the refusal meant, in this order: (1) not this way — the goal stands but the "
	"means was wrong, so reach it another way; (2) unnecessary — the step is not needed, so drop "
	"it, carry on with the rest of the task, and say plainly that it was skipped at the user's "
	"request; (3) stop — end your turn and hand back, reporting what is done, what is blocked, "
	"and what you need from the user."
)

_DENIAL_DROP_OR_STOP: str = (
	"That action has now been refused more than once, so the objection is to the action itself, "
	"not to how you went about it. Stop looking for another route to the same end. Either drop "
	"the step and finish the rest of the task without it — stating that it was skipped at the "
	"user's request — or, if the task cannot proceed without it, stop and hand back with what "
	"is blocked."
)

_DENIAL_HANDBACK: str = (
	"That action has been refused repeatedly and you will not be asked to approve it again. Stop "
	"here. Make no further tool calls toward it: end your turn with what you completed, what the "
	"refusal leaves undone, and what you need from the user to go further."
)


def discovery_nudge_message() -> str:
	return _DISCOVERY_NUDGE


def state_nudge_message(agent: Any, execution_context: dict) -> str:
	state = execution_context.get("workflow_state", "discover")
	if state == "discover":
		return (
			"Workflow state is DISCOVER (gather evidence). Collect enough material — repository context "
			"(list/search/read), symbolic/numeric results, or references — "
			"then, only when grounded, proceed to produce the artifact."
		)
	if state == "edit":
		return (
			"Workflow state is EDIT (produce the artifact). Work from the exact current text "
			"of the target, then apply a focused change. "
			"A successful edit returns its diff and the new line numbers it produced, and "
			"a failed anchor returns the site's current text — use those rather than a "
			"previous draft from memory, and read the file only when neither covers what you "
			"need."
		)
	if state == "validate":
		return (
			"Workflow state is VALIDATE (verify). The modified files are checked here, without "
			"you having to invoke anything; repair whatever that check reports. Exercising the "
			"change on top of it — building it, running it, running the tests around it — is the "
			"part that is yours, wherever it is simply feasible. Going back to editing, here or "
			"elsewhere, is a normal move, not a step out of order."
		)
	if state == "conclude":
		return "Workflow state is CONCLUDE. Summarize completion and residual risks."
	return f"Unknown workflow state '{state}'. Expected one of: {', '.join(WORKFLOW_STATES)}."


def _validation_nudge_message(findings: dict[str, str]) -> str:
	"""Report what the built-in check found, file by file.

	The check itself is no longer asked for: it runs here, in-process, over every
	modified file before the loop may conclude. So this message is not a request to go
	validate — it is the result, and the only thing it asks for is the repair. It names
	no tool for the same reason it names no command: nothing outside had to be installed
	for the defect below to be found.
	"""
	lines = "\n".join(f"- {path}: {detail}" for path, detail in findings.items())
	return (
		"Do not conclude success yet. The mandatory check ran on the files you modified "
		"and rejected these:\n"
		+ lines
		+ "\n\nRead the reported location and repair it. The check runs again on its own "
		"once you stop editing — you do not have to invoke anything. A file that is "
		"truncated or half-written is the usual cause; re-read the region before rewriting it."
	)


def denial_nudge_message(execution_context: dict | None = None) -> str:
	context = execution_context or {}
	stage = worst_denial_stage(context)

	if stage == STAGE_HANDBACK:
		return _DENIAL_HANDBACK
	if stage == STAGE_DROP_OR_STOP:
		return _DENIAL_DROP_OR_STOP

	denied_calls = context.get("denied_tool_calls", [])
	if not denied_calls:
		return _DENIAL_RECONSIDER

	# Alternatives are worth naming only while reading (1) is still on the table.
	fragments: list[str] = []
	for item in denied_calls[:3]:
		tool_name = item.get("tool", "unknown")
		fallback_tools = item.get("fallback_tools", [])
		if fallback_tools:
			fragments.append(f"{tool_name}: try {', '.join(fallback_tools[:3])}")
		else:
			fragments.append(f"{tool_name}: no configured alternative — weigh (2) and (3)")

	return _DENIAL_RECONSIDER + " Alternatives for (1): " + "; ".join(fragments)


# --- Guidance-nudge copy ---------------------------------------------------
# These build the message body for the reasoning-babysitting nudges in
# guardrails.nudges.engine. The *when-to-fire* gating stays in nudge_logic; only
# the wording lives here so all nudge copy is in one place.

def error_recovery_nudge_message(failing_path: str) -> str:
	return (
		f"Repeated edit failures detected on '{failing_path}'. "
		"A failed anchor returns the current text of the site in `anchor_excerpt` — "
		"copy it from there rather than re-reading the file, then apply a smaller, "
		"differently-anchored patch instead of repeating the same edit."
	)


def stuck_repair_nudge_message(streak: int) -> str:
	"""The two rungs of the stuck-repair ladder, chosen by how deep the streak is.

	One function, branching on state, rather than two table rows — the same shape
	``denial_nudge_message`` uses. Neither rung names a cause or proposes a fix: what
	is being corrected is not the model's hypothesis but its refusal to leave the one
	it has, so the copy constrains the *class* of next action and leaves the content
	to the model, which is the only party that knows the code.
	"""
	if streak >= STUCK_REPAIR_CONSTRAIN_AFTER:
		return (
			f"The same thing has now failed {streak} times, and each fix has been a "
			"variation on the last one. Do not edit that target again this turn.\n\n"
			"Produce one new observation first: take the primitive your fix depends on "
			"— the operator, the helper, the boundary case it calls — and check it alone "
			"against a value you know in advance. A layer you never questioned is where a "
			"streak like this usually lives.\n\n"
			"If you are confident the layer below is sound and you still cannot place the "
			"fault, stop and say what you ruled out and what you would try next. That is an "
			"ending too."
		)
	return (
		f"That has failed {streak} times now. Before changing the same code again: the "
		"fault is not necessarily where you are correcting it. Widen the frame — check "
		"whether what you are editing is actually the thing that is wrong, or whether it "
		"is downstream of something you have been assuming is correct."
	)


def regression_nudge_message(untested: list[tuple[str, str]]) -> str:
	preview = "\n".join(f"- {src}  →  {test}" for src, test in untested[:3])
	return (
		"You modified source file(s) that have existing tests you have not run "
		"this session:\n"
		+ preview
		+ "\n\nThe test already exists, so running it is cheap and it is the one thing that "
		"shows the edit did not silently break existing behavior — worth it whenever the "
		"command is one you can issue as the project stands. Run it through the shell, then say "
		"what the run showed — a suite is an execution like any other, and its exit code is not "
		"its result. "
		"If a test is intentionally out of scope, cannot run in this environment, or would "
		"need an environment built before it could start, say so "
		"explicitly instead — that is an ending too, and this is asked once."
	)


def unexercised_code_nudge_message(paths: list[str], route: str = "") -> str:
	"""Everything you wrote is checked, and none of it has ever run.

	Says what the checks did establish before asking for more, because a model told
	"this is not validated" after watching its lint pass reads the loop as broken and
	re-runs the lint. The three routes are listed because "run it" is not actionable
	for a library, and the way out is stated in the same breath: a change with nothing
	to run is a real answer, and this fires once.

	``route`` is the direct command the gate actually found here. Naming it is what makes
	the ask proportionate by construction: this only fires when one exists, so it points at
	that one rather than asking the model to go looking for any of the three.
	"""
	preview = "\n".join(f"- {p}" for p in paths[:5])
	more = f"\n(+{len(paths) - 5} more)" if len(paths) > 5 else ""
	found = f"\nThe direct route from here is {route}. " if route else "\n"
	return (
		"These files are checked — they parse, import and lint:\n"
		+ preview
		+ more
		+ "\n\nThat says they are written correctly. It says nothing about what they "
		"compute, and nothing here has run yet, so there is no output for you or the "
		"user to judge. Building and running it is recommended when it is simply "
		"feasible — one direct command against the project as it stands. It is not, when "
		"getting to that command needs a step of its own (configuring a build, installing "
		"something, creating an environment, fetching data, an allocation)."
		+ found
		+ "Exercise the change one of three ways — run its entry point "
		"directly, run whatever calls into it, or, when neither is reachable, add a "
		"reproducible test to the project's test directory and run that. A test you "
		"add there is a deliverable the user keeps, not a throwaway probe. Then say "
		"what the output showed.\n"
		"If this change genuinely has nothing to run — a comment, a docstring, a "
		"rename — if nothing here can build or run it, or if getting to a first run "
		"would cost more than the change is worth, say which and why, and move on. "
		"All three are complete answers."
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
		"final answer and leave it unchecked. Both are acceptable endings; concluding "
		"without mentioning them at all is the one that is not."
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
    """What the model is told about the files the built-in check rejected.

    No longer a push toward doing the check — the loop already did it. The message is
    the finding, so the branches that used to soften it (mid-refactor, just-edited)
    are gone with the request they were softening.
    """
    execution_context = bootstrap_state_context(execution_context) or {}

    findings = builtin_check_failures(execution_context)
    if not findings:
        # Nothing was rejected: the pending files are ones the sweep has not reached
        # yet — it runs where the loop asks whether it may conclude, not after every
        # write. Say that plainly rather than inventing a defect.
        pending = pending_validation_paths(execution_context)
        return (
            "Do not conclude success yet. These files were modified and are still "
            f"waiting on the mandatory check: {', '.join(pending[:5])}."
        )

    base_message = _validation_nudge_message(findings)

    fail_counts = execution_context.get("validation_fail_count_by_file", {})
    hottest_path = max(findings, key=lambda path: int(fail_counts.get(path, 0)))
    hottest_count = int(fail_counts.get(hottest_path, 0))
    if hottest_count < 2:
        return base_message

    suffix = f" The check has now failed {hottest_count} time(s) for {hottest_path}."
    last_replace_file = execution_context.get("last_replace_file", "")
    last_replace_old_text = execution_context.get("last_replace_old_text", "")
    if last_replace_file == hottest_path and last_replace_old_text:
        return (
            base_message + suffix
            + " Read the local failing region around the last replacement anchor first, "
              "then apply a smaller targeted fix."
        )
    if hottest_count >= VALIDATION_RETRY_BUDGET:
        return (
            base_message + suffix
            + " The retry budget for this file is exhausted; avoid another broad rewrite "
              "unless the user explicitly wants it."
        )
    return base_message + suffix + " Prefer a smaller, localized repair."
