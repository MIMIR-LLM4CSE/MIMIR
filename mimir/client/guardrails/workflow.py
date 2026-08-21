from __future__ import annotations

import warnings

from ..context.signals import SOURCE_FILE_EXTENSIONS
from ..context.execution_context import unwritten_declared_files, weakest_validation_tier
# Re-exported here for backward compatibility; the canonical definition lives in
# config.constants alongside the other agent-loop tuning knobs.
from ..config.constants import VALIDATION_RETRY_BUDGET

WORKFLOW_STATES: tuple[str, ...] = ("discover", "edit", "validate", "conclude")



def pending_validation_paths(execution_context: dict) -> list[str]:
	dirty_files = set(execution_context.get("dirty_written_files", set()))
	validated_files = set(execution_context.get("validated_files", set()))
	return sorted(path for path in dirty_files if path not in validated_files)


def has_pending_validation(execution_context: dict) -> bool:
	return bool(pending_validation_paths(execution_context))


def has_blocking_denials(execution_context: dict) -> bool:
	return bool(execution_context.get("denied_tool_calls", []))


# ── Denial escalation ladder ──────────────────────────────────────────────────
# A refused approval is not a failure to report and not an instruction to re-ask.
# It carries one of three meanings, which the model must weigh in this order:
#
#   1. "not this way"  — the goal stands, the means is wrong: find another route.
#   2. "that's useless" — the step itself is unnecessary: drop it, keep going, and
#                         say it was skipped at the user's request. (Dropping a step
#                         is never the same as getting past the approval gate.)
#   3. "stop there"    — end the turn: report what was done, what is blocked, and
#                         what you need from the user.
#
# The model chooses. These stages are the floor under that choice: they take the
# earlier readings off the table as refusals accumulate, so a wrong first guess
# cannot become a loop. Counting is per approval scope — the same normalised token
# the "always" grants use — so refusing one member of a command family escalates
# the family, while an unrelated action starts fresh.

STAGE_RECONSIDER: str = "reconsider"      # readings 1-3 all open
STAGE_DROP_OR_STOP: str = "drop_or_stop"  # reading 1 spent: drop the step or hand back
STAGE_HANDBACK: str = "handback"          # only reading 3 is left


def denial_scope_count(execution_context: dict, scope: str) -> int:
	"""How many times this approval scope has been refused this query."""
	if not scope:
		return 0
	return sum(
		1 for item in execution_context.get("denial_history", [])
		if item.get("scope") == scope
	)


def denial_stage(execution_context: dict, scope: str = "") -> str:
	"""Which readings of a refusal are still open, for *scope* (or the query overall).

	A refusal already counted for *scope* must be in ``denial_history`` before this
	is called: the stage describes the situation the model is being handed, so the
	first refusal of a scope reports ``reconsider`` (all three readings open).
	"""
	from ..config.constants import (
		DENIAL_QUERY_HANDBACK_TOTAL,
		DENIAL_SCOPE_DROP_AFTER,
		DENIAL_SCOPE_HANDBACK_AFTER,
	)

	history = execution_context.get("denial_history", [])
	total = len(history)
	scoped = denial_scope_count(execution_context, scope)

	# Stopping the run mid-prompt is the user saying "stop" in the plainest way there
	# is; it needs no accumulation.
	if any(item.get("kind") == "cancelled" for item in history):
		return STAGE_HANDBACK
	if scoped >= DENIAL_SCOPE_HANDBACK_AFTER or total >= DENIAL_QUERY_HANDBACK_TOTAL:
		return STAGE_HANDBACK
	if scoped >= DENIAL_SCOPE_DROP_AFTER or total >= DENIAL_QUERY_HANDBACK_TOTAL - 1:
		return STAGE_DROP_OR_STOP
	return STAGE_RECONSIDER


_STAGE_ORDER: tuple[str, ...] = (STAGE_RECONSIDER, STAGE_DROP_OR_STOP, STAGE_HANDBACK)


def worst_denial_stage(execution_context: dict) -> str:
	"""The furthest stage any refused scope has reached this query.

	What the layers that speak about the run as a whole (the nudge, the completion
	report) need: one scope at the end of the ladder governs the turn, even if another
	was only refused once.
	"""
	history = execution_context.get("denial_history", [])
	if not history:
		return STAGE_RECONSIDER
	scopes = {item.get("scope", "") for item in history}
	return max(
		(denial_stage(execution_context, scope) for scope in scopes),
		key=_STAGE_ORDER.index,
	)


def handback_required(execution_context: dict) -> bool:
	"""True once any refused scope has reached the end of the ladder."""
	return bool(execution_context.get("denial_history")) and \
		worst_denial_stage(execution_context) == STAGE_HANDBACK


def handback_scopes(execution_context: dict) -> list[dict]:
	"""The refusal records whose scope reached ``handback``, newest first."""
	history = execution_context.get("denial_history", [])
	seen: set = set()
	out: list[dict] = []
	for item in reversed(history):
		scope = item.get("scope", "")
		if scope in seen:
			continue
		seen.add(scope)
		if denial_stage(execution_context, scope) == STAGE_HANDBACK:
			out.append(item)
	return out


def set_workflow_state(execution_context: dict, new_state: str) -> None:
	if new_state in WORKFLOW_STATES:
		execution_context["workflow_state"] = new_state
	else:
		warnings.warn(
			f"set_workflow_state called with unknown state '{new_state}'; ignored. "
			f"Expected one of: {', '.join(WORKFLOW_STATES)}",
			stacklevel=2,
		)




def unchecked_checklist_items(execution_context: dict) -> list[dict]:
	"""Steps still unticked on the live checklist, or [] if there is no checklist.

	The single reader of checklist state outside the prompt builder, shared by the
	completion issues below, the finalization predicate, and the unfinished-plan
	nudge. Fails closed to ``[]`` on a missing or unreadable file so a run without a
	checklist — the majority — keeps behaving exactly as before, rather than having
	an obligation invented for it.

	Optional steps are included; callers that must not block on them filter on
	``item["optional"]``.
	"""
	todo_fp = execution_context.get("todo_file_path", "")
	if not todo_fp:
		return []
	# Imported lazily: the prompt package pulls in the extension loader, and the
	# guardrails layer is imported during its construction.
	from ..prompt.system_prompt import _load_todo_items
	return [it for it in _load_todo_items(todo_fp) if not it.get("done")]


def _plural_runs(n: int) -> str:
	return "1 run" if n == 1 else f"{n} runs"


def _collect_completion_issues(execution_context: dict) -> tuple[list[str], list[str]]:
	"""Return (issues, completed) describing the current completion state."""
	issues: list[str] = []
	completed: list[str] = []

	pending = pending_validation_paths(execution_context)
	fail_counts: dict = execution_context.get("validation_fail_count_by_file", {})
	dirty: set = execution_context.get("dirty_written_files", set())

	if pending:
		# Split pending files into three sub-buckets for clearer feedback.
		stuck_paths = [p for p in pending if fail_counts.get(p, 0) >= VALIDATION_RETRY_BUDGET]
		retry_paths = [p for p in pending if fail_counts.get(p, 0) > 0 and fail_counts.get(p, 0) < VALIDATION_RETRY_BUDGET]
		fresh_paths = [p for p in pending if fail_counts.get(p, 0) == 0]
		if stuck_paths:
			issues.append("Check budget exhausted (no further retries): " + ", ".join(stuck_paths[:5]))
		if retry_paths:
			issues.append("Checks failing (will retry): " + ", ".join(retry_paths[:5]))
		if fresh_paths:
			issues.append("Modified files never checked: " + ", ".join(fresh_paths[:5]))
	else:
		# Name what a check actually proves. "All modified files validated" is what a
		# model reads back as licence to report the work as verified, when all it means
		# is that the files parse and lint — the answer's correctness lives on the runs.
		tier = weakest_validation_tier(execution_context, dirty) if dirty else None
		if tier:
			completed.append(f"All modified files checked ({tier}) — parses/lints, says nothing about the result")
		else:
			completed.append("All modified files checked")

	# A refusal is only a *blocker* when it ended the run — the hand-back case. A step
	# the user judged unnecessary is not a failure to fix, and listing it as one is what
	# made every refusal come back as "Task is incomplete". It is still reported, in its
	# own section (see refused_action_lines), because a skipped step must never be silent.
	denied_calls = execution_context.get("denied_tool_calls", [])
	if denied_calls and handback_required(execution_context):
		issues.append("Stopped at the user's request: " + ", ".join(refused_action_lines(execution_context)))
	elif not denied_calls:
		completed.append("No denied actions")

	if execution_context.get("code_mutation_started") and execution_context.get("workflow_state") != "conclude":
		issues.append("Workflow not completed: still in '" + str(execution_context.get("workflow_state")) + "' state")
	elif execution_context.get("workflow_state") == "conclude":
		completed.append("Workflow reached conclude state")

	# The plan-vs-implementation check, obtained from state that already exists:
	# declared_edit_set is scraped from the checklist's own step text and until now
	# only fed a state transition. Reporting the difference is what makes a step
	# that was planned and then quietly skipped visible at completion time.
	unwritten = unwritten_declared_files(execution_context)
	if unwritten:
		issues.append("Declared but never written: " + ", ".join(unwritten[:5]))

	# The reminder budget stops asking after two tries; what it must not do is let a run
	# disappear. Every run still open at completion is named here, with what was tried.
	runs = execution_context.get("runs") or {}
	never_judged = [c for c, r in sorted(runs.items()) if r.get("completed") and not r.get("verdict")]
	unresolved = [c for c, r in sorted(runs.items()) if r.get("verdict") == "unknown"]
	broken = [(c, r) for c, r in sorted(runs.items())
	          if not r.get("completed") or r.get("verdict") == "fail"]
	if never_judged:
		issues.append("Ran but never judged: " + ", ".join(never_judged[:5]))
	if unresolved:
		issues.append("Judged inconclusive, still unresolved: " + ", ".join(unresolved[:5]))
	for command, run in broken[:5]:
		spent = int(run.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
		label = "no further retries" if spent else "will retry"
		issues.append(f"Run failing ({label}): {command}")
		# Naming the run says a wall was hit; naming the attempts says what the wall was,
		# which is the part the user needs to take it from here.
		attempts = run.get("attempts") or []
		if spent and attempts:
			issues.append("  tried: " + "; ".join(attempts[:3]))
	if runs and not never_judged and not unresolved and not broken:
		completed.append(_plural_runs(len(runs)) + " judged pass by the model")

	unchecked = [it for it in unchecked_checklist_items(execution_context) if not it.get("optional")]
	if unchecked and execution_context.get("code_mutation_started"):
		issues.append(
			f"Checklist incomplete: {len(unchecked)} step(s) unchecked — "
			+ "; ".join(it["text"] for it in unchecked[:3])
		)

	if not issues and not refused_action_lines(execution_context):
		issues.append("Unknown blocker; explicit completion criteria were not met")

	return issues, completed


def refused_action_lines(execution_context: dict) -> list[str]:
	"""One line per action the user refused, for the completion report.

	Deduplicated, and named by target where there is one: three refusals of the same
	command family printed as the bare tool name three times told the user nothing
	about what they had actually refused.
	"""
	lines: list[str] = []
	for item in execution_context.get("denied_tool_calls", []):
		tool_name = item.get("tool", "unknown")
		# The scope token is the command family / host / package set the refusal was
		# really about; the tool name alone is the least informative part of it.
		target = item.get("path") or (item.get("scope", "").split(":", 2) + ["", "", ""])[2]
		line = f"{tool_name}({target})" if target else tool_name
		if line not in lines:
			lines.append(line)
	return lines[:5]


# Report headlines. The wording is load-bearing: it is what the user reads first and
# what `is_incomplete_answer` matches on, so both live here rather than being spelled
# out at each consumer.
HEADLINE_INCOMPLETE: str = "Task is incomplete."
HEADLINE_HANDBACK: str = "Stopped at your request."
HEADLINE_REFUSED_ONLY: str = "Task complete, except for what you refused."

_INCOMPLETE_HEADLINES: tuple[str, ...] = (HEADLINE_INCOMPLETE, HEADLINE_HANDBACK)


def is_incomplete_answer(answer: str) -> bool:
	"""True when a finalized answer reports the task as unfinished.

	The consumers (the CLI's re-plan offer, the sub-agent's ``completed`` flag) used to
	match the one literal prefix there was. Now that a refusal can end a run three
	different ways, they ask here instead — a run that only skipped what the user
	refused *is* finished, and must not be reported as a failure.
	"""
	return answer.startswith(_INCOMPLETE_HEADLINES)


def finalize_incomplete_answer(
	answer: str, execution_context: dict, ran_out_of_steps: bool = False,
) -> str:
	issues, completed = _collect_completion_issues(execution_context)
	pending = pending_validation_paths(execution_context)
	denied_calls = execution_context.get("denied_tool_calls", [])
	refused = refused_action_lines(execution_context)
	handback = handback_required(execution_context)

	# Refusals alone do not make a task incomplete: reading (2) of a refusal is "this
	# step was not needed", and the honest report of that is a completed task with a
	# named omission — not a failure. Only a hand-back, some *other* open issue, or a
	# run that never reached a final answer ends the run unfinished.
	if handback:
		headline = HEADLINE_HANDBACK
	elif refused and not issues and not ran_out_of_steps:
		headline = HEADLINE_REFUSED_ONLY
	else:
		headline = HEADLINE_INCOMPLETE

	summary = headline + "\n\nCompleted:\n- " + "\n- ".join(completed)
	if issues:
		summary += "\n\nRemaining issues:\n- " + "\n- ".join(issues)
	if refused and not handback:
		summary += (
			"\n\nNot performed (you refused these; they were skipped, not attempted "
			"another way):\n- " + "\n- ".join(refused)
		)

	# Only treat unvalidated source-code files as high-risk; unvalidated non-code
	# files (e.g. .md, .txt) that never go through a code validator are low-risk.
	_pending_code = [p for p in pending if any(p.endswith(ext) for ext in SOURCE_FILE_EXTENSIONS)]
	# Something the model said it would change and then didn't is a silent gap, not
	# a failure — medium, not high, but never "low".
	_unwritten = unwritten_declared_files(execution_context)
	# A refusal the run stopped on stays high risk — work was cut short. A refusal the
	# run absorbed and continued past is a known, named gap: medium, never "low".
	risk_level = (
		"high" if _pending_code or handback
		else ("medium" if pending or _unwritten or denied_calls else "low")
	)
	summary += f"\n\nResidual risk: {risk_level}."
	if answer.strip():
		summary += "\n\nLatest model answer:\n" + answer.strip()
	return summary

# --- Loop-control nudge copy -----------------------------------------------
# Message bodies for the nudges fired directly from query_engine.agent_loop
# (plan-mode control flow, the step-limit reminder, and the mid-tool-loop repeat
# correctives). As with the guidance-nudge copy above, only the wording lives
# here; the *when-to-fire* gating stays in agent_loop. Kept here so all nudge copy
# sits in one module regardless of which loop fires it.

# Plan-mode user nudges.
PLAN_TODO_NUDGE_EARLY = (
	"You have not yet recorded a plan, which is mandatory to produce a valid plan output. "
	"Record it with the plan tool as a written plan document (structured markdown prose — an overview, the approach broken into its main axes with the reasoning and the concrete files/symbols each touches, key decisions/risks, and how you'll validate). "
	"If you need more information to write the plan, you may ask the user for clarification or call the read-only exploration tools (search, read, inspect, platform/memory queries) to gather evidence. "
	"The plan must focus on detailed key actions and validations needed to complete the task (implementations, tests, validations, etc.). "
	"Steps such as discovery, analysis, or information gathering that are not directly part of the implementation/validation plan should be omitted. "
)
PLAN_TODO_NUDGE_LATE = (
	"You should have described a plan to the user by now — record it with the plan tool as a written plan document (structured markdown prose explaining the approach, its main axes, the reasoning, and validation). "
	"If you are unsure about the exact approach, make your best guess based on the information you have, and we can iterate from there. "
	"The plan must focus on detailed key actions and validations needed to complete the task (implementations, tests, validations, etc.). "
	"Steps such as discovery, analysis, or information gathering that are not directly part of the implementation/validation plan should be omitted. "
)
PLAN_DELIVER_ANSWER = (
	"You must now deliver your final answer to the user, explaining the plan you have written. No more tool calls should be made. "
	"The plan should be the main basis of your answer, but you can also include relevant information from the discovery context or tool results if it helps the user understand the reasoning behind the plan. "
)
PLAN_DELIVER_ANSWER_FIRM = (
	"STOP calling tools. The plan is already recorded and the user can see it — re-reading or re-writing it "
	"changes nothing and repeating its text back is not an answer. Reply now, in prose only, with a short "
	"summary (a few sentences) of what the plan does and what you need from the user to proceed. "
	"Do not reproduce the plan document verbatim. "
)
# Tool-role reply used when a redundant plan/todo call is dropped after the plan is
# already recorded (the model is looping on the checklist instead of delivering).
PLAN_ALREADY_RECORDED_ERROR = (
	"The plan is already recorded and unchanged — this call was skipped. "
	"Do not call any more tools: answer the user in prose now."
)
# Advisory, not a rejection: the plan stands either way. Rejecting it used to discard
# the plan the model had just recorded, with nothing guaranteeing it would submit that
# form again — which is how a run could spin without ever delivering anything.
PLAN_EVIDENCE_NUDGE = (
	"Your plan is not grounded in any exploration of the code. The repository structure and platform "
	"summary in your context are orientation only — they do not tell you which files, symbols, or boundaries "
	"this task touches. If locating the relevant code would materially change this plan, use the read-only "
	"exploration tools (search, read, inspect, platform/memory queries) now and refine it. If it would not — "
	"a task that touches no existing code, for instance — keep the plan as it is and say plainly in your "
	"answer that it rests on assumptions rather than on the code."
)

# Plan-approval workflow nudges. After the user reviews a proposed plan they may
# accept it (switch to agent mode and execute), reject it (drop it and stop), ask for a
# rework (re-plan from scratch), or describe specific changes in free text (fold them
# in, then re-present for approval).
PLAN_APPROVED_EXECUTE = (
	"The user has APPROVED your plan. You are now switching to agent mode to carry it out. "
	"No task checklist exists yet: before starting the work, record the ordered implementation/validation "
	"steps of the approved plan with the plan/todo tool, unless the task is small enough not to warrant one. "
	"The steps must be the key actions and validations the task needs (implementations, tests, validations, "
	"etc.); omit discovery, analysis, and information-gathering steps. "
	"Then execute the approved plan end to end, making the necessary edits, running the relevant validations, "
	"and marking each step done with the plan/todo update tool as you complete it. "
	"You may ask for re-plan or re-approval "
	"if you notice any issues as you execute the plan, otherwise act on the plan you already agreed with the user."
)
PLAN_REJECTED_STOP = (
	"The user has REJECTED this plan. Do not execute any part of it and do not write a new plan. "
	"Stop here and wait for the user's next instruction."
)
PLAN_REJECTED_ANSWER = (
	"Plan rejected — nothing was executed. Tell me what you would like to do instead."
)
PLAN_REWORK_NUDGE = (
	"The user has asked you to REWORK this plan. Redo it from scratch: reconsider the approach, question the "
	"assumptions that led to the rejected plan, gather any further evidence you need with the read-only "
	"exploration tools, and then record a revised plan with the plan/todo tool for the user to review again."
)


def plan_revision_nudge(feedback: str) -> str:
	"""Nudge to fold the user's requested changes into a revised plan, then re-present it."""
	detail = f' The user asked for the following changes:\n"{feedback}"\n' if feedback else " "
	return (
		"The user has requested CHANGES to the plan before approving it." + detail +
		"Evaluate each request: integrate the changes that improve the plan, and for anything you "
		"disagree with or that seems out of scope, briefly explain your reasoning rather than blindly "
		"applying it. Then record the revised plan with the plan/todo tool so the user can review it again."
	)


STEP_LIMIT_NUDGE = (
	"You are approaching the step limit. "
	"Consider to stop calling tools and summarise: (1) what has been completed, "
	"(2) what still needs doing, and (3) the next step the user should request."
)


def repeat_corrective_message(tool_name: str, fails: int) -> str:
	"""One-time mid-loop reminder when a non-write call keeps failing identically."""
	return (
		f"[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"
		f"The same operation has now failed {fails} times with identical arguments. "
		"Repeating it unchanged will keep failing. Change something concrete — different "
		"arguments, a different tool, or fix the underlying precondition (e.g. resolve the "
		"environment per the cascade) — or stop and conclude clearly that you cannot proceed, "
		"naming what failed and why. Do not issue the same call again."
	)


def handback_corrective_message(refusals: list[dict]) -> str:
	"""One-time mid-loop stop once refusals reached the end of the denial ladder.

	Unlike the other correctives this one is not advisory framing around a retry: the
	user has refused the same thing to the end of the ladder, and the only remaining
	reading of that is "stop". It carries what was refused so the model reports the
	right thing rather than guessing which of its actions is meant.
	"""
	what = ", ".join(item.get("scope") or item.get("tool", "that action") for item in refusals[:3])
	return (
		"[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"
		f"The user has refused this repeatedly ({what}) and will not be asked to approve it "
		"again. That is no longer a hint to find another way — it is the end of this line of "
		"work. Stop calling tools toward it now and end your turn: report what you completed, "
		"what the refusal leaves undone, and what you need from the user to go further. If "
		"other, unrelated parts of the task remain and are unaffected by the refusal, finish "
		"those first, then hand back."
	)


def redundant_corrective_message(tool_name: str, repeats: int) -> str:
	"""One-time mid-loop reminder when a non-write call keeps returning identical content."""
	return (
		f"[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"
		f"You have already made this exact call {repeats + 1} times and received the same "
		"result each time. The content is unchanged and already in the conversation above — "
		"re-reading it adds nothing. Use what you already have: act on it, read something "
		"different (another file or line range), or conclude. Do not repeat this call."
	)