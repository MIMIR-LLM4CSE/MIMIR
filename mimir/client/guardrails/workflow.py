from __future__ import annotations

import warnings

from ..context.signals import SOURCE_FILE_EXTENSIONS
from ..context.execution_context import (
	failed_runs,
	unwritten_declared_files,
	weakest_validation_tier,
)
# Re-exported here for backward compatibility; the canonical definition lives in
# config.constants alongside the other agent-loop tuning knobs.
from ..config.constants import VALIDATION_RETRY_BUDGET

WORKFLOW_STATES: tuple[str, ...] = ("discover", "edit", "validate", "conclude")



def pending_validation_paths(execution_context: dict) -> list[str]:
	"""Dirty files that still owe a check, minus the ones nothing here can check.

	The mandatory axis is the cheap one — parse, resolve imports, lint — and it is only
	mandatory where it is possible: a Fortran source on a box with no compiler has no
	checker to run, and demanding one turns a completion gate into a dead end. Those
	files leave this list and enter the ledger instead, named as unchecked.
	"""
	dirty_files = set(execution_context.get("dirty_written_files", set()))
	validated_files = set(execution_context.get("validated_files", set()))
	unverifiable = set(execution_context.get("unverifiable_files", set()) or set())
	return sorted(
		path for path in dirty_files
		if path not in validated_files and path not in unverifiable
	)


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


def approval_is_settled(execution_context: dict, scope: str) -> bool:
	"""True when raising an approval card for *scope* would re-ask an answered question.

	The ladder tells the model what a refusal *means*; it cannot stop it re-issuing the
	call, and each retry costs the user another card for a decision already made. From
	the second refusal of one scope, "another route to the same goal" is off the table —
	and the same route most certainly is one, so the gate answers instead of the user.
	The query-wide hand-back keeps its own reach: past that point nothing sensitive is
	worth another prompt, whatever scope it belongs to.
	"""
	from ..config.constants import DENIAL_SCOPE_DROP_AFTER
	return (
		denial_scope_count(execution_context, scope) >= DENIAL_SCOPE_DROP_AFTER
		or denial_stage(execution_context, scope) == STAGE_HANDBACK
	)


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


# ── How the loop ended ────────────────────────────────────────────────────────
# Computed once, where the loop exits, instead of being inferred downstream from
# budgets. A retry budget with room left means "the loop would try again" only while
# the loop is running; read from the completion report it became a promise nobody was
# ever going to keep ("Checks failing (will retry)" on the last line of a finished
# run). The report below therefore speaks only in the past tense, and this says why it
# stopped speaking at all.
TERMINATION_ANSWERED: str = "answered"        # the model produced a final answer
TERMINATION_STEP_LIMIT: str = "step_limit"   # the step budget ran out
TERMINATION_USER_STOPPED: str = "user_stopped"  # the user declined to continue

_TERMINATION_ISSUE: dict[str, str] = {
	TERMINATION_STEP_LIMIT: "Stopped: the step budget ran out before the work was finished.",
	TERMINATION_USER_STOPPED: "Stopped: you declined to continue at the step checkpoint.",
}


def _run_label(run: dict) -> str:
	""""Build" or "Run" — what hit the wall, in the user's vocabulary."""
	return "Build" if run.get("effect") == "build" else "Run"


def _collect_completion_issues(
	execution_context: dict, termination: str = TERMINATION_ANSWERED,
) -> tuple[list[str], list[str]]:
	"""Return (issues, completed) describing the final completion state."""
	issues: list[str] = []
	completed: list[str] = []

	termination_issue = _TERMINATION_ISSUE.get(termination)
	if termination_issue:
		issues.append(termination_issue)

	pending = pending_validation_paths(execution_context)
	fail_counts: dict = execution_context.get("validation_fail_count_by_file", {})
	dirty: set = execution_context.get("dirty_written_files", set())

	if pending:
		# Split pending files into three sub-buckets for clearer feedback.
		stuck_paths = [p for p in pending if fail_counts.get(p, 0) >= VALIDATION_RETRY_BUDGET]
		retry_paths = [p for p in pending if fail_counts.get(p, 0) > 0 and fail_counts.get(p, 0) < VALIDATION_RETRY_BUDGET]
		fresh_paths = [p for p in pending if fail_counts.get(p, 0) == 0]
		if stuck_paths:
			issues.append("Checks failing, budget exhausted: " + ", ".join(stuck_paths[:5]))
		if retry_paths:
			issues.append("Checks failing, unresolved: " + ", ".join(retry_paths[:5]))
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

	# `workflow_state` is deliberately NOT reported: it is the loop's own steering
	# variable, and "still in 'edit' state" names no gap the user can act on. Every
	# real one — a file unchecked, a run failing or unjudged, a step never started —
	# has its own line here.

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
	# A blocked run is red, and stays reported as red — but by a wall this environment put
	# there, not by the change. Charging it here is what turned "I tried the recommended
	# build and gcc is not installed" into "Task is incomplete."; see blocked_run_lines.
	broken = [(c, r) for c, r in sorted(runs.items())
	          if (not r.get("completed") or r.get("verdict") == "fail")
	          and not r.get("blocked")]
	# never_judged / unresolved are reported by unjudged_run_lines, not charged here: a
	# verdict is a recommendation, and charging its absence made the model spend turns
	# producing a label to clear the headline rather than because it had read something.
	# A run that actually *failed* stays an issue below — that is a machine fact.
	for command, run in broken[:5]:
		spent = int(run.get("failures", 0)) >= VALIDATION_RETRY_BUDGET
		label = "budget exhausted" if spent else "unresolved"
		issues.append(f"{_run_label(run)} failing, {label}: {command}")
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

	if (
		not issues
		and not refused_action_lines(execution_context)
		and not blocked_run_lines(execution_context)
		and not unjudged_run_lines(execution_context)
	):
		issues.append("Unknown blocker; explicit completion criteria were not met")

	return issues, completed


def unjudged_run_lines(execution_context: dict) -> list[str]:
	"""One line per run whose output was never read out, for the report.

	Same standing as :func:`blocked_run_lines`: reported, never counted. A verdict says
	what a run showed, and asking for one is a recommendation — so a missing or `unknown`
	verdict is a gap in the record, not a defect in the change. Charging it here is what
	taught the model to emit a label for every command it issued.
	"""
	runs = execution_context.get("runs") or {}
	lines = [
		f"{command} — ran; its output was never judged"
		for command, run in sorted(runs.items())
		if run.get("completed") and not run.get("verdict")
	]
	lines += [
		f"{command} — judged inconclusive"
		for command, run in sorted(runs.items()) if run.get("verdict") == "unknown"
	]
	return lines[:5]


def blocked_run_lines(execution_context: dict) -> list[str]:
	"""One line per run that hit a wall this environment put there, for the report.

	Reported, never counted: a prerequisite that is missing is a limitation the user has to
	see, and never a defect in the change to be charged against completion. Kept out of
	``_collect_completion_issues`` for that reason alone — it is what decides the headline.
	"""
	runs = execution_context.get("runs") or {}
	return [
		f"{_run_label(run)} not attempted — {run['blocked']}: {command}"
		for command, run in sorted(runs.items()) if run.get("blocked")
	][:5]


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
	answer: str, execution_context: dict, termination: str = TERMINATION_ANSWERED,
) -> str:
	issues, completed = _collect_completion_issues(execution_context, termination)
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
	elif refused and not issues and termination == TERMINATION_ANSWERED:
		headline = HEADLINE_REFUSED_ONLY
	else:
		headline = HEADLINE_INCOMPLETE

	summary = headline + "\n\nCompleted:\n- " + "\n- ".join(completed)
	blocked = blocked_run_lines(execution_context)
	if blocked:
		summary += (
			"\n\nNot attempted (a prerequisite this environment does not have):\n- "
			+ "\n- ".join(blocked)
		)
	unjudged = unjudged_run_lines(execution_context)
	if unjudged:
		summary += "\n\nRan, with no verdict on record:\n- " + "\n- ".join(unjudged)
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
	#
	# A run left red is one too: the ledger already prints it as an issue, and the level
	# printed underneath used to contradict it — "Run failing, unresolved" above
	# "Residual risk: low" was observed in the wild. `failed_runs` excludes blocked runs,
	# which keeps a missing toolchain a limitation rather than a defect of the change.
	# Runs merely unjudged are deliberately NOT charged here: a verdict is a
	# recommendation, and charging its absence is the trade POLICY already refused.
	_failed_runs = failed_runs(execution_context)
	risk_level = (
		"high" if _pending_code or handback
		else ("medium" if pending or _unwritten or denied_calls or _failed_runs else "low")
	)
	summary += f"\n\nResidual risk: {risk_level}."
	if answer.strip():
		# Named as a claim, not as the answer: everything above is machine-recorded, and
		# what follows is what the model says about the same run.
		summary += "\n\nWhat the model claims:\n" + answer.strip()
	return summary

# --- Loop-control nudge copy -----------------------------------------------
# Message bodies for the nudges fired directly from query_engine.agent_loop
# (plan-mode control flow, the step-limit reminder, and the mid-tool-loop repeat
# correctives). As with the guidance-nudge copy above, only the wording lives
# here; the *when-to-fire* gating stays in agent_loop. Kept here so all nudge copy
# sits in one module regardless of which loop fires it.

# Plan-mode user nudges.
# Explore phase: the plan-document tool is withheld until the code has actually been
# read, so this nudge asks for the exploration rather than for the plan. Nothing here
# forbids a plan-to-explore — the tool list does, by not offering the document yet.
PLAN_EXPLORE_FIRST = (
	"The plan tool is not available yet: this phase of plan mode is the exploration itself, and the plan is written once it is done. "
	"Use the read-only exploration tools now (search, read, inspect, platform/memory queries) to READ the code this task touches — not just to locate it. "
	"Listing directories and matching file names tells you nothing about what has to change; open the files, follow the symbols, and find the boundaries the work runs into. "
	"Keep going until you could name the concrete files and symbols the change touches and say what happens to each. "
)
# Appended to the above when a delegation capability is connected. Kept separate so the
# reminder stays true when it is not: the fan-out is an instruction only where it can be
# carried out.
PLAN_EXPLORE_DELEGATE = (
	"This exploration parallelises: rather than reading every area yourself, split it into one to three self-contained questions and send them out as read-only sub-agents in the SAME response, so they run at once. "
	"Each returns a conclusion naming the files and symbols it found; you then read for yourself only the places those answers point at. "
)
# Escape hatch: the explore-turn budget is spent and the evidence is still thin. The
# tool is unlocked regardless — plan mode must always reach a plan — so the plan is
# written over what was found, with its gaps stated rather than papered over.
PLAN_EXPLORE_BUDGET_SPENT = (
	"You have spent the exploration budget for this task, so the plan tool is now available whatever you found. "
	"Record the plan over the evidence you actually gathered, and state plainly in it which parts rest on assumptions you could not verify and what would settle them. "
)
PLAN_TODO_NUDGE_EARLY = (
	"You have not yet recorded a plan, which is mandatory to produce a valid plan output. "
	"Record it with the plan tool as a written plan document (structured markdown prose — an overview, the approach broken into its main axes with the reasoning and the concrete files/symbols each touches, key decisions/risks, and how you'll validate). "
	"The exploration is already done and is not part of the plan: write the plan over what you found. "
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


def evidence_handback_message(execution_context: dict) -> str:
	"""Hand the model the machine-recorded state before it writes its conclusion.

	The gap this closes: the completion report is assembled *after* the model has
	written its summary and appended below it, so the model has never seen the ledger
	it is contradicting. "Successfully implemented, complete and correct" printed above
	"Modified files never checked" is not a rhetoric problem to police in the prose —
	it is a missing fact at the moment the prose is written. Sent once per query; if the
	corrected summary still contradicts the record, the report below states the record
	and the two stand side by side.
	"""
	issues, _ = _collect_completion_issues(execution_context)
	return (
		"Before you conclude, here is what this run actually recorded:\n- "
		+ "\n- ".join(issues)
		+ "\n\nThese are machine-recorded facts, not an opinion about your work. Rewrite your "
		"summary so it matches them: say what was done, name each gap above as unfinished or "
		"unverified, and do not describe as complete or correct anything the record does not "
		"support. If you can still close one of these gaps with a tool call, do that instead."
	)
# The plan-grounding check no longer lives here as an after-the-fact advisory: plan
# mode withholds the document tool until the evidence bar is met (see plan_loop), so a
# plan written over nothing is unreachable rather than flagged. PLAN_EXPLORE_FIRST and
# PLAN_EXPLORE_BUDGET_SPENT above are what remains of it.

# Plan-approval workflow nudges. After the user reviews a proposed plan they may
# accept it (switch to agent mode and execute), reject it (drop it and stop), ask for a
# rework (re-plan from scratch), or describe specific changes in free text (fold them
# in, then re-present for approval).
PLAN_APPROVED_EXECUTE = (
	"The user has APPROVED your plan. You are now switching to agent mode to carry it out. "
	"The exploration behind this plan is already done and its findings are in the plan: do not survey the "
	"code again from scratch — re-read a specific file only when you are about to change it. "
	"No task checklist exists yet: before starting the work, record the ordered implementation/validation "
	"steps of the approved plan with the plan/todo tool, unless the task is small enough not to warrant one. "
	"The steps must be the key actions and validations the task needs (implementations, tests, validations, "
	"etc.); omit discovery, analysis, and information-gathering steps. "
	"Then carry out the approved plan, making the necessary edits, running the relevant validations, "
	"and marking each step done with the plan/todo update tool as you complete it. "
	"The plan is what you agreed with the user, not a script: where the code turns out differently than "
	"it assumed, adapt — rewrite the checklist, drop a step that proved unnecessary, add one it missed — "
	"and say what you changed and why in your final answer. Ask for re-plan or re-approval when the "
	"difference is large enough to change what the user approved."
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