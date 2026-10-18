"""Every text the guardrails send the model: the nudges of the cascade, and the
loop-control messages (plan mode, the step limit, empty turns, repeat correctives).

When each one is sent is decided elsewhere; only the wording lives here.
"""
from __future__ import annotations


from ..builtin_check import builtin_check_failures
from ..workflow import (
    collect_completion_issues,
    pending_validation_paths,
    unhonoured_commitments,
    worst_denial_stage,
    STAGE_DROP_OR_STOP,
    STAGE_HANDBACK,
)
from ...config.limits import STUCK_REPAIR_CONSTRAIN_AFTER, VALIDATION_RETRY_BUDGET
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort


# A refusal is an instruction, not an error to report and retry. The three readings
# below are stated in priority order and mirror the tool-result hint the model got
# mid-loop (agent_core.denied_tool_result) and the Non-negotiables line in the system
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


def _validation_nudge_message(findings: dict[str, str]) -> str:
    """Report what the built-in check found, file by file.

    The check runs in-process over every modified file before the loop may conclude, so
    this message is the result, and asks only for the repair. It names no tool and no
    command: nothing outside had to be installed to find the defect.
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


def denial_nudge_message(context: ExecutionContext) -> str:
    stage = worst_denial_stage(context)

    if stage == STAGE_HANDBACK:
        return _DENIAL_HANDBACK
    if stage == STAGE_DROP_OR_STOP:
        return _DENIAL_DROP_OR_STOP

    denied_calls = context.denials.denied_tool_calls
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
# The wording of the guidance nudges; when they fire is ``predicates``' business.

def error_recovery_nudge_message(failing_path: str) -> str:
    return (
        f"Repeated edit failures detected on '{failing_path}'. "
        "A failed anchor returns the current text of the site in `anchor_excerpt` — "
        "copy it from there rather than re-reading the file, then apply a smaller, "
        "differently-anchored patch instead of repeating the same edit."
    )


def stuck_repair_nudge_message(streak: int) -> str:
    """The two rungs of the stuck-repair ladder, chosen by the streak.

    They constrain the class of the next action, never its content: only the model knows
    the code.
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
    """Everything written is checked and none of it ever ran; *route* is the command found.

    Says what the checks established first, and accepts "nothing to run" as an answer.
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


def env_resolution_nudge_message(execution_context: ExecutionContext) -> str:
    modules = sorted(m for m in execution_context.env.unresolved_modules if m)
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


def env_cleanup_nudge_message(execution_context: ExecutionContext) -> str:
    mutations = execution_context.env.env_mutations or []
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


def unwatched_run_nudge_message(execution_context: ExecutionContext) -> str:
    ids = sorted(execution_context.execution.unwatched_runs or set())
    named = ", ".join(ids[:6]) + (f" (+{len(ids) - 6} more)" if len(ids) > 6 else "")
    plural = "s" if len(ids) != 1 else ""
    return (
        f"Before you conclude: the scheduler lists {len(ids)} job{plural} of yours that "
        f"nothing is watching — {named}.\n\n"
        "A job is tracked by the record its submission wrote, and these have none: they "
        "were not submitted through this server. So nothing polls them, nobody is woken "
        "when they end, and this process does not stay alive for them — it stops on its "
        "idle timer with your run still going, and the result reaches no one.\n\n"
        "Adopt each one now — slurm_watch(job_id, note=\"what it is\") for a Slurm job, "
        "bash_watch(pid) for a process on this host — then end your turn: you are resumed "
        "when they finish. If one of them is deliberately not yours to follow (another "
        "user's work, a job you only looked at), say so in your answer and leave it."
    )


def documentation_nudge_message(execution_context: ExecutionContext) -> str:
    written = sorted(execution_context.edits.dirty_written_files)
    file_hint = f" You modified: {', '.join(written[:3])}." if written else ""
    return (
        "Before concluding, quickly check whether any documentation or reference file "
        "(README, *.md, catalogs, server indexes) should be updated to reflect the code change."
        + file_hint
        + " If nothing relevant exists, you can ignore this."
    )


def blast_radius_nudge_message(execution_context: ExecutionContext) -> str:
    read_files = sorted(execution_context.discovery.read_files)
    file_hint = f" You have read: {', '.join(read_files[:3])}." if read_files else ""
    return (
        "Before changing a function/class definition or renaming a symbol, "
        "it may help to quickly search for callers or import sites first."
        + file_hint
        + " If the change is purely local and clearly contained, you can proceed."
    )


def creation_nudge_message(execution_context: ExecutionContext) -> str:
    targets = sorted(execution_context.edits.planned_edit_targets)
    target_hint = f" Declared target(s): {', '.join(targets[:3])}." if targets else ""
    return (
        "A write target was declared earlier in this task but nothing has been "
        "written yet, and you have gathered context since."
        + target_hint
        + " If that work is still intended, write it now and validate it before "
        "concluding. If the request only called for an answer — or you have already "
        "given one — conclude without writing anything."
    )


def todo_nudge_message(execution_context: ExecutionContext) -> str:
    files = sorted(
        execution_context.edits.dirty_written_files
        | execution_context.edits.planned_edit_targets
    )
    file_hint = f" Files touched so far: {', '.join(files[:4])}." if files else ""
    if execution_context.plan.plan_written:
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


def validation_nudge_message(agent: AgentPort, execution_context: ExecutionContext) -> str:
    """What the model is told about the files the built-in check rejected.

    The loop already ran the check: the message is the finding.
    """
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

    fail_counts = execution_context.checks.validation_fail_count_by_file
    hottest_path = max(findings, key=lambda path: int(fail_counts.get(path, 0)))
    hottest_count = int(fail_counts.get(hottest_path, 0))
    if hottest_count < 2:
        return base_message

    suffix = f" The check has now failed {hottest_count} time(s) for {hottest_path}."
    last_replace_file = execution_context.edits.last_replace_file
    last_replace_old_text = execution_context.edits.last_replace_old_text
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


def evidence_handback_message(execution_context: ExecutionContext) -> str:
    """The machine-recorded state, handed to the model before it writes its conclusion.

    The completion report comes after the summary, so without this the model never sees
    the ledger it may contradict. Sent once per query.
    """
    issues, _ = collect_completion_issues(execution_context)
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


# The half both wordings share: what happened. Public because it is the stable way
# to recognise this reminder among others without pinning the advice half.
EMPTY_TURN_OPENING = (
    "Your last turn came back empty — no text and no tool call — so nothing was done and "
    "nothing was said. Pick the work back up where it stood: "
)


# What the reminder adds on the second and third attempt. A retry that re-sends the
# same sentence is not a retry: the loop asked three times with one prompt, and the
# reminder is transient, so the model saw the identical message each round. Escalating
# the *ask* — narrower each time — is what makes the attempts different, and it does it
# without keeping the reminder in history (which is what once put 21 copies of one
# sentence in a single prompt).
_EMPTY_TURN_ESCALATION = (
    " The previous retry also came back empty. Emit a single tool call — the smallest "
    "useful one — rather than a turn of prose.",
    " Two retries have now come back empty. Do not call a tool this turn: write one "
    "plain sentence saying where the work stands.",
)


def empty_turn_retry_message(
    execution_context: ExecutionContext | None = None, *, attempt: int = 1,
) -> str:
    """What to say after a turn that produced nothing, given what the turn still owes.

    "Or write the answer" is offered only when nothing is outstanding. *attempt* (1-based)
    widens the advice.
    """
    if execution_context is not None and unhonoured_commitments(execution_context):
        base = (
            EMPTY_TURN_OPENING
            + "call the tool the next unfinished step needs. Your checklist still has "
            "open steps and files you named are not written, so this is not a turn to "
            "summarise — do the next piece of the work."
        )
    else:
        base = EMPTY_TURN_OPENING + "call the tool the task needs next, or write the answer."
    if attempt <= 1:
        return base
    return base + _EMPTY_TURN_ESCALATION[min(attempt, len(_EMPTY_TURN_ESCALATION) + 1) - 2]


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


def moving_test_corrective_message(test_path: str) -> str:
    """One-time reminder when a test file's failing set keeps changing.

    A failure that moves between runs is usually the test's own oracle being wrong: read
    the test whole and say which side is wrong before touching either.
    """
    return (
        "[automated workflow reminder — not from the user; advisory, apply judgment]\n\n"
        f"The set of failing tests in {test_path} has changed between runs — you are fixing "
        "a different failure each time rather than the same one. That usually means the "
        "test's own oracle is wrong, not the code it exercises. Before editing this file "
        "again, read it in full and check the oracle itself against the implementation: the "
        "expected value and where it comes from, the indexing and strides, the units and "
        "scaling, and the direction of any ratio or comparison. Then state which of the two "
        "— the test or the code — is wrong, and change only that one."
    )


def handback_corrective_message(refusals: list[dict]) -> str:
    """Stop: refusals reached the end of the denial ladder. Lists what was refused."""
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
