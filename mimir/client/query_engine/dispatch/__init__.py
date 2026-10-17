"""Tool-call dispatch for one model step.

A step's calls go through the guards (:mod:`.admission`), are announced as rows, run
— reads in parallel, writes one at a time (:mod:`.execution`) — and come back as one
tool message per call, in the order the model made them. Between two steps, the
correctives (:mod:`.inject`) reach the model.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from typing import Any

from ...config.limits import IDENTICAL_REPEAT_THRESHOLD, SOFT_REPEAT_THRESHOLD
from ...config.servers import SKILL_LOAD_TOOL
from ...context.capabilities import CODE_EXEC, DIVERTIBLE, has_cap, kind_for, label_for
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from ...event_sink import emit
from ...tool_execution.exec_preview import exec_input_preview
from ...context.tool_display import (
    dedup_row_detail,
    shorten_display_args,
    summarize_tool_result,
    tool_arg_preview,
    tool_status_message,
)
from .. import streaming as _streaming
from ..history import bound_step_results
from .admission import Call, admit, is_write
from .execution import run_call
from .inject import _post_dispatch_inject

logger = logging.getLogger(__name__)

__all__ = ["ToolDispatcher", "_dispatch_tool_calls", "_post_dispatch_inject"]


class ToolDispatcher:
    """One step's calls, from what the model issued to the tool messages in history."""

    def __init__(self, agent: AgentPort, messages: list[dict], execution_context: ExecutionContext) -> None:
        self.agent = agent
        self.messages = messages
        self.ec = execution_context

    async def run(self, tool_calls: list) -> None:
        admission = admit(tool_calls, self.agent, self.ec)
        for call in admission.calls:
            self.announce(call)
        results = await self.execute(admission.calls)
        self.account_repeats(admission.calls, results)
        self.append_results(admission, results)

    def announce(self, call: Call) -> None:
        """Put the call's row on screen before it runs."""
        display_name, display_args = self.agent.rewrite_tool_for_context(call.name, call.args)
        # Label precedence: the capability-declared template (server-side
        # `label="Reading file: {path}"`) wins; the client status map is the
        # fallback for tools that declare none.
        # Paths are shortened to their file name for the row: tools carry absolute
        # paths now, and a row reading "Reading file: /shared/.../mimir/client/foo.py"
        # buries the only token the user is scanning for. Approval prompts keep the
        # absolute path — see tool_display._relpath.
        row_args = shorten_display_args(display_name, display_args, self.agent.tool_caps)
        row_label = (label_for(display_name, row_args, self.agent.tool_caps)
                     or tool_status_message(display_name, row_args))
        # The salient argument of the call — the url, the file, the job id, the verdict.
        # Deduplicated against whichever text the row will actually put beside it: the
        # model's description when it wrote one, the label when it did not. Against the
        # label alone it dropped previews the row needed (the basename under "Reading
        # file: x.py", with nothing else left to name the file); against the description
        # alone it kept ones the row then said twice — "Recording the plan: refonte des
        # lignes" followed by "refonte des lignes".
        row_detail = dedup_row_detail(
            call.doing or row_label,
            tool_arg_preview(display_name, row_args))
        call_event = {
            "type": "tool_call",
            "id": call.call_id,
            "name": display_name,
            "label": row_label,
            "detail": row_detail,
            # What the row shows: the tool's work family in place of the derived label,
            # and the model's sentence about this call beside it. The label stays on the
            # wire — approval cards, policy messages and the row's own tooltip read it —
            # but it is no longer what the row says.
            "kind": kind_for(display_name, self.agent.tool_caps),
            "doing": call.doing,
            # Whether the front-end may offer to detach this row while it runs. Read
            # off the registry, like every other row property: the UI must not learn
            # which tool happens to be a shell. DIVERTIBLE, not BACKGROUNDABLE: the
            # question is whether this call publishes a channel while it blocks, not
            # whether the tool can launch something detached. Read the second way, the
            # button also appeared on tools that return the instant they have
            # submitted — where the click had nothing to reach, and said so wrongly.
            "divertible": has_cap(display_name, DIVERTIBLE, self.agent.tool_caps),
        }
        # A run's input is known now; its output only exists when it ends. Sending the
        # IN half here puts the command on screen for the whole run instead of landing
        # the finished IN/OUT panel at the end, after a spinner that said nothing about
        # what was running. Gated on the registry's CODE_EXEC capability — the result
        # preview can detect an exec by the shape of its payload, but at call time
        # there is no payload yet, and a non-exec tool with a `command`-ish argument
        # would otherwise grow a terminal panel.
        if has_cap(display_name, CODE_EXEC, self.agent.tool_caps):
            pending = exec_input_preview(call.args)
            if pending is not None:
                call_event["exec"] = pending
        emit(call_event)

    async def execute(self, calls: list[Call]) -> list[str]:
        """Every call's result, in call order.

        Reads run concurrently. Writes must NOT: each does a snapshot → execute → diff
        sequence, so two writes to one file — or a write racing a read of it — would
        interleave and corrupt the captured diff, or the file. All reads in parallel,
        then the writes strictly in order. A refused call's payload is its result.
        """
        results = [call.refused or "" for call in calls]
        runnable = [i for i, call in enumerate(calls) if call.refused is None]
        reads = [i for i in runnable if not is_write(self.agent, calls[i].name)]
        if reads:
            done = await asyncio.gather(*(run_call(self.agent, calls[i], self.ec) for i in reads))
            for i, result in zip(reads, done):
                results[i] = result
        for i in runnable:
            if is_write(self.agent, calls[i].name):
                results[i] = await run_call(self.agent, calls[i], self.ec)
        return results

    def account_repeats(self, calls: list[Call], results: list[str]) -> None:
        """Count identical calls across steps: failures toward the guards, successes
        toward the IDENTICAL_REPEAT note. Writes have their own dedup; refused calls
        never ran."""
        lc = self.ec.loop_control
        for i, (call, result) in enumerate(zip(calls, results)):
            if call.refused is not None or is_write(self.agent, call.name):
                continue
            key = call.key
            ok, _summary = summarize_tool_result(call.name, result, self.agent.tool_caps)
            if ok:
                # The same call returning the same answer for the third time is spin, and
                # nothing later in the turn will notice it: the failing-call guard counts
                # only failures, and a nudge fires only once the model stops calling
                # tools — which a spinning model never does. Said as an annotation, never
                # a block: guards that withheld a repeated success only sent the model to
                # read the same thing another way.
                digest = hashlib.sha1(str(result).encode("utf-8", "replace")).hexdigest()
                seen, count = lc.call_results.get(key, ("", 0))
                count = count + 1 if seen == digest else 1
                lc.call_results[key] = (digest, count)
                if count >= IDENTICAL_REPEAT_THRESHOLD and key not in lc.repeat_noted:
                    lc.repeat_noted.add(key)
                    results[i] = str(result) + (
                        f"\n\nIDENTICAL_REPEAT: this call has returned exactly this result "
                        f"{count} times in this task. It will not return anything else — "
                        f"decide with what you already have, or ask a different question."
                    )
                continue
            # The first time a failing call crosses the soft threshold, a one-time
            # corrective is staged (consumed by _post_dispatch_inject).
            lc.call_fails[key] = lc.call_fails.get(key, 0) + 1
            if lc.call_fails[key] >= SOFT_REPEAT_THRESHOLD and key not in lc.repeat_warned:
                lc.repeat_warned.add(key)
                self.ec.repeat_alert = (call.name, lc.call_fails[key])

    def append_results(self, admission: Any, results: list[str]) -> None:
        """One tool message per call, in call order, then the calls not run.

        Which files each message concerns is recorded by call id, so the history trim
        can match messages to files structurally instead of by substring scanning. An
        empty list is meaningful: a grep result touches no tracked file, so evicting it
        must not invalidate a read just because a path appears in its text.
        """
        tool_msg_files: dict = self.ec.tool_msg_files
        # Bounded BEFORE it is appended: the trim pass exempts the current step, so
        # afterwards only the force-fit backstop could touch it, once already over.
        results = _bound_results(results, self.agent)
        for call, result in zip(admission.calls, results):
            try:
                tool_msg_files[call.call_id] = self.agent.get_tool_file_targets(call.name, call.args)
            except Exception:
                tool_msg_files[call.call_id] = []
            _pin_loaded_skill(self.agent, call.name, call.args, call.call_id, result, self.ec)
            self.messages.append({"role": "tool", "tool_call_id": call.call_id, "content": result})
        for call_id, why in admission.skipped:
            tool_msg_files[call_id] = []
            self.messages.append({"role": "tool", "tool_call_id": call_id,
                                  "content": json.dumps({"status": "skipped", "note": why})})


async def _dispatch_tool_calls(tool_calls: list, agent: AgentPort, messages: list[dict],
                               execution_context: ExecutionContext) -> None:
    """Run one step's calls (see :class:`ToolDispatcher`)."""
    await ToolDispatcher(agent, messages, execution_context).run(tool_calls)


def _pin_loaded_skill(
    agent: AgentPort,
    name: str,
    args: dict,
    call_id: str,
    result: str,
    execution_context: ExecutionContext,
) -> None:
    """Promote a successful skill pull from reservation to pinned tail message.

    The body stays where it landed — a tool result in the tail — and is protected
    there instead of being folded into ``messages[0]``: the system prompt is the one
    thing a query may not rewrite mid-run without costing the prefix cache for every
    remaining step. What makes the tail safe is the pin, honoured by the three history
    passes (see ``pinned_call_ids`` in history.py).

    A refused pull releases the slot the gate reserved. Otherwise a mistyped name, or
    a skill the file itself declares user-invoked only, would spend one of the three
    the model is allowed — a refusal charging the budget of the thing it refused.
    """
    if name != SKILL_LOAD_TOOL:
        return
    skill = str((args or {}).get("name") or "").strip()
    if not skill:
        return
    loaded: dict = execution_context.skills_loaded
    succeeded, _summary = summarize_tool_result(name, result, agent.tool_caps)
    if not succeeded:
        if loaded.get(skill) == "pending":
            del loaded[skill]
        return
    loaded[skill] = call_id
    execution_context.pinned_call_ids.add(call_id)
    emit({"type": "status", "text": f"  \u2295 Skill loaded: {skill}"})


def _bound_results(results: list[str], agent: AgentPort) -> list[str]:
    """Apply the per-result and per-step context budgets, and say so when they bite.

    Best effort: a backend that cannot count tokens, or a budget that cannot be
    resolved, must not cost the step its results. The history's own passes still run
    afterwards, so a failure here degrades to the behaviour that came before it.
    """
    try:
        backend = _streaming.get_backend()
        bounded, cut = bound_step_results(
            results,
            model=agent.model,
            context_mode=getattr(agent, "context_mode", "full"),
            token_counter=lambda text: backend.count_text_tokens(agent.model, text),
            context_ceiling=int(getattr(agent, "max_context_tokens", 0) or 0) or None,
        )
    except Exception:  # noqa: BLE001 — a budget failure must not drop the evidence
        logger.debug("_bound_results: a best-effort step failed and was skipped", exc_info=True)
        return results
    for before, after in cut:
        emit({"type": "status", "text": (
            f"  \u2702 Context: a tool result of {before:,} tokens was cut to "
            f"{after:,} to fit the step budget."
        )})
    return bounded
