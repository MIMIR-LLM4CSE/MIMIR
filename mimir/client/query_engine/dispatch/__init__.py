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
from ...tools.capabilities import CODE_EXEC, DIVERTIBLE, has_cap, kind_for, label_for
from ...context.execution_context import ExecutionContext
from ...core.ports import AgentPort
from ...event_sink import emit
from ...tools.exec_preview import exec_input_preview
from ...tools.display import (
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
from .inject import post_dispatch_inject

logger = logging.getLogger(__name__)

__all__ = ["ToolDispatcher", "dispatch_tool_calls", "post_dispatch_inject"]


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
        # Label: the declared template first, the status map as fallback. Paths shortened to
        # their file name (approval prompts keep them whole).
        row_args = shorten_display_args(display_name, display_args, self.agent.tool_caps)
        row_label = (label_for(display_name, row_args, self.agent.tool_caps)
                     or tool_status_message(display_name, row_args))
        # The salient argument, deduplicated against the text the row will show beside it: the
        # model's description if it wrote one, else the label.
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
            # Whether the row may offer to detach the run: DIVERTIBLE (it publishes a channel while
            # it blocks), read off the registry.
            "divertible": has_cap(display_name, DIVERTIBLE, self.agent.tool_caps),
        }
        # The IN half of the terminal panel, sent now so the command shows while it runs. Gated
        # on CODE_EXEC: there is no payload yet to recognise an exec by.
        if has_cap(display_name, CODE_EXEC, self.agent.tool_caps):
            pending = exec_input_preview(call.args)
            if pending is not None:
                call_event["exec"] = pending
        emit(call_event)

    async def execute(self, calls: list[Call]) -> list[str]:
        """Every call's result, in call order: reads concurrently, then writes one at a time."""
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
        lc = self.ec.bookkeeping.loop_control
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
            # corrective is staged (consumed by post_dispatch_inject).
            lc.call_fails[key] = lc.call_fails.get(key, 0) + 1
            if lc.call_fails[key] >= SOFT_REPEAT_THRESHOLD and key not in lc.repeat_warned:
                lc.repeat_warned.add(key)
                self.ec.nudges.repeat_alert = (call.name, lc.call_fails[key])

    def append_results(self, admission: Any, results: list[str]) -> None:
        """One tool message per call, in order, then the calls not run; records each message's files by id."""
        tool_msg_files: dict = self.ec.bookkeeping.tool_msg_files
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


async def dispatch_tool_calls(tool_calls: list, agent: AgentPort, messages: list[dict],
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
    """Pin a successful skill pull's result in the tail, or release the reserved slot.

    The tail rather than messages[0], which a query may not rewrite without losing the
    prefix cache; history passes honour the pin (``pinned_call_ids``).
    """
    if name != SKILL_LOAD_TOOL:
        return
    skill = str((args or {}).get("name") or "").strip()
    if not skill:
        return
    loaded: dict = execution_context.bookkeeping.skills_loaded
    succeeded, _summary = summarize_tool_result(name, result, agent.tool_caps)
    if not succeeded:
        if loaded.get(skill) == "pending":
            del loaded[skill]
        return
    loaded[skill] = call_id
    execution_context.bookkeeping.pinned_call_ids.add(call_id)
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
