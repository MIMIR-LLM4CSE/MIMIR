// Tool rows: a call, its result, verdict and progress, a sub-agent's steps under it,
// the diff it made, and the approval it waits on.

import type { ChatMessage, ToolActivity } from "../../types";
import { insertAfterFamily, relabelOrigins, settleFamily } from "../../components/subAgentUtils";
import { type ChatHandlers } from "../chatState";
import { patchToolById } from "./toolRows";
import { appendStamped, flushLive, commitDraft } from "./transcript";

export const TOOL_HANDLERS: ChatHandlers = {
  tool_call: (state, action) => {
    // Tools are tracked in their own stream — independent of thinking.
    const activity: ToolActivity = {
      id: action.id,
      name: action.name,
      label: action.label,
      detail: action.detail,
      kind: action.kind,
      doing: action.doing,
      status: "running",
      divertible: action.divertible,
      // Present for an exec-shaped call: the command, with no output yet. It
      // opens the terminal panel on IN alone, and tool_result completes it.
      exec: action.exec,
      startedAt: Date.now(),
      seq: state.nextSeq,
    };
    // A tool ran → the next token opens a fresh streaming bubble.
    return {
      ...state,
      nextSeq: state.nextSeq + 1,
      liveToolCalls: [...state.liveToolCalls, activity],
      toolCallAfterToken: true,
    };
  },

  tool_result: (state, action) => {
    const { id, ok, summary, error, exec, math, target, duration_ms } = action;
    const patch = (t: ToolActivity): ToolActivity =>
      t.id === id
        ? {
            ...t,
            // A detached run has not succeeded — it has left. Its own ending
            // arrives later, on job_complete.
            status: exec?.running
              ? ("background" as const)
              : ok
              ? ("ok" as const)
              : ("error" as const),
            summary: exec?.running ? "moved to background" : summary,
            // Fall back to the (clipped) summary so a failed row is always
            // expandable, even when the server sent no full error body.
            error: ok ? undefined : error || summary || "The tool call failed.",
            // A result with no exec payload (a failure, or a tool that turned out
            // not to be exec-shaped) must not erase the IN pane the call opened:
            // the command that ran is the row's only trace of what was attempted.
            exec: exec ?? t.exec,
            math,
            // Only a success names a file worth opening.
            target: ok ? target : undefined,
            // The run is over, whatever it was last seen doing. Without this a
            // detached row keeps "building…" and its bar for the rest of the
            // session, describing a moment that has passed.
            phase: undefined,
            percent: undefined,
            durationMs: duration_ms,
          }
        : t;
    const now = Date.now();
    // A delegating call's result also settles whatever its sub-agent left running.
    return (
      patchToolById(state, id, (tools) => settleFamily(tools.map(patch), id, now)) ??
      state
    );
  },

  verdict: (state, action) => {
    // Lands on the row of the run it judges, which may already be frozen into a
    // "tools" message by the time the model gets around to judging it.
    const { id: vId, verdict } = action;
    return (
      patchToolById(state, vId, (tools) =>
        tools.map((t) => (t.id === vId ? { ...t, verdict } : t))
      ) ?? state
    );
  },

  tool_progress: (state, action) => {
    // What the run is doing, while it is still doing it. Only a running row can
    // be mid-anything: a settled one has an outcome, and relabelling it
    // "building…" would describe work that is over.
    const { id: pId, phase, percent } = action;
    return (
      patchToolById(state, pId, (tools) =>
        tools.map((t) =>
          t.id === pId && t.status === "running"
            ? { ...t, phase, percent: typeof percent === "number" ? percent : undefined }
            : t
        )
      ) ?? state
    );
  },

  tool_backgrounded: (state, action) => {
    // The row is done, the work is not. Marked here rather than inferred from the
    // result, because only the client knows whether a watcher actually took the
    // job — and a row that claimed to be tracked by one that declined would wait
    // for an ending nobody was going to report.
    const { id: bId, job_key } = action;
    return (
      patchToolById(state, bId, (tools) =>
        tools.map((t) =>
          t.id === bId
            ? { ...t, jobKey: job_key, status: "background" as const,
                summary: t.summary || "moved to background" }
            : t
        )
      ) ?? state
    );
  },

  diff: (state, action, makeId) => {
    const { file: dFile, patch: dPatch } = action;
    const isNewFile = !dPatch || dPatch.includes("/dev/null");
    // The emitter diffs against the turn's baseline snapshot, so each event
    // already carries the file's full change: coalesce by path instead of
    // stacking, or a file edited N times shows N near-identical cards.
    const pending = [...state.pendingDiffs];
    const pi = pending.findIndex((d) => d.file === dFile);
    if (pi >= 0) pending[pi] = { ...pending[pi], patch: dPatch };
    else pending.push({ file: dFile, patch: dPatch });
    const s = commitDraft({ ...state, pendingDiffs: pending }, makeId);
    const liveIdx = s.messages.findIndex((m) => m.kind === "editing" && m.live);
    let messages: ChatMessage[];
    if (liveIdx >= 0) {
      messages = s.messages.map((m, i) => {
        if (i !== liveIdx) return m;
        const diffs = [...(m.diffs ?? [])];
        const fi = diffs.findIndex((d) => d.file === dFile);
        if (fi >= 0) {
          // is_new is sticky: a file created then edited is still a creation.
          diffs[fi] = { ...diffs[fi], patch: dPatch, is_new: diffs[fi].is_new || isNewFile };
        } else {
          diffs.push({ file: dFile, patch: dPatch, is_new: isNewFile });
        }
        return { ...m, diffs };
      });
    } else {
      messages = appendStamped(s.messages, {
        id: makeId(),
        seq: s.nextSeq,
        role: "agent",
        kind: "editing",
        live: true,
        diffs: [{ file: dFile, patch: dPatch, is_new: isNewFile }],
      });
      return { ...s, nextSeq: s.nextSeq + 1, messages };
    }
    return { ...s, messages };
  },

  approval: (state, action, makeId) => {
    let s = flushLive({ ...state, toolCallAfterToken: false }, makeId);
    s = { ...s, busy: false, pendingDiffs: [] };
    const base = s.messages.map((m) =>
      m.kind === "editing" && m.live ? { ...m, live: false } : m
    );
    // The newest card of THIS conversation. Matching on kind alone would merge cards
    // from different conversations into one — two agents' ids in a single `ids[]`,
    // both then answered by one choice, one of them for a call the user never saw.
    let existingIdx = -1;
    for (let i = base.length - 1; i >= 0; i--) {
      const card = base[i].approval;
      if (base[i].kind === "approval" && card !== undefined &&
          (card.session_id ?? "") === (action.session_id ?? "")) { existingIdx = i; break; }
    }
    let messages: ChatMessage[];
    if (existingIdx >= 0) {
      const existing = base[existingIdx];
      const existingIds = existing.approval!.ids ?? [existing.approval!.id];
      messages = base.map((m, i) =>
        i === existingIdx
          // A card put back on return from another session can meet its own
          // saved copy: the same id twice would answer it twice.
          ? { ...m, approval: { ...action, ids: existingIds.includes(action.id) ? existingIds : [...existingIds, action.id] } }
          : m
      );
    } else {
      messages = [
        ...base,
        {
          id: makeId(),
          role: "agent",
          kind: "approval",
          approval: { ...action, ids: [action.id] },
        },
      ];
    }
    return { ...s, messages };
  },

  approval_response: (state, action) => {
    const { id, choice } = action;
    const messages = state.messages
      .map((m) => {
        if (m.kind === "approval" && m.approval?.id === id) {
          if (choice === "n") {
            const tool = m.approval?.tool ?? "";
            const isBash = /bash|shell|run/i.test(tool);
            const approvalArgs = m.approval?.args ?? {};
            const commandKey = Object.keys(approvalArgs).find((k) => /^(command|cmd|script)$/i.test(k))
              ?? Object.keys(approvalArgs).find((k) => /^(filepath|path|file)$/i.test(k));
            const firstArgValue = commandKey !== undefined
              ? approvalArgs[commandKey]
              : Object.values(approvalArgs).find((v) => typeof v === "string");
            const cmdText = typeof firstArgValue === "string"
              ? firstArgValue.split("\n")[0].trim().slice(0, 80)
              : null;
            const label = isBash && cmdText ? `✕ Blocked: \`${cmdText}\`` : "✕ Blocked";
            return { ...m, kind: "text" as const, text: label, approval: undefined };
          }
          // y / a → remove approval card; editing card stays live until write completes.
          return null;
        }
        // On deny: remove the live editing card too (nothing was written).
        if (choice === "n" && m.kind === "editing" && m.live) return null;
        return m;
      })
      .filter(Boolean) as ChatMessage[];
    return { ...state, messages, busy: true };
  },

  subagent_event: (state, action) => {
    // A sub-agent's step, shown as an ordinary tool row under the call that
    // spawned it — same spinner, same timer, same freezing. Only the indent and
    // the origin badge say it came from somewhere else.
    const parentId = action.parent_id;
    if (action.kind === "tool_call") {
      return (
        patchToolById(state, parentId, (tools) => {
          // The child inherits its parent's stamp: it is spliced in under the
          // call that spawned it, and a stamp of its own could put a reasoning
          // block that arrived meanwhile between the two, splitting the family
          // across two cards.
          const parentSeq = tools.find((t) => t.id === parentId)?.seq;
          const child: ToolActivity = {
            id: action.id ?? `${parentId}:?`,
            name: action.name ?? "",
            label: action.label ?? "",
            detail: action.detail ?? "",
            kind: action.tool_kind,
            doing: action.doing,
            status: "running",
            startedAt: Date.now(),
            parentId,
            seq: parentSeq,
          };
          // A step of its own supersedes the heartbeat: the row now says what
          // the child is actually doing.
          const cleared = tools.map((t) =>
            t.id === parentId ? { ...t, waiting: undefined } : t
          );
          return relabelOrigins(insertAfterFamily(cleared, parentId, child));
        }) ?? state
      );
    }
    if (action.kind === "tool_result") {
      const childId = action.id ?? "";
      return (
        patchToolById(state, childId, (tools) =>
          tools.map((t) =>
            t.id === childId
              ? {
                  ...t,
                  status: action.ok ? ("ok" as const) : ("error" as const),
                  summary: action.summary,
                  target: action.ok ? action.target ?? undefined : undefined,
                  durationMs: action.duration_ms,
                }
              : t
          )
        ) ?? state
      );
    }
    if (action.kind === "heartbeat") {
      // No row of its own: the child is thinking, and the only thing worth saying
      // is that its parent has not stalled. Cleared by the next child step.
      const secs = action.waiting_secs ?? 0;
      return (
        patchToolById(state, parentId, (tools) =>
          tools.map((t) =>
            t.id === parentId ? { ...t, waiting: `working — ${secs}s` } : t
          )
        ) ?? state
      );
    }
    // "end": how much of the child's activity was shed rather than shown.
    return (
      patchToolById(state, parentId, (tools) =>
        tools.map((t) =>
          t.id === parentId ? { ...t, childrenDropped: action.dropped } : t
        )
      ) ?? state
    );
  },
};
