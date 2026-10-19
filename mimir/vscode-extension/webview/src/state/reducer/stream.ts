// What streams in during a step: prose, reasoning, transient status, command
// replies and the edit card a write opens.

import type { ChatMessage, DiffEntry } from "../../types";
import { type ChatHandlers, type ChatState } from "../chatState";
import { appendStamped, flushLive, commitDraft } from "./transcript";

/**
 * Transient status and tool output: dropped (decorrelated from both thinking and
 * tools), but noted, so the next token opens a fresh bubble. Most file-operation
 * statuses do not even count: diffs and approvals drive that part of the UI.
 */
function noteActivity(state: ChatState, text: string): ChatState {
  const t = text.trim().toLowerCase();
  if (
    !t ||
    t.startsWith("✓") ||
    t.startsWith("performing tool") ||
    (t.startsWith("reading") && !/^reading (?:lines? \d|from line \d)/.test(t))
  ) {
    return state;
  }
  return { ...state, toolCallAfterToken: true };
}

export const STREAM_HANDLERS: ChatHandlers = {
  token: (state, action, makeId) => {
    const delta = action.text;
    const isNewLLMStep = state.toolCallAfterToken;
    let s: ChatState = { ...state, toolCallAfterToken: false };
    if (isNewLLMStep) {
      // Only a step that actually produced cards is a boundary worth breaking
      // the prose at: prose written before this step's tools belongs above
      // them, so it is committed first and the cards are frozen in under it.
      // A bare status flagged activity that has nothing to show, and splitting
      // one turn's prose over it would fragment it for nothing.
      const hasCards =
        s.liveToolCalls.length > 0 ||
        s.liveThinkingBlocks.some((b) => b.text.trim().length > 0);
      if (hasCards) {
        s = flushLive(s, makeId);
      } else if (s.draft && !/\s$/.test(s.draft) && !/^\s/.test(delta)) {
        s = { ...s, draft: s.draft + "\n\n" };
      }
    }
    // The stamp of the first token is what later places this prose among the
    // step's cards; it is cleared with the draft, so a flush re-stamps.
    if (s.draftSeq === null) s = { ...s, draftSeq: s.nextSeq, nextSeq: s.nextSeq + 1 };
    return { ...s, draft: s.draft + delta };
  },

  thinking_start: (state, _action, makeId) => {
    const s = { ...state, toolCallAfterToken: true };
    const last = s.liveThinkingBlocks[s.liveThinkingBlocks.length - 1];
    if (last && !last.done) return s; // reuse active block
    return {
      ...s,
      nextSeq: s.nextSeq + 1,
      liveThinkingBlocks: [
        ...s.liveThinkingBlocks,
        { id: makeId(), text: "", done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq },
      ],
    };
  },

  thinking: (state, action, makeId) => {
    const thinkChunk = action.text;
    const s = { ...state, pendingThinking: state.pendingThinking + thinkChunk };
    if (s.liveThinkingBlocks.length === 0) {
      return {
        ...s,
        nextSeq: s.nextSeq + 1,
        liveThinkingBlocks: [{ id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq }],
      };
    }
    const lastIdx = s.liveThinkingBlocks.length - 1;
    const last = s.liveThinkingBlocks[lastIdx];
    if (!last.done) {
      return {
        ...s,
        liveThinkingBlocks: s.liveThinkingBlocks.map((b, i) =>
          i === lastIdx
            ? { ...b, text: b.text + thinkChunk }
            : b
        ),
      };
    }
    return {
      ...s,
      nextSeq: s.nextSeq + 1,
      liveThinkingBlocks: [
        ...s.liveThinkingBlocks,
        { id: makeId(), text: thinkChunk, done: false, collapsed: false, startedAt: Date.now(), seq: s.nextSeq },
      ],
    };
  },

  thinking_end: (state, action) => {
    // Block stays live to accumulate tool calls; closed by next start/answer.
    // Its reported size is banked onto the block so freezing carries it over.
    const tokens = action.tokens;
    const lastIdx = state.liveThinkingBlocks.length - 1;
    if (!tokens || lastIdx < 0) return state;
    return {
      ...state,
      liveThinkingBlocks: state.liveThinkingBlocks.map((b, i) =>
        i === lastIdx ? { ...b, tokens: (b.tokens ?? 0) + tokens } : b
      ),
    };
  },

  toggle_thinking: (state, action) => {
    const { id } = action;
    let liveThinkingBlocks = state.liveThinkingBlocks;
    if (liveThinkingBlocks.some((b) => b.id === id)) {
      liveThinkingBlocks = liveThinkingBlocks.map((b) =>
        b.id === id ? { ...b, collapsed: !b.collapsed } : b
      );
    }
    // 'live' doubles as the "expanded" flag for frozen thinking messages.
    const messages = state.messages.map((m) =>
      m.id === id && m.kind === "thinking" ? { ...m, live: !m.live } : m
    );
    return { ...state, liveThinkingBlocks, messages };
  },

  output: (state, action) => noteActivity(state, action.text),
  status: (state, action) => noteActivity(state, action.text),

  // The answer to a command the user typed. Unlike "output" — transient tool
  // chatter this reducer deliberately drops — it is rendered, because dropping it
  // made "/memory list" print nothing and "/memory clear" wipe the store in
  // silence. Does not touch `busy`: a session command runs beside a turn, not
  // as one.
  command_output: (state, action, makeId) => {
    const title = (action.title ?? "").trim();
    if (!title) return state;
    const s = commitDraft(state, makeId);
    return {
      ...s,
      nextSeq: s.nextSeq + 1,
      messages: appendStamped(s.messages, {
          id: makeId(),
          seq: s.nextSeq,
          role: "agent",
          kind: "command",
          // Carried whole rather than flattened to a line: the card decides how
          // a setting change and a twenty-row listing each want to look.
          command: {
            type: "command_output",
            command: action.command ?? "",
            title,
            items: (action.items ?? []).filter((i) => (i?.label ?? "").trim()),
            note: (action.note ?? "").trim(),
            tone: action.tone ?? "ok",
            transient: action.transient ?? false,
          },
      }),
    };
  },

  file_progress: (state, action, makeId) => {
    const progressDiffs: DiffEntry[] = action.diffs ?? [];
    if (progressDiffs.length === 0) return state;
    const s0 = commitDraft(state, makeId);
    const liveIdx = s0.messages.findIndex((m) => m.kind === "editing" && m.live);
    let messages: ChatMessage[];
    if (liveIdx >= 0) {
      messages = s0.messages.map((m, i) => {
        if (i !== liveIdx) return m;
        const existing = m.diffs ?? [];
        const merged = [...existing];
        for (const d of progressDiffs) {
          const fi = merged.findIndex((e) => e.file === d.file);
          if (fi >= 0) merged[fi] = d;
          else merged.push(d);
        }
        return { ...m, diffs: merged };
      });
    } else {
      // Stamped: the write that produced these diffs is a tool row of the step
      // in flight, and the card belongs under it rather than above the step.
      messages = appendStamped(s0.messages, {
        id: makeId(), seq: s0.nextSeq,
        role: "agent", kind: "editing", live: true, diffs: progressDiffs,
      });
      return { ...s0, nextSeq: s0.nextSeq + 1, messages };
    }
    return { ...s0, messages };
  },
};
