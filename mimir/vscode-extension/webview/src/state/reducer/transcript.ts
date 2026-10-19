// Moving the turn's live state into the transcript, in arrival order.

import type { ChatMessage } from "../../types";
import { orderLiveStream } from "../../components/liveStreamUtils";
import type { ChatState } from "../chatState";

/**
 * Append *msg* to the transcript, keeping stamped messages in arrival order.
 *
 * What lands while a step is in flight (a steer bubble, an edit card, a command reply)
 * is stamped, as is what the step freezes; a new stamped message is inserted into that
 * run. An unstamped one (a query, an answer, an error) goes last.
 */
export function appendStamped(messages: ChatMessage[], msg: ChatMessage): ChatMessage[] {
  const seq = msg.seq;
  if (seq === undefined) return [...messages, msg];
  let at = messages.length;
  while (at > 0) {
    const prev = messages[at - 1].seq;
    if (prev === undefined || prev <= seq) break;
    at--;
  }
  return [...messages.slice(0, at), msg, ...messages.slice(at)];
}

/**
 * Append *msg* above the steer bubbles the ending turn never read: they were typed
 * during it and are answered by a turn of their own. The other half of
 * {@link appendStamped}.
 */
export function appendAboveUnclaimed(
  messages: ChatMessage[], msg: ChatMessage,
): ChatMessage[] {
  let at = messages.length;
  while (at > 0 && messages[at - 1].queued) at--;
  return [...messages.slice(0, at), msg, ...messages.slice(at)];
}

/**
 * Bake the turn's live state into the transcript and clear it, in arrival order.
 *
 * The three streams (prose, reasoning, tools) are ordered by their stamps through
 * orderLiveStream, as the live view is; each reasoning block and each run of tool rows
 * is one message. *provisional* is as in commitDraft.
 */
export function flushLive(
  state: ChatState, makeId: () => string, provisional = false,
): ChatState {
  const blocks = state.liveThinkingBlocks.filter((b) => b.text.trim().length > 0);
  const tools = state.liveToolCalls;
  const hasDraft = state.draft.trim().length > 0;
  if (!hasDraft && blocks.length === 0 && tools.length === 0) {
    // Nothing worth keeping — but a whitespace-only draft and empty reasoning
    // blocks still have to go, or they outlive the step that opened them.
    if (
      state.draft === "" &&
      state.liveThinkingBlocks.length === 0 &&
      state.liveToolCalls.length === 0
    ) return state;
    return { ...state, draft: "", draftSeq: null, liveThinkingBlocks: [], liveToolCalls: [] };
  }

  const frozenAt = Date.now();
  // A draft with no stamp cannot happen through `token`, but placing it first is
  // the safe reading of one: prose is never lost, only possibly placed early.
  const entries = orderLiveStream({
    thinking: blocks,
    tools,
    draftSeq: hasDraft ? state.draftSeq ?? 0 : null,
  });

  const frozen: ChatMessage[] = entries.map((e) => {
    // orderLiveStream also places messages that landed mid-step; the transcript
    // already holds those, and appendStamped below is what keeps them in order.
    if (e.kind === "message") return e.message;
    if (e.kind === "draft") {
      return {
        id: makeId(), seq: e.seq,
        role: "agent" as const, kind: "text" as const, text: state.draft,
        ...(provisional ? { provisional: true } : {}),
      };
    }
    if (e.kind === "thinking") {
      return {
        id: e.block.id,
        seq: e.seq,
        role: "agent" as const,
        kind: "thinking" as const,
        text: e.block.text,
        live: false, // start collapsed — user can click to expand
        thinkingDurationMs: Math.max(0, frozenAt - e.block.startedAt),
        thinkingTokens: e.block.tokens,
      };
    }
    // Any tool still "running" at freeze time never received a result; stamp a
    // final duration and mark it done so its live timer stops ticking instead
    // of counting up forever in the frozen card.
    return {
      id: makeId(),
      seq: e.seq,
      role: "agent" as const,
      kind: "tools" as const,
      live: false,
      tools: e.tools.map((t) =>
        t.status === "running"
          ? { ...t, status: "ok" as const, durationMs: t.durationMs ?? Math.max(0, frozenAt - t.startedAt) }
          : t
      ),
    };
  });

  return {
    ...state,
    draft: "",
    draftSeq: null,
    liveThinkingBlocks: [],
    liveToolCalls: [],
    // Inserted rather than appended: a steer bubble may already sit at the tail,
    // and the prose and cards of the step it interrupted belong above it.
    messages: frozen.reduce(appendStamped, state.messages),
  };
}

/**
 * Move the draft into the transcript before anything is appended under it. Not on
 * `answer` (which supersedes it) nor on `nudge_injected` (the turn was not accepted).
 *
 * *provisional*: the turn parked on a question or lost its connection, and this prose is
 * its only copy; the `answer` that supersedes it drops it again.
 */
export function commitDraft(
  state: ChatState, makeId: () => string, provisional = false,
): ChatState {
  if (!state.draft.trim()) {
    return state.draft ? { ...state, draft: "", draftSeq: null } : state;
  }
  return {
    ...state,
    draft: "",
    draftSeq: null,
    messages: appendStamped(state.messages, {
      id: makeId(), seq: state.draftSeq ?? undefined,
      role: "agent", kind: "text", text: state.draft,
      ...(provisional ? { provisional: true } : {}),
    }),
  };
}

/**
 * True when *answer* is the finished form of the provisional prose *held*: equal, or
 * extending it, on collapsed whitespace.
 */
export function isRewordedBy(answer: string, held: string | undefined): boolean {
  const flat = (t: string) => t.replace(/\s+/g, " ").trim();
  const a = flat(answer ?? "");
  const h = flat(held ?? "");
  return h.length > 0 && a.startsWith(h);
}

/** Settle every provisional bubble: they are ordinary transcript from here on. */
export function clearProvisional(messages: ChatMessage[]): ChatMessage[] {
  if (!messages.some((m) => m.provisional)) return messages;
  return messages.map((m) => {
    if (!m.provisional) return m;
    const { provisional, ...rest } = m;
    return rest;
  });
}

/**
 * Commit the draft and stop the turn's live state, for a turn parked on a question:
 * nothing else marks the moment, and the transcript (the plan up for approval included)
 * only reaches the server when `busy` falls.
 */
export function parkOnPrompt(state: ChatState, makeId: () => string): ChatState {
  const s = flushLive(state, makeId, true);
  return { ...s, busy: false, toolCallAfterToken: false };
}
