// Finding a tool row wherever it is, live or frozen into the transcript, and patching it.

import type { ToolActivity } from "../../types";
import type { ChatState } from "../chatState";

/**
 * Apply *transform* to whichever list holds the row *id*: the live calls or a frozen
 * `tools` message, since late news can land on either side of the freeze. Null when
 * the row is nowhere (a cleared turn's event is dropped).
 */
function patchToolWhere(
  state: ChatState,
  match: (t: ToolActivity) => boolean,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  if (state.liveToolCalls.some(match)) {
    return { ...state, liveToolCalls: transform(state.liveToolCalls) };
  }
  const idx = state.messages.findIndex(
    (m) => m.kind === "tools" && (m.tools ?? []).some(match)
  );
  if (idx >= 0) {
    return {
      ...state,
      messages: state.messages.map((m, i) =>
        i === idx ? { ...m, tools: transform(m.tools ?? []) } : m
      ),
    };
  }
  return null;
}

/**
 * Apply *transform* to every list holding a row that matches *match*: news keyed by a
 * job reaches the launch row, a status call naming the same run, and a re-watched row
 * alike. Null when nothing matched.
 */
export function patchToolsEverywhere(
  state: ChatState,
  match: (t: ToolActivity) => boolean,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  const inLive = state.liveToolCalls.some(match);
  const hits = state.messages.map((m) => m.kind === "tools" && (m.tools ?? []).some(match));
  if (!inLive && !hits.some(Boolean)) return null;
  return {
    ...state,
    liveToolCalls: inLive ? transform(state.liveToolCalls) : state.liveToolCalls,
    messages: hits.some(Boolean)
      ? state.messages.map((m, i) => (hits[i] ? { ...m, tools: transform(m.tools ?? []) } : m))
      : state.messages,
  };
}

/** The common case: the row is known by the call id it was opened with. */
export function patchToolById(
  state: ChatState,
  id: string,
  transform: (tools: ToolActivity[]) => ToolActivity[],
): ChatState | null {
  return patchToolWhere(state, (t) => t.id === id, transform);
}
