// Shell-like recall of sent messages in the chat input.
//
// `index` is null while the user is typing their own text; otherwise it points
// into `entries` (oldest first). `draft` holds what was in the input when the
// browse began, so walking back down past the newest entry restores it.

export interface HistoryState {
  index: number | null;
  draft: string;
}

export const HISTORY_IDLE: HistoryState = { index: null, draft: "" };

/**
 * One step through the history. Returns null when there is nowhere to go,
 * so the key keeps its ordinary behaviour.
 */
export function stepHistory(
  entries: readonly string[],
  state: HistoryState,
  input: string,
  direction: "up" | "down"
): { state: HistoryState; text: string } | null {
  if (direction === "up") {
    if (entries.length === 0) return null;
    if (state.index === null) {
      const index = entries.length - 1;
      return { state: { index, draft: input }, text: entries[index] };
    }
    if (state.index === 0) return null;
    const index = state.index - 1;
    return { state: { ...state, index }, text: entries[index] };
  }
  if (state.index === null) return null;
  if (state.index >= entries.length - 1) {
    return { state: HISTORY_IDLE, text: state.draft };
  }
  const index = state.index + 1;
  return { state: { ...state, index }, text: entries[index] };
}

/** Appends a sent message, skipping an immediate repeat. */
export function pushHistory(entries: readonly string[], text: string): string[] {
  if (entries[entries.length - 1] === text) return [...entries];
  return [...entries, text];
}
