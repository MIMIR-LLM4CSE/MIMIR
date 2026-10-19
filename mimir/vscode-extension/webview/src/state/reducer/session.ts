// A conversation loaded from disk.

import { type ChatHandlers } from "../chatState";

export const SESSION_HANDLERS: ChatHandlers = {
  session_loaded_messages: (state, action) => {
    return {
      ...state,
      // Past every stamp the stored messages carry: a counter behind them gives
      // the next live rows smaller stamps, and the last stored message is then
      // read as part of the step in flight and pinned below it.
      nextSeq: action.messages.reduce(
        (n, m) => (typeof m.seq === "number" && m.seq >= n ? m.seq + 1 : n),
        state.nextSeq),
      // What a run was doing is not worth restoring: it described a moment, and
      // the watcher that could have refreshed it may have died with the process
      // that wrote the file. Left in place, a row saved mid-build would come back
      // claiming to be building, and its dock card would never leave.
      messages: action.messages.map((m) =>
        m.tools?.some((t) => t.phase !== undefined || t.percent !== undefined)
          ? { ...m, tools: m.tools.map((t) => ({ ...t, phase: undefined, percent: undefined })) }
          : m
      ),
      draft: "",
      draftSeq: null,
      liveThinkingBlocks: [],
      liveToolCalls: [],
      busy: false,
      pendingDiffs: [],
      pendingThinking: "",
    };
  },
};
