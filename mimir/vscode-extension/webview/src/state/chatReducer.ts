// The chat's reducer: one table of handlers, gathered from the families in reducer/.

import { JOB_HANDLERS } from "./reducer/jobs";
import { SESSION_HANDLERS } from "./reducer/session";
import { STREAM_HANDLERS } from "./reducer/stream";
import { TOOL_HANDLERS } from "./reducer/tools";
import { TURN_HANDLERS } from "./reducer/turn";
import type { ChatAction, ChatHandlers, ChatState } from "./chatState";

export {
  initialChatState, type ChatAction, type ChatState, type ThinkingBlock,
} from "./chatState";

const HANDLERS: ChatHandlers = {
  ...TURN_HANDLERS, ...STREAM_HANDLERS, ...TOOL_HANDLERS, ...JOB_HANDLERS, ...SESSION_HANDLERS,
};

/**
 * Pure reducer for the streaming / thinking / approval / diff state machine.
 * `makeId` is injected so the reducer stays deterministic under test. A message no
 * family answers (ready, config, todo, sessions, …) leaves the state unchanged.
 */
export function createChatReducer(makeId: () => string) {
  return function reducer(state: ChatState, action: ChatAction): ChatState {
    const handler = HANDLERS[action.type] as
      | ((s: ChatState, a: ChatAction, m: () => string) => ChatState)
      | undefined;
    return handler ? handler(state, action, makeId) : state;
  };
}
