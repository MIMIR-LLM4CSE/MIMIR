// The chat's state: the transcript, the live step, and the actions that change them.

import type { ChatMessage, DiffEntry, ServerMessage, ToolActivity } from "../types";

// A reasoning block — reasoning TEXT only. Tool calls are tracked separately
// (see `liveToolCalls`); thinking and tools are fully decorrelated.
export type ThinkingBlock = {
  id: string;
  text: string;
  done: boolean;
  collapsed: boolean;
  /** Epoch ms when this block opened — drives the live timer and frozen duration. */
  startedAt: number;
  /** Arrival stamp, used to place the block among this step's prose and tool
   *  rows (see orderLiveStream). */
  seq: number;
  /** Reasoning tokens reported by the server as the block closed. Accumulated,
   *  since a block that reopens (reasoning resumed before the answer) is fed by
   *  more than one `thinking_end`. */
  tokens?: number;
};

export interface ChatState {
  messages: ChatMessage[];
  /** Prose of the turn in flight, held OUT of the transcript until the loop
   *  accepts it. A turn only becomes an answer once it ends with no tool call
   *  and no guardrail sends the model back to work, so anything streamed before
   *  that is a draft: putting it straight into the message list is what made a
   *  finished-looking answer appear and then vanish. */
  draft: string;
  /** Live reasoning blocks for the current turn (text only). */
  liveThinkingBlocks: ThinkingBlock[];
  /** Live structured tool invocations for the current turn — independent of
   *  thinking. Frozen into a `kind:"tools"` message at each step boundary. */
  liveToolCalls: ToolActivity[];
  busy: boolean;
  /** The turn waits on its agent being built (backend, MCP servers), not on the
   *  model: set by `agent_starting`, cleared by the worker's `ready`. */
  starting: boolean;
  /** True when tool calls / a thinking phase occurred since the last token, so
   *  the next token starts a new paragraph in the draft. */
  toolCallAfterToken: boolean;
  /** Diffs accumulated for the current turn, attached to the final answer. */
  pendingDiffs: DiffEntry[];
  /** Thinking text accumulated for the current turn. */
  pendingThinking: string;
  /** Next arrival stamp. Handed to each live item — the draft, a reasoning
   *  block, a tool row — so the order they came in survives being collected
   *  into three separate streams. Monotonic for the life of the window. */
  nextSeq: number;
  /** Stamp of the draft's first token; null whenever the draft is empty. */
  draftSeq: number | null;
}

export const initialChatState: ChatState = {
  messages: [],
  draft: "",
  liveThinkingBlocks: [],
  liveToolCalls: [],
  busy: false,
  starting: false,
  toolCallAfterToken: false,
  pendingDiffs: [],
  pendingThinking: "",
  nextSeq: 1,
  draftSeq: null,
};

// The server messages the reducer acts on, and the actions the panel dispatches itself.
export type ChatAction =
  | ServerMessage
  | { type: "submit_query"; text: string }
  | { type: "steer_query"; text: string }
  | { type: "reset" }
  | { type: "connection_lost" }
  | { type: "session_loaded_messages"; messages: ChatMessage[] }
  | { type: "turn_resumed" }
  | { type: "prompt_answered"; resumes: boolean }
  | { type: "toggle_thinking"; id: string }
  | { type: "approval_response"; id: string; choice: "y" | "n" | "a" };

export type ChatActionOf<T extends ChatAction["type"]> = Extract<ChatAction, { type: T }>;

/** One handler per action a family of the reducer answers; *makeId* keeps it deterministic. */
export type ChatHandlers = {
  [T in ChatAction["type"]]?: (state: ChatState, action: ChatActionOf<T>, makeId: () => string) => ChatState;
};
