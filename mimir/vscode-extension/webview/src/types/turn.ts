// A turn as it streams: prose, reasoning, status, the command replies, the answer or error
// that ends it, and what the user sends into it.

import type { Attributed } from "./prompts";

export interface OutputMessage {
  type: "output";
  text: string;
}

export interface StatusMessage {
  type: "status";
  text: string;
}

/** One row of a command answer: a name, optionally what it is. */
export interface CommandItem {
  label: string;
  detail?: string;
}

/**
 * How a command answer should read. "warn" is for something irreversible that has
 * just happened (a wipe), "empty" for a listing with nothing in it.
 */
/**
 * How a command's answer should read.
 *
 * `quiet` is not a weaker `ok`: it renders as a rule across the thread rather than as
 * a result, for the one thing that is neither a listing nor something the user just
 * did — a note about the state they have walked back into. A warning badge on that
 * reads as a problem to deal with, and it is only a fact.
 */
export type CommandTone = "ok" | "warn" | "empty" | "quiet";

/**
 * The answer to a session command the user typed ("/memory list"), structured so a
 * setting and a listing get different shapes. Not an OutputMessage, which the reducer
 * drops as transient.
 */
export interface CommandOutputMessage {
  type: "command_output";
  /** The command that produced this, e.g. "/memory list". */
  command: string;
  /** The headline result, e.g. "3 memories" or "Cleared 4 memories". */
  title: string;
  items?: CommandItem[];
  /** A single line under the rows, e.g. "This cannot be undone." */
  note?: string;
  tone?: CommandTone;
  /**
   * Rendered, never stored: the card describes this moment, and replayed on every load
   * it would read as current.
   */
  transient?: boolean;
}

/** Every agent slot is taken; this conversation's turn starts when one frees.
 *
 *  Rendered rather than dropped like other transient status: a queue nobody can see
 *  reads as a hang, which is the one thing a bounded pool must never look like. */
export interface QueuedMessage extends Attributed {
  type: "queued";
  position: number;
  text: string;
}

export interface AnswerMessage {
  type: "answer";
  text: string;
  cancelled?: boolean;
}

export interface TokenMessage {
  type: "token";
  text: string;
}

export interface ErrorMessage {
  type: "error";
  text: string;
}

export interface ThinkingMessage {
  type: "thinking";
  text: string;
}

export interface ThinkingStartMessage {
  type: "thinking_start";
}

export interface ThinkingEndMessage {
  type: "thinking_end";
  /** Size of the reasoning block just closed, counted server-side with the same
   *  tokenizer as the context bar. Absent on older servers. */
  tokens?: number;
}

/** Confirms a mid-run steer message reached the agent: injected into the running
 *  turn, or — when `starts_turn` is set — carried by a turn the server started for
 *  it because the run it was typed into ended before reading it. */
export interface SteerInjectedMessage {
  type: "steer_injected";
  text: string;
  starts_turn?: boolean;
}

/** A workflow reminder the guardrail layer injected into the agent's user turn.
 *  Signals that the turn the model just streamed was not accepted as its answer.
 *  Not rendered — the reducer uses it to drop the superseded draft. */
export interface NudgeInjectedMessage {
  type: "nudge_injected";
  category: string;
  text: string;
}

/** A whole prose block, recorded for replay only (live clients have its `token`s).
 *
 *  Fed to the reducer as a `token`, so a replayed turn interleaves prose and tool cards
 *  as the live one did. */
export interface AssistantTextMessage {
  type: "assistant_text";
  text: string;
}

export interface ContextUsageMessage {
  type: "context_usage";
  /** Estimated tokens used by the current history. */
  used_tokens: number;
  /** Total context window size for the active mode. */
  total_tokens: number;
  /** Tokens reserved for the model's answer (auto-compact fires below this headroom). */
  reserved_tokens: number;
  /** Fixed per-call overhead (system prompt + tools schema) included in used_tokens. */
  overhead_tokens?: number;
  /** True once `overhead_tokens` is the size the server reported for a real prompt,
   *  false while it is still this client's estimate of the prompt it will send. */
  overhead_measured?: boolean;
  /** True while the conversation has no agent yet: the system prompt and tools schema
   *  are not in `used_tokens` at all, so the figure is a floor and the percentage must
   *  not be read as a verdict. */
  provisional?: boolean;
  /** Messages in the window the model actually sees this turn. */
  history_messages?: number;
  /** Messages in the untrimmed record a resume would start from. Larger than
   *  `history_messages` once the budget has trimmed the window. */
  history_messages_full?: number;
}

export interface QueryMessage {
  type: "query";
  text: string;
}

/** A message typed while the agent is busy — injected into the running turn. */
export interface SteerMessage {
  type: "steer";
  text: string;
}

export interface CommandMessage {
  type: "command";
  text: string;
}
