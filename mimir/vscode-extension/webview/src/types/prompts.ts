// The cards a turn parks on — an approval, a question, the batch of edits held for review —
// and the answers to them.

import type { DiffEntry } from "./chat";

/** Which conversation a card belongs to.
 *
 *  Several conversations run turns at once, and one may raise a card while the user is
 *  reading another. The id routes the answer back to the agent that asked: the server
 *  drops an answer naming no conversation rather than handing it to whoever is on screen,
 *  which would attach the user's approval to a call they never saw. Optional, so a card
 *  from a server that does not attribute them still renders — it counts as the active
 *  conversation's. */
export interface Attributed {
  session_id?: string;
  session_title?: string;
}

export interface ApprovalMessage extends Attributed {
  type: "approval";
  id: string;
  /** All approval IDs when multiple concurrent approvals are merged into one card. */
  ids?: string[];
  tool: string;
  server: string;
  args: Record<string, unknown>;
  risk: string;
  /** Declared undo level for this call: how much of its effect can be taken back.
   *  Drives the severity badge from a value instead of keyword-matching `risk`.
   *  Optional so a card from an older server still renders. */
  reversibility?: "reversible" | "recoverable" | "irreversible";
  scope: string;
  /** Canonical human label ("Proxy exec: run"); absent for tools with no template. */
  label?: string;
  /** The tool's family of work, so the card carries the same icon as the activity row
   *  for this call. Absent on a card from a server that predates the field. */
  kind?: string;
  /** Set when the call is held because it touches paths outside the workspace.
   *  Every outside path the call names travels in this one card — the user judges
   *  the call, not each of its operands. The rest of the card describes the tool
   *  call itself, as usual. */
  oow_paths?: string[];
  /** First of `oow_paths`, kept for a card built by an older server. */
  oow_path?: string;
}

export interface QuestionOption {
  label: string;
  description?: string;
}

export interface QuestionSpec {
  question: string;
  header: string;
  options: QuestionOption[];
  multiSelect: boolean;
}

/**
 * Who raised a card, when it is not the turn on screen.
 *
 * A sub-agent works alongside the turn rather than inside it, and several can be
 * working at once — so "allow this command?" is a question the user cannot weigh
 * without knowing which piece of work it unblocks.
 */
export interface PromptOrigin {
  kind: string;
  label: string;
  session?: string;
}

export interface UserQuestionMessage extends Attributed {
  type: "user_question";
  id: string;
  questions: QuestionSpec[];
  origin?: PromptOrigin;
  /** Seconds the agent waits for an answer before closing the card and going with
   *  the first option. Absent on a card with no wall (plan approval). */
  timeout_secs?: number;
}

/** The wait behind a card ended with nothing answered: close it, here or in the
 *  foreign-prompt strip. The turn is producing again — it went on with what it
 *  recommended — so it is not an answer and nothing is sent back. */
export interface PromptExpiredMessage extends Attributed {
  type: "prompt_expired";
  id: string;
  kind: "user_question";
  timeout_secs?: number;
}

export interface QuestionAnswer {
  selected: string[];
  otherText?: string;
}

export interface BatchStatusMessage {
  type: "batch_status";
  files: DiffEntry[];
}

export interface ApprovalResponseMessage {
  type: "approval_response";
  id: string;
  /** The conversation that asked. Required in practice: the server drops an answer
   *  carrying none rather than guessing which agent it belongs to. */
  session_id?: string;
  choice: "y" | "n" | "a";
  approved_files?: string[];
}

export interface UserQuestionResponseMessage {
  type: "user_question_response";
  id: string;
  session_id?: string;
  answers: QuestionAnswer[];
}

export interface BatchReviewAcceptMessage {
  type: "batch_review_accept";
}

export interface BatchReviewRevertMessage {
  type: "batch_review_revert";
}

export interface BatchReviewAcceptFileMessage {
  type: "batch_review_accept_file";
  file: string;
}

export interface BatchReviewRevertFileMessage {
  type: "batch_review_revert_file";
  file: string;
}
