import React from "react";
import type { ApprovalMessage, UserQuestionMessage } from "../types";

type ForeignPrompt = ApprovalMessage | UserQuestionMessage;

interface Props {
  /** Newest card per conversation, keyed by session id. */
  prompts: Record<string, ForeignPrompt>;
  /** Jump to that conversation — where the card can be read in full and answered. */
  onOpen: (sessionId: string) => void;
  /** Answer an approval without leaving this conversation. */
  onApprove: (prompt: ApprovalMessage, choice: "y" | "n") => void;
}

/**
 * Conversations that are waiting on the user, other than the one on screen.
 *
 * Conversations run turns at once, so a card can be raised by one the user is not reading.
 * It belongs neither in this chat's thread — that is another transcript — nor nowhere: the
 * agent blocks on the answer with no timeout, by design, so an unseen card is a
 * conversation stopped until somebody notices. Hence a strip above the thread, the level
 * that cannot be missed while the panel is open.
 *
 * An approval is answerable here, because yes/no is the whole decision and making the user
 * switch conversations to say "no" is friction with no purpose. "Always" is deliberately
 * absent: it widens a sandbox, which is not a choice to offer beside a one-line summary of
 * a call the user has not read in context. A question is not answerable here at all — its
 * options need the conversation around them — so it offers the jump instead.
 */
export function ForeignPromptStrip({ prompts, onOpen, onApprove }: Props) {
  const entries = Object.entries(prompts);
  if (entries.length === 0) return null;

  return (
    <div className="foreign-prompts" role="region" aria-label="Other conversations waiting">
      {entries.map(([sessionId, prompt]) => {
        const title = prompt.session_title?.trim() || "Another conversation";
        const isApproval = prompt.type === "approval";
        const card = prompt as ApprovalMessage;
        const what = isApproval
          ? `is waiting on you: ${card.label || card.tool}`
          : "has a question for you";
        return (
          <div key={sessionId} className="foreign-prompt">
            <span className="foreign-prompt-icon" aria-hidden="true">
              {isApproval ? "🔐" : "💬"}
            </span>
            <span className="foreign-prompt-text">
              <strong>{title}</strong> {what}
            </span>
            <span className="foreign-prompt-actions">
              {isApproval && (
                <>
                  <button
                    type="button"
                    className="foreign-prompt-btn"
                    onClick={() => onApprove(card, "y")}
                  >
                    Allow
                  </button>
                  <button
                    type="button"
                    className="foreign-prompt-btn"
                    onClick={() => onApprove(card, "n")}
                  >
                    Deny
                  </button>
                </>
              )}
              <button
                type="button"
                className="foreign-prompt-btn foreign-prompt-open"
                onClick={() => onOpen(sessionId)}
              >
                Open
              </button>
            </span>
          </div>
        );
      })}
    </div>
  );
}
