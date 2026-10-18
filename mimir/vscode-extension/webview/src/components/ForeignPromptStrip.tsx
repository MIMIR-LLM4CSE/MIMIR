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
 * Cards raised by conversations other than the one on screen: a turn blocked on the
 * user with no timeout, so shown above the thread.
 *
 * An approval is answered here (no "Always": it widens a sandbox from a one-line
 * summary); a question needs its conversation, so it offers the jump instead.
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
