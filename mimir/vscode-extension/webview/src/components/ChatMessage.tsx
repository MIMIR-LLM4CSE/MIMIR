import React, { useState } from "react";
import type { ChatMessage as IChatMessage } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { FileDiff } from "./FileDiff";
import { countDiffLines } from "./diffUtils";
import { ToolActivityList } from "./ToolActivityList";
import { ThinkingPanel } from "./ThinkingPanel";
import { VerificationLedger } from "./VerificationLedger";
import { splitAnswerLedger } from "./ledgerUtils";
import { vscodePostMessage } from "../hooks/useWebSocket";

interface Props {
  message: IChatMessage;
  onApprovalResponse: (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => void;
  /** When provided, an error card shows a "Retry" button that re-runs the turn. */
  onRetry?: () => void;
}


const ChatMessageInner: React.FC<Props> = ({ message, onApprovalResponse, onRetry }) => {
  // Per-file disclosure state of the editing card, keyed by file path (stable
  // across file_progress merges). Declared before any early return — hooks
  // must run unconditionally.
  const [openDiffs, setOpenDiffs] = useState<Record<string, boolean>>({});

  // Approval cards are now rendered in the GlobalApprovalBar above the input;
  // suppress inline rendering so they don't duplicate.
  if (message.kind === "approval" && message.approval) {
    return null;
  }

  // Frozen tool-activity block (kind="tools") — a step's tool calls baked into
  // the message stream. Rendered without an avatar so it reads as a sub-step.
  if (message.kind === "tools") {
    return <ToolActivityList tools={message.tools ?? []} />;
  }

  // Frozen reasoning block (kind="thinking") — a collapsed, re-openable panel.
  if (message.kind === "thinking") {
    return (
      <ThinkingPanel
        text={message.text ?? ""}
        live={false}
        durationMs={message.thinkingDurationMs}
        tokens={message.thinkingTokens}
      />
    );
  }

  if (message.kind === "editing") {
    const diffs = message.diffs ?? [];
    return (
      <div className="edit-card">
        {diffs.map((d, i) => {
          const { adds, dels } = countDiffLines(d.patch ?? "");
          const name = d.file.split("/").pop() || d.file;
          const isNew = d.is_new;
          const isDelete = d.is_delete;
          const open = !!openDiffs[d.file];
          const hasPatch = !!d.patch;
          return (
            <div key={i} className="edit-file">
              <div className={`edit-file-row${message.live ? " edit-file-row--live" : ""}`}>
                {hasPatch ? (
                  <button
                    className="edit-file-disclose"
                    onClick={() => setOpenDiffs((prev) => ({ ...prev, [d.file]: !prev[d.file] }))}
                    aria-expanded={open}
                    aria-label={open ? `Hide diff: ${d.file}` : `Show diff: ${d.file}`}
                  >
                    <span className={`edit-file-chevron${open ? " edit-file-chevron--open" : ""}`}>▸</span>
                  </button>
                ) : (
                  <span className="edit-file-disclose edit-file-disclose--spacer" aria-hidden="true" />
                )}
                {message.live
                  ? <span className="tb-spinner" style={{ flexShrink: 0 }} aria-hidden="true" />
                  : <span className="edit-icon">{isDelete ? "🗑" : isNew ? "＋" : "✎"}</span>
                }
                <button
                  className="edit-file-name"
                  title={d.file}
                  onClick={() => {
                    if (d.new_content !== undefined) {
                      vscodePostMessage({ type: "open_diff", file: d.file, new_content: d.new_content });
                    } else {
                      vscodePostMessage({ type: "open_diff", file: d.file, patch: d.patch });
                    }
                  }}
                >
                  {name}
                </button>
                <span className="edit-file-path">{d.file}</span>
                {!message.live && (
                  <>
                    <span className="edit-stat edit-stat--add">+{adds}</span>
                    <span className="edit-stat edit-stat--del">−{dels}</span>
                  </>
                )}
              </div>
              {open && hasPatch && (
                <div className="edit-file-diff">
                  <FileDiff diff={d} hideHeader />
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  const isUser = message.role === "user";
  // The verification ledger rides at the end of the answer text (history keeps it for
  // the model); the bubble shows the prose and hands the ledger to its own panel.
  const { body: answerBody, ledger } = splitAnswerLedger(message.text ?? "");

  return (
    <div className={`chat-message ${isUser ? "user" : "agent"}`}>
      <div className="message-body">
        {/* Main content */}
        {message.kind === "error" ? (
          <div className="error-card" role="alert">
            <span className="error-card__icon" aria-hidden="true">⚠</span>
            <div className="error-card__body">
              <div className="error-card__title">Something went wrong</div>
              <div className="error-card__text">{message.text}</div>
              {onRetry && (
                <button className="error-card__retry" onClick={onRetry}>
                  Retry
                </button>
              )}
            </div>
          </div>
        ) : message.kind === "text" && !isUser ? (
          <>
            {/* Reasoning is rendered in-stream via kind="thinking" panels, so it
                is intentionally NOT duplicated on the answer bubble here. */}
            <MarkdownContent text={answerBody} />
            {ledger && <VerificationLedger block={ledger} />}
            {(message.diffs ?? []).length > 0 && (
              <div className="diff-list">
                {(message.diffs ?? []).map((d, i) => (
                  <FileDiff key={i} diff={d} />
                ))}
              </div>
            )}
          </>
        ) : (
          <div className="user-text">
            {message.text}
            {message.queued && (
              <span className="steer-badge" title="Queued — will reach MIMIR at its next step">
                ↯ queued
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

// Memoized so streaming a new bubble doesn't re-render every prior message.
// Props are stable per message id; callbacks from App are useCallback-wrapped.
export const ChatMessage = React.memo(ChatMessageInner);
