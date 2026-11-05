import React, { useState } from "react";
import type { ChatMessage as IChatMessage } from "../types";
import { MarkdownContent } from "./MarkdownContent";
import { FileDiff } from "./FileDiff";
import { countDiffLines } from "./diffUtils";
import { ToolActivityList } from "./ToolActivityList";
import { CommandResult } from "./CommandResult";
import { ThinkingPanel } from "./ThinkingPanel";
import { CompletionReport } from "./CompletionReport";
import { splitAnswerCompletion } from "./completionUtils";
import { splitAnswerLedger } from "./ledgerUtils";
import { vscodePostMessage } from "../hooks/useWebSocket";
import { MentionText } from "./MentionText";
import type { SkillNames } from "./mentionHighlight";
import { MessageEditor } from "./MessageEditor";
import type { Rewind } from "../hooks/useRewind";

interface Props {
  message: IChatMessage;
  onApprovalResponse: (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => void;
  /** When provided, an error card shows a "Retry" button that re-runs the turn. */
  onRetry?: () => void;
  /** Reopen this message for editing (a double-click); offered on the user messages that opened a turn. */
  onEdit?: (id: string) => void;
  /** Set on the one message being edited only, so every other bubble stays memoized. */
  edit?: Rewind;
  /** The skills a "/name" in a user message invokes, drawn as chips. */
  skills?: SkillNames;
}


const ChatMessageInner: React.FC<Props> = ({ message, onApprovalResponse, onRetry, onEdit, edit, skills }) => {
  // Per-file disclosure state of the editing card, keyed by file path (stable
  // across file_progress merges). Declared before any early return — hooks
  // must run unconditionally.
  const [openDiffs, setOpenDiffs] = useState<Record<string, boolean>>({});

  // Approval cards belong to the GlobalApprovalBar above the input; rendering one inline
  // as well would show the same card twice.
  if (message.kind === "approval" && message.approval) {
    return null;
  }

  // Frozen tool-activity block (kind="tools") — a step's tool calls baked into
  // the message stream. Rendered without an avatar so it reads as a sub-step.
  if (message.kind === "tools") {
    return <ToolActivityList tools={message.tools ?? []} />;
  }

  // The answer to a session command the user typed. Not agent prose, so it gets
  // neither the bubble nor the markdown pass — it is a result, and it says which
  // command produced it.
  if (message.kind === "command" && message.command) {
    return <CommandResult result={message.command} />;
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
          // The directory alone, never the path again: the row already shows the
          // file name beside it, so rendering the whole path repeated the name and
          // pushed the part that actually disambiguates off the end of the line.
          // Empty for a file at the root, where there is nothing to disambiguate.
          const dir = d.file.slice(0, d.file.length - name.length).replace(/\/$/, "");
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
                      // A clipped patch no longer reverses to the original: open the file.
                      vscodePostMessage({ type: "open_diff", file: d.file,
                                          patch: d.truncated ? undefined : d.patch });
                    }
                  }}
                >
                  {name}
                </button>
                {dir && <span className="edit-file-path" title={d.file}>{dir}</span>}
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
  // The verification ledger and the completion report both ride at the end of the
  // answer text (history keeps them for the model). The ledger is stripped and not
  // shown; it is appended last, so it comes off first and the report is the tail.
  const { body: withReport } = splitAnswerLedger(message.text ?? "");
  const { body: answerBody, report } = splitAnswerCompletion(withReport);

  // A message that opened a turn is edited by double-clicking it; its tint on hover
  // says so. Editing it goes back to it.
  const editable = isUser && !edit && !!onEdit && !!message.opensTurn && !message.queued
    && message.kind === "text";
  const editProps = editable ? {
    className: "message-body message-body--editable",
    title: "Double-click to edit — goes back to this message",
    tabIndex: 0,
    onDoubleClick: () => onEdit!(message.id),
    // A double-click would otherwise select a word before the editor replaces it.
    onMouseDown: (e: React.MouseEvent) => { if (e.detail > 1) e.preventDefault(); },
    onKeyDown: (e: React.KeyboardEvent) => {
      if (e.key === "Enter" && e.target === e.currentTarget) { e.preventDefault(); onEdit!(message.id); }
    },
  } : { className: "message-body" };

  return (
    <div className={`chat-message ${isUser ? "user" : "agent"}`}>
      <div {...editProps}>
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
            {report && <CompletionReport block={report} />}
            {(message.diffs ?? []).length > 0 && (
              <div className="diff-list">
                {(message.diffs ?? []).map((d, i) => (
                  <FileDiff key={i} diff={d} />
                ))}
              </div>
            )}
          </>
        ) : (
          edit ? (
            <MessageEditor initialText={message.text ?? ""} rewind={edit} />
          ) : (
            <div className="user-text">
              <MentionText text={message.text ?? ""} skills={skills} />
              {message.queued && (
                <span className="steer-badge" title="Queued — will reach MIMIR at its next step">
                  ↯ queued
                </span>
              )}
            </div>
          )
        )}
      </div>
    </div>
  );
};

// Memoized so streaming a new bubble doesn't re-render every prior message.
// Props are stable per message id; callbacks from App are useCallback-wrapped.
export const ChatMessage = React.memo(ChatMessageInner);
