import React from "react";
import type { ChatMessage, ToolActivity } from "../types";
import type { ThinkingBlock } from "../state/chatReducer";
import { ChatMessage as ChatMessageView } from "./ChatMessage";
import { MarkdownContent } from "./MarkdownContent";
import { ToolActivityList } from "./ToolActivityList";
import { ThinkingPanel } from "./ThinkingPanel";
import { GlobalApprovalBar } from "./GlobalApprovalBar";
import { StreamingStatus } from "./StreamingStatus";

interface Props {
  messages: ChatMessage[];
  busy: boolean;
  /** Prose of the turn in flight. Rendered below the transcript, never in it:
   *  the loop may still send the model back to work, and a draft that lived in
   *  the transcript would then have to be deleted out of it. */
  draft?: string;
  /** Tool calls for the current (in-flight) step, not yet frozen into a message. */
  liveToolCalls?: ToolActivity[];
  /** Reasoning blocks for the current (in-flight) step, not yet frozen. */
  liveThinkingBlocks?: ThinkingBlock[];
  /** Rendered in place of the message list when there are no messages. */
  emptyState?: React.ReactNode;
  /** When true, a spinner replaces the thread (e.g. while a session loads). */
  loading?: boolean;
  chatThreadRef: React.RefObject<HTMLDivElement>;
  bottomRef: React.RefObject<HTMLDivElement>;
  lastUserMsgRef: React.RefObject<HTMLDivElement>;
  onScroll: () => void;
  onApprovalResponse: (id: string, choice: "y" | "n" | "a", approvedFiles?: string[]) => void;
  /** Re-runs the last turn; surfaced as a Retry button on the latest error card. */
  onRetry?: () => void;
}

/**
 * Owns the scrollable conversation pane: the message list (dispatched by kind)
 * and the typing indicator. App keeps connection, session and settings
 * orchestration; this component is purely presentational.
 *
 * Frozen `tools` and `thinking` messages are rendered as their own cards
 * (tool-activity list and collapsible reasoning panels). Live, in-flight tool
 * calls and reasoning blocks are rendered at the bottom until they freeze.
 */
export const ChatThread: React.FC<Props> = ({
  messages,
  busy,
  draft = "",
  liveToolCalls = [],
  liveThinkingBlocks = [],
  emptyState,
  loading,
  chatThreadRef,
  bottomRef,
  lastUserMsgRef,
  onScroll,
  onApprovalResponse,
  onRetry,
}) => {
  // All message kinds render via renderMessage; thinking blocks (frozen) and
  // tool lists each have their own render branch in ChatMessage.
  const visible = messages;

  const lastIdx = visible.length - 1;

  // One animated status line, always at the bottom of the thread — while
  // waiting AND while the answer streams. Hidden when live tool rows or a
  // live thinking line already signal activity (they carry their own motion).
  const showStatusLine =
    busy &&
    draft.trim().length === 0 &&
    liveToolCalls.length === 0 &&
    liveThinkingBlocks.length === 0;

  // ── One message → one render path, dispatched by kind ──────────────────────
  const renderMessage = (msg: ChatMessage, idx: number) => {
    if (msg.kind === "approval" && msg.approval) {
      return (
        <div className="inline-approval-wrap" key={msg.id}>
          <GlobalApprovalBar
            approval={msg.approval}
            onRespond={(choice) => onApprovalResponse(msg.approval!.id, choice)}
          />
        </div>
      );
    }

    // Skip approval shells that carry no content.
    if (msg.kind === "approval") return null;

    const isLastUser =
      msg.role === "user" &&
      !visible.slice(idx + 1).some((m) => m.role === "user");
    const isLatestError = msg.kind === "error" && idx === lastIdx;

    return (
      <div key={msg.id} ref={isLastUser ? lastUserMsgRef : undefined}>
        <ChatMessageView
          message={msg}
          onApprovalResponse={onApprovalResponse}
          onRetry={isLatestError ? onRetry : undefined}
        />
      </div>
    );
  };

  return (
    <div
      className="chat-thread"
      ref={chatThreadRef}
      onScroll={onScroll}
      role="log"
      aria-live="polite"
      aria-label="Conversation"
    >
      {visible.length === 0 ? (
        emptyState
      ) : loading ? (
        <div className="session-loading">
          <span className="inline-spinner" aria-hidden="true" />
          Loading session…
        </div>
      ) : (
        <>
          {visible.map(renderMessage)}

          {/* Live reasoning for the in-flight step (streams, then freezes). */}
          {liveThinkingBlocks
            .filter((b) => b.text.trim().length > 0)
            .map((b) => (
              <ThinkingPanel key={b.id} text={b.text} live startedAt={b.startedAt} />
            ))}

          {/* Live tool calls for the in-flight step (not yet frozen). */}
          {liveToolCalls.length > 0 && (
            <ToolActivityList tools={liveToolCalls} />
          )}

          {/* The turn in flight. It joins the transcript only once the loop
              accepts it; until then it lives here, where clearing it reads as
              "still working" rather than as an answer being taken away. */}
          {draft.trim().length > 0 && (
            <div className="chat-message agent chat-draft">
              <div className="message-body">
                <MarkdownContent text={draft} />
                <div className="chat-draft__status">
                  <StreamingStatus />
                </div>
              </div>
            </div>
          )}

          {/* Animated status line — shown while the agent is busy */}
          {showStatusLine && (
            <div className="agent-thinking">
              <StreamingStatus />
            </div>
          )}
        </>
      )}
      <div ref={bottomRef} />
    </div>
  );
};
