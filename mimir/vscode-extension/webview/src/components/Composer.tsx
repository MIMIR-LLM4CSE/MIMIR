// The message box and the controls beside it: attachments, dropdowns, the approval
// mode, and send or stop.

import React from "react";
import type { ApprovalMode } from "../types";
import type { useComposer } from "../hooks/useComposer";
import { ApprovalSwitcher } from "./ApprovalSwitcher";
import { DraftAttachments } from "./DraftAttachments";
import { DraftHighlight } from "./DraftHighlight";
import { MentionAutocomplete } from "./MentionAutocomplete";
import { SlashAutocomplete } from "./SlashAutocomplete";
import { SendIcon, StopIcon } from "./icons";
import { wrapIndex } from "./mentionUtils";

interface Props {
  composer: ReturnType<typeof useComposer>;
  busy: boolean;
  enabled: boolean;
  approvalMode: ApprovalMode;
  onApprovalModeChange: (mode: ApprovalMode) => void;
  onStop: () => void;
}

export const Composer: React.FC<Props> = ({
  composer, busy, enabled, approvalMode, onApprovalModeChange, onStop,
}) => {
  const {
    input, activeEditor, textareaRef,
    mentionOpen, mentionItems, mentionIndex, setMentionIndex, pickMention,
    slashOpen, slashItems, slashIndex, setSlashIndex, pickSlash,
    attachActiveEditor, detachMention, submitQuery, handleKeyDown,
    onChange, onSelect, onBlur,
  } = composer;
  return (
    <>
          {activeEditor && enabled && (
            <div className="active-file-bar">
              <button
                className="active-file-chip"
                onClick={attachActiveEditor}
                title="Attach the active editor file (and selection) to your message"
              >
                <span className="af-icon">📎</span>
                <span className="af-name">
                  {activeEditor.file}
                  {activeEditor.selection ? `:${activeEditor.selection}` : ""}
                </span>
                <span className="af-add">＋</span>
              </button>
            </div>
          )}
          <DraftAttachments text={input} onRemove={detachMention} />
          <div className="input-row">
          {mentionOpen && (
            <MentionAutocomplete
              items={mentionItems}
              activeIndex={wrapIndex(mentionIndex, mentionItems.length)}
              onPick={pickMention}
              onHover={setMentionIndex}
            />
          )}
          {slashOpen && (
            <SlashAutocomplete
              items={slashItems}
              activeIndex={wrapIndex(slashIndex, slashItems.length)}
              onPick={pickSlash}
              onHover={setSlashIndex}
            />
          )}
          <div className="chat-input-wrap">
          <DraftHighlight text={input} inputRef={textareaRef} />
          <textarea
            ref={textareaRef}
            className="chat-input"
            rows={3}
            placeholder={
              busy
                ? "Steer MIMIR while it works…  (Enter to send)"
                : "Message MIMIR  (Enter to send, Shift+Enter for newline, @ to attach, / for commands)"
            }
            value={input}
            disabled={!enabled}
            onChange={onChange}
            onSelect={onSelect}
            onBlur={onBlur}
            onKeyDown={handleKeyDown}
          />
          </div>
          {/* Right-hand column: who answers the approval cards, directly above the
              button that sends the work off — the mode is switched mid-run, so it
              belongs where the user already is rather than behind a settings gear. */}
          <div className="input-actions">
            <ApprovalSwitcher mode={approvalMode} onModeChange={onApprovalModeChange} />
            {busy ? (
              <>
                {input.trim() && (
                  <button
                    className="send-btn"
                    onClick={submitQuery}
                    title="Send steer (Enter) — queued for the agent's next step"
                    aria-label="Send steer message"
                  >
                    <SendIcon />
                  </button>
                )}
                <button
                  className="stop-btn"
                  onClick={onStop}
                  title="Stop (cancel current query)"
                  aria-label="Stop current query"
                >
                  <StopIcon />
                </button>
              </>
            ) : (
              <button
                className="send-btn"
                onClick={submitQuery}
                disabled={!input.trim() || !enabled}
                title="Send (Enter)"
                aria-label="Send message"
              >
                <SendIcon />
              </button>
            )}
          </div>
          </div>
    </>
  );
};
