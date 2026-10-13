import React, { useMemo } from "react";
import { splitMentionSegments } from "./mentionHighlight";
import { vscodePostMessage } from "../hooks/useWebSocket";

interface Props {
  /** The draft as the textarea holds it. */
  text: string;
  /** Remove the mention spanning [start, end) from the draft. */
  onRemove: (start: number, end: number) => void;
}

/**
 * What the message in the box will attach, as chips above the input.
 *
 * The tint inside the textarea says which words are mentions, but a textarea's text
 * cannot be clicked through to anything — so the same mentions are listed here, where a
 * file one opens the file (`open_file`, at the attached line range) and "×" takes it
 * back out of the draft. The chips are derived from the text on every keystroke and own
 * no state: the message is the only record of what it attaches, so a hand-typed path and
 * a path picked from the dropdown appear and behave alike.
 */
export function DraftAttachments({ text, onRemove }: Props) {
  const mentions = useMemo(
    () => splitMentionSegments(text).filter((s) => s.kind === "mention"),
    [text]
  );
  if (mentions.length === 0) return null;

  return (
    <div className="draft-attachments">
      {mentions.map((seg, i) => {
        if (seg.kind !== "mention") return null;
        const label = seg.isFile ? seg.target.split("/").pop() || seg.target : seg.target;
        const range = seg.line
          ? `:${seg.line}${seg.endLine && seg.endLine !== seg.line ? `-${seg.endLine}` : ""}`
          : "";
        return (
          <span className="draft-chip" key={`${seg.start}:${seg.text}`}>
            <button
              className="draft-chip-open"
              disabled={!seg.isFile}
              title={
                seg.isFile
                  ? `Open ${seg.target}${range}`
                  : `${seg.target} — attached as a resource`
              }
              onClick={() =>
                vscodePostMessage({
                  type: "open_file",
                  file: seg.target,
                  line: seg.line,
                  end_line: seg.endLine,
                })
              }
            >
              <span className="draft-chip-icon" aria-hidden="true">
                {seg.isFile ? "📄" : "◈"}
              </span>
              <span className="draft-chip-name">
                {label}
                {range}
              </span>
            </button>
            <button
              className="draft-chip-remove"
              title={`Detach ${seg.text}`}
              aria-label={`Detach ${seg.text}`}
              // onMouseDown so the draft is edited before the textarea's blur runs.
              onMouseDown={(e) => {
                e.preventDefault();
                onRemove(seg.start, seg.end);
              }}
            >
              ×
            </button>
          </span>
        );
      })}
    </div>
  );
}
