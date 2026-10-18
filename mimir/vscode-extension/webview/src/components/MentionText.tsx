import React, { useMemo } from "react";
import { splitMentionSegments } from "./mentionHighlight";
import { vscodePostMessage } from "../hooks/useWebSocket";

interface Props {
  text: string;
}

/**
 * A sent message with its "@"-attachments drawn as chips: a file chip opens the file at
 * the attached lines (`open_file`); a resource mention (`@memory://all`) is marked but
 * has nothing to open.
 */
export function MentionText({ text }: Props) {
  const segments = useMemo(() => splitMentionSegments(text), [text]);

  return (
    <>
      {segments.map((seg, i) => {
        if (seg.kind === "text") return <React.Fragment key={i}>{seg.text}</React.Fragment>;
        if (!seg.isFile) {
          return (
            <span key={i} className="ctx-mention ctx-mention--plain" title={seg.target}>
              {seg.text}
            </span>
          );
        }
        return (
          <button
            key={i}
            className="ctx-mention"
            title={`Open ${seg.target}${seg.line ? `:${seg.line}` : ""}`}
            onClick={() =>
              vscodePostMessage({
                type: "open_file",
                file: seg.target,
                line: seg.line,
                end_line: seg.endLine,
              })
            }
          >
            {seg.text}
          </button>
        );
      })}
    </>
  );
}
