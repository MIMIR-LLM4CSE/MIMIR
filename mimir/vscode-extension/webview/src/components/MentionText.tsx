import React, { useMemo } from "react";
import { splitMentionSegments } from "./mentionHighlight";
import { vscodePostMessage } from "../hooks/useWebSocket";

interface Props {
  text: string;
}

/**
 * A sent message with its "@"-attachments marked out of the prose.
 *
 * What was attached is the part of a message the user comes back to — which file, which
 * lines — and as plain text it is the hardest part to find again. Each mention is drawn
 * as a chip, and a file mention opens the file in a real editor tab, at the attached
 * line range when it named one (`open_file`, the host handler the edit cards use).
 *
 * A resource mention (`@memory://all`) is marked the same way but is not a link: it
 * addresses a server, not a path, and there is nothing to open.
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
