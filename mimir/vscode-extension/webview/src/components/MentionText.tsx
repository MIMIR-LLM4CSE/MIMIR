import React, { useMemo } from "react";
import { splitMentionSegments, type SkillNames } from "./mentionHighlight";
import { vscodePostMessage } from "../hooks/useWebSocket";

interface Props {
  text: string;
  /** The skills a "/name" in the text invokes. */
  skills?: SkillNames;
}

const NO_SKILLS: SkillNames = new Map();

/**
 * A sent message with its "@"-attachments drawn as chips: a file chip opens the file at
 * the attached lines (`open_file`); a resource mention (`@memory://all`) is marked but
 * has nothing to open. A skill it invokes ("/name") gets a chip of its own colour.
 */
export function MentionText({ text, skills = NO_SKILLS }: Props) {
  const segments = useMemo(() => splitMentionSegments(text, skills), [text, skills]);

  return (
    <>
      {segments.map((seg, i) => {
        if (seg.kind === "text") return <React.Fragment key={i}>{seg.text}</React.Fragment>;
        if (seg.kind === "skill") {
          const about = skills.get(seg.name);
          return (
            <span key={i} className="ctx-skill" title={`Skill ${seg.name}${about ? ` — ${about}` : ""}`}>
              {seg.text}
            </span>
          );
        }
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
