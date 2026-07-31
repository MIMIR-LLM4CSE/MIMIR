import React, { useMemo, useState } from "react";
import type { DiffEntry } from "../types";
import { vscodePostMessage } from "../hooks/useWebSocket";
import type { DiffLine } from "./diffUtils";
import { computeWordRanges, highlightFragment, langFromPath, parsePatch } from "./diffUtils";

interface Props {
  diff: DiffEntry;
  /** Render only the diff body (no filename/stats header). Used when the diff is
   *  embedded under a row that already shows the filename, e.g. BatchReviewBar. */
  hideHeader?: boolean;
}

function handleOpenFile(diff: DiffEntry, e: React.MouseEvent) {
  e.stopPropagation();
  if (diff.new_content !== undefined) {
    // Pre-write approval: open VS Code diff editor (current ↔ proposed).
    vscodePostMessage({ type: "open_diff", file: diff.file, new_content: diff.new_content });
  } else {
    // Post-write: reconstruct original from patch and open a VS Code diff editor.
    vscodePostMessage({ type: "open_diff", file: diff.file, patch: diff.patch });
  }
}

// Diffs longer than this clip to a scroll region with a "Show all" toggle.
const CLIP_LINES = 24;
// Skip syntax highlighting beyond this many lines — every line is in the DOM
// (CLIP_LINES only limits the visible height), so hljs cost scales with all of them.
const MAX_HL_LINES = 1500;

/** One diff line's content: intra-line word emphasis + optional syntax colors.
 *
 * Two render paths. Without a language: plain React text nodes split around the
 * word range (React escapes; no innerHTML). With a language: each segment is
 * highlighted separately (hljs escapes its input) and joined, so the word-range
 * span never cuts through hljs markup. Per-line highlighting loses multi-line
 * construct context (block comments, template strings) — accepted, GitHub does
 * the same. If ever needed: highlight whole lines and re-split via a DOM walk.
 */
const DiffLineContent: React.FC<{ line: DiffLine; lang?: string }> = ({ line, lang }) => {
  if (line.type === "hunk" || line.type === "header" || line.type === "meta") {
    // Rendered whole — these lines' leading chars are content, not diff markers.
    return <span className="diff-line-text">{line.text}</span>;
  }
  const content = line.text.slice(1);
  const range = line.wordRange;
  const wordClass = line.type === "add" ? "diff-word--add" : "diff-word--del";

  if (lang) {
    const segments = range
      ? [content.slice(0, range[0]), content.slice(range[0], range[1]), content.slice(range[1])]
      : [content];
    const html = segments.map((seg) => (seg ? highlightFragment(seg, lang) : ""));
    if (!html.some((h) => h === undefined)) {
      const joined = range
        ? `${html[0]}<span class="${wordClass}">${html[1]}</span>${html[2]}`
        : html[0] ?? "";
      return <span className="diff-line-text" dangerouslySetInnerHTML={{ __html: joined }} />;
    }
    // hljs failed on a segment — fall through to the plain path.
  }

  if (range && line.type !== "ctx") {
    return (
      <span className="diff-line-text">
        {content.slice(0, range[0])}
        <span className={wordClass}>{content.slice(range[0], range[1])}</span>
        {content.slice(range[1])}
      </span>
    );
  }
  return <span className="diff-line-text">{content}</span>;
};

export const FileDiff: React.FC<Props> = ({ diff, hideHeader }) => {
  const [open, setOpen] = useState(true);
  const [expanded, setExpanded] = useState(false);
  const lines = useMemo(() => {
    const parsed = parsePatch(diff.patch);
    computeWordRanges(parsed);
    return parsed;
  }, [diff.patch]);
  const lang = useMemo(
    () => (lines.length > MAX_HL_LINES ? undefined : langFromPath(diff.file)),
    [diff.file, lines.length],
  );
  const adds = lines.filter((l) => l.type === "add").length;
  const dels = lines.filter((l) => l.type === "del").length;
  const clipped = lines.length > CLIP_LINES;
  const bodyOpen = hideHeader || open;

  return (
    <div className={`file-diff${hideHeader ? " file-diff--embedded" : ""}`}>
      {!hideHeader && (
        <button className="file-diff-header" onClick={() => setOpen((o) => !o)}>
          <span className="file-diff-chevron">{open ? "▾" : "▸"}</span>
          <button
            className="file-diff-filename"
            title={diff.new_content !== undefined ? `Open diff: ${diff.file}` : `Open: ${diff.file}`}
            onClick={(e) => handleOpenFile(diff, e)}
          >
            {diff.file.split("/").pop() || diff.file}
          </button>
          <span className="file-diff-stat file-diff-stat--add">+{adds}</span>
          <span className="file-diff-stat file-diff-stat--del">−{dels}</span>
        </button>
      )}
      {bodyOpen && (
        <>
          <div className={`file-diff-body${expanded ? " is-expanded" : ""}`}>
            {lines.map((line, i) => (
              <div key={i} className={`diff-line diff-line--${line.type}`}>
                <span className="diff-line-gutter">
                  {line.type === "add" ? "+" : line.type === "del" ? "−" : line.type === "hunk" ? "@@" : ""}
                </span>
                <DiffLineContent line={line} lang={lang} />
              </div>
            ))}
          </div>
          {clipped && (
            <button
              className="show-more-btn"
              onClick={() => setExpanded((e) => !e)}
              aria-expanded={expanded}
            >
              {expanded ? "Show less" : `Show all ${lines.length} lines`}
            </button>
          )}
        </>
      )}
    </div>
  );
};
