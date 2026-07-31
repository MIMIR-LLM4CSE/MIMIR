import React, { useState } from "react";
import type { DiffEntry } from "../types";
import { vscodePostMessage } from "../hooks/useWebSocket";
import { FileDiff } from "./FileDiff";
import { countDiffLines } from "./diffUtils";

interface Props {
  files: DiffEntry[];
  onAccept: () => void;
  onRevert: () => void;
  onAcceptFile: (file: string) => void;
  onRevertFile: (file: string) => void;
}

export const BatchReviewBar: React.FC<Props> = ({ files, onAccept, onRevert, onAcceptFile, onRevertFile }) => {
  const [expanded, setExpanded] = useState(true);
  // Per-file inline-diff disclosure state, keyed by file path.
  const [openDiffs, setOpenDiffs] = useState<Record<string, boolean>>({});
  const toggleDiff = (file: string) =>
    setOpenDiffs((m) => ({ ...m, [file]: !m[file] }));

  const totalAdds = files.reduce((s, f) => s + countDiffLines(f.patch).adds, 0);
  const totalDels = files.reduce((s, f) => s + countDiffLines(f.patch).dels, 0);

  return (
    <div className="brb">
      {/* Header row */}
      <div className="brb-header">
        <button
          className="brb-toggle"
          onClick={() => setExpanded((v) => !v)}
          title={expanded ? "Collapse file list" : "Expand file list"}
        >
          <span className="brb-chevron">{expanded ? "▾" : "▸"}</span>
          <span className="brb-icon">✏️</span>
          <span className="brb-title">
            {files.length === 1 ? "1 file changed" : `${files.length} files changed`}
          </span>
          <span className="brb-stat brb-stat--add">+{totalAdds}</span>
          <span className="brb-stat brb-stat--del">−{totalDels}</span>
        </button>

        <div className="brb-actions">
          <button
            className="brb-btn brb-btn--accept"
            onClick={onAccept}
            title="Keep all changes"
          >
            Keep all
          </button>
          <button
            className="brb-btn brb-btn--revert"
            onClick={onRevert}
            title="Revert all changes"
          >
            Revert all
          </button>
        </div>
      </div>

      {/* File list (collapsible) */}
      {expanded && (
        <div className="brb-file-list">
          {files.map((f, i) => {
            const { adds, dels } = countDiffLines(f.patch);
            const name = f.file.split("/").pop() || f.file;
            const diffOpen = !!openDiffs[f.file];
            return (
              <div key={i} className="brb-file">
                <div className="brb-file-row">
                  <button
                    className="brb-file-disclose"
                    aria-expanded={diffOpen}
                    aria-label={diffOpen ? "Hide diff" : "Show diff"}
                    onClick={() => toggleDiff(f.file)}
                  >
                    <span className={`brb-file-chevron${diffOpen ? " brb-file-chevron--open" : ""}`}>▸</span>
                  </button>
                  <span className="brb-file-icon">{f.is_new ? "🆕" : "✏️"}</span>
                  <button
                    className="brb-file-name"
                    title={`Open diff: ${f.file}`}
                    onClick={() =>
                      // Always open a native VS Code diff editor (red/green line
                      // backgrounds) rather than the raw +/- .diff text: pass the
                      // proposed content when we have it, else the patch (the host
                      // reverse-applies it to reconstruct Before ↔ After).
                      vscodePostMessage(
                        f.new_content !== undefined
                          ? { type: "open_diff", file: f.file, new_content: f.new_content }
                          : { type: "open_diff", file: f.file, patch: f.patch }
                      )
                    }
                  >
                    {name}
                  </button>
                  <span className="brb-file-path">{f.file}</span>
                  <span className="brb-stat brb-stat--add">+{adds}</span>
                  <span className="brb-stat brb-stat--del">−{dels}</span>
                  <div className="brb-file-actions">
                    <button
                      className="brb-file-btn brb-file-btn--accept"
                      title="Keep this file"
                      onClick={() => onAcceptFile(f.file)}
                    >✓</button>
                    <button
                      className="brb-file-btn brb-file-btn--revert"
                      title="Revert this file"
                      onClick={() => onRevertFile(f.file)}
                    >✗</button>
                  </div>
                </div>
                {diffOpen && (
                  <div className="brb-file-diff">
                    <FileDiff diff={f} hideHeader />
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};
