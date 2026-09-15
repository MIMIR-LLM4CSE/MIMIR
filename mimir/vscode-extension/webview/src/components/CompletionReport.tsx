import React from "react";
import { MarkdownContent } from "./MarkdownContent";
import { parseCompletion } from "./completionUtils";
import type { Completion } from "./completionUtils";

interface Props {
  /** The rendered completion block lifted off the answer text. */
  block: string;
}

/** Three headlines, three glyphs — the one thing readable without unfolding. */
const GLYPH: Record<Completion["status"], string> = {
  "incomplete": "⚠",
  "handback": "✋",
  "refused-only": "✔",
};

/**
 * How loud the panel is. A hand-back stopped the work, so it is never quiet whatever
 * the risk line says; otherwise the report's own residual risk decides.
 */
function tone(report: Completion): "ok" | "note" | "warn" {
  if (report.status === "handback") return "warn";
  if (report.risk === "high") return "warn";
  if (report.risk === "low") return "ok";
  return "note";
}

/**
 * The machine's account of how the run ended, as a collapsed disclosure under the
 * answer — the same shape as the verification ledger it sits beside, and reusing its
 * styling for that reason.
 *
 * It used to be bare prose concatenated ahead of the answer, which no front-end could
 * lift off, so the whole report rendered as body text and the model's own words came
 * underneath it. Collapsed, the headline and the residual risk stay in view and the
 * sections are one click away.
 */
export const CompletionReport: React.FC<Props> = ({ block }) => {
  const report = parseCompletion(block);
  if (!report.body) return null;

  return (
    <details className={`ledger ledger--report ledger--${tone(report)}`}>
      <summary
        className="ledger-summary"
        title="Recorded by MIMIR from the run itself — not written by the model"
      >
        <span className="ledger-chevron" aria-hidden="true">▸</span>
        <span className="ledger-glyph" aria-hidden="true">{GLYPH[report.status]}</span>
        <span className="ledger-label">{report.headline}</span>
        {report.risk && <span className="ledger-chip">risk: {report.risk}</span>}
      </summary>
      <div className="ledger-body">
        <MarkdownContent text={report.body} />
        <div className="ledger-foot">machine-recorded, not model-authored</div>
      </div>
    </details>
  );
};
