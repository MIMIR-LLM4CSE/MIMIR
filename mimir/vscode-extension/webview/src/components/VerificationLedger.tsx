import React from "react";
import { inlineSegments, parseLedger, rowLevel, summaryChips } from "./ledgerUtils";
import type { LedgerStatus } from "./ledgerUtils";

interface Props {
  /** The rendered ledger block lifted off the answer text. */
  block: string;
}

const GLYPH: Record<LedgerStatus, string> = { ok: "✔", note: "✳", warn: "⚠" };

const Row: React.FC<{ row: string }> = ({ row }) => (
  <div className={`ledger-row ledger-row--${rowLevel(row)}`}>
    <span className="ledger-row-dot" aria-hidden="true" />
    <span className="ledger-row-text">
      {inlineSegments(row).map((seg, i) =>
        seg.kind === "code" ? (
          <code key={i} className="ledger-path">{seg.text}</code>
        ) : seg.kind === "strong" ? (
          <strong key={i}>{seg.text}</strong>
        ) : (
          <React.Fragment key={i}>{seg.text}</React.Fragment>
        ),
      )}
    </span>
  </div>
);

/**
 * The machine-recorded verification ledger, as a collapsed disclosure under the
 * answer: one status line the eye can skip, the evidence a click away. Native
 * <details> handles the toggle — no state, no wiring.
 */
export const VerificationLedger: React.FC<Props> = ({ block }) => {
  const ledger = parseLedger(block);
  if (ledger.rows.length === 0) return null;
  const chips = summaryChips(ledger.summary);

  return (
    <details className={`ledger ledger--${ledger.status}`}>
      <summary className="ledger-summary" title="Recorded by MIMIR from the run itself — not written by the model">
        <span className="ledger-chevron" aria-hidden="true">▸</span>
        <span className="ledger-glyph" aria-hidden="true">{GLYPH[ledger.status]}</span>
        <span className="ledger-label">Verification</span>
        {chips.map((chip, i) => (
          <span key={i} className="ledger-chip">{chip}</span>
        ))}
      </summary>
      <div className="ledger-body">
        {ledger.rows.map((row, i) => <Row key={i} row={row} />)}
        <div className="ledger-foot">machine-recorded, not model-authored</div>
      </div>
    </details>
  );
};
