// The verification ledger the agent appends to an answer behind a marker comment
// (client/guardrails/verification/ledger.py), so the model's history keeps it. The webview
// lifts it off the prose.

const MARKER = "<!--mimir:ledger";

/**
 * Split an answer into its prose and its ledger block.
 * The ledger is always the tail, so the marker's position ends the prose.
 */
export function splitAnswerLedger(text: string): { body: string; ledger: string | null } {
  const idx = text.lastIndexOf(MARKER);
  if (idx === -1) return { body: text, ledger: null };
  return { body: text.slice(0, idx).trimEnd(), ledger: text.slice(idx).trim() };
}
