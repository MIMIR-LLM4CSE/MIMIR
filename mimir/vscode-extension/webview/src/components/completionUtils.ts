// Completion-report parsing — mirrors client/guardrails/workflow.py.
//
// Same contract as the verification ledger next door: the agent appends the report to
// the answer behind a marker comment so the model's history keeps it, and the webview
// lifts it out of the prose to render it as a collapsed panel (CompletionReport).
//
// The report used to be concatenated *ahead* of the answer as bare prose, with no
// marker to lift it by — so the whole of it rendered as body text and the model's own
// words arrived underneath. Folding it is all that changed; nothing is dropped.

export type CompletionStatus = "incomplete" | "handback" | "refused-only";

export interface Completion {
  status: CompletionStatus;
  /** "low" | "medium" | "high" — the report's own residual-risk line. */
  risk: string;
  /** The one line meant to be readable without unfolding anything. */
  headline: string;
  /** The sections, as markdown. */
  body: string;
}

const MARKER = "<!--mimir:completion";
const MARKER_RE = /<!--mimir:completion([^>]*)-->/;
const ATTR_RE = /(\w+)="([^"]*)"/g;
const FRAMING = "Completion report — machine-recorded, not model-authored:";

/**
 * Split an answer into its prose and its completion block.
 *
 * Both blocks ride at the tail, the ledger appended last, so a caller splits the
 * ledger off first and passes the rest here.
 */
export function splitAnswerCompletion(text: string): { body: string; report: string | null } {
  const idx = text.lastIndexOf(MARKER);
  if (idx === -1) return { body: text, report: null };
  return { body: text.slice(0, idx).trimEnd(), report: text.slice(idx).trim() };
}

/** Recover the header fields and body of a rendered completion block. */
export function parseCompletion(block: string): Completion {
  const m = MARKER_RE.exec(block);
  const attrs: Record<string, string> = {};
  if (m) {
    for (const [, key, value] of m[1].matchAll(ATTR_RE)) attrs[key] = value;
  }
  let body = m ? block.slice(m.index + m[0].length) : block;
  body = body.replace(/^\n+/, "");
  if (body.startsWith(FRAMING)) body = body.slice(FRAMING.length);
  const status = attrs.status;
  return {
    status:
      status === "handback" || status === "refused-only" ? status : "incomplete",
    risk: attrs.risk ?? "",
    headline: attrs.headline ?? "",
    body: body.trim(),
  };
}
