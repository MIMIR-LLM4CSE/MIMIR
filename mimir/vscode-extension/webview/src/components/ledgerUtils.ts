// Verification-ledger parsing — mirrors client/query_engine/verification.py.
//
// The agent appends the ledger to the answer text behind a marker comment, so the
// model's history keeps it. The webview lifts it out of the prose and renders it as a
// collapsed panel (VerificationLedger). Anything the marker doesn't declare is
// recovered from the markdown rows, so an unparseable block still renders.

export type LedgerStatus = "ok" | "note" | "warn";

export interface Ledger {
  status: LedgerStatus;
  /** Number of files written this turn (0 for a checklist-only ledger). */
  files: number;
  /** One-line header summary, " · "-separated chips. */
  summary: string;
  /** Ledger rows as markdown (inline emphasis and `code` spans included). */
  rows: string[];
}

const MARKER = "<!--mimir:ledger";
const MARKER_RE = /<!--mimir:ledger([^>]*)-->/;
const ATTR_RE = /(\w+)="([^"]*)"/g;

/**
 * Split an answer into its prose and its ledger block.
 * The ledger is always the tail, so the marker's position ends the prose.
 */
export function splitAnswerLedger(text: string): { body: string; ledger: string | null } {
  const idx = text.lastIndexOf(MARKER);
  if (idx === -1) return { body: text, ledger: null };
  return { body: text.slice(0, idx).trimEnd(), ledger: text.slice(idx).trim() };
}

/** Recover the header fields and rows of a rendered ledger block. */
export function parseLedger(block: string): Ledger {
  const m = MARKER_RE.exec(block);
  const attrs: Record<string, string> = {};
  if (m) {
    for (const [, key, value] of m[1].matchAll(ATTR_RE)) attrs[key] = value;
  }
  const rows = block
    .split("\n")
    .map((l) => l.trim())
    .filter((l) => l.startsWith("- "))
    .map((l) => l.slice(2).trim());
  const status = attrs.status;
  return {
    status: status === "ok" || status === "warn" ? status : "note",
    files: Number.parseInt(attrs.files ?? "0", 10) || 0,
    summary: attrs.summary ?? "",
    rows,
  };
}

/** The summary split into chips, for the panel header. */
export function summaryChips(summary: string): string[] {
  return summary
    .split("·")
    .map((c) => c.trim())
    .filter(Boolean);
}

/**
 * A row's severity. Bold emphasis is the ledger's convention for "a reader has to
 * act on this" (an unvalidated file, an open checklist step), so it is what tints
 * the row; a file row that carries a tier is settled evidence.
 */
export function rowLevel(row: string): LedgerStatus {
  if (row.includes("**")) return "warn";
  if (row.startsWith("`") && row.includes("validated:")) return "ok";
  return "note";
}

export type Segment = { kind: "text" | "code" | "strong"; text: string };

const INLINE_RE = /`([^`]+)`|\*\*([^*]+)\*\*/g;

/**
 * Split a row into inline segments. Rows only ever use `code` spans and bold, so a
 * two-token scan replaces a markdown pass and keeps the row a single flex line.
 */
export function inlineSegments(row: string): Segment[] {
  const out: Segment[] = [];
  let last = 0;
  for (const m of row.matchAll(INLINE_RE)) {
    const at = m.index ?? 0;
    if (at > last) out.push({ kind: "text", text: row.slice(last, at) });
    // Alternation: group 1 is a code span, group 2 a bold run — never both.
    out.push(m[1] ? { kind: "code", text: m[1] } : { kind: "strong", text: m[2] });
    last = at + m[0].length;
  }
  if (last < row.length) out.push({ kind: "text", text: row.slice(last) });
  return out;
}
