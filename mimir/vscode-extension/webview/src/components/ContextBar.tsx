import React from "react";

export interface ContextUsage {
  used_tokens: number;
  total_tokens: number;
  reserved_tokens: number;
  overhead_tokens?: number;
  /** True once the overhead is the server's own count rather than an estimate. */
  overhead_measured?: boolean;
  /** True while the conversation has no agent yet, so the system prompt and tools are
   *  missing from `used_tokens` — the figure is a floor, not a reading. */
  provisional?: boolean;
  /** Messages in the window the model sees this turn. */
  history_messages?: number;
  /** Messages in the untrimmed record kept on disk — larger once the budget has
   *  trimmed the window. */
  history_messages_full?: number;
}

interface Props {
  usage: ContextUsage;
  contextMode: "compact" | "full";
}

export const ContextBar: React.FC<Props> = ({ usage, contextMode }) => {
  const {
    used_tokens, total_tokens, reserved_tokens, overhead_tokens, overhead_measured,
    history_messages, history_messages_full, provisional,
  } = usage;
  const usable = Math.max(1, total_tokens - reserved_tokens);
  const overBy = used_tokens - usable;
  // An overflow is only claimed when the figure is whole. Before the agent exists the
  // fixed part (system prompt + tools) is not in `used_tokens`, so the number is a floor
  // of the conversation's own size and nothing can be concluded from where it sits
  // against the limit — least of all in red, which is how a resumed session came back
  // looking as though it had already overflowed.
  const over = overBy > 0 && !provisional;
  // Budget consumed, as a share of what history may actually use. This is the
  // number the label and the colour thresholds speak in.
  const pct = Math.min(100, Math.round((used_tokens / usable) * 100));
  // The track maps the *whole* window, with the reserved tail drawn at its end,
  // so the fill has to be measured against the window too. Using `pct` here made
  // the fill run into the reserved region and read as full while budget was left.
  const fillPct = Math.min(100, Math.round((used_tokens / Math.max(1, total_tokens)) * 100));

  // Colour thresholds: green → amber → red. An actual overflow forces red. A
  // provisional figure stays neutral: it has no business raising an alarm it cannot
  // substantiate, and it is replaced by a whole one as soon as the agent is up.
  const colour =
    provisional ? "var(--vscode-descriptionForeground, #8a8a8a)"
    : over || pct >= 90 ? "var(--vscode-editorError-foreground, #f44747)"
    : pct >= 70 ? "var(--vscode-editorWarning-foreground, #cca700)"
    : "var(--accent, #0e9eff)";

  const usedK  = (used_tokens  / 1000).toFixed(1);
  const totalK = (total_tokens / 1000).toFixed(0);
  const resK   = (reserved_tokens / 1000).toFixed(0);
  const overK  = (overBy / 1000).toFixed(1);

  // The prompt overhead is part of `used_tokens`; naming it separately explains
  // why the bar never starts at zero on a fresh session. Whether it was measured
  // or estimated is worth a word: the figure legitimately shifts once the first
  // answer lands and the server's own count replaces the estimate, and a number
  // that moves on its own is otherwise indistinguishable from a wrong one.
  const overheadNote = overhead_tokens
    ? ` · incl. ${(overhead_tokens / 1000).toFixed(1)}K system prompt + tools`
      + ` (${overhead_measured ? "measured" : "estimated"})`
    : provisional
      ? " · the system prompt and tools (~30K) are not counted yet — this conversation"
        + " has no agent running. The figure completes on the first query."
      : "";
  // Trimming is silent otherwise: the bar would sit comfortably under the limit while
  // the earliest turns had already dropped out of what the model can see. The record
  // on disk still holds them, and a resume starts from there.
  const trimmed =
    history_messages !== undefined && history_messages_full !== undefined
      && history_messages_full > history_messages;
  const trimNote = trimmed
    ? ` · ${history_messages_full! - history_messages!} earlier message(s) trimmed from `
      + `the window (kept on disk, restored on resume)`
    : "";
  const title = over
    ? `Context OVERFLOW: ~${usedK}K used vs ${(usable / 1000).toFixed(0)}K usable `
      + `(+${overK}K over) · ${resK}K reserved for answer${overheadNote}${trimNote}`
    : `Context: ${provisional ? "at least" : "~"}${usedK}K / ${totalK}K tokens used`
      + ` · ${resK}K reserved for answer${overheadNote}${trimNote}`;

  return (
    <div className={`ctx-bar${over ? " ctx-bar--over" : ""}`} title={title}>
      <div className="ctx-bar__track">
        {/* Used portion */}
        <div
          className="ctx-bar__fill"
          style={{ width: `${fillPct}%`, background: colour }}
        />
        {/* Reserved portion (shown as a dimmed region at the end) */}
        <div
          className="ctx-bar__reserved"
          style={{ width: `${Math.round((reserved_tokens / total_tokens) * 100)}%` }}
        />
      </div>
      <span className="ctx-bar__label">
        {contextMode === "full" ? `${totalK}K` : "ctx"}&nbsp;
        <span style={{ color: colour }}>
          {over ? `+${overK}K over` : provisional ? `≥${pct}%` : `${pct}%`}
        </span>
        <span className="ctx-bar__detail">&nbsp;{usedK}K/{totalK}K</span>
        {trimmed && <span className="ctx-bar__detail">&nbsp;✂</span>}
      </span>
    </div>
  );
};
