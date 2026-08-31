import React from "react";

export interface ContextUsage {
  used_tokens: number;
  total_tokens: number;
  reserved_tokens: number;
  overhead_tokens?: number;
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
    used_tokens, total_tokens, reserved_tokens, overhead_tokens,
    history_messages, history_messages_full,
  } = usage;
  const usable = Math.max(1, total_tokens - reserved_tokens);
  const overBy = used_tokens - usable;
  const over = overBy > 0;
  // Budget consumed, as a share of what history may actually use. This is the
  // number the label and the colour thresholds speak in.
  const pct = Math.min(100, Math.round((used_tokens / usable) * 100));
  // The track maps the *whole* window, with the reserved tail drawn at its end,
  // so the fill has to be measured against the window too. Using `pct` here made
  // the fill run into the reserved region and read as full while budget was left.
  const fillPct = Math.min(100, Math.round((used_tokens / Math.max(1, total_tokens)) * 100));

  // Colour thresholds: green → amber → red. An actual overflow forces red.
  const colour =
    over || pct >= 90 ? "var(--vscode-editorError-foreground, #f44747)"
    : pct >= 70 ? "var(--vscode-editorWarning-foreground, #cca700)"
    : "var(--accent, #0e9eff)";

  const usedK  = (used_tokens  / 1000).toFixed(1);
  const totalK = (total_tokens / 1000).toFixed(0);
  const resK   = (reserved_tokens / 1000).toFixed(0);
  const overK  = (overBy / 1000).toFixed(1);

  // The prompt overhead is part of `used_tokens`; naming it separately explains
  // why the bar never starts at zero on a fresh session.
  const overheadNote = overhead_tokens
    ? ` · incl. ${(overhead_tokens / 1000).toFixed(1)}K system prompt + tools`
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
    : `Context: ~${usedK}K / ${totalK}K tokens used · ${resK}K reserved for answer`
      + `${overheadNote}${trimNote}`;

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
        <span style={{ color: colour }}>{over ? `+${overK}K over` : `${pct}%`}</span>
        <span className="ctx-bar__detail">&nbsp;{usedK}K/{totalK}K</span>
        {trimmed && <span className="ctx-bar__detail">&nbsp;✂</span>}
      </span>
    </div>
  );
};
