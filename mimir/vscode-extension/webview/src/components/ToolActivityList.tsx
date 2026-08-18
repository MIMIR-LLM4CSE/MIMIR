import React, { useEffect, useRef, useState } from "react";
import type { ExecResult, ToolActivity } from "../types";
import { useElapsed, formatDuration } from "../hooks/useElapsed";

/** Terminal-style in/out panel for an exec-shaped tool result (shell, code
 *  runner, compiler): the command that ran (IN) and its output (OUT). Each pane
 *  scrolls on its own inside its own height cap, so a long command can never
 *  push the output off-panel — IN and OUT are both always in view. Both caps are
 *  restricted by default and grow on demand; stderr is shown in a neutral tone
 *  and only turns red when the command actually failed (non-zero exit), so
 *  routine stderr chatter doesn't read as an error. */
const ExecOutput: React.FC<{ exec: ExecResult }> = ({ exec }) => {
  const [full, setFull] = useState(false);
  const inRef = useRef<HTMLPreElement>(null);
  const outRef = useRef<HTMLDivElement>(null);
  const [overflows, setOverflows] = useState(false);

  const commandLines = (exec.command ?? "").split("\n");
  const noOutput = !exec.stdout && !exec.stderr;
  const failed = exec.returncode !== 0;
  const showNotes = failed || !!exec.truncated || noOutput;

  // Detect whether *either* pane clips its content, so the expand control only
  // appears when it would actually do something (or to collapse back).
  useEffect(() => {
    const clipped = (el: HTMLElement | null) =>
      !!el && el.scrollHeight > el.clientHeight + 2;
    setOverflows(clipped(inRef.current) || clipped(outRef.current));
  }, [exec.command, exec.stdout, exec.stderr, full]);
  const showResize = overflows || full;

  return (
    <div
      className={
        "tool-exec" +
        (full ? " tool-exec--full" : "") +
        (failed ? " tool-exec--failed" : "")
      }
    >
      <div className="tool-exec-body">
        {exec.command && (
          <div className="tool-exec-section tool-exec-section--in">
            <span className="tool-exec-tag tool-exec-tag--in">IN</span>
            <pre
              className="tool-exec-cmd tool-exec-body-col"
              ref={inRef}
              title={exec.cwd ? `cwd: ${exec.cwd}` : undefined}
            >
              {commandLines.map((line, i) => (
                <div key={i}>
                  <span className="tool-exec-prompt" aria-hidden="true">{i === 0 ? "$" : "›"}</span>
                  {line}
                </div>
              ))}
            </pre>
          </div>
        )}
        {!noOutput && (
          <div className="tool-exec-section">
            <span className="tool-exec-tag tool-exec-tag--out">OUT</span>
            <div className="tool-exec-out tool-exec-body-col" ref={outRef}>
              {exec.stdout && <pre className="tool-exec-stdout">{exec.stdout}</pre>}
              {exec.stderr && <pre className="tool-exec-stderr">{exec.stderr}</pre>}
            </div>
          </div>
        )}
      </div>
      {(showNotes || showResize) && (
        <div className="tool-exec-meta">
          {failed && (
            <span className="tool-exec-badge">exit {exec.returncode}</span>
          )}
          {noOutput && <span className="tool-exec-note">(no output)</span>}
          {exec.truncated && <span className="tool-exec-note">output truncated</span>}
          {showResize && (
            <button
              className="tool-exec-resize"
              onClick={() => setFull((f) => !f)}
            >
              {full ? "Collapse" : "Expand"}
            </button>
          )}
        </div>
      )}
    </div>
  );
};

/** Full-width panel carrying the error text of a failed tool call. The row tail
 *  can only show a cropped one-liner, so the real message lives here: wrapped
 *  over as many lines as it needs, scrolling past a bounded height.
 *
 *  Deliberately has NO height toggle of its own: the row's expand/collapse is the
 *  single control. A second "Collapse" button inside the panel only shrank its
 *  height, so clicking it — the obvious target once the panel is open — left the
 *  panel open and read as "the row won't collapse". Its footer button now closes
 *  the row itself, same as clicking the row head. */
const ErrorOutput: React.FC<{ error: string; onCollapse: () => void }> = ({
  error,
  onCollapse,
}) => (
  <div className="tool-error">
    <div className="tool-error-body">
      <span className="tool-exec-tag tool-exec-tag--err">ERR</span>
      <pre className="tool-error-text">{error}</pre>
    </div>
    <div className="tool-exec-meta">
      <button className="tool-exec-resize" onClick={onCollapse}>
        Collapse
      </button>
    </div>
  </div>
);

/** Tool label with its leading verb emphasized: "**Reading** file: x.py".
 *  The verb carries the meaning at a glance; the rest stays dim. */
const ToolLabel: React.FC<{ label: string }> = ({ label }) => {
  const space = label.indexOf(" ");
  if (space <= 0) return <span className="tool-label">{label}</span>;
  return (
    <span className="tool-label">
      <span className="tool-label-verb">{label.slice(0, space)}</span>
      {label.slice(space)}
    </span>
  );
};

interface RowProps {
  tool: ToolActivity;
}

const ToolRow: React.FC<RowProps> = ({ tool }) => {
  const isError = tool.status === "error";
  const hasExec = tool.exec !== undefined;
  const hasError = isError && !!tool.error;
  // Anything the row can reveal — exec panel or error text — makes the head a
  // working toggle. A failed row without exec used to be inert, leaving its only
  // explanation cropped in the tail.
  const canExpand = hasExec || hasError;
  // The tail is a narrow, single-line slot: an error there renders as a fragment
  // cropped mid-sentence. Failed rows keep it clear (the ✕ glyph carries the
  // status) and state the problem on the brief line below instead.
  const tailSummary = isError ? "" : tool.summary;
  // Collapsed, a failed row shows ONE line: the first line of the error, clipped.
  // The full text — stack, hint, every line — appears only once expanded.
  const briefError = (tool.error || tool.summary || "failed").split("\n")[0].trim();
  // Only *successful* exec-shaped rows (shell / code runner / compiler) open by
  // default, so the terminal IN/OUT panel is visible without a click. Failed or
  // blocked rows stay collapsed — the explanation is a dropdown the user opens on
  // demand — and non-exec rows have nothing to reveal.
  const [expanded, setExpanded] = useState(hasExec && !isError);
  // Auto-reveal the terminal panel when exec output arrives on a row that did NOT
  // fail. A live row mounts as "running" with no exec (so the initial state above
  // is false); the exec payload lands later on tool_result. This fires once on
  // that transition — the user can still collapse it afterwards. Failed rows are
  // intentionally excluded so their explanation is not shown until requested.
  useEffect(() => {
    if (hasExec && !isError) setExpanded(true);
  }, [hasExec, isError]);

  const running = tool.status === "running";
  const elapsed = useElapsed(tool.startedAt, running);
  const duration = tool.durationMs ?? elapsed;

  // Success is the norm and needs no mark — a clean row already reads as "fine".
  // Only failure is worth a glyph, so the ✕ stands out instead of being one more
  // symbol in a column of them.
  const statusGlyph = tool.status === "error" ? "✕" : null;

  return (
    <div className={`tool-row tool-row--${tool.status}`}>
      <button
        className="tool-row-head"
        onClick={() => canExpand && setExpanded((e) => !e)}
        disabled={!canExpand}
        title={tool.error || tool.detail || tool.label}
      >
        {running ? (
          <span className="tb-spinner" aria-hidden="true" />
        ) : (
          <span className="tool-icon" aria-hidden="true">{tool.icon}</span>
        )}
        <ToolLabel label={tool.label} />
        {tool.detail && <span className="tool-detail">{tool.detail}</span>}
        <span className="tool-row-tail">
          {tailSummary && <span className="tool-summary">{tailSummary}</span>}
          {statusGlyph && (
            <span className={`tool-status-glyph tool-status-glyph--${tool.status}`}>
              {statusGlyph}
            </span>
          )}
          <span className="tool-duration">{formatDuration(duration)}</span>
          {canExpand && (
            <span className="tool-chevron" aria-hidden="true">
              {expanded ? "▾" : "▸"}
            </span>
          )}
        </span>
      </button>
      {expanded && tool.exec && <ExecOutput exec={tool.exec} />}
      {hasError &&
        (expanded ? (
          <ErrorOutput error={tool.error!} onCollapse={() => setExpanded(false)} />
        ) : (
          // Collapsed: a single full-width line, ellipsized at the row edge — never
          // squeezed into the right-hand tail. Clicking it opens the full panel.
          <button
            className="tool-error-brief"
            onClick={() => setExpanded(true)}
            title={briefError}
          >
            {briefError}
          </button>
        ))}
    </div>
  );
};

interface Props {
  tools: ToolActivity[];
}

/** Renders a list of tool invocations (live or frozen) as a compact card. */
export const ToolActivityList: React.FC<Props> = ({ tools }) => {
  if (tools.length === 0) return null;
  return (
    <div className="tool-card">
      {tools.map((t) => (
        <ToolRow key={t.id} tool={t} />
      ))}
    </div>
  );
};
