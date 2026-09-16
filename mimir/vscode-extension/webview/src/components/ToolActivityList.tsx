import React, { useEffect, useRef, useState } from "react";
import type { CSSProperties } from "react";
import type { ExecResult, ToolActivity } from "../types";
import { subAgentTail } from "./subAgentUtils";
import { useElapsed, formatDuration } from "../hooks/useElapsed";

/** Terminal-style in/out panel for an exec-shaped tool result (shell, code
 *  runner, compiler): the command that ran (IN) and its output (OUT). Each pane
 *  scrolls on its own inside its own height cap, so a long command can never
 *  push the output off-panel — IN and OUT are both always in view. Both caps are
 *  restricted by default and grow on demand; stderr is shown in a neutral tone
 *  and only turns red when the command actually failed (non-zero exit), so
 *  routine stderr chatter doesn't read as an error.
 *
 *  While the row is still running the panel is half-built: the command is known
 *  from the call, the output does not exist yet. `pending` is what tells the two
 *  apart — an empty OUT under a finished run is the fact "(no output)", under a
 *  live one it is simply not in yet, and saying "(no output)" there is wrong. */
const ExecOutput: React.FC<{ exec: ExecResult; pending?: boolean }> = ({
  exec,
  pending,
}) => {
  const [full, setFull] = useState(false);
  const inRef = useRef<HTMLPreElement>(null);
  const outRef = useRef<HTMLDivElement>(null);
  const [overflows, setOverflows] = useState(false);

  const commandLines = (exec.command ?? "").split("\n");
  const noOutput = !exec.stdout && !exec.stderr;
  const awaitingOutput = !!pending && noOutput;
  // A detached run carries no exit code: without the first clause every diverted
  // run would read as a failure, since `undefined !== 0`.
  const failed = exec.returncode !== undefined && exec.returncode !== 0;
  const showNotes =
    failed || !!exec.truncated || (noOutput && !pending) || !!exec.running;

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
        {(!noOutput || awaitingOutput) && (
          <div className="tool-exec-section">
            <span className="tool-exec-tag tool-exec-tag--out">OUT</span>
            <div className="tool-exec-out tool-exec-body-col" ref={outRef}>
              {exec.stdout && <pre className="tool-exec-stdout">{exec.stdout}</pre>}
              {exec.stderr && <pre className="tool-exec-stderr">{exec.stderr}</pre>}
              {/* The pane is kept, empty, rather than left out: dropping it would
                  reflow the panel the moment the output lands. */}
              {awaitingOutput && (
                <pre className="tool-exec-waiting">running…</pre>
              )}
            </div>
          </div>
        )}
      </div>
      {(showNotes || showResize) && (
        <div className="tool-exec-meta">
          {exec.running ? (
            <span
              className="tool-exec-badge tool-exec-badge--bg"
              title="Still running in the background — this is the output it had produced when it was detached"
            >
              background{exec.job_key ? ` · ${exec.job_key}` : ""}
            </span>
          ) : (
            failed && <span className="tool-exec-badge">exit {exec.returncode}</span>
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
  /** Rows a delegating call produced — rendered as its collapsible content. */
  childRows?: ToolActivity[];
  /** Detach this run, keeping what it has already done. Absent on frozen rows. */
  onDivert?: (id: string) => void;
}

const ToolRow: React.FC<RowProps> = ({ tool, childRows = [], onDivert }) => {
  const isError = tool.status === "error";
  const hasExec = tool.exec !== undefined;
  const hasError = isError && !!tool.error;
  const hasChildren = childRows.length > 0;
  // Anything the row can reveal — exec panel, error text or a sub-agent's steps —
  // makes the head a working toggle. A failed row without exec used to be inert,
  // leaving its only explanation cropped in the tail.
  const canExpand = hasExec || hasError || hasChildren;
  // The tail is a narrow, single-line slot: an error there renders as a fragment
  // cropped mid-sentence. Failed rows keep it clear — the ✕ carries the status and
  // the full command/error is one click away.
  const tailSummary = isError ? "" : tool.summary;
  // Only *successful* exec-shaped rows (shell / code runner / compiler) open by
  // default, so the terminal IN/OUT panel is visible without a click. Failed or
  // blocked rows stay collapsed — the explanation is a dropdown the user opens on
  // demand — and non-exec rows have nothing to reveal.
  // A live row mounts as "running" with no exec and the payload lands later, on
  // tool_result, so the panel has to open on that transition. Derived during
  // render rather than set from an effect: an effect opens it in a SECOND commit,
  // and the transcript is pinned to its bottom in the first one — so the pane was
  // pinned to the height of the collapsed row and the terminal panel unrolled
  // below the fold, which is how a bash call came out cut off at the bottom. A
  // click is kept as an override, and until then the default is re-derived, so a
  // row the user has not touched is always shown at its natural size.
  const [toggled, setToggled] = useState<boolean | null>(null);
  const expanded = toggled ?? (hasExec && !isError);
  const setExpanded = (next: boolean | ((prev: boolean) => boolean)) =>
    setToggled((prev) => {
      const current = prev ?? (hasExec && !isError);
      return typeof next === "function" ? next(current) : next;
    });

  // The row's arg preview IS the command line for an exec-shaped row, so while the
  // IN pane below is open it says it twice — cropped up here, in full down there.
  // Dropped only while the panel is open: collapsed, the one-liner is the only
  // trace of what ran.
  const showsCommandBelow = expanded && hasExec && !!tool.exec?.command;

  const running = tool.status === "running";
  // Latched locally rather than waiting for the row to change status: the request
  // travels to the shell server and back, and a control that stays live in the
  // meantime invites a second click on a run already on its way out.
  const [diverting, setDiverting] = useState(false);
  const canDivert = running && !!tool.divertible && !!onDivert;
  // A percentage is only meaningful while the row is running: a settled row's bar
  // would describe a moment that has passed.
  const hasProgress = running && typeof tool.percent === "number";
  const pct = hasProgress ? Math.max(0, Math.min(100, tool.percent as number)) : 0;
  const elapsed = useElapsed(tool.startedAt, running);
  const duration = tool.durationMs ?? elapsed;

  // A run's exit code and its result are different facts: the badge carries what the
  // model said the output showed, and only appears once it has said it.
  const verdictBadge = tool.verdict ? (
    <span
      className={`tool-verdict tool-verdict--${tool.verdict}`}
      title={`The model judged this run's output: ${tool.verdict}`}
    >
      {tool.verdict}
    </span>
  ) : null;

  return (
    <div
      className={`tool-row tool-row--${tool.status}${tool.parentId ? " tool-row--child" : ""}`}
      // How anything outside this list finds the row again: the progress dock
      // watches it to know whether it has scrolled away, and scrolls back to it.
      data-tool-id={tool.id}
    >
      <div
        className={`tool-row-line${hasProgress ? " tool-row-line--progress" : ""}`}
        // The fill is a custom property rather than an element: the row is a button
        // whose hover paints its own background, so a bar *under* it would vanish on
        // hover and one beside it would cost the row a second line. The stylesheet
        // draws it as a translucent overlay from this one number.
        style={hasProgress ? ({ "--tool-progress": `${pct}%` } as CSSProperties) : undefined}
        role={hasProgress ? "progressbar" : undefined}
        aria-valuenow={hasProgress ? pct : undefined}
        aria-valuemin={hasProgress ? 0 : undefined}
        aria-valuemax={hasProgress ? 100 : undefined}
        aria-label={hasProgress ? tool.phase || tool.label : undefined}
      >
      <button
        className="tool-row-head"
        onClick={() => canExpand && setExpanded((e) => !e)}
        disabled={!canExpand}
        title={tool.error || tool.detail || tool.label}
      >
        {/* Success is the norm and needs no mark; only failure gets a glyph, and it
            leads the row so the status is read before the label. */}
        {isError && (
          <span className="tool-status-glyph tool-status-glyph--error" aria-label="failed">
            ✕
          </span>
        )}
        {running ? (
          <span className="tb-spinner" aria-hidden="true" />
        ) : (
          <span className="tool-icon" aria-hidden="true">{tool.icon}</span>
        )}
        {/* Whose work this is. Several sub-agents run at once, and their rows sit in
            the same list as the agent's own — without the badge a reader cannot tell
            which run a line belongs to. */}
        {tool.origin && (
          <span className="tool-origin" title={`Work of sub-agent ${tool.origin}`}>
            {tool.origin}
          </span>
        )}
        <ToolLabel label={tool.label} />
        {tool.detail && !showsCommandBelow && (
          <span className="tool-detail">{tool.detail}</span>
        )}
        <span className="tool-row-tail">
          {/* What the run is doing. The dots are what say it is still doing it: a
              phase that only changes every few minutes would otherwise read as a
              frozen label on a row that has stopped. */}
          {tool.phase && running && (
            <span className="tool-phase" title={tool.phase}>
              {/* The text truncates on its own so the dots are never the part that
                  gets clipped — they are the half that says the run is still alive. */}
              <span className="tool-phase-text">{tool.phase}</span>
              <span className="streaming-dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </span>
            </span>
          )}
          {tool.waiting && tool.status === "running" && (
            <span
              className="tool-summary"
              title="The sub-agent is in a model turn — no step to show yet"
            >
              {tool.waiting}
            </span>
          )}
          {hasChildren && (
            <span className="tool-summary">{subAgentTail([tool, ...childRows], tool.id)}</span>
          )}
          {tool.childrenDropped ? (
            <span
              className="tool-summary"
              title="Sub-agent steps that were shed rather than shown"
            >
              +{tool.childrenDropped} more
            </span>
          ) : null}
          {tailSummary && <span className="tool-summary">{tailSummary}</span>}
          {verdictBadge}
          <span className="tool-duration">{formatDuration(duration)}</span>
          {canExpand && (
            <span className="tool-chevron" aria-hidden="true">
              {expanded ? "▾" : "▸"}
            </span>
          )}
        </span>
      </button>
      {/* Icon-only: the row is dense, and the label repeated what the tooltip and
          the aria-label already say. A sibling of the head rather than a child — the
          head is itself a button, and a button inside a button is not valid markup. */}
      {canDivert && (
        <button
          className="tool-divert"
          disabled={diverting}
          aria-label="Move this run to the background without interrupting it"
          title={
            diverting
              ? "Moving it to the background…"
              : "Move this run to the background without interrupting it"
          }
          onClick={(e) => {
            e.stopPropagation();
            setDiverting(true);
            onDivert!(tool.id);
          }}
        >
          <span className="tool-divert-icon" aria-hidden="true">
            {diverting ? "⋯" : "↗"}
          </span>
        </button>
      )}
      </div>
      {expanded && tool.exec && (
        <ExecOutput exec={tool.exec} pending={running} />
      )}
      {expanded && hasError && (
        <ErrorOutput error={tool.error!} onCollapse={() => setExpanded(false)} />
      )}
      {expanded && hasChildren && (
        <div className="tool-children">
          {childRows.map((c) => (
            <ToolRow key={c.id} tool={c} />
          ))}
        </div>
      )}
    </div>
  );
};

interface Props {
  tools: ToolActivity[];
  /** Detach a running row. Passed only for the live list: frozen rows are settled. */
  onDivert?: (id: string) => void;
}

/** Renders a list of tool invocations (live or frozen) as a compact card.
 *
 *  A sub-agent's steps are its parent's content, not siblings of the agent's own
 *  work: a delegated run can be dozens of rows, and flat they buried the turn. They
 *  are grouped under the delegating row and folded away until asked for. A child
 *  whose parent is gone from this list still gets rendered, flat, rather than lost. */
export const ToolActivityList: React.FC<Props> = ({ tools, onDivert }) => {
  if (tools.length === 0) return null;
  const ids = new Set(tools.map((t) => t.id));
  const childrenOf = new Map<string, ToolActivity[]>();
  for (const t of tools) {
    if (t.parentId && ids.has(t.parentId)) {
      childrenOf.set(t.parentId, [...(childrenOf.get(t.parentId) ?? []), t]);
    }
  }
  return (
    <div className="tool-card">
      {tools
        .filter((t) => !(t.parentId && ids.has(t.parentId)))
        .map((t) => (
          <ToolRow
            key={t.id}
            tool={t}
            childRows={childrenOf.get(t.id)}
            onDivert={onDivert}
          />
        ))}
    </div>
  );
};
