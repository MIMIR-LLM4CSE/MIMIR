import React from "react";
import type { PanelGroup, WatchedRun } from "../types";
import { badgeDetailRows, badgeKey, badgeLabel, kindLabel } from "./panelGroupUtils";

interface Props {
  groups: PanelGroup[];
  open: Record<string, boolean>;
  onToggle: (key: string) => void;
  onPrompt: (text: string) => void;
  /** Stop a run by its key; offered only on a run whose server named how. */
  onCancelRun: (jobKey: string) => void;
  depth?: number;
}

/**
 * A section's nested half: a scheduler, the kinds of machine on it, what one of them is.
 *
 * Renders whatever a server put in `groups` without knowing what it means. Open state is
 * held by the panel, keyed on the server's ids, so a fresh `panel_report` does not
 * collapse what the reader opened.
 */
export const PanelGroupList: React.FC<Props> = ({
  groups,
  open,
  onToggle,
  onPrompt,
  onCancelRun,
  depth = 0,
}) => (
  <>
    {groups.map((group) => {
      const isOpen = !!open[group.id];
      const bodyId = `pg-${group.id}`;
      return (
        <div className="science-group" key={group.id} style={{ ["--depth" as any]: depth }}>
          <button
            className={`science-group-row${isOpen ? " is-open" : ""}`}
            aria-expanded={isOpen}
            aria-controls={bodyId}
            onClick={() => onToggle(group.id)}
          >
            <span className="science-group-chevron" aria-hidden="true">
              {isOpen ? "▾" : "▸"}
            </span>
            <span className="science-group-label" title={group.label}>{group.label}</span>
            {group.value && (
              <span
                className={`science-group-value${group.state ? ` is-${group.state}` : ""}`}
                title={group.value}
              >
                {group.value}
              </span>
            )}
          </button>

          {/* Outside the row's own button: a badge is a control of its own, and a
              button inside a button is not something a browser will render. */}
          {(group.badges?.length ?? 0) > 0 && (
            <div className="science-badges">
              {group.badges!.map((run) => (
                <RunBadge
                  key={run.job_key}
                  run={run}
                  open={!!open[badgeKey(run)]}
                  onToggle={() => onToggle(badgeKey(run))}
                  onCancel={onCancelRun}
                />
              ))}
            </div>
          )}

          {isOpen && (
            <div className="science-group-body" id={bodyId} role="group">
              {(group.lines ?? []).map((line, i) => (
                <div className="science-line" key={i}>
                  <span className="science-line-label" title={line.label}>{line.label}</span>
                  <span
                    className={`science-line-value${line.state ? ` is-${line.state}` : ""}`}
                    title={line.value}
                  >
                    {line.value}
                  </span>
                </div>
              ))}
              {group.detail && <div className="science-detail">{group.detail}</div>}
              {group.action && (
                <GroupAction
                  label={group.action.label}
                  prompt={group.action.prompt}
                  onPrompt={onPrompt}
                />
              )}
              {group.groups && group.groups.length > 0 && (
                <PanelGroupList
                  groups={group.groups}
                  open={open}
                  onToggle={onToggle}
                  onPrompt={onPrompt}
                  onCancelRun={onCancelRun}
                  depth={depth + 1}
                />
              )}
            </div>
          )}
        </div>
      );
    })}
  </>
);

/**
 * A run on this machine, and what it is when opened.
 *
 * The disclosure is a row inside the tree rather than a tooltip: a command is too long
 * to hover over, and a reader using the keyboard gets nothing from a hover at all.
 */
export const RunBadge: React.FC<{
  run: WatchedRun;
  open: boolean;
  onToggle: () => void;
  onCancel: (jobKey: string) => void;
}> = ({ run, open, onToggle, onCancel }) => (
  <>
    <button
      className={`science-badge${run.state === "running" ? " is-running" : ""}`}
      aria-expanded={open}
      title={`${run.kind || "run"} — ${badgeLabel(run)}`}
      onClick={onToggle}
    >
      {badgeLabel(run)}
    </button>
    {open && <RunDetail run={run} onCancel={onCancel} />}
  </>
);

/**
 * An opened run: what it is and how far along, the command it runs, where and whose —
 * and the way to stop it, when its server said how.
 */
const RunDetail: React.FC<{ run: WatchedRun; onCancel: (jobKey: string) => void }> = ({
  run,
  onCancel,
}) => {
  const percent = typeof run.percent === "number"
    ? Math.max(0, Math.min(100, run.percent))
    : null;
  return (
    <div className="science-run">
      <div className="science-run-head">
        <span className="science-run-kind">{kindLabel(run)}</span>
        {run.state && (
          <span className={`science-run-state is-${run.state}`}>{run.state}</span>
        )}
      </div>
      {(run.phase || percent !== null) && (
        <div className="science-run-progress">
          {percent !== null && (
            <div className="science-run-bar" aria-hidden="true">
              <span style={{ width: `${percent}%` }} />
            </div>
          )}
          <span className="science-run-phase">
            {[run.phase, percent !== null ? `${Math.round(percent)}%` : ""]
              .filter(Boolean).join(" · ")}
          </span>
        </div>
      )}
      {run.command && <code className="science-run-command">{run.command}</code>}
      {badgeDetailRows(run).map((row) => (
        <div className="science-line" key={row.label}>
          <span className="science-line-label" title={row.label}>{row.label}</span>
          <span className="science-line-value" title={row.value}>{row.value}</span>
        </div>
      ))}
      {run.cancellable && <CancelRun jobKey={run.job_key} onCancel={onCancel} />}
    </div>
  );
};

/**
 * Stopping a run, asked twice: what it had not written yet is lost. Once sent it stays
 * "Stopping…" until the run leaves the panel, which is the watcher seeing it end.
 */
const CancelRun: React.FC<{ jobKey: string; onCancel: (jobKey: string) => void }> = ({
  jobKey,
  onCancel,
}) => {
  const [step, setStep] = React.useState<"idle" | "asking" | "sent">("idle");
  if (step === "sent") {
    return <div className="science-run-actions"><span className="science-detail">Stopping…</span></div>;
  }
  if (step === "idle") {
    return (
      <div className="science-run-actions">
        <button className="science-group-action is-danger" onClick={() => setStep("asking")}>
          Cancel
        </button>
      </div>
    );
  }
  return (
    <div className="science-group-confirm">
      <div className="science-detail">
        Stop this run? Whatever it has not written yet is lost. MIMIR is told when it ends.
      </div>
      <div className="science-group-confirm-row">
        <button
          className="science-group-action is-danger"
          onClick={() => {
            setStep("sent");
            onCancel(jobKey);
          }}
        >
          Stop it
        </button>
        <button className="science-group-action" onClick={() => setStep("idle")}>
          Keep running
        </button>
      </div>
    </div>
  );
};

/**
 * What the panel offers when an answer is missing: the request that would get it. The
 * click sends the server's sentence to the conversation, so the tool still asks for
 * approval; the confirmation is here because a submission is a decision.
 */
const GroupAction: React.FC<{
  label: string;
  prompt: string;
  onPrompt: (text: string) => void;
}> = ({ label, prompt, onPrompt }) => {
  const [asking, setAsking] = React.useState(false);
  if (!asking) {
    return (
      <button className="science-group-action" onClick={() => setAsking(true)}>
        {label}
      </button>
    );
  }
  return (
    <div className="science-group-confirm">
      <div className="science-detail">{prompt}</div>
      <div className="science-group-confirm-row">
        <button
          className="science-group-action"
          onClick={() => {
            setAsking(false);
            onPrompt(prompt);
          }}
        >
          Ask for it
        </button>
        <button className="science-group-action" onClick={() => setAsking(false)}>
          Cancel
        </button>
      </div>
    </div>
  );
};
