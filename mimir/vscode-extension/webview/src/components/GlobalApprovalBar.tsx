import React from "react";
import type { ApprovalMessage } from "../types";
import { iconForTool } from "../state/chatReducer";

interface Props {
  approval: ApprovalMessage;
  onRespond: (choice: "y" | "n" | "a") => void;
}

// Bookkeeping args that carry no meaning for a human reading the card: the
// dispatch op is already surfaced in the header label, and confirm/booleans are
// plumbing. Everything else is a candidate "subject" for the tool call.
const META_ARGS = new Set(["op", "confirm"]);

// Preference order for the single most meaningful arg to show as the subject —
// the *object* the operation acts on. Command/query text first, then file
// paths, then the named entity op-dispatched tools operate on (proxy_name, …).
const SUBJECT_KEYS = [
  "command", "cmd", "script", "code", "query", "text",
  "path", "filepath", "filePath", "file_path", "file",
  "proxy_name", "reference_name", "suite_name", "benchmark_name",
  "run_id", "name", "url", "pattern",
];

function asText(v: unknown): string | null {
  return typeof v === "string" && v.trim() ? v.trim() : null;
}

/** The single most meaningful string to show under the header, or "" if the
 *  header label already says everything (no descriptive arg to add). */
function getSubject(approval: ApprovalMessage): string {
  const args = approval.args ?? {};
  for (const key of SUBJECT_KEYS) {
    const v = asText(args[key]);
    if (v) return v;
  }
  // Fallback: first non-meta, non-empty string arg.
  for (const [key, val] of Object.entries(args)) {
    if (META_ARGS.has(key)) continue;
    const v = asText(val);
    if (v) return v;
  }
  return "";
}

/** Header label: canonical server-supplied label ("Proxy exec: run") when
 *  present, else a "server · op/tool" fallback derived from the payload. */
function getHeaderLabel(approval: ApprovalMessage): string {
  if (approval.label?.trim()) return approval.label.trim();
  const op = asText(approval.args?.["op"]);
  const tool = approval.tool.replace(/_/g, " ");
  const subject = op ? `${tool}: ${op}` : tool;
  return approval.server && approval.server !== "unknown"
    ? `${approval.server} · ${subject}`
    : subject;
}

export const GlobalApprovalBar: React.FC<Props> = ({ approval, onRespond }) => {
  const risk = (approval.risk ?? "").toLowerCase();
  // Severity comes from the *declared* undo level when the server sends one. It used
  // to be sniffed out of the risk sentence (`/destructive|critical/`), which guessed a
  // severity from prose written for a human — a tool whose note happened to avoid
  // those words rendered as low risk however far its effect reached. The keyword sniff
  // survives only as the fallback for a card from a server that predates the field.
  const level = approval.reversibility;
  const isHigh = level ? level === "irreversible" : /high|destructive|critical/.test(risk);
  const isMed = level
    ? level === "recoverable"
    : !isHigh && /medium|moderate|elevated/.test(risk);
  const riskClass = isHigh ? "high" : isMed ? "med" : "low";
  // The badge names what is at stake, not a severity word: "irreversible" tells the
  // user why they are being asked, where "high" only tells them to worry.
  const badge = level ?? (risk || "risk");

  const header = getHeaderLabel(approval);
  const subject = getSubject(approval);
  // Don't repeat the subject when the header already ends with it.
  const showSubject = subject && !header.endsWith(subject);
  const icon = iconForTool(approval.tool);
  // `oow_paths` is the list; `oow_path` is the single-path payload of an older server.
  const oowPaths = approval.oow_paths ?? (approval.oow_path ? [approval.oow_path] : []);

  return (
    <div className={`ap-row${isHigh ? " ap-row--high" : isMed ? " ap-row--med" : ""}`}>
      <div className="ap-head">
        <span className="ap-icon" aria-hidden="true">{icon}</span>
        <span className="ap-title">{header}</span>
        {(isHigh || isMed) && (
          <span className={`ap-risk ap-risk--${riskClass}`}>{badge}</span>
        )}
      </div>
      {showSubject && <code className="ap-cmd ap-cmd--main">{subject}</code>}
      {/* Why the call is being held, when it's the workspace boundary rather than
          the tool itself: the card above already says what the tool is doing. */}
      {oowPaths.length > 0 && (
        <div className="ap-reason">
          <span className="ap-reason-tag">
            outside workspace{oowPaths.length > 1 ? ` (${oowPaths.length})` : ""}
          </span>
          {/* One row per path: a command names several, and they are approved
              together — hiding all but the first would ask for consent to a
              location the user never saw. */}
          <div className="ap-reason-paths">
            {oowPaths.map((p) => (
              <code className="ap-reason-path" key={p}>{p}</code>
            ))}
          </div>
        </div>
      )}
      <div className="ap-actions">
        <button className="ap-btn ap-btn--allow" onClick={() => onRespond("y")}>Allow</button>
        <button className="ap-btn ap-btn--block" onClick={() => onRespond("n")}>Block</button>
        <button className="ap-btn ap-btn--always" onClick={() => onRespond("a")}
          title={`Always allow for ${approval.scope}`}>Always</button>
      </div>
    </div>
  );
};
