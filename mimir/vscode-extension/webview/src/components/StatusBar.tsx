// The bar above the chat: the drawers' toggles, the connection, the model (a picker when
// the endpoint serves several), and disconnect or cancel.

import React from "react";
import type { ConnectionState } from "../types";

function modelDisplayName(value: string): string {
  const trimmed = value.trim();
  if (!trimmed) return "MIMIR";
  const normalized = trimmed.replace(/\\/g, "/");
  if (!normalized.includes("/")) return trimmed;
  const parts = normalized.split("/").filter(Boolean);
  return parts[parts.length - 1] || trimmed;
}

interface Props {
  connection: ConnectionState;
  model: string;
  servedModels: string[];
  runningSubAgents: number;
  showSessions: boolean;
  showSubAgents: boolean;
  showScience: boolean;
  onToggleSessions: () => void;
  onNewSession: () => void;
  onToggleSubAgents: () => void;
  onToggleScience: () => void;
  onSwitchModel: (model: string) => void;
  onCancelConnect: () => void;
  onDisconnect: () => void;
}

export const StatusBar: React.FC<Props> = ({
  connection, model, servedModels, runningSubAgents, showSessions, showSubAgents,
  showScience, onToggleSessions, onNewSession, onToggleSubAgents, onToggleScience,
  onSwitchModel, onCancelConnect, onDisconnect,
}) => {
  const connectionColor =
    connection === "connected"
      ? "var(--vscode-testing-iconPassed)"
      : connection === "error" || connection === "disconnected"
      ? "var(--vscode-testing-iconFailed)"
      : "var(--vscode-descriptionForeground)";

  return (
            <div className="status-bar">
      {connection === "connected" && (
        <>
          <button
            className={`sessions-toggle-btn ${showSessions ? "active" : ""}`}
            title="Chat history"
            aria-label="Toggle chat history"
            aria-pressed={showSessions}
            onClick={onToggleSessions}
          >
            ☰
          </button>
          {/* New session next to the history toggle — starting a chat
              shouldn't require opening the panel first. */}
          <button
            className="sessions-toggle-btn sessions-new-toggle-btn"
            title="New session"
            aria-label="New session"
            onClick={onNewSession}
          >
            ＋
          </button>
          {/* Always here, with a count when sub-agents are working: a button that
              comes and goes is a button the user has to look for. */}
          <button
            className={`sessions-toggle-btn ${showSubAgents ? "active" : ""}`}
            title={runningSubAgents
              ? `${runningSubAgents} sub-agent(s) working`
              : "Sub-agents of this session"}
            aria-label="Toggle sub-agents"
            aria-pressed={showSubAgents}
            onClick={onToggleSubAgents}
          >
            ⑂
            {runningSubAgents > 0 && (
              <span className="subagent-badge">{runningSubAgents}</span>
            )}
          </button>
          <button
            className={`sessions-toggle-btn ${showScience ? "active" : ""}`}
            title="Scientific computing"
            aria-label="Toggle the scientific-computing panel"
            aria-pressed={showScience}
            onClick={onToggleScience}
          >
            ⚗
          </button>
        </>
      )}
      <span
        className={`connection-dot is-${connection}`}
        style={{ color: connectionColor }}
        title={connection}
        role="img"
        aria-label={`Connection: ${connection}`}
      >
        ●
      </span>
      {/* Model name — a picker when the endpoint reported a real choice, a
          static label otherwise. The current model is always an option even
          if the probe missed it (auto-selected, or probe failed). Shown only
          once an agent is live: before that there is no served model, and the
          probe's list here would name one the connect form has not been told
          to use — a label contradicting the form's own dropdown. While the
          connection is still being made the name is a label, not a picker:
          there is no agent yet to switch. */}
      {connection !== "disconnected" && model && (() => {
        const options = Array.from(new Set([...servedModels, model].filter(Boolean)));
        if (connection === "connected" && options.length > 1) {
          return (
            <select
              className="model-picker"
              value={model}
              title="Switch model"
              aria-label="Switch model"
              onChange={(e) => onSwitchModel(e.target.value)}
            >
              {options.map((name) => (
                <option key={name} value={name}>{modelDisplayName(name)}</option>
              ))}
            </select>
          );
        }
        return <span className="model-name">{modelDisplayName(model)}</span>;
      })()}
      {/* Also while connecting: a connection can hang for the whole backend
          timeout — minutes on a cold vLLM, and forever if the endpoint is
          simply wrong — and offering no way out until it succeeds left the
          only escape a window reload. Cancelling drops the socket and returns
          to the connect form, where the endpoint can be corrected. */}
      {(connection === "connected" || connection === "connecting") && (
        <button
          className="disconnect-btn"
          title={connection === "connecting" ? "Cancel connection" : "Disconnect agent"}
          aria-label={connection === "connecting" ? "Cancel connection" : "Disconnect agent"}
          onClick={connection === "connecting"
            ? onCancelConnect
            : onDisconnect}
        >
          ⏏
        </button>
      )}
    </div>
  );
};
