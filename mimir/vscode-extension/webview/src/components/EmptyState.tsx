// What an empty thread shows: the connect form while disconnected, else where the
// connection and the agent stand.

import React from "react";
import type { ConnectionState } from "../types";
import { MimirIntro } from "./MimirIntro";

interface Props {
  connection: ConnectionState;
  agentReady: boolean;
  /** The connect form, shown while disconnected. */
  form: React.ReactNode;
  onCancelConnect: () => void;
}

export const EmptyState: React.FC<Props> = ({ connection, agentReady, form, onCancelConnect }) => {
  if (connection === "disconnected") {
    return (
      <div className="empty-state">
        <MimirIntro />
        <div className="empty-text">MIMIR</div>
        {form}
      </div>
    );
  }
  // "error" is a failed attempt the host is still retrying, not an agent coming up:
  // ws_closed follows if the retries run out.
  const connecting = connection === "connecting" || connection === "error";
  return (
    <div className="empty-state">
      <MimirIntro loop={connecting} />
      <div className="empty-text">MIMIR</div>
      <div className="empty-hint">
        {connecting
          ? "Connecting to agent…"
          : agentReady
            ? "Type a message to start"
            : "Starting the agent — waiting for the model backend…"}
      </div>
      {(connecting || !agentReady) && <div className="loading-track" aria-hidden="true" />}
      {connecting && (
        <button className="connect-btn cancel-connect-btn" onClick={onCancelConnect}>
          Cancel connection
        </button>
      )}
    </div>
  );
};
