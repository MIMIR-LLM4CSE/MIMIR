// Under a thread with history while not connected: rejoin the run a server is still
// doing, reconnect, or connect anew; and the banner while a connection is coming up.

import React from "react";
import type { ConnectionState, RunningServer } from "../types";

interface Props {
  connection: ConnectionState;
  /** The server this workspace already has, if any. */
  runningServer: RunningServer | null;
  /** The last connect can be repeated as it was. */
  canReconnect: boolean;
  onRejoin: () => void;
  onReconnect: () => void;
  onCancelConnect: () => void;
  /** The connect form. */
  form: React.ReactNode;
}

export const ReconnectPanel: React.FC<Props> = ({
  connection, runningServer, canReconnect, onRejoin, onReconnect, onCancelConnect, form,
}) => {
  if (connection === "connecting") {
    return (
      <div className="reconnect-panel reconnect-panel--connecting">
        <span className="inline-spinner" aria-hidden="true" /> Connecting to agent…
        <button className="reconnect-btn cancel-connect-btn" onClick={onCancelConnect}>
          Cancel
        </button>
      </div>
    );
  }
  if (connection !== "disconnected") return null;
  return (
    <div className="reconnect-panel">
      {runningServer ? (
        <>
          {/* A run is going without this panel. Rejoining it is the only thing worth
              offering: the form below would collect an endpoint that a server already
              serving this workspace does not use. */}
          <div className="reconnect-hint">
            {runningServer.detached
              ? "MIMIR is still working without this window"
              : "This workspace already has a MIMIR server"}
            {runningServer.autonomy && runningServer.autonomy !== "manual"
              ? ` — running under “${runningServer.autonomy}”`
              : ""}
          </div>
          <button className="reconnect-btn" onClick={onRejoin}>
            Rejoin the run
          </button>
        </>
      ) : (
        <>
          <div className="reconnect-hint">Agent disconnected — reconnect to continue</div>
          {canReconnect && (
            <button className="reconnect-btn" onClick={onReconnect}>
              Reconnect
            </button>
          )}
        </>
      )}
      {form}
    </div>
  );
};
