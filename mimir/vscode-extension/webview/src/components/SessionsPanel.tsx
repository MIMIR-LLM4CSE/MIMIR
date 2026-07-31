import React, { useState, useRef, useCallback } from "react";
import type { SessionMeta } from "../types";

interface Props {
  sessions: SessionMeta[];
  activeSessionId: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onRename: (id: string, title: string) => void;
}

function relativeTime(isoString: string): string {
  if (!isoString) return "";
  const now = Date.now();
  const then = new Date(isoString).getTime();
  const diffMs = now - then;
  const diffSec = Math.floor(diffMs / 1000);
  if (diffSec < 60) return "just now";
  const diffMin = Math.floor(diffSec / 60);
  if (diffMin < 60) return `${diffMin}m ago`;
  const diffHr = Math.floor(diffMin / 60);
  if (diffHr < 24) return `${diffHr}h ago`;
  const diffDay = Math.floor(diffHr / 24);
  if (diffDay < 30) return `${diffDay}d ago`;
  return new Date(isoString).toLocaleDateString();
}

/** What the panel shows for a session: the generated description, unless the
 *  user renamed it by hand — then their title wins. Falls back to the first
 *  query while the description is still being generated. */
function sessionLabel(session: SessionMeta): string {
  if (!session.title_custom && session.summary) return session.summary;
  return session.title || session.summary || "New session";
}

export const SessionsPanel: React.FC<Props> = ({
  sessions,
  activeSessionId,
  onSelect,
  onDelete,
  onRename,
}) => {
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");
  const renameInputRef = useRef<HTMLInputElement>(null);

  const startRename = useCallback((session: SessionMeta, e: React.MouseEvent) => {
    e.stopPropagation();
    setRenamingId(session.id);
    setRenameValue(session.title || "");
    setTimeout(() => renameInputRef.current?.select(), 0);
  }, []);

  const commitRename = useCallback(
    (id: string) => {
      const title = renameValue.trim();
      if (title) onRename(id, title);
      setRenamingId(null);
    },
    [renameValue, onRename]
  );

  const handleRenameKey = useCallback(
    (e: React.KeyboardEvent, id: string) => {
      if (e.key === "Enter") {
        e.preventDefault();
        commitRename(id);
      } else if (e.key === "Escape") {
        setRenamingId(null);
      }
    },
    [commitRename]
  );

  return (
    <div className="sessions-panel">
      {/* No new-session button here: the status bar carries the one ＋. */}
      <div className="sessions-panel-header">
        <span className="sessions-panel-title">Chat history</span>
      </div>
      <div className="sessions-list">
        {sessions.length === 0 && (
          <div className="sessions-empty">No sessions yet</div>
        )}
        {sessions.map((session) => (
          <div
            key={session.id}
            className={`session-item ${session.id === activeSessionId ? "active" : ""}`}
            onClick={() => {
              if (renamingId !== session.id) onSelect(session.id);
            }}
          >
            <div className="session-item-body">
              {renamingId === session.id ? (
                <input
                  ref={renameInputRef}
                  className="session-rename-input"
                  value={renameValue}
                  onChange={(e) => setRenameValue(e.target.value)}
                  onBlur={() => commitRename(session.id)}
                  onKeyDown={(e) => handleRenameKey(e, session.id)}
                  onClick={(e) => e.stopPropagation()}
                  autoFocus
                />
              ) : (
                <span
                  className="session-title"
                  onDoubleClick={(e) => startRename(session, e)}
                  title={sessionLabel(session) + "\n\nDouble-click to rename"}
                >
                  {sessionLabel(session)}
                </span>
              )}
              <span className="session-time">
                {relativeTime(session.updated_at)}
              </span>
            </div>
            <button
              className="session-delete-btn"
              title="Delete session"
              onClick={(e) => {
                e.stopPropagation();
                onDelete(session.id);
              }}
            >
              ×
            </button>
          </div>
        ))}
      </div>
    </div>
  );
};
