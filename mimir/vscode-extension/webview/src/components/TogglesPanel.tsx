import React, { useEffect, useRef } from "react";
import type { ToggleItem } from "../types";

interface Props {
  servers: ToggleItem[];
  skills: ToggleItem[];
  onToggleServer: (name: string, enabled: boolean) => void;
  onToggleSkill: (name: string, enabled: boolean) => void;
  onClose: () => void;
}

const Row: React.FC<{ item: ToggleItem; onToggle: (name: string, enabled: boolean) => void }> = ({
  item,
  onToggle,
}) => (
  <button
    type="button"
    className={`toggles-row ${item.enabled ? "enabled" : "disabled"}`}
    title={item.description || item.name}
    aria-pressed={item.enabled}
    onClick={() => onToggle(item.name, !item.enabled)}
  >
    <span className="toggles-check" aria-hidden="true">{item.enabled ? "☑" : "☐"}</span>
    <span className="toggles-name">{item.name}</span>
  </button>
);

/**
 * A scrollable popover listing every MCP server and skill as a clickable checkbox.
 * Ticked = active (advertised to the model / eligible for auto-detection); unticked =
 * soft-hidden. Hover a row for its description. Mirrors the AgentSettings popover.
 */
export const TogglesPanel: React.FC<Props> = ({
  servers,
  skills,
  onToggleServer,
  onToggleSkill,
  onClose,
}) => {
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.closest(".toggles-wrapper")?.contains(e.target as Node)) {
        onClose();
      }
    };
    document.addEventListener("mousedown", handler);
    return () => document.removeEventListener("mousedown", handler);
  }, [onClose]);

  return (
    <div className="settings-popover toggles-popover" ref={ref}>
      <div className="settings-section-label">Servers</div>
      <div className="toggles-list">
        {servers.length === 0 ? (
          <div className="toggles-empty">No servers.</div>
        ) : (
          servers.map((s) => <Row key={s.name} item={s} onToggle={onToggleServer} />)
        )}
      </div>

      <div className="settings-divider" />

      <div className="settings-section-label">Skills</div>
      <div className="toggles-list">
        {skills.length === 0 ? (
          <div className="toggles-empty">No skills.</div>
        ) : (
          skills.map((s) => <Row key={s.name} item={s} onToggle={onToggleSkill} />)
        )}
      </div>
    </div>
  );
};
