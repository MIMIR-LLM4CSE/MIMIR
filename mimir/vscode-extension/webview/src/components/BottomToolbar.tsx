// The row above the message box: the mode, the agent's settings, the servers and skills
// switched on, and — hard right — leaving the run going without this window.

import React, { useState } from "react";
import type { ClientMessage, SubAgentLevel, ToggleItem } from "../types";
import type { useAgentSettings } from "../hooks/useAgentSettings";
import { AgentSettings } from "./AgentSettings";
import { DetachButton } from "./DetachButton";
import { BlocksIcon, GearIcon } from "./icons";
import { ModeSwitcher } from "./ModeSwitcher";
import { TogglesPanel } from "./TogglesPanel";

interface Props {
  settings: ReturnType<typeof useAgentSettings>;
  subAgentLevel: SubAgentLevel;
  onSubAgentLevelChange: (level: SubAgentLevel) => void;
  serverToggles: ToggleItem[];
  skillToggles: ToggleItem[];
  send: (msg: ClientMessage) => void;
  detached: boolean;
  onDetach: () => void;
  onReattach: () => void;
}

export const BottomToolbar: React.FC<Props> = ({
  settings, subAgentLevel, onSubAgentLevelChange, serverToggles, skillToggles, send,
  detached, onDetach, onReattach,
}) => {
  const [settingsOpen, setSettingsOpen] = useState(false);
  const [togglesOpen, setTogglesOpen] = useState(false);

  return (
    <div className="bottom-toolbar">
      {/* Mode — its own control, colour-coded like the rest of the chat */}
      <ModeSwitcher mode={settings.agentMode} onModeChange={settings.changeMode} />

      <div className="settings-wrapper">
        <button
          className="settings-btn"
          title="Agent settings"
          aria-label="Agent settings"
          aria-expanded={settingsOpen}
          onClick={() => setSettingsOpen((o) => !o)}
        >
          {/* The gear alone: a button whose glyphs come and go is a button the eye has
              to re-find, and the popover says what the settings are. */}
          <span className="settings-gear"><GearIcon /></span>
        </button>
        {settingsOpen && (
          <AgentSettings
            thinkingLevel={settings.thinkingLevel}
            thinkingProfile={settings.thinkingProfile}
            streaming={settings.streaming}
            contextMode={settings.contextMode}
            enforcement={settings.enforcement}
            subAgentLevel={subAgentLevel}
            onSubAgentLevelChange={onSubAgentLevelChange}
            temperature={settings.temperature}
            onTemperatureChange={settings.changeTemperature}
            onThinkingLevelChange={settings.changeThinkingLevel}
            onStreamingToggle={settings.changeStreaming}
            onContextModeChange={settings.changeContextMode}
            onEnforcementChange={settings.changeEnforcement}
            onClose={() => setSettingsOpen(false)}
          />
        )}
      </div>

      {/* Server / skill toggles, listed afresh each time the panel opens */}
      <div className="toggles-wrapper">
        <button
          className="settings-btn"
          title="Servers & skills"
          aria-label="Servers and skills"
          aria-expanded={togglesOpen}
          onClick={() => {
            setTogglesOpen((o) => {
              if (!o) send({ type: "list_toggles" });
              return !o;
            });
          }}
        >
          <span className="settings-gear"><BlocksIcon /></span>
        </button>
        {togglesOpen && (
          <TogglesPanel
            servers={serverToggles}
            skills={skillToggles}
            onToggleServer={(name, enabled) => send({ type: "toggle_server", name, enabled })}
            onToggleSkill={(name, enabled) => send({ type: "toggle_skill", name, enabled })}
            onClose={() => setTogglesOpen(false)}
          />
        )}
      </div>

      {/* Hard right, away from the controls that change what the next turn does: this
          one changes who owns the process, and is pressed once a session rather than
          adjusted. It runs under the approval mode set above the send button. */}
      <div className="toolbar-end">
        <DetachButton
          mode={settings.approvalMode}
          detached={detached}
          onDetach={onDetach}
          onReattach={onReattach}
        />
      </div>
    </div>
  );
};
