import { describe, it, expect } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { StatusBar } from "./StatusBar";

const noop = () => {};
const bar = (over: Partial<React.ComponentProps<typeof StatusBar>> = {}) =>
  renderToStaticMarkup(
    <StatusBar
      connection="connected" model="org/big-model" servedModels={[]} runningSubAgents={0}
      showSessions={false} showSubAgents={false} showScience={false}
      onToggleSessions={noop} onNewSession={noop} onToggleSubAgents={noop}
      onToggleScience={noop} onSwitchModel={noop} onCancelConnect={noop} onDisconnect={noop}
      {...over}
    />,
  );

describe("StatusBar", () => {
  it("names the model by its last path segment", () => {
    expect(bar()).toContain(">big-model<");
  });

  it("offers a picker only when connected to an endpoint serving several models", () => {
    expect(bar({ servedModels: ["org/big-model", "org/small"] })).toContain("<select");
    expect(bar({ connection: "connecting", servedModels: ["org/big-model", "org/small"] }))
      .not.toContain("<select");
  });

  it("counts the sub-agents at work on their toggle", () => {
    expect(bar({ runningSubAgents: 2 })).toContain('class="subagent-badge">2<');
  });
});
