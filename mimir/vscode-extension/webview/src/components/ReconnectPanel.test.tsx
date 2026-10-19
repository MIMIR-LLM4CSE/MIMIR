import { describe, expect, it } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { ReconnectPanel } from "./ReconnectPanel";

const noop = () => {};
const render = (over: Partial<React.ComponentProps<typeof ReconnectPanel>> = {}) =>
  renderToStaticMarkup(
    <ReconnectPanel
      connection="disconnected" runningServer={null} canReconnect={false}
      onRejoin={noop} onReconnect={noop} onCancelConnect={noop} form={<form id="the-form" />}
      {...over}
    />,
  );

describe("ReconnectPanel", () => {
  it("offers to rejoin a run the workspace's server is still doing", () => {
    const html = render({
      runningServer: { url: "ws://x", pid: 1, model: "m", autonomy: "auto", detached: true },
    });
    expect(html).toContain("Rejoin the run");
    expect(html).toContain("still working without this window");
    expect(html).toContain("running under “auto”");
  });

  it("offers a reconnect only when there is a connect to repeat", () => {
    expect(render()).not.toContain(">Reconnect<");
    expect(render({ canReconnect: true })).toContain(">Reconnect<");
  });

  it("shows a banner while connecting, and nothing once connected", () => {
    expect(render({ connection: "connecting" })).toContain("Connecting to agent…");
    expect(render({ connection: "connected" })).toBe("");
  });
});
