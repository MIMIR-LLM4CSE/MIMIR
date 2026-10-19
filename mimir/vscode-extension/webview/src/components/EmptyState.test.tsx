import { describe, expect, it, vi } from "vitest";
import React from "react";
import { renderToStaticMarkup } from "react-dom/server";
import { EmptyState } from "./EmptyState";

// MimirIntro reads the brand assets the host injects; none here.
vi.stubGlobal("window", {});

const render = (over: Partial<React.ComponentProps<typeof EmptyState>> = {}) =>
  renderToStaticMarkup(
    <EmptyState
      connection="connected" agentReady={true} form={<form id="the-form" />}
      onCancelConnect={() => {}} {...over}
    />,
  );

describe("EmptyState", () => {
  it("offers the connect form while disconnected", () => {
    expect(render({ connection: "disconnected" })).toContain('id="the-form"');
    expect(render()).not.toContain('id="the-form"');
  });

  it("says the agent is coming up until it is ready, and invites a message once it is", () => {
    expect(render({ agentReady: false })).toContain("waiting for the model backend");
    expect(render()).toContain("Type a message to start");
  });

  it("reads a failed attempt the host still retries as connecting, with a way out", () => {
    const html = render({ connection: "error" });
    expect(html).toContain("Connecting to agent…");
    expect(html).toContain("Cancel connection");
  });
});
