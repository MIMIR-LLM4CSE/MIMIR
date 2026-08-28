import { describe, it, expect } from "vitest";
import {
  insertAfterFamily, originLabel, relabelOrigins, settleFamily, subAgentTail,
} from "./subAgentUtils";
import type { ToolActivity } from "../types";

function row(id: string, extra: Partial<ToolActivity> = {}): ToolActivity {
  return {
    id, name: "grep", icon: "🔍", label: "", detail: "",
    status: "running", startedAt: 1_000, ...extra,
  };
}

describe("originLabel", () => {
  it("names the sub-agent by the role its parent's label already declares", () => {
    expect(originLabel("Sub-agent (explore): where is the router", 2)).toBe("explore #2");
  });

  it("falls back to something true when the label says no role", () => {
    expect(originLabel("Delegating", 1)).toBe("sub-agent #1");
    expect(originLabel(undefined, 1)).toBe("sub-agent #1");
  });

  it("never shows a zeroth sub-agent", () => {
    expect(originLabel("Sub-agent (task): x", 0)).toBe("task #1");
  });
});

describe("relabelOrigins", () => {
  it("numbers sub-agents by where their calls sit, not by who reported first", () => {
    // The wire order of a fan-out is arbitrary; the reading order is not.
    const rows = [
      row("p1", { label: "Sub-agent (explore): A" }),
      row("p2", { label: "Sub-agent (explore): B" }),
      row("p2:c1", { parentId: "p2" }),
      row("p1:c1", { parentId: "p1" }),
    ];
    const out = relabelOrigins(rows);
    expect(out.find((r) => r.id === "p1:c1")!.origin).toBe("explore #1");
    expect(out.find((r) => r.id === "p2:c1")!.origin).toBe("explore #2");
  });

  it("keeps badging children whose parent row is no longer in this list", () => {
    const out = relabelOrigins([row("p1:c1", { parentId: "p1" })]);
    expect(out[0].origin).toBe("sub-agent #1");
  });

  it("leaves the agent's own rows unbadged", () => {
    const out = relabelOrigins([row("t1"), row("p1"), row("p1:c1", { parentId: "p1" })]);
    expect(out[0].origin).toBeUndefined();
    expect(out[1].origin).toBeUndefined();
  });
});

describe("insertAfterFamily", () => {
  it("puts a first child directly under its parent", () => {
    const rows = [row("p1"), row("other")];
    const out = insertAfterFamily(rows, "p1", row("p1:c1", { parentId: "p1" }));
    expect(out.map((r) => r.id)).toEqual(["p1", "p1:c1", "other"]);
  });

  it("keeps siblings together and in order", () => {
    const rows = [row("p1"), row("p1:c1", { parentId: "p1" }), row("p2")];
    const out = insertAfterFamily(rows, "p1", row("p1:c2", { parentId: "p1" }));
    expect(out.map((r) => r.id)).toEqual(["p1", "p1:c1", "p1:c2", "p2"]);
  });

  it("shows a child whose parent is gone rather than dropping it", () => {
    const out = insertAfterFamily([row("x")], "p1", row("p1:c1", { parentId: "p1" }));
    expect(out.map((r) => r.id)).toEqual(["x", "p1:c1"]);
  });
});

describe("settleFamily", () => {
  it("stops the timers of a finished call's children, and only those", () => {
    const rows = [
      row("p1"),
      row("p1:c1", { parentId: "p1" }),
      row("p1:c2", { parentId: "p1", status: "error" }),
      row("p2:c1", { parentId: "p2" }),
    ];
    const out = settleFamily(rows, "p1", 4_000);
    expect(out.map((r) => r.status)).toEqual(["running", "ok", "error", "running"]);
    expect(out[1].durationMs).toBe(3_000);
  });
});

describe("subAgentTail", () => {
  it("counts a call's children", () => {
    const rows = [row("p1"), row("p1:c1", { parentId: "p1" })];
    expect(subAgentTail(rows, "p1")).toBe("1 tool");
    expect(subAgentTail([...rows, row("p1:c2", { parentId: "p1" })], "p1")).toBe("2 tools");
  });

  it("says nothing about a call that delegated nothing", () => {
    expect(subAgentTail([row("p1")], "p1")).toBe("");
  });
});
