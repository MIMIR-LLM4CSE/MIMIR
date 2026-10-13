import { describe, it, expect } from "vitest";
import { MIN_PANEL_WIDTH, clampPanelWidth } from "./panelWidthUtils";

describe("clampPanelWidth", () => {
  it("keeps a drawer readable, however far the handle is dragged left", () => {
    expect(clampPanelWidth(10, 1600)).toBe(MIN_PANEL_WIDTH);
    expect(clampPanelWidth(-400, 1600)).toBe(MIN_PANEL_WIDTH);
  });

  it("leaves the chat a share of the window, however far it is dragged right", () => {
    expect(clampPanelWidth(5000, 1000)).toBe(800);
  });

  it("passes a width between the two through, rounded to the pixel", () => {
    expect(clampPanelWidth(421.4, 1600)).toBe(421);
  });

  it("answers a narrow window with the floor rather than something unreadable", () => {
    // A ceiling computed from a 200px window would be below the floor; the drawer is
    // readable and the window scrolls, which is the lesser of the two failures.
    expect(clampPanelWidth(300, 200)).toBe(MIN_PANEL_WIDTH);
  });

  it("treats a stored value that is not a number as the floor", () => {
    expect(clampPanelWidth(Number.NaN, 1600)).toBe(MIN_PANEL_WIDTH);
  });
});
