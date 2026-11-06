import { afterEach, describe, expect, it, vi } from "vitest";
import { fileQuery, hitsHtml, TargetSearch } from "./targetSearch";

/** A form's search as the page gives it: an input and a list that keep what they are told. */
function form() {
  const on: Record<string, (e: unknown) => void> = {};
  const input = {
    value: "", focused: false,
    addEventListener: (type: string, fn: (e: unknown) => void) => { on[type] = fn; },
    focus() { this.focused = true; }, setSelectionRange() {},
  };
  const list = { innerHTML: "", addEventListener() {} };
  const box = { querySelector: (s: string) => (s === "input" ? input : list) } as unknown as HTMLElement;
  const type = (v: string) => { input.value = v; on.input({}); };
  const key = (k: string) => on.keydown({ key: k, preventDefault() {}, stopPropagation() {} });
  return { box, input, list, type, key };
}

function search(targets: string[] = []) {
  const asked: string[] = [];
  const s = new TargetSearch({
    ask: (q) => asked.push(q), add: (f) => targets.push(f), taken: (f) => targets.includes(f),
  });
  return { s, asked, targets };
}

afterEach(() => vi.unstubAllGlobals());

describe("the form's file search", () => {
  it("reads a query with or without the chat's @", () => {
    expect(fileQuery("  @pkg/sol ")).toBe("pkg/sol");
    expect(fileQuery("solver")).toBe("solver");
  });

  it("escapes the paths it lists and marks the row Enter picks", () => {
    const html = hitsHtml(["src/a<b>.py", "c.py"], 1);
    expect(html).toContain('<span class="nm">a&lt;b&gt;.py</span><span class="sub">src</span>');
    expect(html.indexOf("is-cur")).toBeGreaterThan(html.indexOf("a&lt;b"));
  });

  it("asks the host for a typed query, an @ alone included, and drops an answer the typing overtook", () => {
    const { s, asked } = search(), f = form();
    s.mount(f.box);
    f.type("@");
    f.type("@so");
    expect(asked).toEqual(["", "so"]);
    s.answer("", ["old.py"]);
    expect(f.list.innerHTML).toBe("");
    s.answer("so", ["solver.py"]);
    expect(f.list.innerHTML).toContain("solver.py");
    f.type("");
    expect(f.list.innerHTML).toBe("");
    expect(asked).toHaveLength(2);
  });

  it("leaves out the targets, picks with the arrows and Enter, and clears for the next file", () => {
    const { s, targets } = search(["taken.py"]), f = form();
    s.mount(f.box);
    f.type("py");
    s.answer("py", ["taken.py", "a.py", "b.py"]);
    expect(f.list.innerHTML).not.toContain("taken.py");
    f.key("ArrowDown");
    f.key("Enter");
    expect(targets).toEqual(["taken.py", "b.py"]);
    const next = form();
    s.mount(next.box);
    expect(next.input.value).toBe("");
    expect(next.input.focused).toBe(true);
  });

  it("keeps what was typed, its hits and the focus through a redraw, until no form is open", () => {
    const { s } = search(), f = form();
    s.mount(f.box);
    f.type("so");
    s.answer("so", ["solver.py"]);
    vi.stubGlobal("document", { activeElement: f.input });
    s.leave();
    const again = form();
    s.mount(again.box);
    expect(again.input.value).toBe("so");
    expect(again.list.innerHTML).toContain("solver.py");
    expect(again.input.focused).toBe(true);
    s.mount(null);
    const later = form();
    s.mount(later.box);
    expect(later.input.value).toBe("");
    expect(later.input.focused).toBe(false);
  });
});
