import { describe, expect, it } from "vitest";
import { axisTag, live, newDraft, ready, requestPrompt, targetLabel, touches, variantRows, withTarget, type Request } from "./request";
import { formHtml, requestDetail, requestsSection } from "./requestView";

const draft = () => ({
  ...newDraft([{ file: "pkg/solver.py", function: "relax", line: 12 }], "tok123"),
  mode: "try" as const, idea: "multigrid instead of Jacobi", variants: "V-cycle\n\n W-cycle ",
});

describe("the request's message", () => {
  it("names the targets, the variants, and the call that records it", () => {
    const text = requestPrompt(draft(), () => "30%");
    expect(text).toContain("Request from the code map (token tok123): try — numerical method.");
    expect(text).toContain("- pkg/solver.py: relax (line 12), 30% of the baseline's time");
    expect(text).toContain("1. V-cycle\n2. W-cycle");
    expect(text).toContain('proxy_request(op=\'open\', token="tok123", mode="try", kind="method", '
      + 'targets=[{"file":"pkg/solver.py","function":"relax","line":12}], idea="multigrid instead of Jacobi", '
      + 'variants=["V-cycle","W-cycle"])');
    expect(text).not.toContain("External code");
  });

  it("gives the source only for external code, and a study changes no file", () => {
    const ext = { ...draft(), kind: "external", source: "petsc4py", mode: "study" as const };
    expect(requestPrompt(ext)).toContain("External code: petsc4py");
    expect(requestPrompt(ext)).toContain("Study only: change no file.");
    expect(requestPrompt({ ...draft(), source: "petsc4py" })).not.toContain("petsc4py");
  });

  it("carries what the map found, for a refactor", () => {
    const r = { ...draft(), kind: "refactor", targets: [{ all: true }] };
    const text = requestPrompt(r, () => null, ["module cycle: a → b → a"]);
    expect(text).toContain("- the whole code");
    expect(text).toContain("What the map finds there:\n- module cycle: a → b → a");
    expect(requestPrompt(draft(), () => null, ["module cycle: a → b → a"])).not.toContain("What the map finds");
  });

  it("is ready with an idea and a target, and adds a target once", () => {
    expect(ready(draft())).toBe(true);
    expect(ready({ ...draft(), targets: [] })).toBe(false);
    const t = { module: "pkg/mesh" };
    expect(withTarget(withTarget([], t), { module: "pkg/mesh" })).toHaveLength(1);
    expect(targetLabel(t)).toBe("module pkg/mesh");
  });
});

const request: Request = {
  id: "R3", mode: "try", kind: "method", status: "trying", idea: "solvers",
  targets: [{ file: "pkg/solver.py", function: "relax" }, { component: "mesh" }],
  variants: ["CG", "BiCGStab", "GMRES"], metric: "time_s", start_value: 10,
  steps: [
    { run_id: "r1", variant: 1, axis: "[R3.1] CG", verdict: "accept", value: 8, delta: -0.2 },
    { run_id: "r2", variant: 2, axis: "[R3.2] BiCGStab", verdict: "reject", value: 9, delta: -0.1 },
    { run_id: "r5", variant: null, axis: "[R3] tweak", verdict: "reject", value: 11, delta: 0.1 },
  ],
};

describe("a recorded request", () => {
  it("reads its axis tag", () => {
    expect(axisTag("[R3.2] BiCGStab")).toEqual({ id: "R3", variant: 2 });
    expect(axisTag("[R12] x")).toEqual({ id: "R12", variant: null });
    expect(axisTag("tiling")).toBeNull();
  });

  it("lines its variants up with their runs, then the runs that named none", () => {
    expect(variantRows(request).map((r) => [r.label, r.step?.run_id ?? null])).toEqual([
      ["1. CG", "r1"], ["2. BiCGStab", "r2"], ["3. GMRES", null], ["tweak", "r5"],
    ]);
  });

  it("is drawn on the modules it targets", () => {
    expect(touches(request, "pkg/solver", "pkg", new Set(["pkg/solver.py"]))).toBe(true);
    expect(touches(request, "mesh/io", "mesh", new Set(["mesh/io.py"]))).toBe(true);
    expect(touches(request, "pkg/other", "pkg", new Set(["pkg/other.py"]))).toBe(false);
    expect(live("trying") && !live("done")).toBe(true);
  });

  it("shows its variants against the start, a click on one selecting its step", () => {
    const html = requestDetail(request, (t) => targetLabel(t), (id) => (id === "r1" ? 4 : null));
    expect(html).toContain("Variants against the start (time_s 10)");
    expect(html).toContain('<tr data-step-n="4" tabindex="0"><td>1. CG</td><td><span class="is-ok">✓ kept</span></td><td>−20%</td></tr>');
    expect(html).toContain("<td>3. GMRES</td><td><span class=\"empty\">not run</span>");
    expect(html).toContain("data-ask=");
  });

  it("points to the chat for what the agent found, rather than repeating it", () => {
    const study: Request = { ...request, mode: "study", status: "proposed", token: "tok9", session: "s-42",
      summary: "a long analysis", conclusion: "an outcome", steps: [] };
    const html = requestDetail(study, (t) => targetLabel(t), () => null);
    expect(html).toContain('data-reveal="tok9" data-session="s-42">See the study in the chat</button>');
    expect(html).not.toContain("a long analysis");
    expect(html).not.toContain("an outcome");
  });

  it("lists the ones sent before the recorded ones", () => {
    const html = requestsSection([request], [{ token: "t", idea: "fft", mode: "study", targets: [] }]);
    expect(html.indexOf("fft")).toBeLessThan(html.indexOf("R3"));
    expect(html).toContain('st-sent');
  });
});

describe("the form", () => {
  it("asks for a source only for external code, and can be sent once ready", () => {
    expect(formHtml(draft(), "", [])).not.toContain("rq-source");
    expect(formHtml({ ...draft(), kind: "external" }, "", [])).toContain("rq-source");
    expect(formHtml(draft(), "", [])).toContain("<button class=\"open\" data-send>");
    expect(formHtml({ ...draft(), idea: "" }, "", [])).toContain("data-send disabled");
  });
});
