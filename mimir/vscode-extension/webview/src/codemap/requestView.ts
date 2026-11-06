// The side pane's R&D requests: the form that writes one, the list of those sent, and one
// request whole with its variants side by side. Plain HTML; index.ts binds the events.

import { esc, signed } from "./format";
import {
  dropPrompt, holdsAll, KINDS, kindLabel, live, ready, STATUS_LABEL, targetLabel, tryPrompt, variantRows,
  type Draft, type Request, type Sent, type Target,
} from "./request";

const PLACEHOLDER: Record<string, string> = {
  method: "e.g. replace the Jacobi sweeps by a multigrid V-cycle",
  algorithm: "e.g. a k-d tree instead of the all-pairs search",
  external: "e.g. solve the linear systems with PETSc's KSP",
  parallel: "e.g. OpenMP over the outer loop, then MPI over the domain",
  layout: "e.g. structure of arrays for the particles",
  precision: "e.g. float32 storage with float64 accumulation",
  refactor: "e.g. break the solver ↔ mesh cycle; move the I/O out of the kernels",
  other: "what to study or try",
};

export function statusPill(status: string): string {
  return '<span class="st st-' + esc(status) + '">' + esc(STATUS_LABEL[status] ?? status) + "</span>";
}

/** A button per tier that takes its components into the targets, pressed when all are
 *  there; none when the code has a single tier, which is the whole code. */
export function tierButtons(tiers: [number, string[]][], targets: Target[]): string {
  if (tiers.length < 2) return "";
  return '<div class="ttiers" role="group" aria-label="Whole tiers">' + tiers.map(([n, comps]) =>
    '<button class="ttier" data-tier="' + n + '" aria-pressed="' + holdsAll(targets, comps) + '" title="' + esc(comps.join(", "))
    + '">Tier ' + n + " · " + comps.length + "</button>").join("") + "</div>";
}

/** The form, filled in from *d*. *adders* are what add to the targets (the file search, the
 *  tiers, the targets' costliest functions); *findings* what the map sees in a refactor's scope. */
export function formHtml(d: Draft, adders: string, findings: string[]): string {
  // The label's end (the file, the function) is what tells two targets apart: cut at its start.
  const chips = d.targets.map((t, i) => '<span class="tchip" title="' + esc(targetLabel(t)) + '"><span class="tl"><bdi>'
    + esc(targetLabel(t)) + "</bdi></span>"
    + '<button class="x" data-untarget="' + i + '" aria-label="Remove ' + esc(targetLabel(t)) + '">×</button></span>').join("");
  const field = (label: string, input: string, hint = "") => '<label class="fl"><span>' + esc(label)
    + (hint ? ' <i class="hint">' + esc(hint) + "</i>" : "") + "</span>" + input + "</label>";
  return '<div class="section req-form"><div class="title">New request</div>'
    + '<div class="seg" role="group" aria-label="Mode">'
    + '<button data-mode="study" aria-pressed="' + (d.mode === "study") + '">Study</button>'
    + '<button data-mode="try" aria-pressed="' + (d.mode === "try") + '">Try</button></div>'
    + '<div class="hint">' + (d.mode === "study" ? "The agent reads the code and proposes; it edits nothing."
      : d.kind === "refactor" ? "The agent refactors step by step; a step is kept when it costs no time beyond the noise."
      : "The agent implements each variant from the same start and measures it.") + "</div>"
    + field("Kind", '<select id="rq-kind">' + KINDS.map(([k, label]) => '<option value="' + k + '"'
      + (k === d.kind ? " selected" : "") + ">" + esc(label) + "</option>").join("") + "</select>")
    + '<div class="cap">Targets <i class="hint">Shift+click a band, a module or a function to add it</i></div>'
    + '<div class="tchips">' + (chips || '<span class="empty">none yet</span>') + "</div>"
    + adders
    + (d.kind === "refactor" && findings.length ? '<div class="cap">What the map finds there</div><ul class="finds">'
      + findings.map((f) => "<li>" + esc(f) + "</li>").join("") + "</ul>" : "")
    + field("Idea", '<textarea id="rq-idea" rows="3" placeholder="' + esc(PLACEHOLDER[d.kind] ?? "") + '">' + esc(d.idea) + "</textarea>")
    + field(d.kind === "refactor" ? "Alternatives" : "Variants", '<textarea id="rq-variants" rows="3" placeholder="'
      + (d.kind === "refactor" ? "one design per line, if there are several" : "one per line, e.g. CG / BiCGStab / GMRES")
      + '">' + esc(d.variants) + "</textarea>", "each from the same start")
    + (d.kind === "external" ? field("Source", '<input id="rq-source" placeholder="pip or conda package, local path, git repository" value="'
      + esc(d.source) + '">', "installing asks you first") : "")
    + field("Constraints", '<input id="rq-constraints" placeholder="keep the API, error under 1e-6, …" value="' + esc(d.constraints) + '">')
    + '<div class="actions"><button class="open" data-send' + (ready(d) ? "" : " disabled")
    + ">Send to the agent</button><button class=\"link\" data-cancel>Cancel</button></div></div>";
}

/** The requests, latest first: those the server recorded and those still on their way. */
export function requestsSection(requests: Request[], sent: Sent[]): string {
  if (!requests.length && !sent.length) return "";
  const rows = sent.map((s) => '<div class="row rq-row"><span class="go" title="' + esc(s.idea) + '">· ' + esc(s.idea)
    + "</span>" + statusPill("sent") + "</div>")
    .concat([...requests].reverse().map((r) => '<div class="row rq-row"><button class="go" data-request="' + esc(r.id)
      + '" title="' + esc(r.idea) + '"><b>' + esc(r.id) + "</b> " + esc(r.idea) + "</button>" + statusPill(r.status) + "</div>"));
  return '<div class="section reqs"><div class="title">Requests</div><div class="list">' + rows.join("") + "</div></div>";
}

/** One request whole: what was asked, what the agent found, its variants side by side.
 *  *targetRow* draws a target as a row that selects it on the map. */
export function requestDetail(r: Request, targetRow: (t: Target) => string, stepOf: (runId: string) => number | null): string {
  const delta = (d: number | null) => (d === null ? "–" : signed(d));
  const rows = variantRows(r);
  const table = rows.length ? '<div class="cap">' + (r.kind === "refactor" ? "Steps" : "Variants") + " against the start"
    + (r.start_value !== null && r.start_value !== undefined ? " (" + esc(r.metric ?? "") + " " + esc(r.start_value) + ")" : "")
    + '</div><table class="vt">' + rows.map(({ label, step }) => {
      const n = step ? stepOf(step.run_id) : null;
      return "<tr" + (n !== null ? ' data-step-n="' + n + '" tabindex="0"' : "") + "><td>" + esc(label) + "</td><td>"
        + (step ? (step.verdict === "accept" ? '<span class="is-ok">✓ kept</span>' : '<span class="is-danger">✕ ruled out</span>')
          : '<span class="empty">not run</span>')
        + "</td><td>" + (step ? delta(step.delta) : "") + "</td></tr>";
    }).join("") + "</table>" : "";
  const asks = (r.mode === "study" && r.status === "proposed" ? '<button class="open" data-ask="' + esc(tryPrompt(r)) + '">Try it</button> ' : "")
    + (live(r.status) || r.status === "proposed" ? '<button class="open ghost" data-ask="' + esc(dropPrompt(r)) + '">Drop</button>' : "");
  return '<div class="section sel rq"><div class="title">' + esc(r.id) + " · " + (r.mode === "study" ? "Study" : "Try") + " · "
    + esc(kindLabel(r.kind)) + "</div>" + statusPill(r.status)
    + '<div class="idea">' + esc(r.idea) + "</div>"
    + '<div class="cap">Targets</div><div class="list">' + r.targets.map(targetRow).join("") + "</div>"
    + (r.source ? '<div class="line"><span>source</span><span>' + esc(r.source) + "</span></div>" : "")
    + (r.constraints ? '<div class="line"><span>constraints</span><span>' + esc(r.constraints) + "</span></div>" : "")
    + table
    // What the agent found is its answer in the chat; the request only points there.
    + (r.token ? '<div class="actions"><button class="open ghost" data-reveal="' + esc(r.token) + '" data-session="'
      + esc(r.session ?? "") + '">' + (r.mode === "study" ? "See the study in the chat" : "See it in the chat")
      + "</button></div>" : "")
    + (asks ? '<div class="actions">' + asks + "</div>" : "") + "</div>";
}
