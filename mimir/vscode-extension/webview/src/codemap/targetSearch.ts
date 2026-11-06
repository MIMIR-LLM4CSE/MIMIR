// The request form's file search: the workspace's files as the chat's "@" lists them,
// ranked by the host. A pick adds the file to the request's targets.

import { esc } from "./format";

const SHOWN = 8;

/** What is searched for in what was typed: the "@" that opens it is the chat's habit. */
export const fileQuery = (typed: string) => typed.trim().replace(/^@/, "");

/** The rows of *files*, *at* the one Enter picks. */
export function hitsHtml(files: string[], at: number): string {
  return files.map((f, i) => {
    const cut = f.lastIndexOf("/");
    return '<div class="row two' + (i === at ? " is-cur" : "") + '"><button class="go" data-add-file="' + esc(f) + '" title="Add '
      + esc(f) + ' to the targets"><span class="nm">' + esc(f.slice(cut + 1)) + '</span><span class="sub">'
      + esc(f.slice(0, Math.max(cut, 0))) + "</span></button></div>";
  }).join("");
}

/** What the search needs of the tab. */
export interface SearchHost {
  ask(query: string): void;
  add(file: string): void;
  /** Whether *file* is a target already. */
  taken(file: string): boolean;
}

/** The search outlives the form's redraws: what was typed, its hits and the focus are put
 *  back on the new input by `mount`. */
export class TargetSearch {
  private typed = "";
  private hits: string[] = [];
  private at = 0;
  private input: HTMLInputElement | null = null;
  private list: HTMLElement | null = null;
  private refocus = false;

  constructor(private host: SearchHost) {}

  html(): string {
    return '<div class="tsearch"><input id="rq-file" aria-label="Add a file to the targets" placeholder="@ Add a file of the workspace…" '
      + 'autocomplete="off" spellcheck="false"><div class="list thits"></div></div>';
  }

  /** To call before the form is redrawn: its input goes with the focus. */
  leave(): void {
    this.refocus ||= !!this.input && document.activeElement === this.input;
  }

  /** Binds the form's search, or forgets it all when no form is open. */
  mount(box: HTMLElement | null): void {
    const input = box?.querySelector("input") ?? null;
    this.input = input;
    this.list = box?.querySelector<HTMLElement>(".thits") ?? null;
    if (!input) { this.typed = ""; this.hits = []; this.refocus = false; return; }
    input.value = this.typed;
    this.draw();
    input.addEventListener("input", () => this.type(input.value));
    input.addEventListener("keydown", (e) => this.key(e));
    this.list?.addEventListener("click", (e) => {
      const file = (e.target as Element).closest<HTMLElement>("[data-add-file]")?.dataset.addFile;
      if (file) this.pick(file);
    });
    if (this.refocus) {
      input.focus({ preventScroll: true });
      input.setSelectionRange(input.value.length, input.value.length);
    }
    this.refocus = false;
  }

  /** The host's answer, unless the typing has overtaken it. */
  answer(query: string, files: string[]): void {
    if (!this.typed.trim() || query !== fileQuery(this.typed)) return;
    this.hits = files;
    this.at = 0;
    this.draw();
  }

  private shown(): string[] {
    return this.hits.filter((f) => !this.host.taken(f)).slice(0, SHOWN);
  }

  private draw(): void {
    if (this.list) this.list.innerHTML = hitsHtml(this.shown(), this.at);
  }

  private type(value: string): void {
    this.typed = value;
    if (value.trim()) { this.host.ask(fileQuery(value)); return; }
    this.hits = [];
    this.draw();
  }

  private pick(file: string): void {
    this.typed = "";
    this.hits = [];
    this.refocus = true;
    this.host.add(file);
  }

  private key(e: KeyboardEvent): void {
    const n = this.shown().length;
    if (e.key === "Escape" && this.typed) {
      if (this.input) this.input.value = "";
      this.type("");
    } else if (e.key === "Enter" && n) {
      this.pick(this.shown()[this.at]);
    } else if ((e.key === "ArrowDown" || e.key === "ArrowUp") && n) {
      this.at = (this.at + (e.key === "ArrowDown" ? 1 : n - 1)) % n;
      this.draw();
    } else return;
    e.preventDefault();
    e.stopPropagation();
  }
}
