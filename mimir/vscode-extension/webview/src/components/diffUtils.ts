import hljs from "highlight.js/lib/common";

/** One parsed line of a unified diff.
 *
 * `wordRange` is the [start, end) char range in the line *content* (i.e. in
 * `text.slice(1)`) that actually changed vs its paired add/del line — used for
 * GitHub-style intra-line emphasis. Absent when the whole line reads better.
 */
export interface DiffLine {
  type: "header" | "hunk" | "add" | "del" | "ctx" | "meta";
  text: string; // raw line, marker included
  wordRange?: [number, number];
}

/** Parse a unified diff into typed lines.
 *
 * `+++`/`---` count as file headers only before the first `@@` hunk marker —
 * past that point they are real content lines (e.g. a removed markdown rule).
 * `\ No newline at end of file` markers get their own `meta` type so renderers
 * don't strip their first char like a diff marker.
 */
export function parsePatch(patch: string): DiffLine[] {
  let inBody = false;
  return patch.split("\n").map((line): DiffLine => {
    if (line.startsWith("@@")) {
      inBody = true;
      return { type: "hunk", text: line };
    }
    if (!inBody && (line.startsWith("+++") || line.startsWith("---")))
      return { type: "header", text: line };
    if (line.startsWith("\\")) return { type: "meta", text: line };
    if (line.startsWith("+")) return { type: "add", text: line };
    if (line.startsWith("-")) return { type: "del", text: line };
    return { type: "ctx", text: line };
  });
}

/** Count add/del lines of a raw patch (shared +N/−N stats). */
export function countDiffLines(patch: string): { adds: number; dels: number } {
  let adds = 0, dels = 0;
  for (const l of parsePatch(patch)) {
    if (l.type === "add") adds++;
    else if (l.type === "del") dels++;
  }
  return { adds, dels };
}

/** When the changed region spans more than this share of the longer line, skip
 *  intra-line emphasis — a whole-line tint reads better than a huge highlight. */
const WORD_RANGE_MAX_SHARE = 0.7;

/** Annotate paired del/add lines with the char range that differs (in place).
 *
 * Pairs each maximal run of consecutive `del` lines with the immediately
 * following run of `add` lines, index by index (runs reset at any other line
 * type). For each pair, the common prefix/suffix delimits the changed region.
 * Unpaired surplus lines, pure insertions (`is_new`) and pure deletions
 * (`is_delete`) naturally get no range.
 */
export function computeWordRanges(lines: DiffLine[]): void {
  let i = 0;
  while (i < lines.length) {
    if (lines[i].type !== "del") { i++; continue; }
    const delStart = i;
    while (i < lines.length && lines[i].type === "del") i++;
    const addStart = i;
    while (i < lines.length && lines[i].type === "add") i++;
    const pairs = Math.min(i - addStart, addStart - delStart);
    for (let p = 0; p < pairs; p++) {
      const a = lines[delStart + p].text.slice(1);
      const b = lines[addStart + p].text.slice(1);
      let pre = 0;
      const maxPre = Math.min(a.length, b.length);
      while (pre < maxPre && a[pre] === b[pre]) pre++;
      let suf = 0;
      const maxSuf = maxPre - pre; // prefix and suffix must not overlap
      while (suf < maxSuf && a[a.length - 1 - suf] === b[b.length - 1 - suf]) suf++;
      const rangeA: [number, number] = [pre, a.length - suf];
      const rangeB: [number, number] = [pre, b.length - suf];
      const span = Math.max(rangeA[1] - rangeA[0], rangeB[1] - rangeB[0]);
      if (span === 0) continue; // identical apart from the marker
      if (span > WORD_RANGE_MAX_SHARE * Math.max(a.length, b.length)) continue;
      lines[delStart + p].wordRange = rangeA;
      lines[addStart + p].wordRange = rangeB;
    }
  }
}

// Extension -> highlight.js language name. Only languages present in the
// bundled "common" grammar set; anything else falls back to plain text.
const EXT_TO_LANG: Record<string, string> = {
  ts: "typescript", tsx: "typescript", mts: "typescript", cts: "typescript",
  js: "javascript", jsx: "javascript", mjs: "javascript", cjs: "javascript",
  py: "python", pyw: "python",
  json: "json", jsonc: "json",
  css: "css", scss: "scss", less: "less",
  html: "xml", htm: "xml", xml: "xml", svg: "xml", vue: "xml",
  md: "markdown", markdown: "markdown",
  sh: "bash", bash: "bash", zsh: "bash",
  yml: "yaml", yaml: "yaml",
  c: "c", h: "c",
  cc: "cpp", cpp: "cpp", cxx: "cpp", hpp: "cpp", hh: "cpp", cu: "cpp", cuh: "cpp",
  cs: "csharp", java: "java", kt: "kotlin", rs: "rust", go: "go",
  rb: "ruby", php: "php", sql: "sql", swift: "swift", lua: "lua",
  pl: "perl", r: "r",
  ini: "ini", toml: "ini", cfg: "ini",
  mk: "makefile",
  diff: "diff", patch: "diff",
  graphql: "graphql", gql: "graphql",
};

const BASENAME_TO_LANG: Record<string, string> = {
  makefile: "makefile", gnumakefile: "makefile",
};

/** Resolve the hljs language for a file path, or undefined for plain text. */
export function langFromPath(file: string): string | undefined {
  const base = (file.split("/").pop() || file).toLowerCase();
  let lang = BASENAME_TO_LANG[base];
  if (!lang) {
    const dot = base.lastIndexOf(".");
    if (dot < 0) return undefined;
    lang = EXT_TO_LANG[base.slice(dot + 1)];
  }
  // Validate against the bundled grammars so a stale entry degrades to plain.
  return lang && hljs.getLanguage(lang) ? lang : undefined;
}

/** Highlight a code fragment, returning escaped HTML — or undefined to signal
 *  the caller to render plain text (unknown language or hljs failure). */
export function highlightFragment(code: string, lang: string): string | undefined {
  try {
    return hljs.highlight(code, { language: lang, ignoreIllegals: true }).value;
  } catch {
    return undefined;
  }
}
