/** Normalize LaTeX delimiters to the ones remark-math understands.
 *
 * The math pipeline (remark-math → rehype-katex) only recognizes `$…$` and
 * `$$…$$`. Models routinely emit the other standard LaTeX pair, `\(…\)` and
 * `\[…\]`, and Markdown then reads those backslashes as escapes: `\[` becomes a
 * literal `[`, the `\\` row separators inside an `aligned` block collapse to a
 * single `\`, and the equation reaches the reader as one mangled line of text.
 *
 * So rewrite the delimiters before parsing. Display math is emitted as its own
 * block (blank lines around it) — `\[…\]` is display math even when it appears
 * mid-sentence, and a paragraph break is the correct rendering of that.
 *
 * Code is left strictly alone: a fenced block or an inline span showing LaTeX
 * source, or C code indexing `arr\[i\]`, must survive verbatim.
 */

// Fenced blocks (``` or ~~~, any info string) and inline spans (`…`). Kept as
// capture groups so `split` returns them interleaved with the prose.
const CODE_SEGMENT = /(```[\s\S]*?(?:```|$)|~~~[\s\S]*?(?:~~~|$)|`[^`\n]*`)/g;

// Non-greedy bodies: consecutive equations must not be swallowed into one.
const DISPLAY_MATH = /\\\[([\s\S]+?)\\\]/g;
const INLINE_MATH = /\\\(([\s\S]+?)\\\)/g;

function convert(prose: string): string {
  return prose
    .replace(DISPLAY_MATH, (_m, body: string) => `\n\n$$\n${body.trim()}\n$$\n\n`)
    .replace(INLINE_MATH, (_m, body: string) => `$${body.trim()}$`);
}

export function normalizeMathDelimiters(text: string): string {
  // Cheap bail-out: the overwhelmingly common case has no such delimiter, and
  // this runs on every token of a streaming reply.
  if (!text.includes("\\[") && !text.includes("\\(")) return text;
  return text
    .split(CODE_SEGMENT)
    .map((seg, i) => (i % 2 === 1 ? seg : convert(seg)))   // odd = captured code
    .join("");
}
