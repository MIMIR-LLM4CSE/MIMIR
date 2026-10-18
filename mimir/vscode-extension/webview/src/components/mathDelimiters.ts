/** Normalize LaTeX delimiters to the ones remark-math understands (`$…$`, `$$…$$`).
 *
 * `\(…\)` and `\[…\]` would otherwise reach Markdown as escapes and mangle the equation;
 * display math becomes its own block. Code spans and fences are left untouched.
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
