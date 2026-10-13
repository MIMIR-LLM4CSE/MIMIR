/**
 * Workspace file ranking for the chat's "@" autocomplete.
 *
 * The webview cannot enumerate the workspace: it has no filesystem and its CSP
 * reaches nothing but the extension host. So the host scans once, caches the
 * relative paths, and answers each keystroke from that list — the scoring below.
 *
 * Matching is subsequence-based like VS Code's own Quick Open: the typed letters
 * have to appear in order, not contiguously, so `apcx` finds
 * `webview/src/App.tsx`. A match in the file name outranks one in the directory,
 * because that is the part the user is usually typing.
 *
 * Kept free of the `vscode` module so it is unit-testable (see fileIndex.test.ts);
 * extension.ts owns the scan, the cache and the message plumbing.
 */

/**
 * Directories never worth offering as context, dropped from the index whatever the
 * workspace's own `files.exclude` says. Virtualenvs and dependency trees hold tens
 * of thousands of files that are nobody's question.
 */
const NOISE_SEGMENTS = [
  ".git",
  "node_modules",
  "__pycache__",
  ".mypy_cache",
  ".pytest_cache",
  ".ruff_cache",
  "dist",
  "build",
  ".egg-info",
  "site-packages",
];

/** Whether a workspace-relative path lies in one of the trees above. */
export function isNoisePath(relPath: string): boolean {
  const segments = relPath.split("/");
  return segments.some((seg, i) =>
    // The last segment is the file itself: a file named `dist` is fine, a
    // directory named `dist` is not.
    i < segments.length - 1 &&
    NOISE_SEGMENTS.some((noise) => seg === noise || seg.endsWith(noise))
  );
}

/**
 * Score `relPath` against `query`, or null when the query is not a subsequence of it.
 *
 * Higher is better. The parts that make a match good, in the order they matter:
 * the whole query sitting in the file name, matching from the name's first letter,
 * contiguous letters, and a short path (a file near the root is more likely the one
 * meant than a namesake six directories down).
 */
export function scoreFilePath(relPath: string, query: string): number | null {
  const q = query.toLowerCase();
  if (!q) return 0;
  const hay = relPath.toLowerCase();
  const nameStart = hay.lastIndexOf("/") + 1;

  let score = 0;
  let from = 0;
  let prevIndex = -2;
  let firstIndex = -1;
  for (const ch of q) {
    const at = hay.indexOf(ch, from);
    if (at === -1) return null;
    if (firstIndex === -1) firstIndex = at;
    if (at === prevIndex + 1) score += 6;        // contiguous run
    if (at >= nameStart) score += 4;             // inside the file name
    if (at === nameStart) score += 10;           // file name starts here
    if (at > 0 && /[/_\-.]/.test(hay[at - 1])) score += 3; // word boundary
    prevIndex = at;
    from = at + 1;
  }

  // A contiguous hit anywhere beats a scattered one; in the name, more so.
  const literal = hay.indexOf(q);
  if (literal !== -1) score += literal >= nameStart ? 25 : 10;
  // Shorter paths first, and shallower ones before deep namesakes.
  score -= relPath.split("/").length * 2;
  score -= Math.min(relPath.length, 80) / 20;
  return score;
}

/**
 * The best `limit` matches for `query`, best first.
 *
 * An empty query is the list the dropdown opens with, right after the "@" and before
 * anything is typed: shallowest paths first, alphabetical within a depth, so it reads
 * as the top of the workspace rather than an arbitrary slice of the scan order.
 */
export function rankFiles(paths: string[], query: string, limit = 40): string[] {
  const q = query.trim();
  if (!q) {
    return [...paths]
      .sort((a, b) => {
        const depth = a.split("/").length - b.split("/").length;
        return depth !== 0 ? depth : a.localeCompare(b);
      })
      .slice(0, limit);
  }
  const scored: { path: string; score: number }[] = [];
  for (const path of paths) {
    const score = scoreFilePath(path, q);
    if (score !== null) scored.push({ path, score });
  }
  scored.sort((a, b) => (b.score - a.score) || a.path.localeCompare(b.path));
  return scored.slice(0, limit).map((s) => s.path);
}
