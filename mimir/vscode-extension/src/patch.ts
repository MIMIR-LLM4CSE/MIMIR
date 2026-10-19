/**
 * Reconstruct the "before" content of a file by reverse-applying a unified diff to its
 * current content: added lines (+) are dropped, removed lines (-) restored. Free of
 * `vscode`, so unit-testable.
 */
export function reversePatch(current: string, patch: string): string {
  const currentLines = current.split("\n");
  const result: string[] = [];
  const patchLines = patch.split("\n");

  let ci = 0; // index into currentLines

  for (const line of patchLines) {
    if (line.startsWith("---") || line.startsWith("+++") || line.startsWith("diff ")) continue;
    if (line.startsWith("@@")) {
      // @@ -oldStart,oldCount +newStart,newCount @@
      const m = line.match(/\+(\d+)(?:,(\d+))?/);
      if (m) {
        const newStart = parseInt(m[1], 10) - 1; // 0-based
        // Emit unchanged lines up to this hunk from current
        while (ci < newStart) {
          result.push(currentLines[ci++]);
        }
      }
      continue;
    }
    if (line.startsWith("+")) {
      // Added by the patch: present in the current file, absent from the original
      ci++;
    } else if (line.startsWith("-")) {
      // Removed by the patch: restored to the original
      result.push(line.slice(1));
    } else if (line.startsWith(" ")) {
      // Context: present in both
      result.push(currentLines[ci++]);
    }
  }
  // Emit any remaining lines after the last hunk
  while (ci < currentLines.length) {
    result.push(currentLines[ci++]);
  }
  return result.join("\n");
}
