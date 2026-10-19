// The batch review's file list: what a diff does to it. Free of React, so unit-testable.

import type { DiffEntry } from "../types";

/**
 * *files* with the diff of *file* folded in: the emitter diffs against the turn's
 * baseline, so a file already listed takes the new patch rather than a second entry.
 */
export function withDiff(files: DiffEntry[], file: string, patch: string): DiffEntry[] {
  const isNew = !patch || patch.includes("/dev/null");
  const idx = files.findIndex((d) => d.file === file);
  if (idx >= 0) return files.map((d, i) => (i === idx ? { ...d, patch, is_new: isNew } : d));
  return [...files, { file, patch, is_new: isNew }];
}
