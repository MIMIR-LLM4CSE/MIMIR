import React from "react";
import type { ReactNode } from "react";

/** The GitHub mark, as the one icon in the set that is not an emoji.
 *
 *  Octicon `mark-github` (MIT). Drawn in `currentColor` so it takes the row's own
 *  colour in either theme, and sized in `em` so it sits in the same box as the emoji
 *  beside it. An emoji octopus says "GitHub" only to someone who already knows the
 *  Octocat; the mark itself is what a reader recognises. GitHub's brand guidelines
 *  allow it for identifying GitHub content, which is exactly this use. */
const GitHubMark: React.FC = () => (
  <svg viewBox="0 0 16 16" role="img" aria-label="GitHub" focusable="false">
    <path
      fill="currentColor"
      d="M8 0C3.58 0 0 3.58 0 8c0 3.54 2.29 6.53 5.47 7.59.4.07.55-.17.55-.38 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21 1.87.87 2.33.66.07-.52.28-.87.51-1.07-1.78-.2-3.64-.89-3.64-3.95 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82.64-.18 1.32-.27 2-.27s1.36.09 2 .27c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73.54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8.012 8.012 0 0 0 16 8c0-4.42-3.58-8-8-8Z"
    />
  </svg>
);

/** Work family → icon. The families are declared by the servers (`TOOL_KINDS` in
 *  servers/_shared/capabilities.py) and the row shows the same word it draws the glyph
 *  from, so the two cannot drift apart — which is what a table keyed on tool *names*
 *  did, with ten of its twelve keys naming tools that no longer exist.
 *
 *  Emoji throughout, with the one exception above: a bare text glyph (`❯`, `∑`) takes
 *  the text colour and not the width of its neighbours, and breaks the column. */
export const KIND_ICONS: Record<string, ReactNode> = {
  // discovery
  read: "📖", search: "🔍", list: "📂", outline: "🧭",
  // file mutation
  write: "📝", edit: "✏️", delete: "🗑️",
  // local execution
  shell: "💻", job: "⏱️", verdict: "⚖️",
  // calculation
  eval: "🧮", symbolic: "📐", string: "🔤", date: "📅",
  // cluster & environment
  slurm: "🛰️", env: "📦", modules: "🧩",
  // the proxy harness
  proxy: "🧪", "proxy eval": "⚡",
  // agent state
  memory: "🧠", plan: "📋", skill: "📘", agent: "🤝", ask: "❓",
  // outside the workspace
  web: "🌐", github: <GitHubMark />, system: "🖥️",
  // the unknown default
  tool: "🔧",
};

/** The icon for a work family, falling back to the generic tool glyph.
 *
 *  A family this build has no icon for still gets a row: the kind is a word the server
 *  chose, and an extension-pack server may name one of its own. The word carries it;
 *  the glyph is generic. */
export function iconForKind(kind: string | undefined): ReactNode {
  return (kind && KIND_ICONS[kind]) || KIND_ICONS.tool;
}
