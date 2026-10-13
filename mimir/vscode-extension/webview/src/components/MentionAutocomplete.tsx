import React, { useEffect, useRef } from "react";
import type { MentionItem } from "./mentionUtils";

interface Props {
  items: MentionItem[];
  activeIndex: number;
  onPick: (item: MentionItem) => void;
  onHover: (index: number) => void;
}

/**
 * Copilot/Claude-style "@" dropdown listing what can be attached: the MCP resources a
 * connected server exposes, and the workspace's files. Presentational: the parent owns
 * the query text, the item list and keyboard nav (Up/Down/Enter/Esc); this just renders
 * and reports clicks/hover. Positioned above the input by CSS.
 */
export function MentionAutocomplete({ items, activeIndex, onPick, onHover }: Props) {
  const listRef = useRef<HTMLUListElement>(null);

  // Keep the active row scrolled into view during keyboard nav.
  useEffect(() => {
    const el = listRef.current?.children[activeIndex] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (items.length === 0) return null;

  return (
    <div className="mention-popup" role="listbox" aria-label="Attach a file or resource">
      <ul ref={listRef} className="mention-list">
        {items.map((item, i) => (
          <li
            key={`${item.kind}:${item.insert}`}
            role="option"
            aria-selected={i === activeIndex}
            className={"mention-item" + (i === activeIndex ? " active" : "")}
            // onMouseDown (not onClick) so the pick fires before the textarea blurs.
            onMouseDown={(e) => {
              e.preventDefault();
              onPick(item);
            }}
            onMouseEnter={() => onHover(i)}
          >
            <span className="mention-icon" aria-hidden="true">
              {item.kind === "file" ? "📄" : "◈"}
            </span>
            <span className="mention-name">{item.label}</span>
            <span className="mention-uri" title={item.detail}>{item.detail}</span>
            {item.kind === "resource" && item.description && (
              <span className="mention-desc">{item.description}</span>
            )}
          </li>
        ))}
      </ul>
      <div className="mention-hint">
        ↑↓ to navigate · Enter to attach · Esc to dismiss · add a line range with
        <code> @src/foo.py:10-20</code>
      </div>
    </div>
  );
}
