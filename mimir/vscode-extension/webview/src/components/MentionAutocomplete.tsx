import React, { useEffect, useRef } from "react";
import type { ResourceItem } from "../types";

interface Props {
  items: ResourceItem[];
  activeIndex: number;
  onPick: (item: ResourceItem) => void;
  onHover: (index: number) => void;
}

/**
 * Copilot/Claude-style "@" dropdown listing attachable MCP resources. Presentational:
 * the parent owns the query text, filtered items and keyboard nav (Up/Down/Enter/Esc);
 * this just renders and reports clicks/hover. Positioned above the input by CSS.
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
    <div className="mention-popup" role="listbox" aria-label="Attach a resource">
      <ul ref={listRef} className="mention-list">
        {items.map((item, i) => (
          <li
            key={item.uri}
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
            <span className="mention-name">@{item.name || item.uri}</span>
            <span className="mention-uri">{item.uri}</span>
            {item.description && (
              <span className="mention-desc">{item.description}</span>
            )}
          </li>
        ))}
      </ul>
      <div className="mention-hint">
        ↑↓ to navigate · Enter to attach · Esc to dismiss · or type a file path like
        <code> @src/foo.py:10-20</code>
      </div>
    </div>
  );
}
