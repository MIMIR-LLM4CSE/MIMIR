import React, { useEffect, useRef } from "react";
import type { ToggleItem } from "../types";

interface Props {
  items: ToggleItem[];
  activeIndex: number;
  onPick: (item: ToggleItem) => void;
  onHover: (index: number) => void;
}

/**
 * Copilot/Claude-style "/" dropdown listing skills that can be invoked as a slash
 * command. Presentational: the parent owns the query text, filtered items and keyboard
 * nav (Up/Down/Enter/Esc); this just renders and reports clicks/hover. Reuses the
 * "@"-mention popup CSS (.mention-*). Positioned above the input by CSS.
 */
export function SlashAutocomplete({ items, activeIndex, onPick, onHover }: Props) {
  const listRef = useRef<HTMLUListElement>(null);

  // Keep the active row scrolled into view during keyboard nav.
  useEffect(() => {
    const el = listRef.current?.children[activeIndex] as HTMLElement | undefined;
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (items.length === 0) return null;

  return (
    <div className="mention-popup" role="listbox" aria-label="Run a skill command">
      <ul ref={listRef} className="mention-list">
        {items.map((item, i) => (
          <li
            key={item.name}
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
            <span className="mention-name">/{item.name}</span>
            {item.description && (
              <span className="mention-desc">{item.description}</span>
            )}
          </li>
        ))}
      </ul>
      <div className="mention-hint">
        ↑↓ to navigate · Enter to run · Esc to dismiss
      </div>
    </div>
  );
}
