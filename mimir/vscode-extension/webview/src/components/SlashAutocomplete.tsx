import React, { useEffect, useRef } from "react";
import type { SlashItem } from "./slashUtils";

interface Props {
  items: SlashItem[];
  activeIndex: number;
  onPick: (item: SlashItem) => void;
  onHover: (index: number) => void;
}

const GROUP_LABEL: Record<SlashItem["kind"], string> = {
  command: "Session commands",
  skill: "Skills",
};

/**
 * Copilot/Claude-style "/" dropdown listing the session commands and the skills that
 * can be invoked as a slash command, each kind under its own heading. Presentational:
 * the parent owns the query text, filtered items and keyboard nav (Up/Down/Enter/Esc);
 * this just renders and reports clicks/hover. Reuses the "@"-mention popup CSS
 * (.mention-*). Positioned above the input by CSS.
 */
export function SlashAutocomplete({ items, activeIndex, onPick, onHover }: Props) {
  const listRef = useRef<HTMLUListElement>(null);

  // Keep the active row scrolled into view during keyboard nav. Headings sit among the
  // rows, so the row is found by its index rather than its position in the list.
  useEffect(() => {
    const el = listRef.current?.querySelector(`[data-index="${activeIndex}"]`);
    el?.scrollIntoView({ block: "nearest" });
  }, [activeIndex]);

  if (items.length === 0) return null;

  return (
    <div className="mention-popup" role="listbox" aria-label="Run a command or a skill">
      <ul ref={listRef} className="mention-list">
        {items.map((item, i) => (
          <React.Fragment key={`${item.kind}:${item.name}`}>
            {(i === 0 || items[i - 1].kind !== item.kind) && (
              <li role="presentation" className="slash-group">
                {GROUP_LABEL[item.kind]}
              </li>
            )}
            <li
              role="option"
              data-index={i}
              aria-selected={i === activeIndex}
              className={
                "mention-item" +
                (i === activeIndex ? " active" : "") +
                (item.kind === "skill" && !item.enabled ? " disabled" : "")
              }
              // onMouseDown (not onClick) so the pick fires before the textarea blurs.
              onMouseDown={(e) => {
                e.preventDefault();
                onPick(item);
              }}
              onMouseEnter={() => onHover(i)}
            >
              <span className="mention-name">/{item.name}</span>
              <span className={`slash-kind slash-kind-${item.kind}`}>
                {item.kind === "skill" ? "skill" : "command"}
              </span>
              {item.description && (
                <span className="mention-desc">{item.description}</span>
              )}
            </li>
          </React.Fragment>
        ))}
      </ul>
      <div className="mention-hint">
        ↑↓ to navigate · Enter to run · Esc to dismiss
      </div>
    </div>
  );
}
