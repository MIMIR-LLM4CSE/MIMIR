import React from "react";
import type { CommandOutputMessage } from "../types";

/**
 * The answer to a session command ("/memory list", "/memory clear", "/proxy list").
 *
 * Two shapes from one payload, chosen here rather than by the backend — which has no
 * business knowing how wide the panel is:
 *
 *  * a SETTING change is one value with nothing to explain, so it reads as a single
 *    line: `✓ /mode  Mode  agent`.
 *  * a LISTING, or anything with a note under it, gets a card with rows.
 *
 * Both carry the command that produced them, because an answer arriving in a chat
 * transcript with no question above it is a sentence with no subject — the command
 * itself never appears as a user bubble.
 */

const TONE_ICON: Record<string, string> = { ok: "✓", warn: "⚠", empty: "·" };

export const CommandResult: React.FC<{ result: CommandOutputMessage }> = ({
  result,
}) => {
  const items = (result.items ?? []).filter((i) => (i?.label ?? "").trim());
  const tone = result.tone ?? "ok";
  const note = (result.note ?? "").trim();
  const icon = TONE_ICON[tone] ?? TONE_ICON.ok;

  const compact =
    !note && items.length <= 1 && items.every((i) => !(i.detail ?? "").trim());

  if (compact) {
    return (
      <div className={`cmd-line cmd-line--${tone}`}>
        <span className="cmd-line__icon" aria-hidden="true">
          {icon}
        </span>
        {result.command && <code className="cmd-chip">{result.command}</code>}
        <span className="cmd-line__title">{result.title}</span>
        {items[0] && <span className="cmd-value">{items[0].label}</span>}
      </div>
    );
  }

  return (
    <div className={`cmd-card cmd-card--${tone}`}>
      <div className="cmd-card__head">
        <span className="cmd-card__icon" aria-hidden="true">
          {icon}
        </span>
        {result.command && <code className="cmd-chip">{result.command}</code>}
        <span className="cmd-card__title">{result.title}</span>
      </div>
      {items.length > 0 && (
        <ul className="cmd-card__items">
          {items.map((item, i) => (
            <li key={i} className="cmd-card__item">
              <span className="cmd-card__label">{item.label}</span>
              {(item.detail ?? "").trim() && (
                <span className="cmd-card__detail">{item.detail}</span>
              )}
            </li>
          ))}
        </ul>
      )}
      {note && <div className="cmd-card__note">{note}</div>}
    </div>
  );
};
