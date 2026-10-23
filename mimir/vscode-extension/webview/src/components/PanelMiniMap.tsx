import React, { useLayoutEffect, useRef, useState } from "react";
import type { PanelMiniMap as MiniMap } from "../types";

interface Props {
  map: MiniMap;
  /** Open the full view the section offers. */
  onOpen: () => void;
}

interface Line { d: string; breaks: boolean; key: string }

/**
 * A section's small map: its components as cards in bands, one band per tier with the
 * highest on top, and the links between them drawn once the cards have their places.
 * The server chose what to put in it; this draws it and opens the full view on a click.
 */
export const PanelMiniMap: React.FC<Props> = ({ map, onOpen }) => {
  const root = useRef<HTMLDivElement>(null);
  const [lines, setLines] = useState<Line[]>([]);
  const tiers = [...new Set(map.components.map((c) => c.tier))].sort((a, b) => b - a);

  // The links depend on where the cards landed, which the drawer's width decides.
  useLayoutEffect(() => {
    const el = root.current;
    if (!el) return;
    const draw = () => {
      const base = el.getBoundingClientRect();
      const box = (id: string) => el.querySelector<HTMLElement>(`[data-comp="${CSS.escape(id)}"]`)?.getBoundingClientRect();
      const out: Line[] = [];
      for (const link of map.links) {
        const a = box(link.from), b = box(link.to);
        if (!a || !b || b.top < a.bottom - 1) continue;    // only down the stack
        const ax = a.left + a.width / 2 - base.left, bx = b.left + b.width / 2 - base.left;
        const y1 = a.bottom - base.top, y2 = b.top - base.top, k = Math.max(6, (y2 - y1) / 2);
        out.push({ d: `M${ax},${y1} C${ax},${y1 + k} ${bx},${y2 - k} ${bx},${y2}`,
                   breaks: link.breaks > 0, key: link.from + "|" + link.to });
      }
      setLines(out);
    };
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(el);
    return () => observer.disconnect();
  }, [map]);

  return (
    <div className="science-minimap" ref={root}>
      <svg className="science-minimap-links" aria-hidden="true">
        {lines.map((l) => (
          <path key={l.key} d={l.d} className={l.breaks ? "is-break" : ""} />
        ))}
      </svg>
      {tiers.map((tier) => (
        <div className="science-minimap-band" key={tier}>
          {map.components.filter((c) => c.tier === tier).map((c) => (
            <button
              key={c.id}
              className="science-minimap-card"
              data-comp={c.id}
              title={`${c.id} · tier ${c.tier} · ${c.files} files`}
              onClick={onOpen}
            >
              <span className="science-minimap-name">{c.name}</span>
              <span className="science-minimap-sub">
                {c.files} files
                {c.breaks > 0 && <span className="is-break" title="includes that break a rule"> ✕ {c.breaks}</span>}
                {c.cycles > 0 && <span className="is-cycle" title="module cycles inside"> ↻ {c.cycles}</span>}
              </span>
            </button>
          ))}
        </div>
      ))}
    </div>
  );
};
