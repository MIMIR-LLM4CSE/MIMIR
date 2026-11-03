import React, { useLayoutEffect, useRef, useState } from "react";
import type { PanelChart } from "../types";

interface Props {
  chart: PanelChart;
}

const HEIGHT = 120;
const PAD = { top: 8, right: 8, bottom: 16, left: 38 };

/** A value in three significant figures; thousands grouped rather than in exponents. */
export function fmt(v: number | null | undefined): string {
  if (v == null || !Number.isFinite(v)) return "–";
  if (Math.abs(v) >= 1000) return Math.round(v).toLocaleString("en");
  return String(Number(v.toPrecision(3)));
}

/** Where the best stands against the baseline: the relative change, the factor gained, and
 *  whether that is the way the goal points. Null until both are numbers. */
export function gain(chart: PanelChart): { delta: string; factor: string; better: boolean } | null {
  const { baseline: base, best, goal } = chart;
  if (base == null || best == null || base === 0 || best === 0) return null;
  const rel = (best - base) / Math.abs(base);
  const better = goal === "max" ? rel > 0 : rel < 0;
  const factor = goal === "max" ? best / base : base / best;
  const pct = Math.round(rel * 100);
  return {
    delta: `${pct > 0 ? "+" : pct < 0 ? "−" : "±"}${Math.abs(pct)}%`,
    factor: factor > 0 ? `${factor.toFixed(factor >= 10 ? 0 : 2)}×` : "",
    better,
  };
}

const KIND_LABEL: Record<string, string> = {
  kept: "kept",
  "not kept": "not kept",
  failed: "failed",
  elsewhere: "other machine",
};

/**
 * The optimisation as a curve: every run's value in the order it settled, the best so far
 * as a step line from the baseline, and the room won between them shaded. A run that
 * measured nothing sits on the floor as a cross. Hovering a run reads it out below.
 */
export const PanelCurve: React.FC<Props> = ({ chart }) => {
  const root = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(280);
  const [hover, setHover] = useState<number | null>(null);

  // The drawer's width decides the plot's; circles stay round, unlike a stretched viewBox.
  useLayoutEffect(() => {
    const el = root.current;
    if (!el || typeof ResizeObserver === "undefined") return;
    const observe = new ResizeObserver(() => setWidth(Math.max(160, el.clientWidth)));
    observe.observe(el);
    return () => observe.disconnect();
  }, []);

  const pts = chart.points;
  if (pts.length === 0) return null;
  const won = gain(chart);

  const values = [chart.baseline, ...pts.map((p) => p.value), ...pts.map((p) => p.best)]
    .filter((v): v is number => v != null && Number.isFinite(v));
  let lo = Math.min(...values), hi = Math.max(...values);
  if (!values.length) { lo = 0; hi = 1; }
  const span = hi - lo || Math.abs(hi) * 0.1 || 1;
  lo -= span * 0.08; hi += span * 0.08;

  const x0 = pts[0].n - 1, x1 = pts[pts.length - 1].n;
  const plotW = width - PAD.left - PAD.right, plotH = HEIGHT - PAD.top - PAD.bottom;
  const x = (n: number) => PAD.left + ((n - x0) / Math.max(1, x1 - x0)) * plotW;
  const y = (v: number) => PAD.top + (1 - (v - lo) / (hi - lo)) * plotH;
  const floor = PAD.top + plotH;

  // The best so far, stepping at each kept run.
  const start = chart.baseline ?? pts.find((p) => p.best != null)?.best ?? null;
  let step = "", area = "";
  if (start != null) {
    let prev = start;
    step = `M${x(x0)},${y(prev)}`;
    for (const p of pts) {
      const cur = p.best ?? prev;
      step += ` L${x(p.n)},${y(prev)} L${x(p.n)},${y(cur)}`;
      prev = cur;
    }
    if (chart.baseline != null) {
      area = `${step} L${x(x1)},${y(chart.baseline)} L${x(x0)},${y(chart.baseline)} Z`;
    }
  }

  const shown = hover != null ? pts[hover] : pts[pts.length - 1];
  const colW = plotW / Math.max(1, x1 - x0);
  const kinds = [...new Set(pts.map((p) => p.kind))];

  return (
    <div className="science-curve" ref={root}>
      <div className="science-curve-head">
        <div className="science-curve-best">
          <span className="science-curve-caption">best {chart.metric}</span>
          <span className="science-curve-value">{fmt(chart.best)}</span>
        </div>
        <div className="science-curve-side">
          {won && (
            <span className={`science-curve-gain ${won.better ? "is-better" : "is-worse"}`}>
              {won.delta}
              {won.better && won.factor ? ` · ${won.factor}` : ""}
            </span>
          )}
          <span className="science-curve-count">
            {chart.tried} tried · {chart.kept} kept
          </span>
        </div>
      </div>

      <svg
        className="science-curve-plot"
        viewBox={`0 0 ${width} ${HEIGHT}`}
        role="img"
        aria-label={`${chart.metric} over ${chart.tried} runs, baseline ${fmt(chart.baseline)}, best ${fmt(chart.best)}`}
        onMouseLeave={() => setHover(null)}
      >
        <line className="science-curve-axis" x1={PAD.left} x2={width - PAD.right} y1={floor} y2={floor} />
        <text className="science-curve-tick" x={PAD.left - 5} y={PAD.top + 4} textAnchor="end">{fmt(hi)}</text>
        <text className="science-curve-tick" x={PAD.left - 5} y={floor} textAnchor="end">{fmt(lo)}</text>
        <text className="science-curve-tick" x={PAD.left} y={HEIGHT - 3}>
          {x0 === 0 ? "baseline" : `#${x0}`}
        </text>
        <text className="science-curve-tick" x={width - PAD.right} y={HEIGHT - 3} textAnchor="end">#{x1}</text>

        {area && <path className="science-curve-area" d={area} />}
        {chart.baseline != null && (
          <line className="science-curve-baseline" x1={PAD.left} x2={width - PAD.right}
            y1={y(chart.baseline)} y2={y(chart.baseline)} />
        )}
        {step && <path className="science-curve-best-line" d={step} />}

        {shown && <line className="science-curve-guide" x1={x(shown.n)} x2={x(shown.n)} y1={PAD.top} y2={floor} />}

        {pts.map((p, i) => {
          const cx = x(p.n);
          const active = shown === p;
          if (p.value == null) {
            const cy = floor - 4;
            return (
              <path key={p.run || i} className={`science-curve-point is-failed${active ? " is-active" : ""}`}
                d={`M${cx - 3},${cy - 3} L${cx + 3},${cy + 3} M${cx - 3},${cy + 3} L${cx + 3},${cy - 3}`} />
            );
          }
          const cls = p.kind === "kept" ? "is-kept" : p.kind === "elsewhere" ? "is-elsewhere" : "is-not-kept";
          return (
            <circle key={p.run || i} className={`science-curve-point ${cls}${active ? " is-active" : ""}`}
              cx={cx} cy={y(p.value)} r={p.kind === "kept" ? 4 : 3} />
          );
        })}

        {/* Hit targets: a whole column per run, wider than the mark. */}
        {pts.map((p, i) => (
          <rect key={`hit-${p.run || i}`} className="science-curve-hit"
            x={x(p.n) - colW / 2} y={PAD.top} width={Math.max(colW, 6)} height={plotH}
            onMouseEnter={() => setHover(i)} />
        ))}
      </svg>

      {shown && (
        <div className="science-curve-readout">
          <span className="science-curve-run">#{shown.n}{shown.axis ? ` · ${shown.axis}` : ""}</span>
          <span className={`science-curve-kind is-${shown.kind.replace(" ", "-")}`}>
            {fmt(shown.value)} · {KIND_LABEL[shown.kind] ?? shown.kind}
          </span>
        </div>
      )}

      <div className="science-curve-legend" aria-hidden="true">
        <span><i className="science-curve-key is-best" />best so far</span>
        {chart.baseline != null && <span><i className="science-curve-key is-baseline" />baseline</span>}
        {kinds.map((k) => (
          <span key={k}><i className={`science-curve-key is-${k.replace(" ", "-")}`} />{KIND_LABEL[k] ?? k}</span>
        ))}
      </div>
    </div>
  );
};
