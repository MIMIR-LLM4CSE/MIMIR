import React from "react";

/** Words cycled in the live "typing" indicator while the agent is working. */
export const STREAMING_PHASES = [
  "Mimiring",
  "Reasoning",
  "Analysing",
  "Working",
  "Thinking",
  "Pondering",
  "Synthesising",
  "Connecting the dots",
  "Drawing from the well",
  "Reading the runes",
  "Weaving threads",
  "Distilling",
  "Cross-checking",
  "Untangling",
  "Piecing it together",
  "Mulling it over",
  "Consulting the ravens",
  "Sharpening the thought",
];

/** A random index other than `current`, so the same word never shows twice in a row. */
function nextPhase(current: number): number {
  const n = STREAMING_PHASES.length;
  return (current + 1 + Math.floor(Math.random() * (n - 1))) % n;
}

/**
 * Elder Futhark runes, each drawn as exactly four strokes ([x1, y1, x2, y2] in a
 * 24×24 box) so that any rune can morph into the next one stroke by stroke. A
 * rune with fewer strokes repeats one of them.
 */
type Stroke = [number, number, number, number];
const RUNES: Stroke[][] = [
  // Ansuz ᚨ — Odin's rune, the word and wisdom
  [[8, 3, 8, 21], [8, 4, 16, 9], [8, 10, 16, 15], [8, 3, 8, 21]],
  // Mannaz ᛗ
  [[6, 3, 6, 21], [18, 3, 18, 21], [6, 3, 18, 13], [18, 3, 6, 13]],
  // Othala ᛟ
  [[12, 3, 17, 8], [12, 3, 7, 8], [17, 8, 6, 21], [7, 8, 18, 21]],
  // Ingwaz ᛜ
  [[12, 4, 18, 12], [18, 12, 12, 20], [12, 20, 6, 12], [6, 12, 12, 4]],
  // Dagaz ᛞ
  [[6, 4, 6, 20], [18, 4, 18, 20], [6, 4, 18, 20], [18, 4, 6, 20]],
  // Raidho ᚱ
  [[8, 3, 8, 21], [8, 3, 15, 8], [15, 8, 8, 12], [8, 12, 16, 21]],
  // Kenaz ᚲ
  [[15, 4, 8, 12], [8, 12, 15, 20], [15, 4, 8, 12], [8, 12, 15, 20]],
  // Sowilo ᛊ
  [[15, 3, 9, 10], [9, 10, 15, 14], [15, 14, 9, 21], [9, 10, 15, 14]],
  // Algiz ᛉ
  [[12, 3, 12, 21], [12, 11, 6, 4], [12, 11, 18, 4], [12, 3, 12, 21]],
  // Tiwaz ᛏ
  [[12, 3, 12, 21], [12, 3, 6, 9], [12, 3, 18, 9], [12, 3, 12, 21]],
  // Fehu ᚠ
  [[8, 3, 8, 21], [8, 9, 16, 4], [8, 14, 16, 9], [8, 3, 8, 21]],
];

const runePath = (strokes: Stroke[]): string =>
  strokes.map(([x1, y1, x2, y2]) => `M${x1} ${y1}L${x2} ${y2}`).join("");

const RUNE_HOLD = 0.9; // s a rune stays still
const RUNE_MORPH = 0.45; // s to reshape into the next one

/** SMIL `values`/`keyTimes`/`keySplines` for the hold → morph cycle over all runes. */
const RUNE_ANIM = (() => {
  const step = RUNE_HOLD + RUNE_MORPH;
  const total = RUNES.length * step;
  const values: string[] = [];
  const times: number[] = [];
  RUNES.forEach((r, i) => {
    values.push(runePath(r), runePath(r));
    times.push((i * step) / total, (i * step + RUNE_HOLD) / total);
  });
  values.push(runePath(RUNES[0]));
  times.push(1);
  const splines = times.slice(1).map((_, i) => (i % 2 ? "0.65 0 0.35 1" : "0 0 1 1"));
  return {
    dur: `${total}s`,
    values: values.join(";"),
    keyTimes: times.map((t) => t.toFixed(4)).join(";"),
    keySplines: splines.join(";"),
  };
})();

/**
 * Small "work in progress" glyph: a rune stroked with the logo's colours that
 * reshapes itself into the next rune, over and over, while the gradient turns.
 */
export const MimirSpinner: React.FC = () => {
  const id = `rune-grad-${React.useId().replace(/:/g, "")}`;
  // SMIL ignores CSS, so the reduced-motion preference is honoured here.
  const still = React.useMemo(
    () => window.matchMedia?.("(prefers-reduced-motion: reduce)").matches ?? false,
    [],
  );
  return (
    <svg className="mimir-spinner" viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <defs>
        <linearGradient id={id} gradientUnits="userSpaceOnUse" x1="4" y1="4" x2="20" y2="20">
          <stop offset="0%" stopColor="#3fe0f5" />
          <stop offset="35%" stopColor="#1f3fd6" />
          <stop offset="65%" stopColor="#c21a4a" />
          <stop offset="100%" stopColor="#f78a8a" />
          {!still && <animateTransform
            attributeName="gradientTransform"
            type="rotate"
            from="0 12 12"
            to="360 12 12"
            dur="6s"
            repeatCount="indefinite"
          />}
        </linearGradient>
      </defs>
      <path className="mimir-spinner__rune" d={runePath(RUNES[0])} stroke={`url(#${id})`}>
        {!still && <animate
          attributeName="d"
          dur={RUNE_ANIM.dur}
          values={RUNE_ANIM.values}
          keyTimes={RUNE_ANIM.keyTimes}
          keySplines={RUNE_ANIM.keySplines}
          calcMode="spline"
          repeatCount="indefinite"
        />}
      </path>
    </svg>
  );
};

interface StreamingStatusProps {
  /** Show the three trailing bouncing dots. Off when a spinner is already shown. */
  showDots?: boolean;
}

/**
 * Animated status shown while a bubble is streaming or the model is thinking.
 * The current word's letters fall in from the top one-by-one (left → right),
 * hold, then keep falling out downward in the same order before the next word
 * enters. Led by a morphing rune glyph, optionally trailed by three gently
 * bouncing dots.
 */
export const StreamingStatus: React.FC<StreamingStatusProps> = ({ showDots = true }) => {
  const [idx, setIdx] = React.useState(() => Math.floor(Math.random() * STREAMING_PHASES.length));
  const [phase, setPhase] = React.useState<"in" | "out">("in");

  const word = STREAMING_PHASES[idx];
  const LETTER_STAGGER = 45; // ms between successive letters
  const LETTER_ANIM = 320; // ms for a single letter's fall
  const HOLD = 2200; // ms the fully-shown word lingers AFTER it finished entering

  // Time for every letter (last one included) to finish its staggered fall.
  const sweepDuration = (word.length - 1) * LETTER_STAGGER + LETTER_ANIM;

  // Wait for the full entrance sweep, then hold, then trigger the exit. Tying
  // the delay to the word length prevents the exit from starting while long
  // words are still falling in.
  React.useEffect(() => {
    if (phase !== "in") return;
    const t = window.setTimeout(() => setPhase("out"), sweepDuration + HOLD);
    return () => window.clearTimeout(t);
  }, [phase, idx, sweepDuration]);

  // Once the exit sweep (last letter included) finishes, advance the word.
  React.useEffect(() => {
    if (phase !== "out") return;
    const t = window.setTimeout(() => {
      setIdx(nextPhase);
      setPhase("in");
    }, sweepDuration);
    return () => window.clearTimeout(t);
  }, [phase, idx, sweepDuration]);

  return (
    <span className="streaming-status" aria-live="polite">
      <MimirSpinner />
      <span className="streaming-status__word" aria-label={word}>
        {word.split("").map((ch, i) => (
          <span
            key={`${idx}-${i}`}
            className={`streaming-status__letter streaming-status__letter--${phase}`}
            style={{ animationDelay: `${i * LETTER_STAGGER}ms` }}
            aria-hidden="true"
          >
            {ch === " " ? "\u00A0" : ch}
          </span>
        ))}
      </span>
      {showDots && (
        <span className="streaming-dots" aria-hidden="true">
          <span /><span /><span />
        </span>
      )}
    </span>
  );
};
