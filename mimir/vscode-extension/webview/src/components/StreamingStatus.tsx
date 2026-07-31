import React from "react";

/** Words cycled in the live "typing" indicator while the agent is working. */
const STREAMING_PHASES = [
  "Reasoning",
  "Analysing",
  "Working",
  "Thinking",
  "Synthesising",
  "Connecting the dots",
  "Almost there",
];

interface StreamingStatusProps {
  /** Show the three trailing bouncing dots. Off when a spinner is already shown. */
  showDots?: boolean;
}

/**
 * Animated status shown while a bubble is streaming or the model is thinking.
 * The current word's letters fall in from the top one-by-one (left → right),
 * hold, then keep falling out downward in the same order before the next word
 * enters. Optionally trailed by three gently bouncing dots.
 */
export const StreamingStatus: React.FC<StreamingStatusProps> = ({ showDots = true }) => {
  const [idx, setIdx] = React.useState(0);
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
      setIdx((i) => (i + 1) % STREAMING_PHASES.length);
      setPhase("in");
    }, sweepDuration);
    return () => window.clearTimeout(t);
  }, [phase, idx, sweepDuration]);

  return (
    <span className="streaming-status" aria-live="polite">
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
