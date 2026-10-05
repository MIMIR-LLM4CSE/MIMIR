import React, { useEffect, useState } from "react";
import type { PromptOrigin, QuestionAnswer, QuestionSpec } from "../types";

interface Props {
  questions: QuestionSpec[];
  onSubmit: (answers: QuestionAnswer[]) => void;
  /** Who raised this card, when it was not the turn on screen (a sub-agent). */
  origin?: PromptOrigin;
  /** Epoch ms the agent stops waiting at, when the question carries a wall. */
  expiresAt?: number;
}

/** A remaining-seconds count as `m:ss`. */
const countdown = (secs: number): string =>
  `${Math.floor(secs / 60)}:${String(secs % 60).padStart(2, "0")}`;

/**
 * Modal asking the user one or more clarification questions, Claude-style.
 *
 * Questions are shown one at a time: submitting an answer advances to the next
 * one, and the last submit returns all answers together. Mirrors
 * the resume card (same `resume-*` overlay/modal classes). In single-select
 * mode a click on an option submits immediately; in multi-select mode options
 * toggle and a Submit button confirms. A free-text "Other" field is always
 * available. A progress indicator ("2 / 3") shows when there are several questions.
 *
 * A question raised by `ask_user_question` carries a deadline: the card shows how
 * long is left, and the agent closes it (a `prompt_expired` event) when nothing has
 * been answered by then, carrying on with the option it listed first. The countdown
 * is shown so that closing reads as the stated behaviour rather than a card
 * vanishing. The server owns the wall — this only displays it.
 */
export const UserQuestion: React.FC<Props> = ({
  questions, onSubmit, origin, expiresAt,
}) => {
  const [index, setIndex] = useState(0);
  const [answers, setAnswers] = useState<QuestionAnswer[]>([]);
  const [selected, setSelected] = useState<string[]>([]);
  const [otherText, setOtherText] = useState("");
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!expiresAt) return;
    const tick = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(tick);
  }, [expiresAt]);
  const secsLeft = expiresAt ? Math.max(0, Math.round((expiresAt - now) / 1000)) : null;

  const current = questions[index];
  if (!current) return null;
  const { question, header, options, multiSelect } = current;
  // The plan-approval card keeps plan mode's red even after the decision flips
  // the agent back to another mode. Header text mirrors plan_loop._request_plan_decision.
  const isPlanApproval = (header ?? "").trim().toLowerCase() === "plan approval";
  const total = questions.length;
  const isLast = index === total - 1;

  const advance = (labels: string[], other: string) => {
    const trimmed = other.trim();
    const all = trimmed ? [...labels, trimmed] : labels;
    if (all.length === 0) return;

    const answer: QuestionAnswer = {
      selected: labels,
      otherText: trimmed || undefined,
    };
    const nextAnswers = [...answers, answer];

    if (isLast) {
      onSubmit(nextAnswers);
      return;
    }
    // Advance to the next question, resetting per-question input state.
    setAnswers(nextAnswers);
    setIndex(index + 1);
    setSelected([]);
    setOtherText("");
  };

  const toggle = (label: string) => {
    if (multiSelect) {
      setSelected((prev) =>
        prev.includes(label) ? prev.filter((l) => l !== label) : [...prev, label],
      );
    } else {
      // Single-select: a click is a final answer (include any typed "Other").
      advance([label], otherText);
    }
  };

  return (
    <div className="resume-overlay resume-overlay--inline">
      <div className={`resume-modal uq-modal${isPlanApproval ? " uq-modal--plan" : ""}`}>
        {total > 1 && (
          <div className="uq-progress">
            {index + 1} / {total}
          </div>
        )}
        {/* Whose question this is, when it is not the turn on screen. Above the
            header, because "which piece of work is this?" has to be read before the
            question can be weighed at all. */}
        {origin && (
          <div className="uq-origin" title={origin.session || undefined}>
            ⑂ {origin.label}
          </div>
        )}
        {header && <div className="uq-header">{header}</div>}
        {secsLeft !== null && (
          <div className="uq-countdown">
            {countdown(secsLeft)} left — unanswered, MIMIR goes with the first option
          </div>
        )}
        <div className="resume-title">{question}</div>

        <div className="uq-options">
          {options.map((opt) => {
            const active = selected.includes(opt.label);
            return (
              <button
                key={opt.label}
                className={`uq-option${active ? " uq-option--active" : ""}`}
                onClick={() => toggle(opt.label)}
              >
                <span className="uq-option-label">
                  {multiSelect && (
                    <span className="uq-check">{active ? "☑" : "☐"}</span>
                  )}
                  {opt.label}
                </span>
                {opt.description && (
                  <span className="uq-option-desc">{opt.description}</span>
                )}
              </button>
            );
          })}
        </div>

        <input
          className="uq-other"
          type="text"
          placeholder="Other… (type your own answer)"
          value={otherText}
          onChange={(e) => setOtherText(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter") advance(selected, otherText);
          }}
        />

        {multiSelect && (
          <div className="resume-actions">
            <button
              className="resume-btn resume-btn--yes"
              onClick={() => advance(selected, otherText)}
            >
              {isLast ? "Submit" : "Next"}
            </button>
          </div>
        )}
      </div>
    </div>
  );
};
