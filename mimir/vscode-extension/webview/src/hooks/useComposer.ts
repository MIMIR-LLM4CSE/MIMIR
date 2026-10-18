// The message box: what is typed, the @-mention and /-command dropdowns over it, the
// sent-message history behind the arrow keys, and what Enter sends.

import React, { useCallback, useMemo, useRef, useState } from "react";
import type { ResourceItem, ToggleItem } from "../types";
import type { ServerHandlers } from "../state/serverMessages";
import {
  detectMentionQuery,
  buildMentionItems,
  applyMention,
  wrapIndex,
  type MentionQuery,
  type MentionItem,
} from "../components/mentionUtils";
import {
  detectSlashQuery,
  filterSlashItems,
  isSessionCommand,
  applySlash,
  type SlashQuery,
} from "../components/slashUtils";
import { removeMention } from "../components/mentionHighlight";
import {
  HISTORY_IDLE,
  pushHistory,
  stepHistory,
  type HistoryState,
} from "../components/historyUtils";
import { caretOnFirstVisualLine, caretOnLastVisualLine } from "../components/caretLine";
import { vscodePostMessage } from "./useWebSocket";

interface Senders {
  /** The skills offered by the "/" dropdown. */
  skillToggles: ToggleItem[];
  /** A session command ("/memory clear"): it acts on the session, never reaches the model. */
  sendCommand: (text: string) => void;
  /** A message for the model, or a steer while a turn runs. */
  sendPrompt: (text: string) => void;
}

export function useComposer({ skillToggles, sendCommand, sendPrompt }: Senders) {
  const [input, setInput] = useState("");
  // ── @-mention autocomplete (attach MCP resources and workspace files) ─────
  const [resources, setResources] = useState<ResourceItem[]>([]);
  // Workspace files matching the mention being typed, ranked by the extension host —
  // the webview has no filesystem of its own to search. `query` is the ask they answer,
  // so a reply overtaken by further typing can be ignored rather than shown.
  const [fileMatches, setFileMatches] = useState<{ query: string; files: string[] }>(
    { query: "", files: [] }
  );
  const [mention, setMention] = useState<MentionQuery | null>(null);
  const [mentionIndex, setMentionIndex] = useState(0);
  const mentionItems = useMemo(
    () =>
      mention
        ? buildMentionItems(
            resources,
            // Only the answer to this very query: showing the previous one's files
            // under a narrower query puts rows in the list that do not match it.
            fileMatches.query === mention.query ? fileMatches.files : [],
            mention.query
          )
        : [],
    [mention, resources, fileMatches]
  );
  const mentionOpen = mention !== null && mentionItems.length > 0;
  // ── "/"-command autocomplete (run a skill) ─────────────────────────────────
  const [slash, setSlash] = useState<SlashQuery | null>(null);
  const [slashIndex, setSlashIndex] = useState(0);
  // The active VS Code editor file/selection, pushed by the extension host — shown as
  // an opt-in click-to-attach chip above the input.
  const [activeEditor, setActiveEditor] = useState<{ file: string; selection: string | null } | null>(null);
  const slashItems = useMemo(
    () => (slash ? filterSlashItems(skillToggles, slash.query) : []),
    [slash, skillToggles]
  );
  const slashOpen = slash !== null && slashItems.length > 0;
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  /** The mention query the host was last asked about, so one ask per keystroke. */
  const mentionFileQueryRef = useRef<string | null>(null);
  // Messages sent from this input, oldest first, recalled with the arrow keys.
  const sentHistoryRef = useRef<string[]>([]);
  const historyStateRef = useRef<HistoryState>(HISTORY_IDLE);

  const submitQuery = useCallback(() => {
    const text = input.trim();
    if (!text) return;
    sentHistoryRef.current = pushHistory(sentHistoryRef.current, text);
    historyStateRef.current = HISTORY_IDLE;
    setInput("");
    setMention(null);
    setSlash(null);
    // A session command acts on the session, not on the conversation: it goes out as
    // `type: "command"` for ws_session._handle_command. Sent as a query it would simply
    // reach the model as text — which is what used to happen, so "/memory clear" asked
    // the model to clear memory instead of clearing it, and typed session commands were
    // indistinguishable from ordinary prose.
    if (isSessionCommand(text)) {
      sendCommand(text);
      return;
    }
    sendPrompt(text);
  }, [input, sendCommand, sendPrompt]);

  // Recompute the @-mention query from the textarea's current value + caret.
  const syncMention = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return;
    const caret = el.selectionStart ?? el.value.length;
    const next = detectMentionQuery(el.value, caret);
    setMention(next);
    setMentionIndex(0);
    // Each keystroke inside a mention asks the host to rank the workspace against it.
    // Cheap on its side (one cached scan, then a sort), and asking per keystroke is
    // what keeps the list honest: it is the only place the file names exist.
    if (next && next.query !== mentionFileQueryRef.current) {
      mentionFileQueryRef.current = next.query;
      vscodePostMessage({ type: "file_search", query: next.query });
    } else if (!next) {
      mentionFileQueryRef.current = null;
    }
  }, []);

  // Recompute the "/"-command query from the textarea's current value + caret.
  const syncSlash = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return;
    const caret = el.selectionStart ?? el.value.length;
    setSlash(detectSlashQuery(el.value, caret));
    setSlashIndex(0);
  }, []);

  // Insert the chosen resource's token at the caret and refocus the input.
  const pickMention = useCallback(
    (item: MentionItem) => {
      const el = textareaRef.current;
      if (!el || !mention) return;
      const caret = el.selectionStart ?? el.value.length;
      const { text, caret: newCaret } = applyMention(
        el.value,
        mention.start,
        caret,
        item.insert
      );
      setInput(text);
      setMention(null);
      // The next mention asks again, even for the same letters: the workspace may
      // have gained or lost a file since.
      mentionFileQueryRef.current = null;
      requestAnimationFrame(() => {
        const t = textareaRef.current;
        if (t) {
          t.focus();
          t.setSelectionRange(newCaret, newCaret);
        }
      });
    },
    [mention]
  );

  // Insert the chosen skill as a "/name " slash command and refocus the input.
  const pickSlash = useCallback(
    (item: ToggleItem) => {
      const el = textareaRef.current;
      if (!el || !slash) return;
      const caret = el.selectionStart ?? el.value.length;
      const { text, caret: newCaret } = applySlash(
        el.value,
        slash.start,
        caret,
        item.name
      );
      setInput(text);
      setSlash(null);
      requestAnimationFrame(() => {
        const t = textareaRef.current;
        if (t) {
          t.focus();
          t.setSelectionRange(newCaret, newCaret);
        }
      });
    },
    [slash]
  );

  // Click-to-attach the active editor file (+ selection) — inserts an @path[:a-b]
  // token at the caret, reusing the same mention mechanism the backend resolves.
  const attachActiveEditor = useCallback(() => {
    if (!activeEditor) return;
    const token = activeEditor.selection
      ? `${activeEditor.file}:${activeEditor.selection}`
      : activeEditor.file;
    const el = textareaRef.current;
    const caret = el?.selectionStart ?? input.length;
    const { text, caret: newCaret } = applyMention(input, caret, caret, token);
    setInput(text);
    requestAnimationFrame(() => {
      const t = textareaRef.current;
      if (t) {
        t.focus();
        t.setSelectionRange(newCaret, newCaret);
      }
    });
  }, [activeEditor, input]);

  // Take an attachment back out of the draft, from its chip's "×". The draft is the
  // record of what the turn attaches, so detaching is an edit of the text.
  const detachMention = useCallback(
    (start: number, end: number) => {
      const { text, caret } = removeMention(input, start, end);
      setInput(text);
      requestAnimationFrame(() => {
        const t = textareaRef.current;
        if (t) {
          t.focus();
          t.setSelectionRange(caret, caret);
        }
      });
    },
    [input]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
      // While the slash-command dropdown is open, arrows/Enter/Tab/Esc drive it.
      if (slashOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setSlashIndex((i) => wrapIndex(i + 1, slashItems.length));
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setSlashIndex((i) => wrapIndex(i - 1, slashItems.length));
          return;
        }
        if (e.key === "Enter" || e.key === "Tab") {
          e.preventDefault();
          const item = slashItems[wrapIndex(slashIndex, slashItems.length)];
          if (item) pickSlash(item);
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          setSlash(null);
          return;
        }
      }
      // While the mention dropdown is open, arrows/Enter/Tab/Esc drive it.
      if (mentionOpen) {
        if (e.key === "ArrowDown") {
          e.preventDefault();
          setMentionIndex((i) => wrapIndex(i + 1, mentionItems.length));
          return;
        }
        if (e.key === "ArrowUp") {
          e.preventDefault();
          setMentionIndex((i) => wrapIndex(i - 1, mentionItems.length));
          return;
        }
        if (e.key === "Enter" || e.key === "Tab") {
          e.preventDefault();
          const item = mentionItems[wrapIndex(mentionIndex, mentionItems.length)];
          if (item) pickMention(item);
          return;
        }
        if (e.key === "Escape") {
          e.preventDefault();
          setMention(null);
          return;
        }
      }
      // Up on the first line recalls the previous sent message; Down on the last
      // line walks forward, and past the newest one gives back the draft.
      if ((e.key === "ArrowUp" || e.key === "ArrowDown") && !e.shiftKey && !e.altKey && !e.ctrlKey && !e.metaKey) {
        const el = e.currentTarget;
        const up = e.key === "ArrowUp";
        const atEdge = up
          ? caretOnFirstVisualLine(el)
          : caretOnLastVisualLine(el);
        const step = atEdge
          ? stepHistory(sentHistoryRef.current, historyStateRef.current, el.value, up ? "up" : "down")
          : null;
        if (step) {
          e.preventDefault();
          historyStateRef.current = step.state;
          setInput(step.text);
          // The caret goes to the end, unless that is below the first displayed line
          // while walking up: then it goes to the start, so the next Up keeps going.
          requestAnimationFrame(() => {
            const t = textareaRef.current;
            if (!t) return;
            t.setSelectionRange(t.value.length, t.value.length);
            if (up && !caretOnFirstVisualLine(t)) t.setSelectionRange(0, 0);
          });
          return;
        }
      }
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        submitQuery();
      }
    },
    [
      slashOpen,
      slashItems,
      slashIndex,
      pickSlash,
      mentionOpen,
      mentionItems,
      mentionIndex,
      pickMention,
      submitQuery,
    ]
  );


  /** The textarea's own events: typing ends a history browse and re-reads the dropdowns. */
  const onChange = useCallback((e: React.ChangeEvent<HTMLTextAreaElement>) => {
    historyStateRef.current = HISTORY_IDLE;
    setInput(e.target.value);
    syncMention(e.target);
    syncSlash(e.target);
  }, [syncMention, syncSlash]);
  const onSelect = useCallback((e: React.SyntheticEvent<HTMLTextAreaElement>) => {
    syncMention(e.currentTarget);
    syncSlash(e.currentTarget);
  }, [syncMention, syncSlash]);
  const onBlur = useCallback(() => {
    setMention(null);
    mentionFileQueryRef.current = null;
    setSlash(null);
  }, []);

  const on: ServerHandlers = useMemo(() => ({
    resources: (msg) => setResources(msg.resources ?? []),
    file_results: (msg) => setFileMatches({ query: msg.query ?? "", files: msg.files ?? [] }),
    // The active editor, offered as a click-to-attach chip above the input.
    active_editor: (msg) => setActiveEditor({ file: msg.file, selection: msg.selection }),
  }), []);

  return {
    input, activeEditor, textareaRef,
    mentionOpen, mentionItems, mentionIndex, setMentionIndex, pickMention,
    slashOpen, slashItems, slashIndex, setSlashIndex, pickSlash,
    attachActiveEditor, detachMention, submitQuery, handleKeyDown,
    onChange, onSelect, onBlur,
    on,
  };
}
