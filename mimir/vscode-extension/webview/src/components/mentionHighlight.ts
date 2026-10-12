// Splitting a sent message into prose and the "@"-mentions it attached, so the chat
// bubble can mark the attached context instead of burying it in a wall of text.
//
// The scan mirrors the backend's rule (`_MENTION_RE` in client/context/
// resource_context.py): "@" then a run of non-whitespace, and only where the "@" opens
// a token — at the start of the message or after whitespace — so "mail@host" and a
// mid-word "@" are prose, exactly as the server reads them.
//
// Kept free of React so it is unit-testable (see mentionHighlight.test.ts).

/** Trailing characters that belong to the sentence, not to the mention. */
const TRAILING = ".,;:!?)]}\"'";

/** ``path:start`` or ``path:start-end`` — the backend's line-range suffix. */
const RANGE_RE = /^(.+?):(\d+)(?:-(\d+))?$/;

export type MentionSegment =
  | { kind: "text"; text: string }
  | {
      kind: "mention";
      /** The mention as typed, "@" included — what the bubble shows. */
      text: string;
      /** The path or resource token, range suffix stripped. */
      target: string;
      /** First line of the attached range, when the mention named one. */
      line?: number;
      endLine?: number;
      /** Whether `target` reads as a workspace path, i.e. something to open. */
      isFile: boolean;
      /** Where `text` sits in the message — what the composer removes to detach it. */
      start: number;
      end: number;
    };

/**
 * Whether a mention token names a file rather than an MCP resource.
 *
 * Decided on shape alone: the webview has no filesystem and no registry to ask, and a
 * wrong guess costs little in either direction — an unopenable path says so when
 * clicked, and a resource mention is still highlighted, just not a link. A path has a
 * separator or an extension; a URI scheme (``memory://all``) never counts, as the
 * ``//`` would otherwise make every resource look like a path.
 */
export function looksLikeFile(target: string): boolean {
  if (!target || /^[a-z][a-z0-9+.-]*:\/\//i.test(target)) return false;
  return target.includes("/") || /\.[A-Za-z0-9_]{1,12}$/.test(target);
}

/**
 * Split `text` into prose and mention segments, in order.
 *
 * Adjacent prose is merged, so a message with no mention comes back as a single text
 * segment and renders exactly as it did before.
 */
export function splitMentionSegments(text: string): MentionSegment[] {
  const out: MentionSegment[] = [];
  let prose = "";
  const pushProse = () => {
    if (prose) out.push({ kind: "text", text: prose });
    prose = "";
  };

  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    // An "@" only opens a mention at the start of the text or after whitespace.
    if (ch !== "@" || (i > 0 && /\S/.test(text[i - 1]))) {
      prose += ch;
      i += 1;
      continue;
    }
    let end = i + 1;
    while (end < text.length && /\S/.test(text[end])) end += 1;
    let token = text.slice(i + 1, end);
    // Punctuation closing the sentence goes back to the prose.
    let trailing = "";
    while (token && TRAILING.includes(token[token.length - 1])) {
      trailing = token[token.length - 1] + trailing;
      token = token.slice(0, -1);
    }
    if (!token) {
      prose += text.slice(i, end);
      i = end;
      continue;
    }
    pushProse();
    const range = RANGE_RE.exec(token);
    const target = range ? range[1] : token;
    out.push({
      kind: "mention",
      text: `@${token}`,
      target,
      line: range ? Number(range[2]) : undefined,
      endLine: range && range[3] ? Number(range[3]) : range ? Number(range[2]) : undefined,
      isFile: looksLikeFile(target),
      start: i,
      end: i + 1 + token.length,
    });
    prose = trailing;
    i = end;
  }
  pushProse();
  return out;
}

/**
 * Take the mention spanning [start, end) out of `text`, with the caret to put back.
 *
 * Detaching is editing the message, so it goes through the text the user typed rather
 * than through a list kept beside it: there is one source for what a turn attaches, and
 * it is the box. One of the spaces that flanked the mention goes with it, so removing
 * one from the middle of a sentence does not leave a double space behind.
 */
export function removeMention(
  text: string,
  start: number,
  end: number
): { text: string; caret: number } {
  let from = start;
  let to = end;
  if (text[to] === " " || text[to] === "\t") to += 1;
  else if (from > 0 && (text[from - 1] === " " || text[from - 1] === "\t")) from -= 1;
  return { text: text.slice(0, from) + text.slice(to), caret: from };
}
