// A sent message split into prose, the "@"-mentions it attached, by the backend's rule
// (`_MENTION_RE` in client/context/resource_context.py): "@" opening a token, and the
// skills it invokes, "/name" (client/query_engine/skill_invocation.py). Free of React,
// so unit-testable.

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
    }
  | {
      kind: "skill";
      /** The invocation as typed, "/" included. */
      text: string;
      /** The skill it names. */
      name: string;
    };

/** The skills a "/name" may invoke: name → one-line description. */
export type SkillNames = ReadonlyMap<string, string>;

/** A skill name as the backend reads it after "/". */
const SKILL_NAME_RE = /^[A-Za-z0-9][\w-]*$/;

/**
 * Whether a mention token names a file rather than an MCP resource, by shape alone: a
 * separator or an extension, and never a URI scheme (``memory://all``).
 */
export function looksLikeFile(target: string): boolean {
  if (!target || /^[a-z][a-z0-9+.-]*:\/\//i.test(target)) return false;
  return target.includes("/") || /\.[A-Za-z0-9_]{1,12}$/.test(target);
}

/**
 * Split `text` into prose, mention and skill segments, in order.
 *
 * A "/name" is a skill segment only when `skills` holds the name; anything else stays
 * prose. Adjacent prose is merged, so a message with neither comes back as a single
 * text segment.
 */
export function splitMentionSegments(
  text: string,
  skills: Pick<SkillNames, "has"> = new Map()
): MentionSegment[] {
  const out: MentionSegment[] = [];
  let prose = "";
  const pushProse = () => {
    if (prose) out.push({ kind: "text", text: prose });
    prose = "";
  };

  let i = 0;
  while (i < text.length) {
    const ch = text[i];
    // "@" and "/" only open a token at the start of the text or after whitespace.
    if ((ch !== "@" && ch !== "/") || (i > 0 && /\S/.test(text[i - 1]))) {
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
    if (ch === "/") {
      if (SKILL_NAME_RE.test(token) && skills.has(token)) {
        pushProse();
        out.push({ kind: "skill", text: `/${token}`, name: token });
        prose = trailing;
      } else {
        prose += text.slice(i, end);
      }
      i = end;
      continue;
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
