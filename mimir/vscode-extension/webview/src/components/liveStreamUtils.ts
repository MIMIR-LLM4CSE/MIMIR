import type { ToolActivity } from "../types";

/**
 * One element of the turn in flight, placed where it arrived.
 *
 * The three live streams — prose, reasoning blocks, tool rows — are collected
 * separately, so nothing in them records that a tool ran *after* the paragraph
 * above it. Each carries an arrival stamp (`seq`) instead, and this module reads
 * those stamps back into a single ordered list.
 */
export type LiveEntry<B, M = never> =
  | { kind: "thinking"; seq: number; block: B }
  | { kind: "draft"; seq: number }
  | { kind: "tools"; seq: number; tools: ToolActivity[] }
  /** A transcript message that landed mid-step: a steer bubble, an edit card, a
   *  session-command reply. It belongs where it arrived, not above the whole step. */
  | { kind: "message"; seq: number; message: M };

/**
 * Order the live streams of one step by arrival.
 *
 * Tool rows keep their **array order** rather than being sorted on `seq`: a
 * sub-agent's row is spliced in under the call that spawned it (insertAfterFamily),
 * so the array — not the stamp — is what says where a row belongs inside its
 * family. Stamps only decide where the *group* sits relative to the prose and the
 * reasoning blocks around it, and a family shares its parent's stamp so it is
 * never split across two groups.
 *
 * Consecutive tool rows stay one group, so a step that called four tools in a row
 * still shows one card rather than four.
 *
 * Both the live view and the freeze go through here, which is what keeps them
 * agreeing: a card cannot sit below the prose while it streams and above it once
 * the step ends.
 */
export function orderLiveStream<
  B extends { seq: number },
  M extends { seq?: number } = never,
>(input: {
  thinking: B[];
  tools: ToolActivity[];
  /** Stamp of the draft's first token, or null when there is no prose to place. */
  draftSeq: number | null;
  /** Messages that landed during this step, in the order the transcript holds them. */
  messages?: M[];
}): LiveEntry<B, M>[] {
  const { thinking, tools, draftSeq, messages = [] } = input;

  // Everything that is not a tool row, in arrival order. These are the dividers:
  // a tool group belongs between the two of them its stamp falls between.
  const marks: LiveEntry<B, M>[] = thinking.map((block) => ({
    kind: "thinking" as const, seq: block.seq, block,
  }));
  for (const message of messages) {
    marks.push({ kind: "message", seq: message.seq ?? 0, message });
  }
  if (draftSeq !== null) marks.push({ kind: "draft", seq: draftSeq });
  marks.sort((a, b) => a.seq - b.seq);

  // Bucket each tool row by how many dividers precede it. Rows that land in the
  // same bucket ran with nothing between them, so they render as one card.
  const buckets: ToolActivity[][] = Array.from({ length: marks.length + 1 }, () => []);
  for (const tool of tools) {
    const seq = tool.seq ?? 0;
    let bucket = 0;
    while (bucket < marks.length && marks[bucket].seq < seq) bucket++;
    buckets[bucket].push(tool);
  }

  const out: LiveEntry<B, M>[] = [];
  for (let i = 0; i <= marks.length; i++) {
    if (buckets[i].length > 0) {
      out.push({ kind: "tools", seq: buckets[i][0].seq ?? 0, tools: buckets[i] });
    }
    if (i < marks.length) out.push(marks[i]);
  }
  return out;
}
