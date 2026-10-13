/** What the composer shows for a draft's attachments: a chip per mention, and the
 *  tints drawn behind the textarea's own text. */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { DraftAttachments } from "./DraftAttachments";
import { DraftHighlight } from "./DraftHighlight";

const chips = (text: string) =>
  renderToStaticMarkup(<DraftAttachments text={text} onRemove={() => {}} />);

const mirror = (text: string) =>
  renderToStaticMarkup(
    <DraftHighlight text={text} inputRef={{ current: null }} />
  );

describe("DraftAttachments", () => {
  it("shows nothing for a draft with no mention", () => {
    expect(chips("fix the solver")).toBe("");
  });

  it("names the file and its range, and offers to detach it", () => {
    const out = chips("look at @mimir/client/loop.py:10-20");
    expect(out).toContain("loop.py:10-20");
    expect(out).toContain("Open mimir/client/loop.py:10");
    expect(out).toContain("Detach @mimir/client/loop.py:10-20");
  });

  it("lists a resource mention but cannot open it", () => {
    const out = chips("@memory://all");
    expect(out).toContain("memory://all");
    expect(out).toContain("disabled");
  });

  it("shows one chip per mention", () => {
    expect(chips("@a.py @b.py").match(/draft-chip-open/g)).toHaveLength(2);
  });
});

describe("DraftHighlight", () => {
  it("draws nothing when there is no mention to tint", () => {
    expect(mirror("fix the solver")).toBe("");
  });

  it("tints the mention and leaves the prose around it", () => {
    const out = mirror("look at @src/foo.py now");
    expect(out).toContain('class="chat-input-mark"');
    expect(out).toContain("@src/foo.py");
    expect(out).toContain("look at ");
  });

  it("marks a resource mention a shade lighter", () => {
    expect(mirror("@memory://all")).toContain("chat-input-mark--plain");
  });
});
