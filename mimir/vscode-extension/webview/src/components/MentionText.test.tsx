/** What an attached mention looks like in a sent message: a chip, and for a file a
 *  button that can be clicked to open it. */
import { describe, expect, it } from "vitest";
import { renderToStaticMarkup } from "react-dom/server";
import React from "react";
import { MentionText } from "./MentionText";

const html = (text: string) => renderToStaticMarkup(<MentionText text={text} />);

describe("MentionText", () => {
  it("leaves a message with no mention as plain text", () => {
    expect(html("just a message")).toBe("just a message");
  });

  it("renders a file mention as a button carrying the whole token", () => {
    const out = html("check @src/foo.py:10-20 now");
    expect(out).toContain('class="ctx-mention"');
    expect(out).toContain("<button");
    expect(out).toContain("@src/foo.py:10-20");
    expect(out).toContain("Open src/foo.py:10");
  });

  it("renders an invoked skill as its own chip, described on hover", () => {
    const out = renderToStaticMarkup(
      <MentionText text="do it /fix-bug" skills={new Map([["fix-bug", "repair a defect"]])} />
    );
    expect(out).toContain('class="ctx-skill"');
    expect(out).toContain("/fix-bug");
    expect(out).toContain("repair a defect");
    expect(out).not.toContain("ctx-mention");
  });

  it("renders a resource mention as a marked span, not a link", () => {
    const out = html("@memory://all");
    expect(out).toContain("ctx-mention--plain");
    expect(out).not.toContain("<button");
  });
});
