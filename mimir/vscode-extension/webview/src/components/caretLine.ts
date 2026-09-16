// Where the caret sits among the lines a textarea displays. A long paragraph
// wraps onto several lines without a single "\n", so the text alone cannot say
// whether the caret is on the first or last line: the layout is measured.

const MIRRORED = [
  "boxSizing", "width", "paddingTop", "paddingRight", "paddingBottom", "paddingLeft",
  "borderTopWidth", "borderRightWidth", "borderBottomWidth", "borderLeftWidth",
  "fontFamily", "fontSize", "fontWeight", "fontStyle", "fontVariant", "fontStretch",
  "letterSpacing", "wordSpacing", "lineHeight", "textIndent", "textTransform", "tabSize",
] as const;

/** Vertical offsets of the characters at `positions`, laid out as `el` lays them out. */
function lineTops(el: HTMLTextAreaElement, positions: number[]): number[] {
  const style = window.getComputedStyle(el);
  const mirror = document.createElement("div");
  for (const prop of MIRRORED) mirror.style[prop] = style[prop];
  Object.assign(mirror.style, {
    position: "absolute",
    visibility: "hidden",
    top: "0",
    left: "-9999px",
    whiteSpace: "pre-wrap",
    overflowWrap: "break-word",
    borderStyle: "solid",
    overflow: "hidden",
  });
  // A scrollbar narrows the lines; clientWidth excludes it, so the mirror wraps the same.
  const padding = parseFloat(style.paddingLeft) + parseFloat(style.paddingRight);
  Object.assign(mirror.style, {
    boxSizing: "content-box",
    borderLeftWidth: "0",
    borderRightWidth: "0",
    width: `${el.clientWidth - padding}px`,
  });
  document.body.appendChild(mirror);
  try {
    return positions.map((pos) => {
      mirror.textContent = el.value.slice(0, pos);
      const marker = document.createElement("span");
      // The marker holds the character at `pos`, so it lands where that character wraps.
      marker.textContent = el.value.slice(pos, pos + 1).replace("\n", "") || "​";
      mirror.appendChild(marker);
      return marker.offsetTop;
    });
  } finally {
    mirror.remove();
  }
}

/** True when nothing is selected and the caret is on the first displayed line. */
export function caretOnFirstVisualLine(el: HTMLTextAreaElement): boolean {
  if (el.selectionStart !== el.selectionEnd) return false;
  if (el.selectionStart === 0) return true;
  const [start, caret] = lineTops(el, [0, el.selectionStart]);
  return caret <= start;
}

/** True when nothing is selected and the caret is on the last displayed line. */
export function caretOnLastVisualLine(el: HTMLTextAreaElement): boolean {
  if (el.selectionStart !== el.selectionEnd) return false;
  const end = el.value.length;
  if (el.selectionStart === end) return true;
  const [caret, last] = lineTops(el, [el.selectionStart, end]);
  return caret >= last;
}
