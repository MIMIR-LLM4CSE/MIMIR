// SVG elements, and text fitted to a width measured on a canvas in the boxes' font.

const NS = "http://www.w3.org/2000/svg";

export function svg<K extends keyof SVGElementTagNameMap>(tag: K, attrs: Record<string, string | number>, parent?: Element): SVGElementTagNameMap[K] {
  const e = document.createElementNS(NS, tag);
  for (const k in attrs) e.setAttribute(k, String(attrs[k]));
  parent?.appendChild(e);
  return e;
}

/** The font an SVG text of class *cls* in a module's box is drawn in, read once. */
const fonts = new Map<string, string>();
function fontOf(cls: string): string {
  let f = fonts.get(cls);
  if (f === undefined) {
    const probe = svg("svg", { width: 0, height: 0 }, document.body);
    const cs = getComputedStyle(svg("text", { class: cls }, svg("g", { class: "m" }, probe)));
    f = `${cs.fontStyle} ${cs.fontWeight} ${cs.fontSize} ${cs.fontFamily}`;
    probe.remove();
    fonts.set(cls, f);
  }
  return f;
}
/** *text* shortened by *cut* until it holds in *max* px, measured on a canvas: an SVG text
 *  measured as it is drawn lays the page out at every try. */
const fitted = new Map<string, string>();
let pen: CanvasRenderingContext2D | null = null;
export function fitText(text: string, max: number, cls: string, cut: (s: string) => string): string {
  const k = cls + "|" + max + "|" + text;
  let out = fitted.get(k);
  if (out !== undefined) return out;
  pen ??= document.createElement("canvas").getContext("2d");
  out = text;
  if (pen) {
    pen.font = fontOf(cls);
    while (pen.measureText(out).width > max && out.length > 4) out = cut(out);
  }
  if (fitted.size > 5000) fitted.clear();
  fitted.set(k, out);
  return out;
}
