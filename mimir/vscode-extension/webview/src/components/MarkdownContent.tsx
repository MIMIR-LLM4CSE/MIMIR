import React, { useState, useRef, useCallback } from "react";
import ReactMarkdown from "react-markdown";
import type { Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import rehypeHighlight from "rehype-highlight";
import { normalizeMathDelimiters } from "./mathDelimiters";

interface Props {
  text: string;
}

// ── Code block with copy button ──────────────────────────────────────────────
// Reproduces the DOM the previous hand-rolled renderer emitted (.code-block-wrap
// / .code-block-header / .code-copy-btn / .code-block) so existing CSS keeps
// working. The inner <code> element (carrying the `hljs language-*` classes that
// rehype-highlight added) is passed through untouched so highlight.js theme CSS
// applies.

const CodeBlock: React.FC<{ lang: string; children: React.ReactNode }> = ({
  lang,
  children,
}) => {
  const [copied, setCopied] = useState(false);
  const preRef = useRef<HTMLPreElement>(null);

  const onCopy = useCallback(() => {
    const code = preRef.current?.textContent ?? "";
    navigator.clipboard
      .writeText(code)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1500);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="code-block-wrap">
      <div className="code-block-header">
        <span className="code-block-lang">{lang || "text"}</span>
        <button
          className="code-copy-btn"
          aria-label="Copy code"
          onClick={onCopy}
          style={copied ? { color: "var(--accent)" } : undefined}
        >
          {copied ? "Copied!" : "Copy"}
        </button>
      </div>
      <pre className="code-block" ref={preRef}>
        {children}
      </pre>
    </div>
  );
};

const components: Components = {
  // Fenced code blocks: react-markdown wraps the <code> in a <pre>. We replace
  // that <pre> with our copy-button chrome, keeping the inner <code> element
  // (and its hljs classes) intact.
  pre({ children }) {
    const child = React.Children.only(children) as React.ReactElement<{
      className?: string;
    }>;
    const childClass = child.props?.className ?? "";
    const m = /language-(\w+)/.exec(childClass);
    const lang = m?.[1] ?? "";
    // code() may have tagged a no-language block as inline — undo that here,
    // where we know for certain it is block-level.
    const fixed = React.cloneElement(child, {
      className: childClass.replace(/\binline-code\b/, "").trim() || undefined,
    });
    return <CodeBlock lang={lang}>{fixed}</CodeBlock>;
  },
  // Both inline and block code reach this. Block code carries a `language-*` or
  // `hljs` class (added by rehype-highlight); everything else is inline.
  // `node` is the hast node react-markdown passes — drop it so it isn't spread
  // onto the DOM element.
  code({ node: _node, className, children, ...props }) {
    const isBlock = /\blanguage-|\bhljs\b/.test(className ?? "");
    return (
      <code className={isBlock ? className : "inline-code"} {...props}>
        {children}
      </code>
    );
  },
  a({ node: _node, href, children, ...props }) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer" {...props}>
        {children}
      </a>
    );
  },
};

// ── Component ────────────────────────────────────────────────────────────────
// react-markdown sanitizes by default (no raw HTML, javascript: URLs stripped),
// so the previous dangerouslySetInnerHTML XSS surface is gone.

const MarkdownContentInner: React.FC<Props> = ({ text }) => {
  return (
    <div className="markdown-body">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex, rehypeHighlight]}
        components={components}
      >
        {normalizeMathDelimiters(text)}
      </ReactMarkdown>
    </div>
  );
};

// Memoized so streaming only re-renders the active bubble, not every prior one.
export const MarkdownContent = React.memo(MarkdownContentInner);
