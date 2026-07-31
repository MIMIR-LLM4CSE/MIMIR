import React from "react";
import { createRoot } from "react-dom/client";
import { App } from "./App";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import "./styles/main.css";

const root = createRoot(document.getElementById("root")!);
root.render(<App />);
