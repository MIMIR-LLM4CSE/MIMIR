// @ts-check
"use strict";

const path = require("path");

/** @type {import('webpack').Configuration[]} */
module.exports = [
  // ── Extension host bundle ──────────────────────────────────────────────────
  {
    name: "extension",
    target: "node",
    mode: "none",
    entry: "./src/extension.ts",
    output: {
      path: path.resolve(__dirname, "dist"),
      filename: "extension.js",
      libraryTarget: "commonjs2",
    },
    externals: { vscode: "commonjs vscode", bufferutil: "commonjs bufferutil", "utf-8-validate": "commonjs utf-8-validate" },
    resolve: { extensions: [".ts", ".js"] },
    module: {
      rules: [{ test: /\.ts$/, loader: "ts-loader", exclude: /node_modules/ }],
    },
  },
  // ── Webview bundle (React app) ─────────────────────────────────────────────
  {
    name: "webview",
    target: "web",
    mode: "none",
    entry: "./webview/src/index.tsx",
    output: {
      path: path.resolve(__dirname, "dist"),
      filename: "webview.js",
    },
    resolve: { extensions: [".tsx", ".ts", ".js"] },
    module: {
      rules: [
        {
          test: /\.tsx?$/,
          loader: "ts-loader",
          exclude: /node_modules/,
          options: {
            configFile: path.resolve(__dirname, "webview/tsconfig.json"),
          },
        },
        {
          test: /\.css$/,
          use: ["style-loader", "css-loader"],
        },
        {
          // KaTeX fonts (referenced by katex.min.css) — inline as data: URIs so
          // they load under the webview CSP without exposing dist file paths.
          test: /\.(woff2?|ttf)$/,
          type: "asset/inline",
        },
      ],
    },
    performance: { hints: false },
  },
];
