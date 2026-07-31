"""Minimal synchronous LSP (Language Server Protocol) client over stdio.

Just enough of the protocol to drive a language server for code navigation and
diagnostics:

    initialize -> initialized -> didOpen -> {workspace/symbol,
    textDocument/documentSymbol, textDocument/definition,
    textDocument/references, textDocument/hover} + publishDiagnostics

Deliberately dependency-free (stdlib only) and **best-effort**: every public
method returns ``None`` / ``[]`` on any failure, so the caller
(server_code_intel) degrades gracefully to ctags when no language server is
installed or the server misbehaves. Nothing here imports ``mcp``.

A small registry maps file extensions to candidate server commands; the first
command whose executable is found on ``PATH`` is used. A started server is
cached per (command, root) for the lifetime of the owning process.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from typing import Any

# extension -> ordered list of candidate server argv (first available wins)
_SERVERS: dict[str, list[list[str]]] = {
    # Python
    ".py": [["pyright-langserver", "--stdio"], ["pylsp"]],
    ".pyi": [["pyright-langserver", "--stdio"], ["pylsp"]],
    # C / C++ / CUDA -> clangd
    ".c": [["clangd"]],
    ".h": [["clangd"]],
    ".cpp": [["clangd"]],
    ".cc": [["clangd"]],
    ".cxx": [["clangd"]],
    ".hpp": [["clangd"]],
    ".hh": [["clangd"]],
    ".cu": [["clangd"]],
    ".cuh": [["clangd"]],
    # Fortran -> fortls
    ".f": [["fortls"]],
    ".f90": [["fortls"]],
    ".f95": [["fortls"]],
    ".f03": [["fortls"]],
    ".f08": [["fortls"]],
    ".for": [["fortls"]],
    ".fpp": [["fortls"]],
}

_LANG_IDS = {
    ".py": "python", ".pyi": "python",
    ".c": "c", ".h": "c",
    ".cpp": "cpp", ".cc": "cpp", ".cxx": "cpp", ".hpp": "cpp", ".hh": "cpp",
    ".cu": "cuda", ".cuh": "cuda",
    ".f": "fortran", ".f90": "fortran", ".f95": "fortran", ".f03": "fortran",
    ".f08": "fortran", ".for": "fortran", ".fpp": "fortran",
}


def uri_for(path: str) -> str:
    return "file://" + os.path.abspath(path)


def path_for(uri: str) -> str:
    return uri[7:] if uri.startswith("file://") else uri


def language_id(path: str) -> str:
    return _LANG_IDS.get(os.path.splitext(path)[1].lower(), "plaintext")


def server_command_for(path: str) -> list[str] | None:
    """First candidate language-server argv whose executable exists, or None."""
    for argv in _SERVERS.get(os.path.splitext(path)[1].lower(), []):
        if shutil.which(argv[0]):
            return argv
    return None


class LSPClient:
    """One language-server subprocess; methods are best-effort and never raise."""

    def __init__(self, command: list[str], root: str, *, timeout: float = 8.0):
        self.command = command
        self.root = os.path.abspath(root)
        self.timeout = timeout
        self._proc: subprocess.Popen | None = None
        self._id = 0
        self._lock = threading.Lock()
        self._opened: set[str] = set()
        self._diagnostics: dict[str, list] = {}
        self._ready = False
        self._dead = False

    # ── lifecycle ────────────────────────────────────────────────────────────
    def start(self) -> bool:
        if self._ready:
            return True
        if self._dead:
            return False
        try:
            self._proc = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=self.root,
                bufsize=0,
            )
        except (OSError, ValueError):
            self._dead = True
            return False
        try:
            self._request("initialize", {
                "processId": os.getpid(),
                "rootUri": uri_for(self.root),
                "capabilities": {},
                "workspaceFolders": [{"uri": uri_for(self.root), "name": "root"}],
            })
            self._notify("initialized", {})
            self._ready = True
            return True
        except Exception:
            self.stop()
            self._dead = True
            return False

    def stop(self) -> None:
        proc, self._proc, self._ready = self._proc, None, False
        if proc is not None:
            try:
                proc.terminate()
            except Exception:
                pass

    # ── wire protocol ────────────────────────────────────────────────────────
    def _write(self, payload: dict) -> None:
        data = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
        self._proc.stdin.write(header + data)  # type: ignore[union-attr]
        self._proc.stdin.flush()  # type: ignore[union-attr]

    def _read_message(self, deadline: float) -> dict | None:
        out = self._proc.stdout  # type: ignore[union-attr]
        headers: dict[bytes, bytes] = {}
        while True:
            if time.monotonic() > deadline:
                return None
            line = out.readline()
            if line == b"":
                return None  # EOF
            line = line.strip()
            if line == b"":
                break
            if b":" in line:
                k, v = line.split(b":", 1)
                headers[k.strip().lower()] = v.strip()
        length = int(headers.get(b"content-length", b"0") or 0)
        body = b""
        while len(body) < length:
            chunk = out.read(length - len(body))
            if not chunk:
                return None
            body += chunk
        try:
            return json.loads(body.decode("utf-8"))
        except Exception:
            return None

    def _notify(self, method: str, params: dict) -> None:
        self._write({"jsonrpc": "2.0", "method": method, "params": params})

    def _drain_notification(self, msg: dict) -> None:
        if msg.get("method") == "textDocument/publishDiagnostics":
            p = msg.get("params", {}) or {}
            self._diagnostics[p.get("uri", "")] = p.get("diagnostics", []) or []
        elif "id" in msg and "method" in msg:
            # server -> client request (e.g. client/registerCapability): ack empty
            try:
                self._write({"jsonrpc": "2.0", "id": msg["id"], "result": None})
            except Exception:
                pass

    def _request(self, method: str, params: dict) -> Any:
        with self._lock:
            self._id += 1
            rid = self._id
            self._write({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
            deadline = time.monotonic() + self.timeout
            while True:
                msg = self._read_message(deadline)
                if msg is None:
                    raise TimeoutError(f"LSP {method} timed out")
                if msg.get("id") == rid:
                    if "error" in msg:
                        raise RuntimeError(msg["error"])
                    return msg.get("result")
                self._drain_notification(msg)

    # ── document sync ────────────────────────────────────────────────────────
    def open(self, path: str) -> bool:
        uri = uri_for(path)
        if uri in self._opened:
            return True
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                text = fh.read()
        except OSError:
            return False
        try:
            self._notify("textDocument/didOpen", {"textDocument": {
                "uri": uri,
                "languageId": language_id(path),
                "version": 1,
                "text": text,
            }})
        except Exception:
            return False
        self._opened.add(uri)
        return True

    # ── high-level queries (None on any failure) ─────────────────────────────
    def workspace_symbol(self, name: str) -> list[dict] | None:
        if not self.start():
            return None
        try:
            res = self._request("workspace/symbol", {"query": name})
        except Exception:
            return None
        return res if isinstance(res, list) else None

    def document_symbol(self, path: str) -> list[dict] | None:
        if not self.start() or not self.open(path):
            return None
        try:
            res = self._request("textDocument/documentSymbol", {
                "textDocument": {"uri": uri_for(path)},
            })
        except Exception:
            return None
        return res if isinstance(res, list) else None

    def references(self, path: str, line: int, character: int) -> list[dict] | None:
        if not self.start() or not self.open(path):
            return None
        try:
            res = self._request("textDocument/references", {
                "textDocument": {"uri": uri_for(path)},
                "position": {"line": line, "character": character},
                "context": {"includeDeclaration": False},
            })
        except Exception:
            return None
        return res if isinstance(res, list) else None

    def hover(self, path: str, line: int, character: int) -> dict | None:
        if not self.start() or not self.open(path):
            return None
        try:
            res = self._request("textDocument/hover", {
                "textDocument": {"uri": uri_for(path)},
                "position": {"line": line, "character": character},
            })
        except Exception:
            return None
        return res if isinstance(res, dict) else None

    def diagnostics(self, path: str, *, settle: float = 2.5) -> list[dict] | None:
        """Open *path* and collect publishDiagnostics until things settle.

        Returns the diagnostics list for the document, or ``None`` if the server
        never reports any within the settle window (caller treats None as "no LSP
        signal" and falls back to the compiler/linter path).
        """
        if not self.start() or not self.open(path):
            return None
        uri = uri_for(path)
        deadline = time.monotonic() + settle
        seen = uri in self._diagnostics
        while time.monotonic() < deadline:
            msg = self._read_message(time.monotonic() + 0.4)
            if msg is None:
                if seen:
                    break
                continue
            if msg.get("id") is not None and "result" in msg:
                continue
            self._drain_notification(msg)
            if uri in self._diagnostics:
                seen = True
        return self._diagnostics.get(uri)


class LSPPool:
    """Lazily-started, reused language servers keyed by (command, root)."""

    def __init__(self, root: str):
        self.root = os.path.abspath(root)
        self._clients: dict[tuple[str, ...], LSPClient] = {}
        self._lock = threading.Lock()

    def client_for(self, path: str) -> LSPClient | None:
        argv = server_command_for(path)
        if argv is None:
            return None
        key = tuple(argv)
        with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = LSPClient(argv, self.root)
                self._clients[key] = client
        return client if client.start() else None

    def shutdown(self) -> None:
        for client in self._clients.values():
            client.stop()
        self._clients.clear()
