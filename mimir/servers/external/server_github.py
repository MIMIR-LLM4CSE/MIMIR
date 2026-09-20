"""
MCP GitHub Server
=================
Read-only GitHub access via the public GitHub REST API.

Authentication
--------------
If GITHUB_TOKEN is set in the environment, requests use it to raise rate limits
and access private repositories available to that token. Without a token, only
public data is available and rate limits are lower.
"""

import base64
import difflib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '_shared'))

from mcp.server.fastmcp import FastMCP
from capabilities import tool_caps, EXTERNAL_FETCH
from responses import err, ok

mcp = FastMCP(
    "GitHubServer",
    debug=False,
    log_level="ERROR",
)

_API_BASE = "https://api.github.com"
_MAX_FILE_BYTES = 256 * 1024
# A file above the API's own inline limit is still readable through its raw URL; the
# ceiling below bounds what we read off that, the way server_web bounds a page.
_MAX_RAW_BYTES = 2 * 1024 * 1024
_MAX_READ_LINES = 400          # per call, mirroring workspace/server_search.py
_MAX_RESULTS = 25
# A 404 on a file path is almost always a guessed name, not a missing repo. The
# error walks back up to the deepest directory that does exist and names what is
# in it, so the next call is informed instead of another guess.
_MAX_SIBLINGS = 40
_MAX_WALK_UP = 4
_TOKEN = os.environ.get("GITHUB_TOKEN")


def _headers() -> dict:
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "mcp-github-server",
    }
    if _TOKEN:
        headers["Authorization"] = f"Bearer {_TOKEN}"
    return headers


def _request(path: str, params: dict | None = None) -> dict:
    url = _API_BASE + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, headers=_headers(), method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            body = resp.read().decode("utf-8", errors="replace")
            return ok({"data": json.loads(body)})
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        try:
            payload = json.loads(body)
        except Exception:
            payload = {"message": body or e.reason}
        return err(
            payload.get("message", e.reason),
            hint="Check repository owner/name, token permissions, or GitHub rate limits.",
            http_status=e.code,
        )
    except Exception as e:
        return err(str(e))


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH]))
def github_repo_info(owner: str, repo: str) -> dict:
    """Return metadata for a GitHub repository.

    Use this first when you need repository description, default branch, stars,
    topics, open issues count, clone URL, or visibility.

    Args:
        owner: GitHub owner or organization name.
        repo: Repository name.
    """
    result = _request(f"/repos/{owner}/{repo}")
    if result["status"] != "ok":
        return result
    data = result["data"]
    return ok({
        "name": data.get("full_name"),
        "description": data.get("description"),
        "default_branch": data.get("default_branch"),
        "private": data.get("private"),
        "stars": data.get("stargazers_count"),
        "forks": data.get("forks_count"),
        "open_issues": data.get("open_issues_count"),
        "topics": data.get("topics", []),
        "url": data.get("html_url"),
        "clone_url": data.get("clone_url"),
    })


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH]))
def github_list_branches(owner: str, repo: str, limit: int = 20) -> dict:
    """List branches for a repository.

    Args:
        owner: GitHub owner or organization name.
        repo: Repository name.
        limit: Maximum number of branches to return.
    """
    limit = max(1, min(limit, _MAX_RESULTS))
    result = _request(f"/repos/{owner}/{repo}/branches", {"per_page": limit})
    if result["status"] != "ok":
        return result
    branches = [
        {"name": branch.get("name"), "sha": branch.get("commit", {}).get("sha")}
        for branch in result["data"]
    ]
    return ok({"branches": branches, "count": len(branches)})


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH]))
def github_list_issues(owner: str, repo: str, state: str = "open", limit: int = 20) -> dict:
    """List repository issues.

    Pull requests are excluded from the returned list.

    Args:
        owner: GitHub owner or organization name.
        repo: Repository name.
        state: One of open, closed, all.
        limit: Maximum number of issues to return.
    """
    limit = max(1, min(limit, _MAX_RESULTS))
    result = _request(
        f"/repos/{owner}/{repo}/issues",
        {"state": state, "per_page": limit},
    )
    if result["status"] != "ok":
        return result
    issues = []
    for item in result["data"]:
        if "pull_request" in item:
            continue
        issues.append(
            {
                "number": item.get("number"),
                "title": item.get("title"),
                "state": item.get("state"),
                "user": item.get("user", {}).get("login"),
                "created_at": item.get("created_at"),
                "url": item.get("html_url"),
            }
        )
    return ok({"issues": issues, "count": len(issues)})


def _list_directory(owner: str, repo: str, dirpath: str, ref: str) -> list[str] | None:
    """Names in one repository directory, or None when it is not a listable directory."""
    params = {"ref": ref} if ref else None
    quoted = "/".join(urllib.parse.quote(part) for part in dirpath.split("/") if part)
    result = _request(f"/repos/{owner}/{repo}/contents/{quoted}", params)
    if result["status"] != "ok" or not isinstance(result["data"], list):
        return None
    return [entry.get("name", "") for entry in result["data"]]


def _not_found(owner: str, repo: str, path: str, ref: str, fallback: dict) -> dict:
    """Turn a bare 404 on ``path`` into the listing of its closest existing parent."""
    parts = [part for part in path.split("/") if part]
    # Walk up from the file's own directory: the first level that lists is where the
    # guessed path stopped matching the repository.
    for depth in range(1, min(len(parts), _MAX_WALK_UP) + 1):
        parent = "/".join(parts[:-depth])
        names = _list_directory(owner, repo, parent, ref)
        if names is None:
            continue
        shown = sorted(names)[:_MAX_SIBLINGS]
        where = f"'{parent}'" if parent else "the repository root"
        near = difflib.get_close_matches(parts[-1], names, n=3, cutoff=0.6)
        # Near matches lead: they are the actionable part, and a long listing is the
        # first thing a downstream truncation would cut.
        hint = f"Closest names to '{parts[-1]}': {', '.join(near)}. " if near else ""
        hint += f"{where} exists and contains: {', '.join(shown)}."
        if len(names) > len(shown):
            hint += f" ({len(names) - len(shown)} more not shown.)"
        return err(
            f"'{path}' does not exist in {owner}/{repo}" + (f" at ref '{ref}'." if ref else "."),
            hint=hint,
            http_status=404,
            listed_path=parent,
            entries=shown,
            near_matches=near,
        )
    return fallback


def _raw_text(url: str) -> str:
    """The file straight off its raw URL, bounded. "" when it cannot be read."""
    if not url:
        return ""
    try:
        req = urllib.request.Request(url, headers=_headers(), method="GET")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.read(_MAX_RAW_BYTES).decode("utf-8", errors="replace")
    except Exception:
        return ""


def _line_window(text: str, start_line: int, end_line: int) -> dict:
    """A line range of *text*, with the keys ``read_file_lines`` established.

    The same names on purpose: `truncated`, `total_lines`, `next_start_line` and
    `line_cap` are what the client's `_build_continuation_hint` already reads to tell
    the model its read stopped short, so a paged GitHub read inherits that for free
    rather than inventing a second vocabulary for the same idea.
    """
    lines = text.splitlines(keepends=True)
    start = max(1, int(start_line or 1))
    requested_end = len(lines) if not end_line or end_line <= 0 else int(end_line)
    capped_end = min(requested_end, start + _MAX_READ_LINES - 1)
    actual_end = min(capped_end, len(lines))
    selected = lines[start - 1:actual_end] if start <= len(lines) else []
    out = {
        "start_line": start,
        "end_line": actual_end,
        "total_lines": len(lines),
        "content": "".join(selected),
        "lines_returned": len(selected),
    }
    if actual_end < len(lines):
        out["truncated"] = True
        out["next_start_line"] = actual_end + 1
    if capped_end < requested_end:
        out["line_cap"] = _MAX_READ_LINES
    return out


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH], label="Fetching from GitHub: {path}"))
def github_get_file(owner: str, repo: str, path: str, ref: str = "",
                    start_line: int = 1, end_line: int = 0) -> dict:
    """Fetch a text file from a GitHub repository and decode its content.

    Best for README files, configuration files, source files, and docs. The reply is
    a line window: at most 400 lines per call, and when it stops short it says so
    (`truncated`) and names where to resume (`next_start_line`), so a large file is
    read in pages instead of being refused.

    Args:
        owner:      GitHub owner or organization name.
        repo:       Repository name.
        path:       File path inside the repository.
        ref:        Branch, tag, or commit SHA. Empty means the default branch.
        start_line: First line to return (1-based).
        end_line:   Last line to return, inclusive. 0 (the default) means "to the end
                    of the file", up to the per-call cap.
    """
    params = {"ref": ref} if ref else None
    quoted_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    result = _request(f"/repos/{owner}/{repo}/contents/{quoted_path}", params)
    if result["status"] != "ok":
        if result.get("http_status") == 404:
            return _not_found(owner, repo, path, ref, result)
        return result
    data = result["data"]
    if data.get("type") != "file":
        return err(
            f"'{path}' is not a regular file.",
            hint="Use a file path, not a directory path.",
        )
    size = data.get("size", 0)
    encoding = data.get("encoding")
    content = data.get("content", "")
    if size > _MAX_FILE_BYTES or encoding != "base64" or not content:
        # Too large for the contents API to inline, or inlined in something we do not
        # decode. This used to be a refusal, which cost the caller the file entirely —
        # a total loss of information to avoid a large one. The raw URL has the same
        # bytes, and the window below is what makes reading them affordable.
        decoded = _raw_text(data.get("download_url") or "")
        if not decoded:
            return err(
                f"File could not be read ({size} bytes, encoding {encoding!r}).",
                hint="Inspect the repository structure, or fetch the raw URL directly.",
            )
        source = "raw"
    else:
        decoded = base64.b64decode(content).decode("utf-8", errors="replace")
        source = "api"
    return ok({
        "path": data.get("path"),
        "sha": data.get("sha"),
        "size": size,
        "source": source,
        "download_url": data.get("download_url"),
        **_line_window(decoded, start_line, end_line),
    })


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH]))
def github_search_repositories(query: str, limit: int = 10) -> dict:
    """Search public GitHub repositories.

    Useful when the user does not know the exact repository name.

    Args:
        query: GitHub search query, e.g. 'llama.cpp language:C++'.
        limit: Maximum number of repositories to return.
    """
    limit = max(1, min(limit, _MAX_RESULTS))
    result = _request("/search/repositories", {"q": query, "per_page": limit})
    if result["status"] != "ok":
        return result
    items = result["data"].get("items", [])
    repos = []
    for item in items:
        repos.append(
            {
                "full_name": item.get("full_name"),
                "description": item.get("description"),
                "stars": item.get("stargazers_count"),
                "language": item.get("language"),
                "url": item.get("html_url"),
            }
        )
    return ok({"repositories": repos, "count": len(repos)})


if __name__ == "__main__":
    mcp.run()
