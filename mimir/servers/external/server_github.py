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
_MAX_RESULTS = 25
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


@mcp.tool(**tool_caps(caps=[EXTERNAL_FETCH], label="Fetching from GitHub: {path}"))
def github_get_file(owner: str, repo: str, path: str, ref: str = "") -> dict:
    """Fetch a text file from a GitHub repository and decode its content.

    Best for README files, configuration files, source files, and docs.
    Large files are rejected to avoid overloading the agent context.

    Args:
        owner: GitHub owner or organization name.
        repo: Repository name.
        path: File path inside the repository.
        ref: Branch, tag, or commit SHA. Empty means the default branch.
    """
    params = {"ref": ref} if ref else None
    quoted_path = "/".join(urllib.parse.quote(part) for part in path.split("/"))
    result = _request(f"/repos/{owner}/{repo}/contents/{quoted_path}", params)
    if result["status"] != "ok":
        return result
    data = result["data"]
    if data.get("type") != "file":
        return err(
            f"'{path}' is not a regular file.",
            hint="Use a file path, not a directory path.",
        )
    size = data.get("size", 0)
    if size > _MAX_FILE_BYTES:
        return err(
            f"File is too large ({size} bytes).",
            hint="Fetch a smaller file or inspect the repository structure first.",
        )
    encoding = data.get("encoding")
    content = data.get("content", "")
    if encoding != "base64":
        return err(f"Unsupported content encoding '{encoding}'.")
    decoded = base64.b64decode(content).decode("utf-8", errors="replace")
    return ok({
        "path": data.get("path"),
        "sha": data.get("sha"),
        "size": size,
        "content": decoded,
        "download_url": data.get("download_url"),
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
