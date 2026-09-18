"""The Python package and the VS Code extension carry one version.

The extension drives the server of the same checkout, so a bump that reaches one
file and not the other ships a pair nobody tested together. CHANGELOG.md must
also describe the version being shipped.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXT = ROOT / "mimir" / "vscode-extension"


def _pyproject_version() -> str:
    text = (ROOT / "pyproject.toml").read_text()
    match = re.search(r'^version = "([^"]+)"', text, re.MULTILINE)
    assert match, "pyproject.toml has no version line"
    return match.group(1)


def test_extension_matches_package():
    assert json.loads((EXT / "package.json").read_text())["version"] == _pyproject_version()


def test_lockfile_matches_package():
    lock = json.loads((EXT / "package-lock.json").read_text())
    assert lock["version"] == _pyproject_version()
    assert lock["packages"][""]["version"] == _pyproject_version()


def test_changelog_has_an_entry():
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert f"## [{_pyproject_version()}]" in changelog
