"""mini-swe-agent, opencode and GitHub Copilot CLI, each pointed at the meter.

The flags and config keys are those their documentation gives (read 2026-11-24); the
pilot checks them against the installed versions (``--help``) before any graded run.
"""
from __future__ import annotations

import json
from pathlib import Path

from mimir.runner import BenchTask

from . import Endpoint, ExternalAgent


class MiniSweAgent(ExternalAgent):
    """mini-swe-agent: bash only, no harness — the floor a harness is measured against.

    Its model is litellm's ``hosted_vllm/<model>`` at the meter; ``-c mini.yaml`` keeps its
    own defaults, the second ``-c`` overrides them (JSON is YAML).
    """

    name = "mini-swe-agent"
    executable = "mini"

    def _config(self, home: Path) -> Path:
        return home / "mimir-run.yaml"

    def prepare(self, task: BenchTask, workspace: str, home: Path, endpoint: Endpoint) -> None:
        self._config(home).write_text(json.dumps({
            "model": {"model_name": f"hosted_vllm/{endpoint.model}",
                      "model_kwargs": {"api_base": endpoint.base_url,
                                       "api_key": endpoint.run_tag},
                      "cost_tracking": "ignore_errors"},
            "agent": {"step_limit": task.max_steps},
            "environment": {"cwd": workspace},
        }), encoding="utf-8")

    def command(self, task: BenchTask, workspace: str, home: Path,
                endpoint: Endpoint) -> list[str]:
        return [self.executable, "-t", task.query, "-m", f"hosted_vllm/{endpoint.model}",
                "-y", "--exit-immediately", "-c", "mini.yaml", "-c", str(self._config(home)),
                "-o", str(home / "trajectory.json")]

    def environment(self, home: Path, endpoint: Endpoint) -> dict[str, str]:
        return {"MSWEA_COST_TRACKING": "ignore_errors"}

    def answer(self, stdout: str, home: Path) -> str:
        try:
            trajectory = json.loads((home / "trajectory.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return stdout.strip()
        return str((trajectory.get("info") or {}).get("submission") or stdout).strip()


class OpenCode(ExternalAgent):
    """opencode, its provider the meter (``@opencode/ai/providers/openai-compatible``),
    every tool allowed, updates off; its own auto-compaction stays as configured."""

    name = "opencode"
    executable = "opencode"
    provider = "meter"

    def __init__(self, context_tokens: int = 128_000, output_tokens: int = 16_000,
                 compaction: bool = True) -> None:
        self.context_tokens = context_tokens
        self.output_tokens = output_tokens
        self.compaction = compaction

    def prepare(self, task: BenchTask, workspace: str, home: Path, endpoint: Endpoint) -> None:
        config = home / ".config" / "opencode" / "opencode.json"
        config.parent.mkdir(parents=True, exist_ok=True)
        config.write_text(json.dumps({
            "$schema": "https://opencode.ai/config.json",
            "model": f"{self.provider}/{endpoint.model}",
            "update": "disable",
            "providers": {self.provider: {
                "name": "MIMIR benchmark meter",
                "env": ["MIMIR_METER_KEY"],
                "package": "@opencode/ai/providers/openai-compatible",
                "settings": {"baseURL": endpoint.base_url},
                "models": {endpoint.model: {
                    "name": endpoint.model,
                    "limit": {"context": self.context_tokens, "output": self.output_tokens}}},
            }},
            "permissions": [{"action": "*", "resource": "*", "effect": "allow"}],
            "compaction": {"auto": self.compaction},
        }), encoding="utf-8")

    def command(self, task: BenchTask, workspace: str, home: Path,
                endpoint: Endpoint) -> list[str]:
        return [self.executable, "run", task.query]

    def environment(self, home: Path, endpoint: Endpoint) -> dict[str, str]:
        return {"MIMIR_METER_KEY": endpoint.run_tag}


class Copilot(ExternalAgent):
    """GitHub Copilot CLI on our own model (BYOK): the ``COPILOT_PROVIDER_*`` variables,
    offline from GitHub, every tool allowed, its answer alone on stdout (``-s``)."""

    name = "copilot"
    executable = "copilot"

    def command(self, task: BenchTask, workspace: str, home: Path,
                endpoint: Endpoint) -> list[str]:
        return [self.executable, "-p", task.query, "--allow-all-tools", "-s"]

    def environment(self, home: Path, endpoint: Endpoint) -> dict[str, str]:
        return {"COPILOT_PROVIDER_BASE_URL": endpoint.base_url,
                "COPILOT_PROVIDER_TYPE": "openai",
                "COPILOT_PROVIDER_API_KEY": endpoint.run_tag,
                "COPILOT_MODEL": endpoint.model,
                "COPILOT_OFFLINE": "true",
                "COPILOT_AUTO_UPDATE": "false",
                "COPILOT_HOME": str(home / ".copilot")}
