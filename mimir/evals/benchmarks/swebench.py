"""SWE-bench (Lite, Verified, full) as a ``BatchAdapter``: the agent fixes an issue in the
repository at its base commit, fetched alone at depth 1 from a bare clone under *work_dir*
so no later commit is among its objects; its diff is graded by SWE-bench's harness in
Docker, every run of a batch at once. Without the ``swebench`` package or Docker the runs
are skipped, their predictions kept for a later ``swebench.harness.run_evaluation``.
"""
from __future__ import annotations

import glob
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from mimir.evals.benchmarks._hub import flag, load_rows
from mimir.runner import BenchTask, CheckContext, RunResult

DATASETS = {
    "lite": "SWE-bench/SWE-bench_Lite",
    "verified": "SWE-bench/SWE-bench_Verified",
    "full": "SWE-bench/SWE-bench",
}
DEFAULT_WORK_DIR = Path.home() / ".cache" / "mimir" / "bench" / "swebench"
GIT_TIMEOUT_SECS = 1800

PROMPT = """{problem_statement}

The issue above is about the repository checked out in the current directory, at the \
commit where it was reported. Change its non-test files so that the issue is resolved. The change \
is graded by the repository's own tests, which are not in the workspace: do not edit the \
existing tests to make them pass. The repository's dependencies may not be installed \
here; when a test cannot run, reason from the code."""


def _git(*args: str, cwd: str | Path | None = None) -> str:
    proc = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True,
                          timeout=GIT_TIMEOUT_SECS)
    if proc.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {proc.stderr.strip()[-400:]}")
    return proc.stdout


class SWEBench:
    """SWE-bench as a ``BatchAdapter``; every argument a string, as ``--adapter-arg`` gives it.

    *dataset* is ``lite``, ``verified``, ``full`` or a Hugging Face dataset name;
    *instances* a comma-separated list of instance ids; *repo_url* the template the
    mirrors are cloned from (``{repo}`` is ``owner/name``).
    """

    def __init__(self, dataset: str = "lite", split: str = "test", instances: str = "",
                 work_dir: str = "", max_steps: str = "100", timeout_secs: str = "",
                 max_workers: str = "4", eval_timeout_secs: str = "1800",
                 model_name: str = "mimir", hints: str = "false",
                 repo_url: str = "https://github.com/{repo}.git") -> None:
        self.dataset = DATASETS.get(dataset, dataset)
        self.split = split
        self.instances = [i for i in (s.strip() for s in instances.split(",")) if i]
        self.work_dir = Path(work_dir).expanduser() if work_dir else DEFAULT_WORK_DIR
        self.max_steps = int(max_steps)
        self.timeout_secs = float(timeout_secs) if timeout_secs else None
        self.max_workers = int(max_workers)
        self.eval_timeout_secs = int(eval_timeout_secs)
        self.model_name = model_name
        self.hints = flag(hints)
        self.repo_url = repo_url
        self.name = "swebench_" + self.dataset.rsplit("/", 1)[-1].lower()

    def rows(self) -> list[dict[str, Any]]:
        """The split's rows, from Hugging Face."""
        return load_rows(self.dataset, self.split)

    def load_tasks(self, limit: int | None = None) -> list[BenchTask]:
        rows = self.rows()
        if self.instances:
            wanted = set(self.instances)
            rows = [r for r in rows if r["instance_id"] in wanted]
            missing = wanted - {r["instance_id"] for r in rows}
            if missing:
                raise SystemExit(f"not in {self.dataset}/{self.split}: {', '.join(sorted(missing))}")
        tasks = [self._task(r) for r in rows]
        return tasks[:limit] if limit is not None else tasks

    def _task(self, row: dict[str, Any]) -> BenchTask:
        statement = row["problem_statement"]
        if self.hints and row.get("hints_text"):
            statement += "\n\nHints from the issue's discussion:\n" + row["hints_text"]

        def _setup(workspace: str, repo: str = row["repo"], commit: str = row["base_commit"]) -> None:
            self.checkout(repo, commit, workspace)

        return BenchTask(
            id=row["instance_id"], query=PROMPT.format(problem_statement=statement),
            max_steps=self.max_steps, timeout_secs=self.timeout_secs, setup=_setup,
            requires=("git",),
            meta={"repo": row["repo"], "base_commit": row["base_commit"]},
        )

    def mirror(self, repo: str, commit: str) -> Path:
        """The bare clone of *repo* under *work_dir*, cloned once, *commit* fetched into it
        when it lacks it."""
        path = self.work_dir / "repos" / (repo.replace("/", "__") + ".git")
        if not path.is_dir():
            path.parent.mkdir(parents=True, exist_ok=True)
            _git("clone", "--quiet", "--bare", self.repo_url.format(repo=repo), str(path))
            # The workspaces fetch a commit by its id, which no branch may name.
            _git("config", "uploadpack.allowAnySHA1InWant", "true", cwd=path)
        has = subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=path,
                             capture_output=True)
        if has.returncode != 0:
            _git("fetch", "--quiet", "origin", commit, cwd=path)
        return path

    def checkout(self, repo: str, commit: str, workspace: str) -> None:
        """*repo* at *commit* in the empty *workspace*, with no other commit."""
        mirror = self.mirror(repo, commit)
        _git("init", "--quiet", cwd=workspace)
        _git("fetch", "--quiet", "--depth", "1", f"file://{mirror}", commit, cwd=workspace)
        _git("-c", "advice.detachedHead=false", "checkout", "--quiet", "--detach", "FETCH_HEAD",
             cwd=workspace)
        (Path(workspace) / ".git" / "FETCH_HEAD").unlink(missing_ok=True)

    async def collect(self, task: BenchTask, answer: str, ctx: CheckContext) -> dict[str, Any]:
        """The workspace's change from the base commit, new files included, as a patch."""
        _git("add", "-A", cwd=ctx.workspace)
        patch = _git("diff", "--cached", task.meta["base_commit"], cwd=ctx.workspace)
        return {"model_patch": patch}

    async def grade(self, runs: list[RunResult], label: str
                    ) -> dict[str, tuple[bool, str] | tuple[bool, str, bool]]:
        """Write the batch's predictions and run SWE-bench's harness over them."""
        run_dir = self.work_dir / "runs" / label
        run_dir.mkdir(parents=True, exist_ok=True)
        predictions = run_dir / "predictions.jsonl"
        with open(predictions, "w", encoding="utf-8") as fh:
            for run in runs:
                fh.write(json.dumps({"instance_id": run.task_id,
                                     "model_name_or_path": self.model_name,
                                     "model_patch": run.artifact.get("model_patch", "")}) + "\n")
        command = [sys.executable, "-m", "swebench.harness.run_evaluation",
                   "--dataset_name", self.dataset, "--split", self.split,
                   "--predictions_path", str(predictions), "--run_id", label,
                   "--max_workers", str(self.max_workers),
                   "--timeout", str(self.eval_timeout_secs),
                   "--instance_ids", *(run.task_id for run in runs)]
        unavailable = self._unavailable()
        if unavailable:
            detail = (f"not graded: {unavailable}; predictions in {predictions}, "
                      f"graded later by: python {' '.join(command[1:])}")
            return {run.task_id: (False, detail, True) for run in runs}
        proc = subprocess.run(command, cwd=run_dir, capture_output=True, text=True)
        (run_dir / "harness.log").write_text(proc.stdout + proc.stderr, encoding="utf-8")
        summary = self._summary(run_dir, label)
        if summary is None:
            detail = f"the harness wrote no report (exit {proc.returncode}); see {run_dir}/harness.log"
            return {run.task_id: (False, detail) for run in runs}
        return {run.task_id: self._verdict(run.task_id, summary) for run in runs}

    @staticmethod
    def _unavailable() -> str:
        """Why the harness cannot run here, or ``""``."""
        if importlib.util.find_spec("swebench") is None:
            return "the swebench package is not installed (pip install 'mimir-agent[bench]')"
        try:
            docker = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=60)
        except (OSError, subprocess.TimeoutExpired):
            return "docker is not installed or does not answer"
        if docker.returncode != 0:
            return f"docker is not reachable: {docker.stderr.strip().splitlines()[-1:]}"
        return ""

    @staticmethod
    def _summary(run_dir: Path, label: str) -> dict[str, Any] | None:
        """The harness's run report: ``<log dir>/<label>/results.json``, or the
        ``<model>.<label>.json`` that versions before 5 write in the working directory."""
        found = (glob.glob(str(run_dir / "**" / label / "results.json"), recursive=True)
                 or glob.glob(str(run_dir / f"*.{label}.json")))
        if not found:
            return None
        with open(found[0], encoding="utf-8") as fh:
            return json.load(fh)

    @staticmethod
    def _verdict(instance_id: str, summary: dict[str, Any]) -> tuple[bool, str]:
        def among(key: str) -> bool:
            return instance_id in (summary.get(key) or [])

        if among("resolved_ids"):
            return True, "resolved"
        if among("empty_patch_ids"):
            return False, "empty patch: the workspace was left unchanged"
        if among("unresolved_ids"):
            return False, "unresolved: the tests that check the issue still fail"
        if among("error_ids"):
            return False, "the harness could not evaluate the patch (it may not apply)"
        return False, "not in the harness's report"
