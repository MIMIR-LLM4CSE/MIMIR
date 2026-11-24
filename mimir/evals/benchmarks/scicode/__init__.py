"""SciCode as a ``BenchmarkAdapter``: one task per problem, its sub-steps the functions of
``solution.py`` for the agent to write, each graded by SciCode's tests against the values
of its ``test_data.h5``. *data_dir* is SciCode's ``eval/data``: that file and the code it
ships for the steps without tests (``13.6.txt``…), which later steps build on. A problem
passes when every tested sub-step does; the detail gives the sub-steps passed.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from mimir.evals.benchmarks._hub import flag, load_rows
from mimir.runner import BenchTask, CheckContext

DATASET = "SciCode1/SciCode"
DEFAULT_DATA_DIR = Path.home() / ".cache" / "mimir" / "bench" / "scicode"
H5_NAME = "test_data.h5"
H5_SOURCE = ("https://drive.google.com/drive/folders/1W5GZW6_bdiDAiipuFMqdUhvUaHIj6-pR "
             "(linked from https://github.com/scicode-bench/SciCode)")
SOLUTION = "solution.py"
TARGETS = Path(__file__).resolve().parent / "targets.py"

PROMPT = """{description}

{io}

Implement this in `solution.py`, one step at a time: each step is a function or class \
already declared there, with its docstring, whose body you write. Keep every name and \
signature as declared; a later step may call an earlier one. Use only the imports at the \
top of the file. Steps marked as provided are already written.

{steps}"""


def _stub(header: str) -> str:
    """*header* with a body that raises, indented as its docstring closes."""
    lines = header.rstrip().splitlines()
    last = next((ln for ln in reversed(lines) if ln.strip()), "")
    indent = last[: len(last) - len(last.lstrip())] or "    "
    return header.rstrip() + f"\n{indent}raise NotImplementedError\n"


class SciCode:
    """SciCode as a ``BenchmarkAdapter``; every argument a string, as ``--adapter-arg``
    gives it. *problems* is a comma-separated list of problem ids; *background* adds each
    step's scientific background to the query."""

    name = "scicode"

    def __init__(self, data_dir: str = "", split: str = "test", problems: str = "",
                 background: str = "false", max_steps: str = "80", timeout_secs: str = "",
                 step_timeout_secs: str = "1800") -> None:
        self.data_dir = Path(data_dir).expanduser() if data_dir else DEFAULT_DATA_DIR
        self.split = split
        self.problems = [p for p in (s.strip() for s in problems.split(",")) if p]
        self.background = flag(background)
        self.max_steps = int(max_steps)
        self.timeout_secs = float(timeout_secs) if timeout_secs else None
        self.step_timeout_secs = float(step_timeout_secs)

    @property
    def h5(self) -> Path:
        return self.data_dir / H5_NAME

    def rows(self) -> list[dict[str, Any]]:
        """The split's rows, from Hugging Face."""
        return load_rows(DATASET, self.split)

    def load_tasks(self, limit: int | None = None) -> list[BenchTask]:
        if not self.h5.is_file():
            raise SystemExit(f"SciCode's expected values are not at {self.h5}: download "
                             f"{H5_NAME} from {H5_SOURCE}, or pass data_dir=<SciCode>/eval/data")
        rows = self.rows()
        if self.problems:
            wanted = set(self.problems)
            rows = [r for r in rows if str(r["problem_id"]) in wanted]
        unshipped = [s["step_number"] for r in rows for s in r["sub_steps"]
                     if not s["test_cases"] and self.provided(s) is None]
        if unshipped:
            raise SystemExit(f"steps {', '.join(unshipped)} have no test and their code is not "
                             f"in {self.data_dir} ({', '.join(f'{s}.txt' for s in unshipped)}, "
                             "from SciCode's eval/data): the steps after them build on it")
        tasks = [self._task(r) for r in rows]
        return tasks[:limit] if limit is not None else tasks

    def provided(self, step: dict[str, Any]) -> str | None:
        """The code shipped for a step that has no test, or ``None`` for a step to write."""
        if step["test_cases"]:
            return None
        if step.get("ground_truth_code"):
            return step["ground_truth_code"]
        shipped = self.data_dir / f"{step['step_number']}.txt"
        return shipped.read_text(encoding="utf-8") if shipped.is_file() else None

    def solution(self, row: dict[str, Any]) -> str:
        """``solution.py`` as the workspace starts: the imports, then each step's code or stub."""
        parts = [row["required_dependencies"].strip(), ""]
        for step in row["sub_steps"]:
            code = self.provided(step)
            parts += [f"# Step {step['step_number']}" + (" (provided)" if code else ""),
                      (code.rstrip() + "\n") if code else _stub(step["function_header"]), ""]
        return "\n".join(parts)

    def _query(self, row: dict[str, Any]) -> str:
        steps = []
        for step in row["sub_steps"]:
            text = step["step_description_prompt"].strip()
            if self.background and step.get("step_background"):
                text += "\n\n" + step["step_background"].strip()
            mark = " (provided)" if self.provided(step) else ""
            steps.append(f"### Step {step['step_number']}{mark}\n\n{text}")
        description = row["problem_description_main"].strip()
        if self.background and row.get("problem_background_main"):
            description += "\n\n" + row["problem_background_main"].strip()
        return PROMPT.format(description=description, io=row["problem_io"].strip(),
                             steps="\n\n".join(steps))

    def _task(self, row: dict[str, Any]) -> BenchTask:
        source = self.solution(row)

        def _setup(workspace: str) -> None:
            Path(workspace, SOLUTION).write_text(source, encoding="utf-8")

        tested = [{"step": s["step_number"], "tests": list(s["test_cases"])}
                  for s in row["sub_steps"] if s["test_cases"]]
        return BenchTask(
            id=f"scicode_{row['problem_id']}", query=self._query(row),
            max_steps=self.max_steps, timeout_secs=self.timeout_secs, setup=_setup,
            requires=("python3",), modules=("numpy", "scipy", "h5py"),
            meta={"steps": tested},
        )

    def step_script(self, solution: str, step: str, tests: list[str]) -> str:
        """The test script of one step: the solution, its expected values, its tests."""
        lines = [solution, "", "from scicode_targets import targets as _scicode_targets",
                 f"targets = _scicode_targets({str(self.h5)!r}, {step!r}, {len(tests)})"]
        for i, test in enumerate(tests):
            lines += ["", f"target = targets[{i}]", test]
        return "\n".join(lines) + "\n"

    def run_step(self, solution: str, step: str, tests: list[str]) -> tuple[bool, str]:
        """Run one step's tests in a fresh interpreter: ``(passed, why not)``."""
        with tempfile.TemporaryDirectory(prefix="scicode_check_") as tmp:
            shutil.copy(TARGETS, Path(tmp, "scicode_targets.py"))
            script = Path(tmp, f"test_{step}.py")
            script.write_text(self.step_script(solution, step, tests), encoding="utf-8")
            try:
                proc = subprocess.run([sys.executable, str(script)], cwd=tmp, capture_output=True,
                                      text=True, timeout=self.step_timeout_secs)
            except subprocess.TimeoutExpired:
                return False, f"timed out after {self.step_timeout_secs:.0f}s"
        if proc.returncode == 0:
            return True, ""
        last = (proc.stderr.strip().splitlines() or ["exit " + str(proc.returncode)])[-1]
        return False, last[:200]

    async def score(self, task: BenchTask, answer: str, ctx: CheckContext) -> tuple[bool, str]:
        path = Path(ctx.workspace, SOLUTION)
        if not path.is_file():
            return False, f"{SOLUTION} is gone from the workspace"
        solution = path.read_text(encoding="utf-8")
        failed = []
        for entry in task.meta["steps"]:
            ok, why = self.run_step(solution, entry["step"], entry["tests"])
            if not ok:
                failed.append(f"{entry['step']} ({why})")
        total = len(task.meta["steps"])
        detail = f"sub-steps {total - len(failed)}/{total} passed"
        if failed:
            detail += "; failed: " + "; ".join(failed)
        return not failed, detail
