"""The test written passes on the second-order scheme and fails on a first-order one."""
import json
import os
import shutil
import subprocess
import sys
import tempfile

FIRST_ORDER = '''"""Finite-difference derivatives."""


def central_derivative(f, x, h):
    """A forward difference, put in place of the central one: error O(h)."""
    return (f(x + h) - f(x)) / h
'''


def _pytest(directory):
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "tests/test_convergence.py"],
        cwd=directory, capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": directory})
    return proc.returncode, (proc.stdout + proc.stderr)[-300:]


def main(workspace):
    if not os.path.isfile(os.path.join(workspace, "tests", "test_convergence.py")):
        return False, "tests/test_convergence.py was not written"
    code, out = _pytest(workspace)
    if code != 0:
        return False, f"the test fails on the second-order scheme: {out}"
    with tempfile.TemporaryDirectory() as copy:
        shutil.copytree(workspace, copy, dirs_exist_ok=True)
        with open(os.path.join(copy, "schemes.py"), "w") as fh:
            fh.write(FIRST_ORDER)
        code, _ = _pytest(copy)
    if code == 0:
        return False, "the test also passes on a first-order scheme: it does not check the order"
    return True, "passes at second order, fails at first order"


if __name__ == "__main__":
    passed, detail = main(sys.argv[1])
    print(json.dumps({"passed": passed, "detail": detail}))
