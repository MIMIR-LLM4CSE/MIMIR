"""Component-scaffold op: generate reference + test harness files for a component."""

from __future__ import annotations

import ast
import os
import re

from _ops import err, ok
from _lib.store import scaffolds_dir


def scaffold(
    proxy_path: str,
    component_hint: str,
    harness_type: str = "kernel",
    output_path: str = "",
) -> dict:
    """Locate a component in a proxy source file and generate two harness files."""
    if not os.path.isfile(proxy_path):
        return err(f"proxy_path not found: {proxy_path}")
    if harness_type not in ("kernel", "subsystem"):
        return err("harness_type must be 'kernel' or 'subsystem'.")

    # Detect language
    ext = os.path.splitext(proxy_path)[1].lower()
    lang = "python" if ext == ".py" else (
        "fortran" if ext in (".f", ".f90", ".f95", ".f03", ".f08", ".for") else
        "c_cpp" if ext in (".c", ".cpp", ".cxx", ".cc", ".h", ".hpp") else
        "unknown"
    )

    try:
        with open(proxy_path, encoding="utf-8", errors="replace") as fh:
            source = fh.read()
    except OSError as exc:
        return err(f"Could not read proxy_path: {exc}")

    # Find the component in source
    found_name: str | None = None
    found_signature: str = ""

    if lang == "python":
        try:
            tree = ast.parse(source)
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    if component_hint.lower() in node.name.lower():
                        found_name = node.name
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                            args = [a.arg for a in node.args.args]
                            found_signature = f"def {node.name}({', '.join(args)})"
                        else:
                            found_signature = f"class {node.name}"
                        break
        except SyntaxError:
            pass

    if found_name is None:
        # Language-agnostic regex fallback (Fortran subroutine/function, C/C++ function)
        patterns = [
            rf"(?:subroutine|function|void|int|float|double|auto)\s+({re.escape(component_hint)}\w*)\s*\(",
            rf"def\s+({re.escape(component_hint)}\w*)\s*\(",
            rf"({re.escape(component_hint)}\w*)\s*\(",
        ]
        for pat in patterns:
            m = re.search(pat, source, re.IGNORECASE)
            if m:
                found_name = m.group(1)
                found_signature = f"{found_name}(...)"
                break

    if found_name is None:
        found_name = component_hint
        found_signature = f"{component_hint}(...)"

    # Determine output directory
    out_dir = output_path if output_path else os.path.join(scaffolds_dir(), found_name)
    os.makedirs(out_dir, exist_ok=True)

    ref_path  = os.path.join(out_dir, f"{found_name}_ref_harness.py")
    test_path = os.path.join(out_dir, f"{found_name}_test_harness.py")

    abs_proxy = os.path.abspath(proxy_path)

    if harness_type == "kernel":
        ref_content = _kernel_ref_harness(found_name, found_signature, abs_proxy, lang)
        test_content = _kernel_test_harness(found_name, found_signature, ref_path)
    else:
        ref_content = _subsystem_ref_harness(found_name, found_signature, abs_proxy, lang)
        test_content = _subsystem_test_harness(found_name, found_signature, ref_path)

    with open(ref_path, "w", encoding="utf-8") as fh:
        fh.write(ref_content)
    with open(test_path, "w", encoding="utf-8") as fh:
        fh.write(test_content)

    suggested_ref_register = (
        f"proxy_manage(\n"
        f"    op='register',\n"
        f"    name='{found_name}_ref',\n"
        f"    executable_path='{ref_path}',\n"
        f"    run_cmd_template='python3 {{executable}} --n {{n}} --output {{output_file}}',\n"
        f"    output_format='npz',\n"
        f"    confirm=True,\n"
        f")"
    )
    suggested_test_register = (
        f"proxy_manage(\n"
        f"    op='register',\n"
        f"    name='{found_name}_test',\n"
        f"    executable_path='{test_path}',\n"
        f"    run_cmd_template='python3 {{executable}} --n {{n}} --output {{output_file}}',\n"
        f"    output_format='npz',\n"
        f"    confirm=True,\n"
        f")"
    )
    workflow_steps = [
        f"1. Review {ref_path} and {test_path}.",
        f"2. proxy_manage(op='register', name='{found_name}_ref', ...) — register reference harness.",
        f"3. proxy_exec(op='reference', proxy_name='{found_name}_ref', "
        f"reference_name='{found_name}_ref_nN', param_overrides={{'n': <N>}}, "
        f"confirm=True) — seal reference.",
        f"4. Implement new method in {test_path}.",
        f"5. proxy_manage(op='register', name='{found_name}_test', ...) — register test harness.",
        f"6. proxy_manage(op='suite_define', name='{found_name}_suite', cases=[...], confirm=True).",
        f"7. proxy_exec(op='suite', suite_name='{found_name}_suite', confirm=True).",
    ]

    return ok({
        "component":               found_name,
        "language":                lang,
        "source_signature":        found_signature,
        "harness_type":            harness_type,
        "ref_harness_path":        ref_path,
        "test_harness_path":       test_path,
        "suggested_ref_register":  suggested_ref_register,
        "suggested_test_register": suggested_test_register,
        "workflow_steps":          workflow_steps,
        "ref_harness_preview":     ref_content[:800] + ("..." if len(ref_content) > 800 else ""),
        "test_harness_preview":    test_content[:800] + ("..." if len(test_content) > 800 else ""),
    })


# ── template generators ───────────────────────────────────────────────────────

def _metrics_block_footer(found_name: str) -> str:
    return (
        "    elapsed = time.perf_counter() - t0\n"
        "    # Save output field\n"
        "    np.savez(output_file, field=result)\n"
        "    # Emit metrics block\n"
        "    print('PROXY_METRICS_BEGIN')\n"
        "    print(f'time_s={elapsed:.6f}')\n"
        "    print(f'output_file={output_file}')\n"
        "    # Add custom metrics below, e.g.:\n"
        "    # print(f'iterations={proxy_iters}')\n"
        "    print('PROXY_METRICS_END')\n"
    )


def _kernel_ref_harness(name: str, sig: str, proxy_path: str, lang: str) -> str:
    return (
        f'"""Reference harness for component: {name}\n'
        f'Generated by proxy_manage(op=\'scaffold\').\n'
        f'Source: {proxy_path}\n'
        f'Detected signature: {sig}\n'
        f'\n'
        f'This harness wraps the ORIGINAL implementation.\n'
        f'Register it and seal the ground truth with proxy_exec(op=\'reference\').\n'
        f'"""\n'
        f'import argparse\n'
        f'import sys\n'
        f'import time\n'
        f'import numpy as np\n'
        f'\n'
        f'# Add the proxy source directory to the path (Python only)\n'
        f'sys.path.insert(0, {repr(str(os.path.dirname(proxy_path)))})\n'
        f'\n'
        f'# TODO: import the component from the proxy source\n'
        f'# from {os.path.splitext(os.path.basename(proxy_path))[0]} import {name}\n'
        f'\n'
        f'\n'
        f'def main():\n'
        f'    parser = argparse.ArgumentParser()\n'
        f'    parser.add_argument("--n", type=int, required=True,\n'
        f'                        help="Grid resolution parameter")\n'
        f'    parser.add_argument("--output", required=True,\n'
        f'                        help="Path for output .npz field file")\n'
        f'    args = parser.parse_args()\n'
        f'\n'
        f'    n = args.n\n'
        f'    output_file = args.output\n'
        f'\n'
        f'    # TODO: set up inputs and call the original component\n'
        f'    t0 = time.perf_counter()\n'
        f'    result = np.zeros((n, n))  # TODO: replace with real call to {name}(...)\n'
        f'    # result = {name}(n, ...)  # ← call the original here\n'
        f'{_metrics_block_footer(name)}'
        f'\n'
        f'\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )


def _kernel_test_harness(name: str, sig: str, ref_path: str) -> str:
    return (
        f'"""Test harness skeleton for component: {name}\n'
        f'Generated by proxy_manage(op=\'scaffold\').\n'
        f'Reference harness: {ref_path}\n'
        f'\n'
        f'Implement your new method below, then register it with\n'
        f'proxy_manage(op=\'register\') and run it via proxy_exec(op=\'suite\').\n'
        f'"""\n'
        f'import argparse\n'
        f'import time\n'
        f'import numpy as np\n'
        f'\n'
        f'\n'
        f'def {name}_new_method(n: int) -> np.ndarray:\n'
        f'    """TODO: implement new method for {name}.\n'
        f'\n'
        f'    Args:\n'
        f'        n: Grid resolution parameter.\n'
        f'\n'
        f'    Returns:\n'
        f'        result: Output field array comparable to the reference harness.\n'
        f'    """\n'
        f'    raise NotImplementedError("Implement new method here")\n'
        f'\n'
        f'\n'
        f'def main():\n'
        f'    parser = argparse.ArgumentParser()\n'
        f'    parser.add_argument("--n", type=int, required=True)\n'
        f'    parser.add_argument("--output", required=True)\n'
        f'    args = parser.parse_args()\n'
        f'\n'
        f'    n = args.n\n'
        f'    output_file = args.output\n'
        f'\n'
        f'    t0 = time.perf_counter()\n'
        f'    result = {name}_new_method(n)\n'
        f'{_metrics_block_footer(name)}'
        f'\n'
        f'\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )


def _subsystem_ref_harness(name: str, sig: str, proxy_path: str, lang: str) -> str:
    return (
        f'"""Reference subsystem harness for: {name}\n'
        f'Generated by proxy_manage(op=\'scaffold\').\n'
        f'Source: {proxy_path}\n'
        f'\n'
        f'This harness drives the ORIGINAL subsystem via a param file.\n'
        f'Template placeholders used: {{n}}, {{output_file}}, {{param_file}}\n'
        f'\n'
        f'Register with:\n'
        f'  run_cmd_template = "python3 {{{{executable}}}} {{{{param_file}}}}"\n'
        f'  param_file_template = "n={{{{n}}}}\\noutput_file={{{{output_file}}}}\\n"\n'
        f'"""\n'
        f'import sys\n'
        f'import time\n'
        f'import numpy as np\n'
        f'\n'
        f'sys.path.insert(0, {repr(str(os.path.dirname(proxy_path)))})\n'
        f'\n'
        f'# TODO: import the subsystem from the proxy source\n'
        f'# from {os.path.splitext(os.path.basename(proxy_path))[0]} import {name}\n'
        f'\n'
        f'\n'
        f'def _read_params(param_file: str) -> dict:\n'
        f'    params = {{}}\n'
        f'    with open(param_file) as fh:\n'
        f'        for line in fh:\n'
        f'            line = line.strip()\n'
        f'            if "=" in line and not line.startswith("#"):\n'
        f'                k, v = line.split("=", 1)\n'
        f'                params[k.strip()] = v.strip()\n'
        f'    return params\n'
        f'\n'
        f'\n'
        f'def main():\n'
        f'    if len(sys.argv) < 2:\n'
        f'        print("Usage: harness.py <param_file>", file=sys.stderr)\n'
        f'        sys.exit(1)\n'
        f'    params = _read_params(sys.argv[1])\n'
        f'    n = int(params["n"])\n'
        f'    output_file = params["output_file"]\n'
        f'\n'
        f'    # TODO: call the original subsystem here\n'
        f'    t0 = time.perf_counter()\n'
        f'    result = np.zeros((n, n))  # TODO: replace with real call to {name}(...)\n'
        f'{_metrics_block_footer(name)}'
        f'\n'
        f'\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )


def _subsystem_test_harness(name: str, sig: str, ref_path: str) -> str:
    return (
        f'"""Test subsystem harness skeleton for: {name}\n'
        f'Generated by proxy_manage(op=\'scaffold\').\n'
        f'Reference harness: {ref_path}\n'
        f'\n'
        f'Implement your new subsystem below.\n'
        f'"""\n'
        f'import sys\n'
        f'import time\n'
        f'import numpy as np\n'
        f'\n'
        f'\n'
        f'def _read_params(param_file: str) -> dict:\n'
        f'    params = {{}}\n'
        f'    with open(param_file) as fh:\n'
        f'        for line in fh:\n'
        f'            line = line.strip()\n'
        f'            if "=" in line and not line.startswith("#"):\n'
        f'                k, v = line.split("=", 1)\n'
        f'                params[k.strip()] = v.strip()\n'
        f'    return params\n'
        f'\n'
        f'\n'
        f'def {name}_new_method(n: int) -> np.ndarray:\n'
        f'    """TODO: implement new method for subsystem {name}."""\n'
        f'    raise NotImplementedError("Implement new subsystem here")\n'
        f'\n'
        f'\n'
        f'def main():\n'
        f'    if len(sys.argv) < 2:\n'
        f'        print("Usage: harness.py <param_file>", file=sys.stderr)\n'
        f'        sys.exit(1)\n'
        f'    params = _read_params(sys.argv[1])\n'
        f'    n = int(params["n"])\n'
        f'    output_file = params["output_file"]\n'
        f'\n'
        f'    t0 = time.perf_counter()\n'
        f'    result = {name}_new_method(n)\n'
        f'{_metrics_block_footer(name)}'
        f'\n'
        f'\n'
        f'if __name__ == "__main__":\n'
        f'    main()\n'
    )
