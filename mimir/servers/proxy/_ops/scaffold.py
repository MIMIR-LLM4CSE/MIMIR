"""Component-scaffold op: generate reference + test harness files for a component."""

from __future__ import annotations

import ast
import os
import re

from _ops import _with_next, err, ok
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

    ref_content  = _ref_harness(found_name, found_signature, abs_proxy, harness_type)
    test_content = _test_harness(found_name, found_signature, ref_path, harness_type)

    with open(ref_path, "w", encoding="utf-8") as fh:
        fh.write(ref_content)
    with open(test_path, "w", encoding="utf-8") as fh:
        fh.write(test_content)

    # Kernel harnesses take CLI flags; subsystem harnesses read a param file.
    def _suggest_register(reg_name: str, exe_path: str) -> str:
        if harness_type == "kernel":
            cmd_line = ("    run_cmd_template='python3 {executable} "
                        "--n {n} --output {output_file}',\n")
            param_lines = ""
        else:
            cmd_line = "    run_cmd_template='python3 {executable} {param_file}',\n"
            param_lines = ("    param_file_template='n={n}\\noutput_file="
                           "{output_file}\\n',\n")
        return (
            "proxy_manage(\n"
            "    op='register',\n"
            f"    name='{reg_name}',\n"
            f"    executable_path='{exe_path}',\n"
            f"{cmd_line}"
            f"{param_lines}"
            "    output_format='npz',\n"
            "    confirm=True,\n"
            ")"
        )

    suggested_ref_register  = _suggest_register(f"{found_name}_ref", ref_path)
    suggested_test_register = _suggest_register(f"{found_name}_test", test_path)
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

    return ok(_with_next({
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
    }, f"review {ref_path}, then run the proxy_manage(op='register') call in "
       "suggested_ref_register to register the reference harness."))


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


_PARAM_READER = (
    'def _read_params(param_file: str) -> dict:\n'
    '    params = {}\n'
    '    with open(param_file) as fh:\n'
    '        for line in fh:\n'
    '            line = line.strip()\n'
    '            if "=" in line and not line.startswith("#"):\n'
    '                k, v = line.split("=", 1)\n'
    '                params[k.strip()] = v.strip()\n'
    '    return params\n'
    '\n'
    '\n'
)

# Kernel harnesses take CLI flags; subsystem harnesses read a param file. Both
# must end up with `n` and `output_file` bound before the timed section.
_KERNEL_INPUTS = (
    'def main():\n'
    '    parser = argparse.ArgumentParser()\n'
    '    parser.add_argument("--n", type=int, required=True,\n'
    '                        help="Grid resolution parameter")\n'
    '    parser.add_argument("--output", required=True,\n'
    '                        help="Path for output .npz field file")\n'
    '    args = parser.parse_args()\n'
    '\n'
    '    n = args.n\n'
    '    output_file = args.output\n'
)
_SUBSYSTEM_INPUTS = (
    'def main():\n'
    '    if len(sys.argv) < 2:\n'
    '        print("Usage: harness.py <param_file>", file=sys.stderr)\n'
    '        sys.exit(1)\n'
    '    params = _read_params(sys.argv[1])\n'
    '    n = int(params["n"])\n'
    '    output_file = params["output_file"]\n'
)


def _harness(name: str, harness_type: str, docstring: str,
             imports: list[str], preamble: str, compute: str) -> str:
    """Assemble one harness: docstring, imports, preamble, main(), metrics block."""
    inputs = _KERNEL_INPUTS if harness_type == "kernel" else _SUBSYSTEM_INPUTS
    reader = "" if harness_type == "kernel" else _PARAM_READER
    return (
        f'"""{docstring}"""\n'
        + "".join(f"import {m}\n" for m in imports)
        + "\n"
        + preamble
        + reader
        + inputs
        + "\n"
        + compute
        + _metrics_block_footer(name)
        + '\n\nif __name__ == "__main__":\n    main()\n'
    )


def _ref_preamble(name: str, proxy_path: str) -> str:
    module = os.path.splitext(os.path.basename(proxy_path))[0]
    return (
        f'sys.path.insert(0, {os.path.dirname(proxy_path)!r})\n'
        f'\n'
        f'# TODO: import the component from the proxy source\n'
        f'# from {module} import {name}\n'
        f'\n'
        f'\n'
    )


def _ref_harness(name: str, sig: str, proxy_path: str, harness_type: str) -> str:
    kind = "component" if harness_type == "kernel" else "subsystem"
    doc = (f'Reference harness for {kind}: {name}\n'
           f"Generated by proxy_manage(op='scaffold').\n"
           f'Source: {proxy_path}\n'
           f'Detected signature: {sig}\n'
           f'\n'
           f'This harness wraps the ORIGINAL implementation; seal the ground truth\n'
           f"with proxy_exec(op='reference') once it calls into {name}.\n")
    imports = (["argparse", "sys", "time", "numpy as np"] if harness_type == "kernel"
               else ["sys", "time", "numpy as np"])
    compute = (f'    # TODO: set up inputs and call the original {kind}\n'
               f'    t0 = time.perf_counter()\n'
               f'    result = np.zeros((n, n))  # TODO: replace with real call\n'
               f'    # result = {name}(n, ...)  # \u2190 call the original here\n')
    return _harness(name, harness_type, doc, imports,
                    _ref_preamble(name, proxy_path), compute)


def _test_harness(name: str, sig: str, ref_path: str, harness_type: str) -> str:
    kind = "component" if harness_type == "kernel" else "subsystem"
    doc = (f'Test harness skeleton for {kind}: {name}\n'
           f"Generated by proxy_manage(op='scaffold').\n"
           f'Reference harness: {ref_path}\n'
           f'\n'
           f'Implement your new method below, then register it with\n'
           f"proxy_manage(op='register') and run it via proxy_exec(op='suite').\n")
    imports = (["argparse", "time", "numpy as np"] if harness_type == "kernel"
               else ["sys", "time", "numpy as np"])
    preamble = (
        f'def {name}_new_method(n: int) -> np.ndarray:\n'
        f'    """TODO: implement the new method for {kind} {name}.\n'
        f'\n'
        f'    Args:\n'
        f'        n: Grid resolution parameter.\n'
        f'\n'
        f'    Returns:\n'
        f'        Output field array comparable to the reference harness.\n'
        f'    """\n'
        f'    raise NotImplementedError("Implement the new method here")\n'
        f'\n'
        f'\n'
    )
    compute = (f'    t0 = time.perf_counter()\n'
               f'    result = {name}_new_method(n)\n')
    return _harness(name, harness_type, doc, imports, preamble, compute)
