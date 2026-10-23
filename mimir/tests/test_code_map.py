"""The code map of a workspace: Python, C/C++ and Fortran read alike, the edges within a
language and across languages, how sure each is, how files group, and the rules an edge
breaks. Each case is a small tree written to a temporary directory."""

import json
import os
import tempfile
import textwrap
import unittest

from mimir.shared import code_map


def _tree(files: dict[str, str]) -> tempfile.TemporaryDirectory:
    tmp = tempfile.TemporaryDirectory()
    for rel, text in files.items():
        path = os.path.join(tmp.name, rel)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as fh:
            fh.write(textwrap.dedent(text))
    return tmp


def _edges(m: dict) -> set[tuple[str, str, str, str]]:
    """(from, to, kind, sure) for every edge, the shape the assertions read."""
    return {(e["from"], e["to"], e["kind"], e["sure"]) for e in m["edges"]}


def _vias(m: dict) -> dict[tuple[str, str], str]:
    return {(e["from"], e["to"]): e["via"] for e in m["edges"]}


class PythonTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = _tree({
            "pkg/__init__.py": "",
            "pkg/a.py": """\
                from . import b
                from typing import TYPE_CHECKING
                import numpy
                if TYPE_CHECKING:
                    from .c import Thing
                """,
            "pkg/b.py": """\
                from .a import x
                def later():
                    from pkg import c
                    import importlib
                    importlib.import_module(name)
                """,
            "pkg/c.py": "",
        })
        self.m = code_map.scan(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_relative_and_absolute_imports_land_on_their_files(self) -> None:
        edges = _edges(self.m)
        self.assertIn(("pkg/a.py", "pkg/b.py", "import", "exact"), edges)
        self.assertIn(("pkg/b.py", "pkg/a.py", "import", "exact"), edges)
        self.assertIn(("pkg/b.py", "pkg/c.py", "import", "exact"), edges)

    def test_a_type_checking_import_is_no_dependency(self) -> None:
        self.assertNotIn("pkg/c.py", {e["to"] for e in self.m["edges"] if e["from"] == "pkg/a.py"})

    def test_an_import_inside_a_function_is_deferred(self) -> None:
        deferred = {(e["from"], e["to"]) for e in self.m["edges"] if e["deferred"]}
        self.assertEqual(deferred, {("pkg/b.py", "pkg/c.py")})

    def test_third_party_is_external_and_a_computed_import_unresolved(self) -> None:
        self.assertEqual(self.m["components"][0]["external"].get("numpy"), 1)
        self.assertEqual(self.m["unresolved"].get("computed_import"), 1)


class CppTests(unittest.TestCase):
    FILES = {
        "CMakeLists.txt": "add_subdirectory(src)\n",
        "src/core/CMakeLists.txt": "add_library(core INTERFACE)\n",
        "src/core/include/types.h": "#pragma once\n",
        "src/mesh/CMakeLists.txt": """\
            add_library(mesh mesh.cc)
            add_library(proj::mesh ALIAS mesh)
            target_link_libraries(mesh PUBLIC core)
            """,
        "src/mesh/api/mesh.h": '#include "types.h"\n#include "../impl/grid.h"\n',
        "src/mesh/impl/grid.h": "#include <vector>\n",
        "src/mesh/mesh.cc": '#include "api/mesh.h"\n#include "solve.h"\n',
        "src/solver/CMakeLists.txt": "add_library(solver s.cc)\ntarget_link_libraries(solver proj::mesh)\n",
        "src/solver/include/solve.h": '#include "mesh.h"\n',
        "src/solver/s.cc": '#include "solve.h"\n#include "util.h"\n',
        "src/solver/a/util.h": "", "src/solver/b/util.h": "",
    }

    def test_includes_resolve_by_path_then_by_a_unique_basename(self) -> None:
        with _tree(self.FILES) as root:
            edges = _edges(code_map.scan(root))
        self.assertIn(("src/mesh/api/mesh.h", "src/mesh/impl/grid.h", "include", "exact"), edges)
        self.assertIn(("src/solver/s.cc", "src/solver/include/solve.h", "include", "exact"), edges)

    def test_an_ambiguous_name_is_counted_and_a_system_header_external(self) -> None:
        with _tree(self.FILES) as root:
            m = code_map.scan(root)
        self.assertEqual(m["unresolved"]["ambiguous_include"], 1)
        mesh = next(c for c in m["components"] if c["name"] == "mesh")
        self.assertEqual(mesh["external"], {"vector": 1})

    def test_compile_commands_include_dirs_are_searched(self) -> None:
        files = dict(self.FILES)
        files["build/compile_commands.json"] = json.dumps([{
            "directory": "BUILD", "file": "src/solver/s.cc",
            "arguments": ["c++", "-I../src/solver/a", "-c", "s.cc"]}])
        with _tree(files) as root:
            path = os.path.join(root, "build/compile_commands.json")
            with open(path) as fh:
                text = fh.read().replace("BUILD", os.path.join(root, "build"))
            with open(path, "w") as fh:
                fh.write(text)
            edges = _edges(code_map.scan(root))
        self.assertIn(("src/solver/s.cc", "src/solver/a/util.h", "include", "exact"), edges)

    def test_the_build_s_links_and_the_api_rule(self) -> None:
        with _tree(self.FILES) as root:
            m = code_map.scan(root)
        comps = {c["name"]: c for c in m["components"]}
        self.assertEqual(comps["solver"]["declared"], ["src/mesh"])
        breaks = {(e["from"], e["to"]): e["breaks"] for e in m["edges"] if e["breaks"]}
        self.assertEqual(breaks, {
            ("src/mesh/api/mesh.h", "src/mesh/impl/grid.h"): "an api reaches its own implementation",
            ("src/mesh/mesh.cc", "src/solver/include/solve.h"): "mesh does not link solver",
        })
        # The include that breaks the rule does not stack mesh above solver.
        self.assertGreater(comps["solver"]["tier"], comps["mesh"]["tier"])
        self.assertGreater(comps["mesh"]["tier"], comps["core"]["tier"])

    def test_a_cycle_closed_by_a_breaking_include_is_still_a_cycle(self) -> None:
        with _tree({"src/a/CMakeLists.txt": "add_library(a x.cc)\n",
                    "src/b/CMakeLists.txt": "add_library(b y.cc)\ntarget_link_libraries(b a)\n",
                    "src/a/x.h": '#include "y.h"\n', "src/b/y.h": '#include "x.h"\n'}) as root:
            m = code_map.scan(root)
        self.assertEqual([e["breaks"] for e in m["edges"] if e["breaks"]], ["a does not link b"])
        self.assertEqual(m["cycles"], [["src/a", "src/b"]])


class FortranTests(unittest.TestCase):
    FILES = {
        "Makefile": "all:\n\tgfortran -c src/*.f90\n",
        "src/kinds.f90": """\
            module kinds
              use, intrinsic :: iso_fortran_env
              integer, parameter :: dp = kind(1.d0)
            end module kinds
            """,
        "src/solver.F90": """\
            module solver
              use kinds   ! precision
              use mpi
              use grid
              implicit none
              include 'consts.inc'
            #include "interfaces.h90"
            contains
              subroutine step(u) ; real(dp) :: u(:) ; end subroutine step
            end module solver
            """,
        "src/solver_impl.f90": """\
            submodule (solver) solver_impl
            end submodule solver_impl
            """,
        "src/mesh/grid.f90": "module grid\n  use solver\nend module grid\n",
        "src/consts.inc": "",
        "headers/interfaces.h90": "",
        "legacy/old.f": """\
            c     a fixed-form file
                  subroutine old(x)
                  use kinds
                  end
            """,
    }

    def setUp(self) -> None:
        self.tmp = _tree(self.FILES)
        self.m = code_map.scan(self.tmp.name)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_use_submodule_and_include(self) -> None:
        edges = _edges(self.m)
        self.assertIn(("src/solver.F90", "src/kinds.f90", "use", "exact"), edges)
        self.assertIn(("src/solver.F90", "src/consts.inc", "include", "exact"), edges)
        self.assertIn(("src/solver.F90", "headers/interfaces.h90", "include", "exact"), edges)
        self.assertIn(("src/solver_impl.f90", "src/solver.F90", "use", "exact"), edges)
        self.assertIn(("legacy/old.f", "src/kinds.f90", "use", "exact"), edges)

    def test_intrinsic_and_library_modules_are_external(self) -> None:
        external = {k: v for c in self.m["components"] for k, v in c["external"].items()}
        self.assertEqual(external.get("iso_fortran_env"), 1)
        self.assertEqual(external.get("mpi"), 1)

    def test_a_use_cycle_between_modules_is_found(self) -> None:
        self.assertEqual(self.m["cycles"], [["src", "src/mesh"]])

    def test_a_component_of_tests_says_so(self) -> None:
        with _tree({"src/a/x.py": "", "src/b/y.py": "", "tests/test_x.py": ""}) as root:
            m = code_map.scan(root)
        self.assertEqual({c["name"]: c["tests"] for c in m["components"]},
                         {"a": False, "b": False, "tests": True})

    def test_a_tree_without_cmake_groups_by_directory(self) -> None:
        self.assertEqual({c["name"] for c in self.m["components"]}, {"src", "legacy", "headers"})


class BindingTests(unittest.TestCase):
    """One case per direction the plan names, each with how sure it is."""

    FILES = {
        "CMakeLists.txt": """\
            pybind11_add_module(fast cpp/bindings.cpp)
            add_library(cshim SHARED cpp/shim.c)
            install(DIRECTORY python/ DESTINATION mypkg)
            """,
        "Makefile": "wrap:\n\tf90wrap -m wrapped fortran/physics.f90\n",
        "cpp/bindings.cpp": "#include <pybind11/pybind11.h>\nPYBIND11_MODULE(fast, m) {}\n",
        "cpp/shim.c": """\
            #include <Python.h>
            double c_solve(double x) { return x; }
            void run(void) {
                PyImport_ImportModule("tools.helpers");
                f_kernel(1.0);
                legacy_(2.0);
            }
            """,
        "cython/ext.pyx": 'cdef extern from "../cpp/api.h":\n    double c_solve(double)\n',
        "cpp/api.h": "double c_solve(double);\n",
        "fortran/flux.pyf": """\
            python module flux
              interface
                subroutine compute(n)
                end subroutine compute
              end interface
            end python module flux
            """,
        "fortran/compute.f90": "subroutine compute(n)\nend subroutine compute\n",
        "fortran/physics.f90": """\
            module physics
              use forpy_mod
              interface
                function c_solve(x) bind(c, name="c_solve")
                end function
              end interface
            contains
              subroutine kernel(x) bind(c, name="f_kernel")
              end subroutine
              subroutine legacy(x)
              end subroutine
              subroutine call_py()
                ierror = import_py(mod, "tools.helpers")
              end subroutine
            end module physics
            """,
        "tools/__init__.py": "",
        "tools/helpers.py": "",
        "tools/use.py": """\
            import io
            from mypkg import fast
            import flux
            import wrapped
            import ctypes
            lib = ctypes.CDLL("./libnative.so")
            lib.f_kernel(1.0)
            other = ctypes.CDLL(path)
            shim = ctypes.cdll.LoadLibrary("libcshim.so")
            """,
    }

    def setUp(self) -> None:
        self.tmp = _tree(self.FILES)
        self.m = code_map.scan(self.tmp.name)
        self.edges = _edges(self.m)
        self.via = _vias(self.m)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_python_to_an_extension_module_under_the_package_the_build_names(self) -> None:
        self.assertIn(("tools/use.py", "cpp/bindings.cpp", "binding", "exact"), self.edges)
        self.assertEqual(self.via[("tools/use.py", "cpp/bindings.cpp")], "extension module")

    def test_a_standard_library_name_never_matches_an_extension_module(self) -> None:
        with _tree({"m/CMakeLists.txt": "pybind11_add_module(io io.cpp)\n",
                    "m/io.cpp": "PYBIND11_MODULE(io, m) {}\n",
                    "p/use.py": "import io\nfrom scipy import io\n"}) as root:
            self.assertEqual(code_map.scan(root)["edges"], [])

    def test_python_to_fortran_through_f2py_and_f90wrap(self) -> None:
        self.assertIn(("tools/use.py", "fortran/flux.pyf", "binding", "exact"), self.edges)
        self.assertEqual(self.via[("tools/use.py", "fortran/flux.pyf")], "f2py")
        self.assertIn(("fortran/flux.pyf", "fortran/compute.f90", "binding", "exact"), self.edges)
        self.assertIn(("tools/use.py", "fortran/physics.f90", "binding", "exact"), self.edges)
        with _tree({"Makefile": "w:\n\tf90wrap -m wrapped phys.f90\n", "phys.f90": "module phys\nend module\n",
                    "use.py": "import wrapped\n"}) as root:
            m = code_map.scan(root)
        self.assertEqual(_vias(m), {("use.py", "phys.f90"): "f90wrap"})

    def test_cython_to_a_header(self) -> None:
        self.assertIn(("cython/ext.pyx", "cpp/api.h", "binding", "exact"), self.edges)
        self.assertEqual(self.via[("cython/ext.pyx", "cpp/api.h")], "cython")

    def test_ctypes_to_a_bind_c_symbol_and_to_a_library_by_name(self) -> None:
        self.assertIn(("tools/use.py", "fortran/physics.f90", "binding", "exact"), self.edges)
        self.assertIn(("tools/use.py", "cpp/shim.c", "binding", "by_name"), self.edges)
        self.assertEqual(self.m["unresolved"].get("computed_library"), 1)

    def test_c_and_fortran_embed_python(self) -> None:
        self.assertIn(("cpp/shim.c", "tools/helpers.py", "binding", "exact"), self.edges)
        self.assertIn(("fortran/physics.f90", "tools/helpers.py", "binding", "exact"), self.edges)
        self.assertEqual(self.via[("fortran/physics.f90", "tools/helpers.py")], "forpy")

    def test_fortran_and_c_call_each_other(self) -> None:
        self.assertIn(("fortran/physics.f90", "cpp/shim.c", "binding", "exact"), self.edges)
        self.assertIn(("cpp/shim.c", "fortran/physics.f90", "binding", "exact"), self.edges)

    def test_a_trailing_underscore_call_is_by_name_only(self) -> None:
        with _tree({"a.c": "void go(void) { legacy_(1.0); }\n",
                    "b.f90": "subroutine legacy(x)\nend subroutine\n"}) as root:
            edges = _edges(code_map.scan(root))
        self.assertEqual(edges, {("a.c", "b.f90", "binding", "by_name")})


class ScanTests(unittest.TestCase):
    def test_the_map_says_the_root_its_paths_are_relative_to(self) -> None:
        with _tree({"a.py": ""}) as root:
            m = code_map.scan(root)
        self.assertEqual(m["root"], os.path.abspath(root))
        self.assertEqual([f["id"] for f in m["files"]], ["a.py"])

    def test_a_snapshot_reads_through_the_callable_it_is_given(self) -> None:
        texts = {"p/a.py": "from . import b\n", "p/b.py": "", "p/__init__.py": ""}
        m = code_map.scan("/nowhere", read=texts.get, files=texts)
        self.assertEqual(_edges(m), {("p/a.py", "p/b.py", "import", "exact")})

    def test_build_trees_and_hidden_directories_are_not_read(self) -> None:
        with _tree({"src/a.c": "", "build/gen.c": "", ".venv/x.py": "",
                    "out/CMakeCache.txt": "", "out/gen2.c": ""}) as root:
            self.assertEqual([f["id"] for f in code_map.scan(root)["files"]], ["src/a.c"])


if __name__ == "__main__":
    unittest.main()
