"""Tests for the persisted, searchable module catalogue on the platform server.

Three contracts drive most of what is asserted here:

* **Hermetic** — the development machine and CI have no module system at all, so the
  "none" path must be a normal empty answer that writes nothing to disk.
* **Deterministic freshness** — the catalogue rebuilds when the module tree changes
  and at no other time. In particular, cluster *occupancy* must never move the signal.
* **Useful at the floor** — with no descriptions and no embedding backend, a search by
  module name must still return the right modules in the right order.
"""

import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
_SHARED = _ROOT / "mimir" / "servers" / "_shared"
if str(_SHARED) not in sys.path:
    sys.path.insert(0, str(_SHARED))

from mimir.servers._shared import embed, slurm_nodes


def _load_server_platform():
    spec = importlib.util.spec_from_file_location(
        "server_platform_under_test",
        _ROOT / "mimir" / "servers" / "hpc" / "server_platform.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


AVAIL_LMOD = """\
/opt/apps/modulefiles:
cuda/11.8
cuda/12.2(D)
gcc/11.3.0
openmpi/4.1.5

/site/contrib/modulefiles:
hdf5/1.14.0
-------------------------------------------
Use "module spider" to find all possible modules.
"""

WHATIS_LOOP = """\
@@cuda/11.8
cuda/11.8: Description: NVIDIA CUDA toolkit 11.8
@@cuda/12.2
cuda/12.2: Description: NVIDIA CUDA toolkit 12.2
@@gcc/11.3.0
gcc/11.3.0: GNU Compiler Collection
@@openmpi/4.1.5
openmpi/4.1.5: Open MPI message passing library
@@hdf5/1.14.0
hdf5/1.14.0: HDF5 parallel IO library
"""

# Tcl Modules' argument-less `module whatis` dumps the whole available set at once.
WHATIS_FLAT = """\
cuda/11.8            : NVIDIA CUDA toolkit 11.8
cuda/12.2            : NVIDIA CUDA toolkit 12.2
gcc/11.3.0           : GNU Compiler Collection
openmpi/4.1.5        : Open MPI message passing library
hdf5/1.14.0          : HDF5 parallel IO library
"""

SPIDER_SOFTWARE_PAGE = json.dumps([
    {
        "package": "cuda",
        "description": "NVIDIA CUDA toolkit",
        "categories": ["compiler"],
        "keywords": ["gpu", "nvidia"],
        "versions": [
            {"full": "cuda/11.8", "versionName": "11.8", "path": "/opt/cuda/11.8.lua"},
            {"full": "cuda/12.2", "versionName": "12.2", "path": "/opt/cuda/12.2.lua",
             "markedDefault": True},
        ],
    },
    {
        "package": "hdf5",
        "description": "HDF5 parallel IO library",
        "versions": [{"full": "hdf5/1.14.0", "path": "/opt/hdf5.lua"}],
    },
])

SPIDER_MAPPING = json.dumps({
    "cuda": {
        "/opt/cuda/12.2.lua": {
            "fullName": "cuda/12.2",
            "Description": "NVIDIA CUDA toolkit",
            "markedDefault": True,
        },
    },
})

SCONTROL_TWO_TYPES = (
    "NodeName=cn001 Arch=x86_64 CPUTot=64 CPUAlloc=0 RealMemory=256000 "
    "Sockets=2 CoresPerSocket=16 ThreadsPerCore=2 Gres=(null) "
    "AvailableFeatures=(null) State=IDLE Partitions=compute\n"
    "NodeName=gn001 Arch=aarch64 CPUTot=128 CPUAlloc=0 RealMemory=512000 "
    "Sockets=2 CoresPerSocket=32 ThreadsPerCore=2 Gres=gpu:a100:8(S:0-1) "
    "AvailableFeatures=(null) State=IDLE Partitions=gpu\n"
)


class _Shell:
    """Stands in for the module shell, dispatching on what the script asks for."""

    def __init__(self, flavour="lmod", avail=AVAIL_LMOD, whatis_loop=WHATIS_LOOP,
                 whatis_flat=WHATIS_FLAT, spider=None, modulepath=""):
        self.flavour = flavour
        self.avail = avail
        self.whatis_loop = whatis_loop
        self.whatis_flat = whatis_flat
        self.spider = spider
        self.modulepath = modulepath
        self.scripts: list[str] = []

    def __call__(self, script, timeout=10):
        self.scripts.append(script)
        if "echo none" in script:
            return self._ok(self.flavour)
        if "${MODULEPATH:-}" in script:
            return self._ok(self.modulepath)
        if "spider -o" in script or "$SPIDER" in script:
            return self._ok(self.spider or "")
        if "MIMIR_EOF" in script:
            return self._ok(self.whatis_loop)
        if "module whatis" in script:
            return self._ok(self.whatis_flat)
        if "module -t list" in script:
            return self._ok("cuda/12.2\n")
        if "module -t avail" in script:
            return self._ok(self.avail)
        return self._ok("")

    @staticmethod
    def _ok(stdout):
        return {"ok": True, "returncode": 0, "stdout": stdout, "stderr": ""}

    def ran(self, needle):
        return any(needle in s for s in self.scripts)


class CatalogueTestCase(unittest.TestCase):
    """Shared fixture: a fake module tree, a stubbed shell, no background threads."""

    flavour = "lmod"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        root = Path(self._tmp.name)

        # A real directory tree, so the freshness fingerprint stats something real.
        self.tree = root / "modulefiles"
        for family, versions in (("cuda", ("11.8", "12.2")), ("gcc", ("11.3.0",)),
                                 ("openmpi", ("4.1.5",)), ("hdf5", ("1.14.0",))):
            for version in versions:
                (self.tree / family).mkdir(parents=True, exist_ok=True)
                (self.tree / family / version).write_text("#%Module\n")

        self.state = root / "state"
        self.state.mkdir()

        self.mod = _load_server_platform()
        self.mod._MODULES_DIR = str(self.state)
        self.mod._CATALOGUE_FILE = str(self.state / "catalogue-test.json")
        self.mod._EMBEDDINGS_FILE = str(self.state / "embeddings-test.json")

        self.shell = _Shell(flavour=self.flavour, modulepath=str(self.tree))
        self.mod._run_shell = self.shell

        # Background work is captured, not spawned: enrichment runs when a test says so.
        self.pending = []
        self.mod._schedule_background = self.pending.append

        self._reset_module_state()
        embed._reset_availability_cache()
        self.mod._embed.is_available = lambda: False

    def tearDown(self):
        embed._reset_availability_cache()

    def _reset_module_state(self):
        self.mod._module_flavour.cache_clear()
        self.mod._effective_modulepath.cache_clear()
        self.mod._collect_modules.cache_clear()
        self.mod._CATALOGUE_CACHE = None
        self.mod._ENRICHING = False
        self.mod._LAST_BUILD_MONOTONIC = None

    def run_pending(self):
        """Run the enrichment that would have gone to a daemon thread."""
        jobs, self.pending[:] = list(self.pending), []
        for job in jobs:
            job()
        return len(jobs)


class NoModuleSystemTests(CatalogueTestCase):
    flavour = "none"

    def test_search_is_ok_and_empty(self):
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["modules"], [])
        self.assertEqual(result["module_system"], "none")
        self.assertIn("note", result)

    def test_nothing_is_written_to_disk(self):
        self.mod.platform_search("cuda")
        self.assertEqual(os.listdir(self.state), [])

    def test_collect_modules_reports_absence(self):
        info = self.mod._collect_modules()
        self.assertFalse(info["available"])
        self.assertEqual(info["module_system"], "none")

    def test_status_says_not_indexed(self):
        status = self.mod.platform_catalogue_status()
        self.assertFalse(status["indexed"])
        self.assertEqual(status["module_system"], "none")


class AvailParsingTests(CatalogueTestCase):
    def test_headers_defaults_and_noise(self):
        entries = self.mod._parse_avail_terse(AVAIL_LMOD)
        loads = [e["load"] for e in entries]
        self.assertEqual(
            loads, ["cuda/11.8", "cuda/12.2", "gcc/11.3.0", "openmpi/4.1.5", "hdf5/1.14.0"])
        # Section headers set the source root; they never become modules themselves.
        self.assertNotIn("/opt/apps/modulefiles", loads)
        by_load = {e["load"]: e for e in entries}
        self.assertTrue(by_load["cuda/12.2"]["default"])
        self.assertFalse(by_load["cuda/11.8"]["default"])
        self.assertEqual(by_load["cuda/12.2"]["source"], "/opt/apps/modulefiles/cuda/12.2")
        self.assertEqual(by_load["hdf5/1.14.0"]["source"], "/site/contrib/modulefiles/hdf5/1.14.0")
        self.assertEqual(by_load["gcc/11.3.0"]["name"], "gcc")
        self.assertEqual(by_load["gcc/11.3.0"]["version"], "11.3.0")

    def test_loaded_and_default_markers(self):
        entries = self.mod._parse_avail_terse("root:\ncuda/12.2(L,D)\nfoo/1.0(L)\n")
        by_load = {e["load"]: e for e in entries}
        self.assertTrue(by_load["cuda/12.2"]["default"])
        self.assertFalse(by_load["foo/1.0"]["default"])

    def test_versionless_module(self):
        entries = self.mod._parse_avail_terse("root:\ncmake\n")
        self.assertEqual(entries[0]["name"], "cmake")
        self.assertEqual(entries[0]["version"], "")


class TierSelectionTests(CatalogueTestCase):
    def test_first_search_is_tier_c_and_defers_enrichment(self):
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["catalogue"]["tier"], "name")
        self.assertTrue(result["catalogue"]["enriching"])
        # Nothing expensive ran on the request path.
        self.assertFalse(self.shell.ran("spider"))
        self.assertFalse(self.shell.ran("MIMIR_EOF"))
        self.assertEqual(len(self.pending), 1)

    def test_enrichment_fills_descriptions_and_marks_done(self):
        self.mod.platform_search("cuda")
        self.assertEqual(self.run_pending(), 1)
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["catalogue"]["tier"], "whatis")
        self.assertFalse(result["catalogue"]["enriching"])
        top = result["modules"][0]
        self.assertEqual(top["load"], "cuda/12.2")
        self.assertIn("CUDA", top["description"])
        on_disk = json.loads(Path(self.mod._CATALOGUE_FILE).read_text())
        self.assertTrue(on_disk["enrich_done"])

    def test_spider_is_used_when_available(self):
        self.shell.spider = SPIDER_SOFTWARE_PAGE
        self.mod.platform_search("cuda")
        self.run_pending()
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["catalogue"]["tier"], "spider")
        top = result["modules"][0]
        self.assertEqual(top["source"], "/opt/cuda/12.2.lua")
        self.assertEqual(top["category"], "compiler")
        self.assertIn("gpu", top["keywords"])

    def test_spider_mapping_shape_also_parses(self):
        parsed = self.mod._parse_spider_json(SPIDER_MAPPING)
        self.assertIn("cuda/12.2", parsed)
        self.assertTrue(parsed["cuda/12.2"]["default"])

    def test_malformed_spider_degrades_to_whatis(self):
        self.shell.spider = "not json at all {{{"
        self.mod.platform_search("cuda")
        self.run_pending()
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["catalogue"]["tier"], "whatis")
        self.assertIn("CUDA", result["modules"][0]["description"])


class TclModulesTests(CatalogueTestCase):
    flavour = "tmod"

    def test_spider_is_never_invoked(self):
        """`module spider` is Lmod-only; Tcl Environment Modules has no such command."""
        self.mod.platform_search("cuda")
        self.run_pending()
        self.assertFalse(self.shell.ran("spider"))

    def test_bulk_whatis_is_used(self):
        self.mod.platform_search("cuda")
        self.run_pending()
        self.assertTrue(self.shell.ran("module whatis"))
        self.assertFalse(self.shell.ran("MIMIR_EOF"))
        result = self.mod.platform_search("cuda")
        self.assertIn("CUDA", result["modules"][0]["description"])


class FreshnessSignalTests(CatalogueTestCase):
    def _build_count(self):
        return sum(1 for s in self.shell.scripts if "module -t avail" in s)

    def test_unchanged_tree_is_not_rebuilt(self):
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 1)
        self.mod.platform_search("gcc")
        self.assertEqual(self._build_count(), 1)

    def test_disk_catalogue_survives_a_fresh_process(self):
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 1)
        self._reset_module_state()  # as if a new session opened
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 1)

    def test_touching_the_tree_forces_a_rebuild(self):
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 1)
        future = time.time() + 120
        os.utime(self.tree / "cuda", (future, future))
        self._reset_module_state()
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 2)

    def test_changed_modulepath_forces_a_rebuild(self):
        self.mod.platform_search("cuda")
        other = Path(self._tmp.name) / "other"
        other.mkdir()
        self.shell.modulepath = str(other)
        self._reset_module_state()
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 2)

    def test_stale_schema_is_rejected(self):
        self.mod.platform_search("cuda")
        stale = json.loads(Path(self.mod._CATALOGUE_FILE).read_text())
        stale["schema"] = 0
        Path(self.mod._CATALOGUE_FILE).write_text(json.dumps(stale))
        self._reset_module_state()
        self.mod.platform_search("cuda")
        self.assertEqual(self._build_count(), 2)

    def test_enrichment_result_is_discarded_when_the_tree_moved(self):
        self.mod.platform_search("cuda")
        future = time.time() + 120
        os.utime(self.tree / "cuda", (future, future))
        self.mod._effective_modulepath.cache_clear()
        self.run_pending()
        on_disk = json.loads(Path(self.mod._CATALOGUE_FILE).read_text())
        self.assertFalse(on_disk["enrich_done"])
        self.assertEqual(on_disk["tier"], "name")

    def test_completed_partial_enrichment_is_not_retried(self):
        """A Tcl site too large for _WHATIS_MAX stays partial; retrying every session
        would spawn a thread each time for a result that cannot improve."""
        self.shell.whatis_loop = ""  # nothing gets a description
        self.mod.platform_search("cuda")
        self.run_pending()
        self.mod.platform_search("cuda")
        self.assertEqual(self.pending, [])
        on_disk = json.loads(Path(self.mod._CATALOGUE_FILE).read_text())
        self.assertTrue(on_disk["enrich_done"])
        self.assertTrue(on_disk["partial"])

    def test_only_one_enrichment_runs_at_a_time(self):
        self.mod.platform_search("cuda")
        self.mod.platform_search("gcc")
        self.assertEqual(len(self.pending), 1)


class SearchTests(CatalogueTestCase):
    def _indexed(self):
        self.mod.platform_search("cuda")
        self.run_pending()

    def test_name_match_without_descriptions_or_backend(self):
        result = self.mod.platform_search("cuda")
        loads = [m["load"] for m in result["modules"]]
        self.assertEqual(loads, ["cuda/12.2", "cuda/11.8"])  # default first
        self.assertTrue(all(m["match"] == "name" for m in result["modules"]))
        self.assertFalse(os.path.exists(self.mod._EMBEDDINGS_FILE))

    def test_newest_version_first_among_non_defaults(self):
        entries = self.mod._parse_avail_terse("root:\nfoo/1.9\nfoo/1.10\nfoo/1.2\n")
        ordered = [e["load"] for e in self.mod._name_matches(entries, "foo")]
        self.assertEqual(ordered, ["foo/1.10", "foo/1.9", "foo/1.2"])

    def test_exact_name_beats_substring(self):
        entries = self.mod._parse_avail_terse("root:\nmpi/1.0\nopenmpi/4.1.5\n")
        ordered = [e["load"] for e in self.mod._name_matches(entries, "mpi")]
        self.assertEqual(ordered[0], "mpi/1.0")

    def test_capability_search_falls_back_to_lexical(self):
        self._indexed()
        result = self.mod.platform_search("parallel IO library")
        self.assertTrue(result["modules"])
        self.assertEqual(result["modules"][0]["load"], "hdf5/1.14.0")
        self.assertEqual(result["modules"][0]["match"], "lexical")
        self.assertIsNone(result["modules"][0]["score"])

    def test_semantic_path_scores_and_persists_vectors(self):
        self._indexed()
        vocab = ("cuda", "nvidia", "toolkit", "hdf5", "parallel", "io", "mpi", "gnu")

        def vec(text):
            low = text.lower()
            return [1.0 if word in low else 0.0 for word in vocab]

        self.mod._embed.is_available = lambda: True
        self.mod._embed.embed_texts = lambda texts: [vec(t) for t in texts]
        self.mod._embed.embed_one = vec
        self.mod._embed.embed_model_id = lambda: "fake-model"

        result = self.mod.platform_search("parallel IO library")
        top = result["modules"][0]
        self.assertEqual(top["load"], "hdf5/1.14.0")
        self.assertEqual(top["match"], "semantic")
        self.assertIsNotNone(top["score"])
        store = json.loads(Path(self.mod._EMBEDDINGS_FILE).read_text())
        self.assertTrue(store)
        self.assertLessEqual(len(store), self.mod._SEMANTIC_POOL)
        self.assertEqual(next(iter(store.values()))["model"], "fake-model")

    def test_embedding_model_change_revectorises(self):
        self._indexed()
        vocab = ("hdf5", "parallel", "io")
        calls = []

        def vec(text):
            low = text.lower()
            return [1.0 if w in low else 0.0 for w in vocab]

        self.mod._embed.is_available = lambda: True
        self.mod._embed.embed_texts = lambda texts: (calls.append(len(texts)) or
                                                     [vec(t) for t in texts])
        self.mod._embed.embed_one = vec
        self.mod._embed.embed_model_id = lambda: "model-a"
        self.mod.platform_search("parallel IO")
        self.assertEqual(len(calls), 1)

        self.mod._embed.embed_model_id = lambda: "model-b"
        self.mod.platform_search("parallel IO")
        self.assertEqual(len(calls), 2)  # vectors under the old model are not reused
        store = json.loads(Path(self.mod._EMBEDDINGS_FILE).read_text())
        self.assertEqual(next(iter(store.values()))["model"], "model-b")

    def test_limit_is_clamped(self):
        self._indexed()
        self.assertLessEqual(len(self.mod.platform_search("c", limit=999)["modules"]), 50)
        self.assertLessEqual(len(self.mod.platform_search("c", limit=0)["modules"]), 10)

    def test_empty_query_neither_builds_nor_errors(self):
        result = self.mod.platform_search("   ")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["modules"], [])
        self.assertFalse(os.path.exists(self.mod._CATALOGUE_FILE))

    def test_result_entry_shape(self):
        self._indexed()
        entry = self.mod.platform_search("cuda")["modules"][0]
        for field in ("kind", "key", "load", "name", "version", "default",
                      "description", "source", "tier", "score", "match"):
            self.assertIn(field, entry)

    def test_payload_avoids_the_discovery_transition_keys(self):
        """A module lookup must not read as "workspace discovery finished"."""
        self._indexed()
        payload = self.mod.platform_search("cuda")
        for key in ("results", "matches", "files", "entries"):
            self.assertNotIn(key, payload)


class RefreshTests(CatalogueTestCase):
    def _build_count(self):
        return sum(1 for s in self.shell.scripts if "module -t avail" in s)

    def test_refresh_rebuilds_a_fresh_catalogue(self):
        self.mod.platform_search("cuda")
        self.run_pending()
        self.mod._LAST_BUILD_MONOTONIC = time.monotonic() - 120  # past the debounce
        self.mod.platform_search("cuda", refresh=True)
        self.assertEqual(self._build_count(), 2)

    def test_repeat_refresh_is_debounced(self):
        self.mod.platform_search("cuda")
        self.run_pending()
        self.mod._LAST_BUILD_MONOTONIC = time.monotonic() - 120
        self.mod.platform_search("cuda", refresh=True)
        self.run_pending()
        self.mod.platform_search("cuda", refresh=True)  # immediately again
        self.assertEqual(self._build_count(), 2)


class DurabilityTests(CatalogueTestCase):
    def test_no_temporary_file_is_left_behind(self):
        self.mod.platform_search("cuda")
        self.run_pending()
        self.assertFalse([f for f in os.listdir(self.state) if f.endswith(".tmp")])

    def test_read_only_state_dir_still_answers(self):
        os.chmod(self.state, 0o500)
        self.addCleanup(os.chmod, self.state, 0o700)
        result = self.mod.platform_search("cuda")
        self.assertEqual(result["status"], "ok")
        self.assertTrue(result["modules"])

    def test_probe_never_builds_the_catalogue(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("platform_probe must not build the catalogue")

        self.mod._build_tier_c = explode
        info = self.mod._collect_modules()
        self.assertTrue(info["available"])
        self.assertFalse(info["catalogue"]["indexed"])
        self.assertEqual(info["loaded"], ["cuda/12.2"])
        self.assertNotIn("sample", info)

    def test_effective_modulepath_is_memoised(self):
        for _ in range(3):
            self.mod._effective_modulepath()
        probes = [s for s in self.shell.scripts if "${MODULEPATH:-}" in s]
        self.assertEqual(len(probes), 1)


class DigestTests(CatalogueTestCase):
    def setUp(self):
        super().setUp()
        self.scontrol = SCONTROL_TWO_TYPES
        self.mod._cmd_exists = lambda name: name in ("scontrol", "sinfo")
        self.mod._collect_toolchains = lambda: {"gcc": "gcc (GCC) 11.3.0"}
        self.mod._collect_cpu = lambda: {"arch": "x86_64", "logical_cpus": 8}
        self.mod._collect_memory = lambda: {"total_gb": 32.0}
        self.mod._run = self._run

    def _run(self, cmd, timeout=8):
        if cmd[:3] == ["scontrol", "show", "node"]:
            return {"ok": True, "returncode": 0, "stdout": self.scontrol, "stderr": ""}
        if cmd[0] == "sinfo":
            return {"ok": True, "returncode": 0,
                    "stdout": "compute|up|1-00:00:00|64|64|256000\n", "stderr": ""}
        return {"ok": False, "returncode": 1, "stdout": "", "stderr": "no"}

    def test_digest_records_node_types_and_partitions(self):
        digest = self.mod._build_digest()
        self.assertEqual(len(digest["node_types"]), 2)
        arches = {t["arch"] for t in digest["node_types"]}
        self.assertEqual(arches, {"x86_64", "aarch64"})
        self.assertEqual(digest["partitions"][0]["partition"], "compute")
        self.assertEqual(digest["host"]["arch"], "x86_64")
        self.assertIn("gcc", digest["toolchains"])

    def test_occupancy_alone_does_not_move_the_signature(self):
        """The whole point of the stable projection: a busy cluster must not look
        like a changed cluster, or the fingerprint would churn every few seconds."""
        before = self.mod._build_digest()["signature"]
        self.scontrol = self.scontrol.replace("CPUAlloc=0", "CPUAlloc=48").replace(
            "State=IDLE", "State=MIXED")
        after = self.mod._build_digest()["signature"]
        self.assertEqual(before, after)

    def test_new_hardware_does_move_the_signature(self):
        before = self.mod._build_digest()["signature"]
        self.scontrol += (
            "NodeName=cn002 Arch=x86_64 CPUTot=192 CPUAlloc=0 RealMemory=768000 "
            "Sockets=2 CoresPerSocket=48 ThreadsPerCore=2 Gres=(null) "
            "AvailableFeatures=(null) State=IDLE Partitions=bigmem\n"
        )
        self.assertNotEqual(before, self.mod._build_digest()["signature"])

    def test_digest_is_empty_without_slurm(self):
        self.mod._cmd_exists = lambda name: False
        digest = self.mod._build_digest()
        self.assertEqual(digest["node_types"], [])
        self.assertEqual(digest["partitions"], [])
        self.assertEqual(digest["signature"], "")

    def test_digest_reaches_the_catalogue_and_the_status(self):
        self.mod.platform_search("cuda")
        self.run_pending()
        status = self.mod.platform_catalogue_status()
        self.assertEqual(len(status["digest"]["node_types"]), 2)
        self.assertTrue(status["signal_fresh"])


class SharedSlurmNodesTests(unittest.TestCase):
    """The parser moved to _shared so two servers could use it; it must be unchanged."""

    def test_parse_and_aggregate_round_trip(self):
        nodes = slurm_nodes.parse_scontrol_nodes(SCONTROL_TWO_TYPES)
        self.assertEqual([n["node"] for n in nodes], ["cn001", "gn001"])
        self.assertEqual(nodes[1]["gres"], "gpu:a100:8")
        self.assertEqual(nodes[0]["features"], "")
        types = slurm_nodes.aggregate_node_types(nodes)
        self.assertEqual(len(types), 2)
        self.assertEqual(sum(t["nodes_total"] for t in types), 2)

    def test_stable_types_drops_occupancy(self):
        types = slurm_nodes.aggregate_node_types(
            slurm_nodes.parse_scontrol_nodes(SCONTROL_TWO_TYPES))
        for stable in slurm_nodes.stable_types(types):
            self.assertNotIn("by_state", stable)
            self.assertNotIn("cpus_free_total", stable)
            self.assertNotIn("example_nodes", stable)


if __name__ == "__main__":
    unittest.main()
