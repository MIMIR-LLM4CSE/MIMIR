"""The per-host hardware digest: what a scheduler's machines are, kept once per process.

The digest exists so that choosing a cluster does not cost two ssh round trips per
candidate, and so that the choice can be made on architecture rather than on hope.
These tests pin the two properties that make it safe to keep: it is blind to occupancy,
and a cluster that did not answer is never reported as a cluster with nothing in it.

Run:
    python -m unittest mimir.tests.test_slurm_digest -v
"""

import sys
import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"
for _p in (_SERVERS / "_shared",):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import slurm_digest  # noqa: E402
import slurm_site  # noqa: E402


SCONTROL = (
    "NodeName=cn001 Arch=x86_64 CPUTot=64 CPUAlloc=0 RealMemory=256000 "
    "Sockets=2 CoresPerSocket=32 ThreadsPerCore=1 Gres=(null) "
    "AvailableFeatures=(null) State=IDLE Partitions=compute\n"
    "NodeName=gn001 Arch=aarch64 CPUTot=72 CPUAlloc=0 RealMemory=579000 "
    "Sockets=1 CoresPerSocket=72 ThreadsPerCore=1 Gres=gpu:h100:2(S:0) "
    "AvailableFeatures=gpu State=IDLE Partitions=gpu\n"
)

SINFO_NODES = (
    "cn001|compute*|idle|64|256000|200000|(null)|2|32|1\n"
    "cn001|debug|idle|64|256000|200000|(null)|2|32|1\n"
)

PARTITIONS = "compute*|up|1-00:00:00|12|64|256000\n"


class _Capture:
    """Stands in for the subprocess seam, dispatching on the script it is handed."""

    def __init__(self, scontrol=SCONTROL, sinfo_nodes=SINFO_NODES,
                 partitions=PARTITIONS):
        self.scontrol = scontrol
        self.sinfo_nodes = sinfo_nodes
        self.partitions = partitions
        self.scripts: list[tuple[str, str]] = []

    def __call__(self, script, host, timeout):
        self.scripts.append((script, host))
        if script.startswith("scontrol"):
            return self.scontrol
        if script.startswith("sinfo -N"):
            return self.sinfo_nodes
        if script.startswith("sinfo "):
            return self.partitions
        return None


class _DigestTestCase(unittest.TestCase):
    def setUp(self):
        self.capture = _Capture()
        self._real = slurm_digest._capture
        slurm_digest._capture = self.capture
        self.addCleanup(setattr, slurm_digest, "_capture", self._real)
        slurm_digest.forget_cluster_digest()
        self.addCleanup(slurm_digest.forget_cluster_digest)
        self._had = slurm_site._HAS_SLURM
        slurm_site._HAS_SLURM = True
        self.addCleanup(setattr, slurm_site, "_HAS_SLURM", self._had)


class DigestTests(_DigestTestCase):
    def test_it_records_node_types_and_partitions(self):
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["scheduler"], "host-a")
        self.assertEqual(digest["source"], "scontrol")
        self.assertEqual({t["arch"] for t in digest["node_types"]},
                         {"x86_64", "aarch64"})
        self.assertEqual(digest["partitions"][0]["partition"], "compute")
        self.assertEqual(digest["degraded"], "")

    def test_occupancy_alone_does_not_move_the_signature(self):
        """What makes the digest keepable at all: a busy cluster is not a changed one."""
        before = slurm_digest.cluster_digest("host-a")["signature"]
        self.capture.scontrol = SCONTROL.replace("CPUAlloc=0", "CPUAlloc=48").replace(
            "State=IDLE", "State=MIXED")
        after = slurm_digest.cluster_digest("host-a")["signature"]
        self.assertEqual(before, after)

    def test_new_hardware_does_move_the_signature(self):
        before = slurm_digest.cluster_digest("host-a")["signature"]
        self.capture.scontrol += (
            "NodeName=bm001 Arch=x86_64 CPUTot=192 CPUAlloc=0 RealMemory=768000 "
            "Sockets=2 CoresPerSocket=96 ThreadsPerCore=1 Gres=(null) "
            "AvailableFeatures=(null) State=IDLE Partitions=bigmem\n")
        self.assertNotEqual(before, slurm_digest.cluster_digest("host-a")["signature"])

    def test_sinfo_answers_where_scontrol_is_restricted(self):
        """A site that locks scontrol down is described, with its gaps named.

        The alternative this replaces is an empty node list, which reads as a cluster
        with no machines in it — a worse answer than a partial one that says so.
        """
        self.capture.scontrol = None
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["source"], "sinfo")
        self.assertEqual(len(digest["node_types"]), 1)
        self.assertEqual(digest["node_types"][0]["arch"], "")
        self.assertEqual(sorted(digest["node_types"][0]["partitions"]),
                         ["compute", "debug"])
        self.assertIn("architecture", digest["degraded"])

    def test_a_host_that_said_nothing_is_not_a_host_with_no_nodes(self):
        self.capture.scontrol = None
        self.capture.sinfo_nodes = None
        self.capture.partitions = None
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["source"], "")
        self.assertEqual(digest["node_types"], [])
        self.assertEqual(digest["signature"], "")

    def test_a_machine_without_slurm_is_settled_without_running_anything(self):
        """A workstation is the common case, and it must not pay two login shells.

        ``slurm_is_local`` is the memoised answer to the same question, already paid
        for by the inventory, so the local branch is decided before any probe.
        """
        slurm_site._HAS_SLURM = False
        digest = slurm_digest.cluster_digest(slurm_site.LOCAL)
        self.assertEqual(digest["scheduler"], "local")
        self.assertEqual(digest["source"], "")
        self.assertEqual(self.capture.scripts, [])

    def test_a_remote_host_is_probed_whatever_this_machine_has(self):
        slurm_site._HAS_SLURM = False
        self.assertEqual(slurm_digest.cluster_digest("host-a")["source"], "scontrol")


class MemoTests(_DigestTestCase):
    def test_one_host_is_probed_once(self):
        first = slurm_digest.cached_cluster_digest("host-a")
        count = len(self.capture.scripts)
        second = slurm_digest.cached_cluster_digest("host-a")
        self.assertIs(first, second)
        self.assertEqual(len(self.capture.scripts), count)

    def test_each_host_is_probed_for_itself(self):
        slurm_digest.cached_cluster_digest("host-a")
        slurm_digest.cached_cluster_digest("host-b")
        self.assertEqual({host for _, host in self.capture.scripts},
                         {"host-a", "host-b"})

    def test_a_host_that_did_not_answer_is_asked_again(self):
        """A cluster briefly unreachable must not read as featureless until restart."""
        self.capture.scontrol = None
        self.capture.sinfo_nodes = None
        self.assertEqual(slurm_digest.cached_cluster_digest("host-a")["source"], "")
        self.capture.scontrol = SCONTROL
        self.assertEqual(slurm_digest.cached_cluster_digest("host-a")["source"],
                         "scontrol")

    def test_forgetting_one_host_keeps_the_others(self):
        slurm_digest.cached_cluster_digest("host-a")
        slurm_digest.cached_cluster_digest("host-b")
        slurm_digest.forget_cluster_digest("host-a")
        count = len(self.capture.scripts)
        slurm_digest.cached_cluster_digest("host-b")
        self.assertEqual(len(self.capture.scripts), count)
        slurm_digest.cached_cluster_digest("host-a")
        self.assertGreater(len(self.capture.scripts), count)


if __name__ == "__main__":
    unittest.main()
