"""The per-host hardware digest, kept once per process: blind to occupancy, and a cluster
that did not answer is never reported empty.
"""

import unittest
from pathlib import Path

_SERVERS = Path(__file__).resolve().parents[1] / "servers"

from mimir.servers.hpc import cli, slurm_digest, slurm_nodes
from mimir.tests.support.stubs import stub_attr
from mimir.shared import slurm_site

from mimir.tests.support.scheduler import local_slurm


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

PARTITIONS = ("compute*|up|1-00:00:00|12|64|256000\n"
              "gpu|up|1-00:00:00|4|72|579000\n"
              "debug|up|1:00:00|2|64|256000\n"
              "bigmem|up|1-00:00:00|1|192|768000\n")


ACCESS = (
    "PartitionName=compute AllowAccounts=ALL DenyAccounts=(null) RootOnly=NO State=UP\n"
    "PartitionName=gpu AllowAccounts=gpu-users RootOnly=NO State=UP\n"
    "@@mimir@@\nphys\n@@mimir@@\nalice\n"
    "@@mimir@@\nPrivateData             = jobs,usage\n")


class _Capture:
    """Stands in for the subprocess seam, dispatching on the script it is handed."""

    def __init__(self, scontrol=SCONTROL, sinfo_nodes=SINFO_NODES,
                 partitions=PARTITIONS, access=None, squeue=None):
        self.access = access
        self.squeue = squeue
        self.scontrol = scontrol
        self.sinfo_nodes = sinfo_nodes
        self.partitions = partitions
        self.scripts: list[tuple[str, str]] = []

    def __call__(self, script, host, timeout):
        self.scripts.append((script, host))
        if script.startswith("scontrol show partition"):
            return self.access
        if script.startswith("squeue"):
            return self.squeue
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
        capture = self.capture
        stub_attr(self, cli, "capture", lambda script, timeout, *, host="": capture(script, host, timeout))
        slurm_digest.forget_cluster_digest()
        self.addCleanup(slurm_digest.forget_cluster_digest)
        local_slurm(self)


class DigestTests(_DigestTestCase):
    def test_it_records_node_types_and_partitions(self):
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["scheduler"], "host-a")
        self.assertEqual(digest["source"], "scontrol")
        self.assertEqual({t["arch"] for t in digest["node_types"]},
                         {"x86_64", "aarch64"})
        self.assertEqual(digest["partitions"][0]["partition"], "compute")
        self.assertEqual(digest["degraded"], "")

    def test_each_node_type_names_its_nodes(self):
        """The panel labels a kind by its nodes and reads probe results by node name."""
        types = slurm_digest.cluster_digest("host-a")["node_types"]
        for t in types:
            self.assertTrue(t["example_nodes"])
            self.assertEqual(t["nodes_total"], len(t["example_nodes"]))
        self.assertNotIn("by_state", types[0])

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
        local_slurm(self, False)
        digest = slurm_digest.cluster_digest(slurm_site.LOCAL)
        self.assertEqual(digest["scheduler"], "local")
        self.assertEqual(digest["source"], "")
        self.assertEqual(self.capture.scripts, [])

    def test_a_remote_host_is_probed_whatever_this_machine_has(self):
        local_slurm(self, False)
        self.assertEqual(slurm_digest.cluster_digest("host-a")["source"], "scontrol")


class AccessTests(_DigestTestCase):
    """Only what this user can submit to is described; the rest is counted."""

    def test_a_partition_sinfo_does_not_list_is_not_offered(self):
        """sinfo without --all already hides what the user's groups cannot use."""
        self.capture.partitions = "compute*|up|1-00:00:00|12|64|256000\n"
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual([t["arch"] for t in digest["node_types"]], ["x86_64"])
        self.assertEqual(digest["left_out"]["nodes"], 1)

    def test_a_partition_closed_to_the_user_s_accounts_is_left_out(self):
        self.capture.access = ACCESS
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual({p["partition"] for p in digest["partitions"]},
                         {"compute", "debug", "bigmem"})
        self.assertEqual(digest["left_out"]["partitions"],
                         {"gpu": "not open to your accounts"})
        self.assertEqual(len(digest["node_types"]), 1)

    def test_a_site_hiding_other_users_jobs_is_known(self):
        """With PrivateData=jobs squeue shows this user only their own jobs."""
        self.assertIsNone(slurm_digest.cluster_digest("host-a")["private_jobs"])
        self.capture.access = ACCESS
        self.assertTrue(slurm_digest.cluster_digest("host-a")["private_jobs"])
        self.capture.access = ACCESS.replace("jobs,usage", "none")
        self.assertFalse(slurm_digest.cluster_digest("host-a")["private_jobs"])

    def test_a_node_in_no_partition_is_not_a_compute_node(self):
        self.capture.scontrol = SCONTROL + (
            "NodeName=login1 Arch=x86_64 CPUTot=32 CPUAlloc=0 RealMemory=128000 "
            "Sockets=2 CoresPerSocket=16 ThreadsPerCore=1 Gres=(null) "
            "AvailableFeatures=(null) State=IDLE\n")
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["left_out"]["nodes"], 1)
        self.assertNotIn(32, [t["cpus"] for t in digest["node_types"]])

    def test_a_down_partition_is_left_out(self):
        self.capture.partitions = PARTITIONS.replace("gpu|up", "gpu|down")
        digest = slurm_digest.cluster_digest("host-a")
        self.assertEqual(digest["left_out"]["partitions"], {"gpu": "down"})

    def test_without_a_partition_listing_nothing_is_guessed_away(self):
        self.capture.partitions = None
        self.assertEqual(len(slurm_digest.cluster_digest("host-a")["node_types"]), 2)


class LoadTests(_DigestTestCase):
    def test_the_load_reads_the_nodes_and_the_queue_now(self):
        self.capture.squeue = ("gpu|PENDING|8|1|gres/gpu:2|Resources\n"
                               "gpu|PENDING|4|1|N/A|Dependency\n"
                               "compute|RUNNING|64|1|N/A|None\n")
        load = slurm_digest.cluster_load("host-a", usable={"compute"})
        self.assertEqual([n["node"] for n in load["nodes"]], ["cn001"])
        self.assertEqual(load["queue"]["gpu"]["waiting"], 1)
        self.assertEqual(load["queue"]["gpu"]["waiting_gpus"], 2)
        self.assertEqual(load["queue"]["gpu"]["held"], 1)
        self.assertEqual(load["queue"]["compute"]["running"], 1)

    def test_a_queue_that_did_not_answer_is_unknown_not_empty(self):
        self.assertIsNone(slurm_digest.cluster_load("host-a")["queue"])


class RoomTests(unittest.TestCase):
    def test_out_of_service_states(self):
        for state in ("DOWN", "IDLE+DRAIN", "MIXED+DRAIN", "down*", "idle*", "drng",
                      "RESERVED", "maint", "IDLE+NOT_RESPONDING"):
            self.assertTrue(slurm_nodes.out_of_service(state), state)
        for state in ("IDLE", "MIXED", "ALLOCATED", "idle~", "IDLE+POWERED_DOWN", "mix"):
            self.assertFalse(slurm_nodes.out_of_service(state), state)

    def test_free_cores_with_no_memory_left_are_not_room(self):
        node = {"state": "MIXED", "cpus": 64, "cpus_allocated": 8, "mem_mb": 256000,
                "mem_alloc_mb": 256000, "gpus": 0, "gpus_allocated": None}
        self.assertEqual(slurm_nodes.node_room(node)["cpus_free"], 0)

    def test_allocated_gpus_are_read_from_alloc_tres(self):
        [node] = slurm_nodes.parse_scontrol_nodes(
            "NodeName=g1 CPUTot=64 CPUAlloc=16 RealMemory=512000 AllocMem=64000 "
            "Gres=gpu:a100:4(S:0-1) AllocTRES=cpu=16,mem=64000M,gres/gpu=3 "
            "State=MIXED Partitions=gpu\n")
        room = slurm_nodes.node_room(node)
        self.assertEqual((room["cpus_free"], room["gpus_free"]), (48, 1))

    def test_a_mixed_node_read_from_sinfo_has_unknown_room(self):
        node = {"state": "MIX", "cpus": 64, "cpus_allocated": None, "mem_mb": 1,
                "mem_alloc_mb": None, "gpus": 2, "gpus_allocated": None}
        room = slurm_nodes.node_room(node)
        self.assertIsNone(room["cpus_free"])
        self.assertIsNone(room["gpus_free"])

    def test_gpu_counts(self):
        self.assertEqual(slurm_nodes.gpu_count("gpu:a100:4,gpu:v100:2"), 6)
        self.assertEqual(slurm_nodes.gpu_count("gpu:2"), 2)
        self.assertEqual(slurm_nodes.gpu_count("mps:100"), 0)
        self.assertEqual(slurm_nodes.gpu_label("gpu:a100:4(S:0-1)"), "4× a100")
        self.assertEqual(slurm_nodes.gpu_label("gpu:a100:4,gpu:v100:2"), "4× a100 + 2× v100")
        self.assertEqual(slurm_nodes.gpu_label("mps:100"), "")


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
