import unittest

from mimir.client.query_engine.toollist import (
    domains_signaled_by_text,
    inactive_domain_prefixes,
)


class DomainPruningTest(unittest.TestCase):
    def test_pure_prose_prunes_all_domains(self):
        # A non-technical query activates no specialized domain; every prefix is pruned.
        pruned = inactive_domain_prefixes("write a poem about the sea")
        for prefix in ("slurm_", "salloc_", "platform_", "benchmark_", "ft_", "proxy_", "symbolic"):
            self.assertIn(prefix, pruned)

    def test_hpc_query_keeps_platform_and_benchmark(self):
        pruned = inactive_domain_prefixes("optimize the kernel for performance")
        self.assertNotIn("platform_", pruned)
        self.assertNotIn("benchmark_", pruned)

    def test_slurm_query_keeps_cluster_tools(self):
        pruned = inactive_domain_prefixes("submit an sbatch job to the cluster")
        self.assertNotIn("slurm_", pruned)
        self.assertNotIn("salloc_", pruned)
        self.assertNotIn("module_", pruned)

    def test_finetune_query_keeps_ft_tools(self):
        pruned = inactive_domain_prefixes("fine-tune the model with a lora adapter")
        self.assertNotIn("ft_", pruned)

    def test_proxy_query_keeps_proxy_tools(self):
        pruned = inactive_domain_prefixes("build a surrogate proxy model")
        self.assertNotIn("proxy_", pruned)

    def test_symbolic_query_keeps_symbolic_tool(self):
        pruned = inactive_domain_prefixes("integrate this polynomial and simplify")
        self.assertNotIn("symbolic", pruned)

    def test_symbolic_query_keeps_symbolic_tool_for_proof_vocabulary(self):
        # The group listed "derivative"/"integral" but not the proof/derivation
        # register, so a derivation query lost the algebra tools it needs.
        for query in (
            "derive a tighter complexity bound",
            "prove this lemma",
            "state the theorem and its proof",
        ):
            with self.subTest(query=query):
                self.assertNotIn("symbolic", inactive_domain_prefixes(query))

    def test_word_boundary_no_substring_false_activation(self):
        # "compiler" must not activate the mpi/perf domains via substring ("mpi" is
        # not in "compiler"; but "perf" IS a substring of "performance"). Verify a
        # query with only "encore" does not activate the platform domain via "core".
        pruned = inactive_domain_prefixes("sing an encore tonight")
        self.assertIn("platform_", pruned)
        self.assertIn("benchmark_", pruned)

    def test_empty_query_prunes_all(self):
        pruned = inactive_domain_prefixes("")
        self.assertIn("platform_", pruned)
        self.assertIn("symbolic", pruned)

    def test_rearmed_group_is_no_longer_pruned(self):
        query = "clean up the parser"
        self.assertIn("slurm_", inactive_domain_prefixes(query))
        pruned = inactive_domain_prefixes(query, {"slurm_"})
        self.assertNotIn("slurm_", pruned)
        self.assertNotIn("salloc_", pruned)   # whole group, not just the key prefix
        self.assertIn("ft_", pruned)          # unrelated groups stay pruned


class DomainRearmSignalTest(unittest.TestCase):
    """``domains_signaled_by_text`` — recovering a domain the query never signaled."""

    def test_detects_domain_absent_from_the_query(self):
        signaled = domains_signaled_by_text(
            query="run the test suite",
            text="error: sbatch: command not found; no allocation on this login node",
            rearmed=set(),
        )
        self.assertIn("slurm_", signaled)

    def test_domain_already_visible_is_not_reported(self):
        # The query itself activated the group, so there is nothing to unlock.
        signaled = domains_signaled_by_text(
            query="submit an sbatch job",
            text="sbatch: submitted batch job 12345",
            rearmed=set(),
        )
        self.assertNotIn("slurm_", signaled)

    def test_already_rearmed_domain_is_not_reported_again(self):
        signaled = domains_signaled_by_text(
            query="run the test suite",
            text="sbatch: command not found",
            rearmed={"slurm_"},
        )
        self.assertEqual(signaled, set())

    def test_empty_text_signals_nothing(self):
        self.assertEqual(domains_signaled_by_text("anything", "", set()), set())
        self.assertEqual(domains_signaled_by_text("anything", "   ", set()), set())


if __name__ == "__main__":
    unittest.main()
