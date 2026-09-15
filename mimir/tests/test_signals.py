import unittest

from mimir.client.context.signals import (
    query_matches_any,
    query_requires_repo_discovery,
)


class QueryMatchesAnyTest(unittest.TestCase):
    def test_matches_whole_word(self):
        self.assertTrue(query_matches_any("please create a file", ("create",)))

    def test_no_substring_false_positive(self):
        # "create" must not fire inside "creative", nor "add" inside "address".
        self.assertFalse(query_matches_any("creative writing task", ("create",)))
        self.assertFalse(query_matches_any("update the address book", ("add",)))
        self.assertFalse(query_matches_any("sing encore", ("core",)))

    def test_case_insensitive(self):
        self.assertTrue(query_matches_any("REFACTOR the module", ("refactor",)))

    def test_multiword_phrase(self):
        self.assertTrue(query_matches_any("please speed up the loop", ("speed up",)))
        self.assertFalse(query_matches_any("speeding upstream", ("speed up",)))

    def test_accented_french_boundaries(self):
        self.assertTrue(query_matches_any("améliore le code", ("améliore",)))
        self.assertTrue(query_matches_any("accélère ce noyau", ("accélère",)))

    def test_empty_inputs(self):
        self.assertFalse(query_matches_any("", ("create",)))
        self.assertFalse(query_matches_any("create", ()))


class QueryRequiresRepoDiscoveryTest(unittest.TestCase):
    """The one surviving query predicate: a coarse exit filter for plan-mode explore.

    The edit/create/informational classifiers this class used to cover were removed
    with the nudge conditions that read them — a keyword over a natural-language
    request guesses at intent, and a nudge has to rest on a fact.
    """

    def test_discovery_intent(self):
        self.assertTrue(query_requires_repo_discovery("analyze the repository"))
        self.assertTrue(query_requires_repo_discovery("optimize the kernel"))

    def test_discovery_not_triggered_by_pure_theory(self):
        # Pure-theory terms are intentionally excluded from discovery signals.
        self.assertFalse(query_requires_repo_discovery("prove this theorem"))

    def test_discovery_covers_french_explanatory_queries(self):
        # The discovery vocabulary used to be almost English-only, so a French
        # session skipped the plan-mode explore phase on purely explanatory questions.
        self.assertTrue(query_requires_repo_discovery("explique moi ce module"))
        self.assertTrue(query_requires_repo_discovery("montre moi la structure du projet"))
        self.assertTrue(query_requires_repo_discovery("a quoi sert cette classe ?"))

    def test_discovery_still_excludes_chit_chat(self):
        # The predicate is an exit filter: broad on repo-touching work, but it must
        # keep excluding theory and conversation.
        self.assertFalse(query_requires_repo_discovery("merci beaucoup"))
        self.assertFalse(query_requires_repo_discovery("hello there"))


if __name__ == "__main__":
    unittest.main()
