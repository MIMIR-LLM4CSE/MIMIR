import unittest

from mimir.client.context.signals import (
    query_has_unnegated_match,
    query_matches_any,
)
from mimir.client.context.signals import (
    query_is_informational,
    query_prefers_existing_file_edits,
    query_prefers_new_file_creation,
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


class QueryHasUnnegatedMatchTest(unittest.TestCase):
    def test_plain_match_is_unnegated(self):
        self.assertTrue(query_has_unnegated_match("create a module", ("create",)))

    def test_english_negation(self):
        self.assertFalse(query_has_unnegated_match("do not create a file", ("create",)))
        self.assertFalse(query_has_unnegated_match("without creating", ("creating",)))

    def test_french_discontinuous_negation(self):
        self.assertFalse(
            query_has_unnegated_match("ne crée pas de nouveau fichier", ("crée",))
        )
        self.assertFalse(query_has_unnegated_match("sans créer de fichier", ("créer",)))

    def test_negation_only_affects_windowed_token(self):
        # Negator far away (beyond the window) must not suppress a later match.
        self.assertTrue(
            query_has_unnegated_match(
                "no problem at all, please create the file", ("create",)
            )
        )

    def test_mixed_negated_and_unnegated(self):
        # One create is negated, another is not -> still an active create intent.
        self.assertTrue(
            query_has_unnegated_match(
                "do not create a test, but create the module", ("create",)
            )
        )


class QueryPredicatesTest(unittest.TestCase):
    def test_edit_intent(self):
        self.assertTrue(query_prefers_existing_file_edits("refactor the parser"))
        self.assertTrue(query_prefers_existing_file_edits("fix the bug in utils"))

    def test_edit_intent_suppressed_by_create(self):
        self.assertFalse(
            query_prefers_existing_file_edits("create and refactor a new module")
        )

    def test_edit_intent_with_negated_create(self):
        # "ne crée pas ... modifie l'existant" -> edit intent, not create.
        query = "ne crée pas de nouveau fichier, modifie l'existant"
        self.assertTrue(query_prefers_existing_file_edits(query))
        self.assertFalse(query_prefers_new_file_creation(query))

    def test_no_substring_false_positive_in_predicates(self):
        # "creative" must not register as a create intent.
        self.assertFalse(query_prefers_new_file_creation("a creative essay"))

    def test_create_intent(self):
        self.assertTrue(query_prefers_new_file_creation("add a new helper"))

    def test_informational_question_is_not_create_intent(self):
        # Regression: "a new machine" is a *question about* the repo, not an order
        # to write anything. Misreading it as create intent made the creation nudge
        # fire after a correct answer and push the model into writing a script.
        query = "what are the pip packages required to install the kb on a new machine?"
        self.assertTrue(query_is_informational(query))
        self.assertFalse(query_prefers_new_file_creation(query))
        self.assertFalse(query_prefers_existing_file_edits(query))

    def test_informational_question_french(self):
        query = "quels packages faut-il pour une nouvelle machine ?"
        self.assertTrue(query_is_informational(query))
        self.assertFalse(query_prefers_new_file_creation(query))

    def test_question_with_action_verb_keeps_intent(self):
        # A question that still orders an action is not informational.
        self.assertFalse(query_is_informational("can you add a test for the parser?"))
        self.assertTrue(query_prefers_new_file_creation("can you add a test for the parser?"))

    def test_weak_create_signal_needs_artifact_noun(self):
        # "new" alone qualifies a non-writable thing -> no create intent...
        self.assertFalse(query_prefers_new_file_creation("deploy this on a new cluster"))
        # ...but promotes to create intent in front of a writable artifact.
        self.assertTrue(query_prefers_new_file_creation("a new script for the cluster"))
        self.assertTrue(query_prefers_new_file_creation("un nouveau fichier de config"))

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
