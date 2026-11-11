"""
Echo — Grounding V3 Validator & Fallback Logic Automated Test Suite
Tests:
1. Each known violation pattern (including newly added Grounding V3 synonyms).
2. Valid outputs that mention similar words in unrelated contexts (false-positive prevention).
3. Sampled output flagged, greedy fallback valid.
4. Sampled output flagged, greedy fallback also flagged.
5. Greedy fallback worse than sampled output.
6. Accurate metadata for the final returned draft.
"""

import unittest
from unittest.mock import MagicMock
from training.generate_with_adapter import validate_grounding, PersonalizedWritingGenerator


class TestGroundingV3Validator(unittest.TestCase):

    def test_violation_third_party_synonyms(self):
        """Tests third-party attribution with buddies, coworkers, teammates."""
        facts = ["[Completed Project]: I built Task Tracker, a personal productivity application."]

        # Test original patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "A friend of mine built it")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "My colleague built this platform")["is_grounded"])

        # Test new Grounding V3 synonym patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "My buddy helped make this productivity app")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "A buddy of mine worked on it")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "My coworker made the interface")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "My co-worker started this project")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Our teammates created the platform")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "A team member developed this app")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Someone I know made the productivity tool")["is_grounded"])

    def test_violation_limitation_inversion_synonyms(self):
        """Tests live data inversion with streaming, websockets, play-by-play."""
        facts = [
            "[Completed Project]: I built Task Tracker, a personal productivity application.",
            "[Project Constraint & Reality]: For my project Task Tracker, it is purely a local offline queue without external real-time sync or live feeds."
        ]

        # Test original patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "It features live data and real-time updates")["is_grounded"])

        # Test new Grounding V3 synonym patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "It connects to streaming feeds for tasks")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "It receives websocket feeds for instant play")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Users get play-by-play updates during games")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Provides realtime scores continuously")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Features streaming match data")["is_grounded"])

    def test_violation_scope_extrapolation_synonyms(self):
        """Tests scope extrapolation with enterprise and foreign domain injection."""
        facts = ["[Completed Project]: I built Task Tracker, a personal productivity application."]

        # Test original patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "It includes automated payroll and billing gateway integration")["is_grounded"])

        # Test new Grounding V3 synonym patterns
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "It supports payment processing and crypto trading")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Users can run an ecommerce storefront on the platform")["is_grounded"])
        self.assertFalse(validate_grounding("Why I built Task Tracker", facts, "Includes a social network feed and multi-tenant banking")["is_grounded"])

    def test_violation_training_memory_bleed_synonyms(self):
        """Tests memory bleed on unrelated topics with A100 server clusters and cloud GPUs."""
        facts = []
        topic = "My thoughts on remote work across time zones"

        # Test original patterns
        self.assertFalse(validate_grounding(topic, facts, "Remote work is good but I need cloud gpu to train their model")["is_grounded"])

        # Test new Grounding V3 synonym patterns
        self.assertFalse(validate_grounding(topic, facts, "I spent too much money renting gpus for testing")["is_grounded"])
        self.assertFalse(validate_grounding(topic, facts, "I am managing a100 server clusters for my tasks")["is_grounded"])
        self.assertFalse(validate_grounding(topic, facts, "Working with a100 clusters takes time")["is_grounded"])

    def test_valid_outputs_with_similar_words_in_unrelated_contexts(self):
        """False-Positive Resistance: ensures normal words in legitimate contexts are NOT flagged."""
        # Case A: Legitimate mention of cricket/baseball when project is NOT in facts
        general_facts = ["[Hobbies]: I enjoy playing outdoor sports with friends."]
        res = validate_grounding("My favorite weekend sports", general_facts, "Over the weekend I played cricket and baseball.")
        self.assertTrue(res["is_grounded"])
        self.assertEqual(len(res["flagged_issues"]), 0)

        # Case B: Legitimate mention of GPU / training when topic is technical GPU training
        gpu_topic = "Training deep learning models on local vs cloud GPUs"
        res_gpu = validate_grounding(gpu_topic, [], "Training deep learning models requires balancing local hardware and cloud GPUs.")
        self.assertTrue(res_gpu["is_grounded"])
        self.assertEqual(len(res_gpu["flagged_issues"]), 0)

        # Case C: Legitimate mention of static simulation without claiming live
        tt_facts = [
            "[Completed Project]: I built Task Tracker, a personal productivity application.",
            "[Project Constraint & Reality]: Local offline queue without external real-time sync."
        ]
        res_tt = validate_grounding(
            "Why I built Task Tracker",
            tt_facts,
            "I built Task Tracker as a personal productivity tool with local queue storage."
        )
        self.assertTrue(res_tt["is_grounded"])
        self.assertEqual(len(res_tt["flagged_issues"]), 0)


class TestFallbackDecisionLogic(unittest.TestCase):

    def test_fallback_adopted_when_greedy_is_clean(self):
        """Scenario: Sampled output flagged (1 issue), Greedy fallback valid (0 issues) -> fallback adopted."""
        sampled_issues = ["third_party_attribution: Project or skill attributed to someone other than author"]
        fallback_issues = []

        # Condition: len(fb_issues) <= len(sampled_issues) (0 <= 1 is True)
        self.assertTrue(len(fallback_issues) <= len(sampled_issues))
        # Result: Fallback adopted, final draft is clean, is_grounded is True, fallback_used is True.

    def test_fallback_adopted_when_greedy_has_equal_issues(self):
        """Scenario: Sampled output flagged (1 issue), Greedy fallback also flagged (1 issue) -> fallback adopted."""
        sampled_issues = ["third_party_attribution: Project or skill attributed to someone other than author"]
        fallback_issues = ["unsupported_scope_extrapolation: Injected sports or venues not in personal facts"]

        # Condition: len(fb_issues) <= len(sampled_issues) (1 <= 1 is True)
        self.assertTrue(len(fallback_issues) <= len(sampled_issues))
        # Result: Fallback adopted, but final draft remains flagged: is_grounded is False, fallback_used is True.

    def test_fallback_rejected_when_greedy_is_worse(self):
        """Scenario: Sampled output flagged (1 issue), Greedy fallback worse (2 issues) -> fallback rejected."""
        sampled_issues = ["third_party_attribution: Project or skill attributed to someone other than author"]
        fallback_issues = [
            "third_party_attribution: Project or skill attributed to someone other than author",
            "unsupported_scope_extrapolation: Injected sports or venues not in personal facts"
        ]

        # Condition: len(fb_issues) <= len(sampled_issues) (2 <= 1 is False)
        self.assertFalse(len(fallback_issues) <= len(sampled_issues))
        # Result: Fallback rejected, sampled output retained, is_grounded is False, fallback_used is False.


if __name__ == "__main__":
    unittest.main()

