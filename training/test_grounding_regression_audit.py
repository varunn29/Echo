"""
Echo — Grounding & Generation-Source Attribution Regression Tests
Reproduces the hallucination failure from the user's real generation test,
tests valid paraphrasing, and verifies generation-source attribution.
"""

import unittest
from training.generate_with_adapter import (
    validate_grounding,
    build_grounded_fallback_draft,
)


class TestGroundingRegressionAudit(unittest.TestCase):
    def setUp(self):
        self.match_point_facts = [
            "[Project Architecture & Reality]: Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
            "[Completed Project]: I built Match Point, a betting platform while applying to some iGaming company; it was my first complete project.",
        ]
        self.topic = "Casual social media post about Match Point"

    def test_exact_hallucinated_output_is_rejected(self):
        """The exact hallucination from the user's test must fail grounding validation."""
        hallucinated_draft = (
            "I started by building something simple—something that felt like a real app. "
            "So when we were told there would be no live database or live chat, "
            "I thought 'what if this is how apps actually work?'"
        )

        res = validate_grounding(self.topic, self.match_point_facts, hallucinated_draft)
        self.assertFalse(res["is_grounded"])
        self.assertGreater(len(res["flagged_issues"]), 0)

        issues_str = " ".join(res["flagged_issues"])
        # Must catch missing project subject
        self.assertIn("missing_project_subject", issues_str)
        # Must catch unsupported third-party / collective instruction phrasing ("we were told")
        self.assertIn("third_party_attribution", issues_str)
        # Must catch unsupported live database / live chat infrastructure
        self.assertIn("unsupported_technical_capability", issues_str)

    def test_valid_grounded_natural_paraphrases_pass(self):
        """Valid paraphrased statements must pass grounding validation."""
        valid_draft_1 = (
            "I built Match Point as a betting platform. "
            "Because I couldn't get live match data, it operates as a static simulation without live match connectivity."
        )
        res1 = validate_grounding(self.topic, self.match_point_facts, valid_draft_1)
        self.assertTrue(res1["is_grounded"], f"Valid draft 1 failed: {res1['flagged_issues']}")

        valid_draft_2 = (
            "Recently I built Match Point, a betting platform I developed while applying to an iGaming company. "
            "Since live match data was unavailable, I built a static simulation with simulated match progression."
        )
        res2 = validate_grounding(self.topic, self.match_point_facts, valid_draft_2)
        self.assertTrue(res2["is_grounded"], f"Valid draft 2 failed: {res2['flagged_issues']}")

        valid_draft_3 = (
            "Match Point is a betting platform I built as my first complete project. "
            "It runs entirely on static match data rather than live external updates."
        )
        res3 = validate_grounding(self.topic, self.match_point_facts, valid_draft_3)
        self.assertTrue(res3["is_grounded"], f"Valid draft 3 failed: {res3['flagged_issues']}")

    def test_missing_project_subject_fails(self):
        """Draft that omits Match Point or betting platform fails."""
        draft = (
            "I built an app with static match data and simulated match progression "
            "without live match connectivity."
        )
        res = validate_grounding(self.topic, self.match_point_facts, draft)
        self.assertFalse(res["is_grounded"])
        issues_str = " ".join(res["flagged_issues"])
        self.assertIn("missing_project_subject", issues_str)

    def test_unsupported_live_chat_or_database_fails(self):
        """Draft mentioning live database or live chat fails."""
        draft = (
            "I built Match Point, a betting platform with no live database and no live chat."
        )
        res = validate_grounding(self.topic, self.match_point_facts, draft)
        self.assertFalse(res["is_grounded"])
        issues_str = " ".join(res["flagged_issues"])
        self.assertIn("unsupported_technical_capability", issues_str)

    def test_we_were_told_fails(self):
        """Draft with collective or instructional phrasing fails."""
        draft = (
            "I built Match Point, a betting platform, but we were told to use static match data."
        )
        res = validate_grounding(self.topic, self.match_point_facts, draft)
        self.assertFalse(res["is_grounded"])
        issues_str = " ".join(res["flagged_issues"])
        self.assertIn("third_party_attribution", issues_str)

    def test_deterministic_fallback_uses_only_verified_facts_no_tennis_fans(self):
        """Deterministic fallback for user facts must not inject 'tennis fans'."""
        fallback = build_grounded_fallback_draft(
            self.topic, self.match_point_facts, length_mode="medium"
        )
        self.assertIn("Match Point", fallback)
        self.assertIn("betting platform", fallback)
        self.assertIn("static match data", fallback)
        # Strictly verify 'tennis fans' is not present
        self.assertNotIn("tennis fans", fallback.lower())
        self.assertNotIn("tennis betting", fallback.lower())


if __name__ == "__main__":
    unittest.main()

