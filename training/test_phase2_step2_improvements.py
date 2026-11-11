"""
Unit tests for Phase 2, Step 2 & 2B:
- Length-Aware Generation (token budgets, prompt instructions, complete thoughts).
- Zero-Fact Personal Topic Safety (autobiographical detection, deterministic safe responses, zero fabrication).
- Knowledge Fidelity and Limitation Preservation (Match Point static data, IndexedDB preference, no reversal, no leaks).
- Complete-Thought Handling (syntax validation, conjunction trimming, preposition trimming, special tokens).
"""

import unittest
from training.generate_with_adapter import (
    to_affirmative_constraint,
    format_affirmative_facts,
    is_autobiographical_topic,
    is_complete_thought,
    ensure_complete_thoughts,
    build_zero_fact_autobiographical_response,
    build_grounded_fallback_draft,
    validate_grounding,
    PersonalizedWritingGenerator,
)


class TestAffirmativeKnowledgeFormatting(unittest.TestCase):
    """Tests for converting negative project constraints into affirmative operational realities."""

    def test_static_data_affirmative_conversion(self):
        raw = "Match Point does not have live match data or real-time score updates, only static data."
        formatted = to_affirmative_constraint(raw)
        self.assertIn("Match Point currently uses static match data and simulated match progression", formatted)
        self.assertIn("It does not have live match connectivity", formatted)

    def test_simulated_match_affirmative_conversion(self):
        raw = "The simulator cannot connect to real bookmakers or fetch live odds; it uses simulated odds."
        formatted = to_affirmative_constraint(raw)
        self.assertIn("Match Point currently uses static match data and simulated match progression", formatted)
        self.assertIn("It does not have live match connectivity", formatted)

    def test_local_offline_queue_affirmative_conversion(self):
        raw = "Task Tracker could not rely on continuous internet connectivity, so built a local offline queue that syncs when reconnected."
        formatted = to_affirmative_constraint(raw)
        self.assertIn("Task Tracker currently operates as an offline sync queue", formatted)
        self.assertIn("does not have real-time websocket access or live external sync", formatted)

    def test_format_affirmative_facts_list(self):
        facts = [
            "Learned: HTML, CSS, JavaScript, React, TypeScript",
            "Project Constraint: Match Point does not have live match data, it uses static tables.",
        ]
        formatted = format_affirmative_facts(facts)
        self.assertEqual(len(formatted), 2)
        self.assertEqual(formatted[0], facts[0])
        self.assertIn("Match Point currently uses static match data", formatted[1])


class TestAutobiographicalTopicClassification(unittest.TestCase):
    """Tests for distinguishing private autobiographical queries from general conceptual topics."""

    def test_autobiographical_topics(self):
        # Explicit requirements from Step 2B
        self.assertTrue(is_autobiographical_topic("My experience with pets"))
        self.assertTrue(is_autobiographical_topic("My morning routine"))
        self.assertTrue(is_autobiographical_topic("How I spend my mornings"))
        self.assertTrue(is_autobiographical_topic("My daily morning routine and coffee habits"))
        self.assertTrue(is_autobiographical_topic("My experience with owning pets"))
        self.assertTrue(is_autobiographical_topic("My pet dog"))
        self.assertTrue(is_autobiographical_topic("My childhood memories"))
        self.assertTrue(is_autobiographical_topic("Where I grew up and my hometown"))

    def test_general_conceptual_topics_remain_unaffected(self):
        # Explicit requirements from Step 2B: Must NOT be classified as autobiographical
        self.assertFalse(is_autobiographical_topic("Why do people keep pets?"))
        self.assertFalse(is_autobiographical_topic("Why are morning routines useful?"))
        self.assertFalse(is_autobiographical_topic("Benefits of morning routines"))
        self.assertFalse(is_autobiographical_topic("The history of domestic pets"))

        # Technical philosophies and work reflections must NEVER be blocked
        self.assertFalse(is_autobiographical_topic("Why I prefer learning by building projects rather than just reading documentation"))
        self.assertFalse(is_autobiographical_topic("My thoughts on remote work and collaborating across different time zones"))
        self.assertFalse(is_autobiographical_topic("Why writing automated tests for every feature saves time in the long run"))
        self.assertFalse(is_autobiographical_topic("IndexedDB"))
        self.assertFalse(is_autobiographical_topic("Why user chat data should stay stored locally in the browser's IndexedDB instead of being uploaded to cloud databases for training"))


class TestDeterministicZeroFactSafeResponse(unittest.TestCase):
    """Tests for deterministic safe responses when zero facts exist for autobiographical topics."""

    def test_morning_routine_safe_response_all_modes(self):
        topic = "My daily morning routine and coffee habits"
        for mode in ["short", "medium", "long"]:
            resp = build_zero_fact_autobiographical_response(topic, length_mode=mode)
            self.assertIn("haven't recorded any personal notes", resp)
            # Must NOT invent fictional personal hours, wake-up times, or remote work claims
            self.assertNotIn("50% of my time", resp)
            self.assertNotIn("wake up at", resp.lower())
            self.assertNotIn("remotely from home", resp)
            is_comp, _ = is_complete_thought(resp)
            self.assertTrue(is_comp, f"Response in {mode} mode must be a complete thought")

    def test_pet_ownership_safe_response_all_modes(self):
        topic = "My experience with owning pets"
        for mode in ["short", "medium", "long"]:
            resp = build_zero_fact_autobiographical_response(topic, length_mode=mode)
            self.assertIn("haven't recorded any personal experiences", resp)
            # Must NOT claim personal pets, fictional dogs/cats, or private activities
            self.assertNotIn("my dog", resp.lower())
            self.assertNotIn("my cat", resp.lower())
            self.assertNotIn("pet-related activities", resp)
            is_comp, _ = is_complete_thought(resp)
            self.assertTrue(is_comp, f"Response in {mode} mode must be a complete thought")


class TestKnowledgeFidelityAndLimitationPreservation(unittest.TestCase):
    """Tests for preserving project limitations, opinions, and preventing reversals/leaks."""

    def test_match_point_limitation_inversion_flagged(self):
        """A project limitation must not become a claimed capability."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        # Claiming live match feeds or real-time odds
        bad_draft = "I built Match Point as a betting platform with live match feeds and real-time score updates."
        res = validate_grounding(topic, facts, bad_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("limitation_inversion" in i for i in res["flagged_issues"]))

    def test_match_point_limitation_omission_flagged(self):
        """When discussing Match Point, the static-data limitation must not silently disappear."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        # Omitting static/simulated limitation entirely
        omitted_draft = "I built Match Point as a betting platform for tennis fans to place predictions easily."
        res = validate_grounding(topic, facts, omitted_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("limitation_omission" in i for i in res["flagged_issues"]))

    def test_match_point_limitation_preserved_valid(self):
        """Preserving static match data and simulated progression passes validation."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        good_draft = "I built Match Point as a betting platform for tennis fans using static match data and simulated match progression without live connectivity."
        res = validate_grounding(topic, facts, good_draft)
        self.assertTrue(res["is_grounded"])
        self.assertEqual(len(res["flagged_issues"]), 0)

    def test_indexeddb_opinion_reversal_flagged(self):
        """An author's stated preference for local IndexedDB storage must not be reversed."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in IndexedDB"

        # Contradicting/reversing author's stated opinion
        reversed_draft = "In my opinion, why would we store user chat history in our local database when we can just use online AI models?"
        res = validate_grounding(topic, facts, reversed_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("opinion_reversal" in i for i in res["flagged_issues"]))

        # Suggesting uploading to cloud servers
        cloud_draft = "We should upload user chat data to cloud servers like AWS to centralize training."
        res_cloud = validate_grounding(topic, facts, cloud_draft)
        self.assertFalse(res_cloud["is_grounded"])
        self.assertTrue(any("opinion_reversal" in i for i in res_cloud["flagged_issues"]))

    def test_indexeddb_preference_preserved_valid(self):
        """Preserving local storage in IndexedDB passes validation."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in IndexedDB"

        good_draft = "My perspective on IndexedDB is that user chat data should stay stored locally in the browser so users can train their own models privately."
        res = validate_grounding(topic, facts, good_draft)
        self.assertTrue(res["is_grounded"])
        self.assertEqual(len(res["flagged_issues"]), 0)

    def test_unrelated_project_facts_must_not_leak(self):
        """Unrelated project facts (e.g. Task Tracker offline queue) must not leak into Match Point."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        leaked_draft = "I built Match Point using static match data, and I also built Task Tracker with an offline sync queue."
        res = validate_grounding(topic, facts, leaked_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("unrelated_project_leak" in i for i in res["flagged_issues"]))

    def test_grounded_fallback_synthesis(self):
        """Fallback synthesis produces perfectly grounded drafts across length modes."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        for mode in ["short", "medium", "long"]:
            draft = build_grounded_fallback_draft(topic, facts, length_mode=mode)
            res = validate_grounding(topic, facts, draft)
            self.assertTrue(res["is_grounded"], f"Fallback in {mode} mode must be fully grounded")
            self.assertIn("static match data", draft)
            self.assertIn("simulated match progression", draft)


class TestCompleteThoughtHandling(unittest.TestCase):
    """Tests for complete thought validation and robust sentence trimming."""

    def test_complete_sentence(self):
        text = "I built Task Tracker to manage productivity offline. It stores queued updates locally."
        is_comp, reason = is_complete_thought(text)
        self.assertTrue(is_comp)
        self.assertIsNone(reason)

    def test_incomplete_sentence_no_terminal_punctuation(self):
        text = "I built Task Tracker to manage productivity offline and I"
        is_comp, reason = is_complete_thought(text)
        self.assertFalse(is_comp)
        self.assertEqual(reason, "missing_terminal_punctuation")

    def test_output_ending_with_conjunction(self):
        # Ends with conjunction before punctuation
        text = "I built the application because."
        is_comp, reason = is_complete_thought(text)
        self.assertFalse(is_comp)
        self.assertEqual(reason, "dangling_conjunction_because")

        text_and = "I learned React, and."
        is_comp_and, reason_and = is_complete_thought(text_and)
        self.assertFalse(is_comp_and)
        self.assertEqual(reason_and, "dangling_conjunction_and")

    def test_output_ending_with_unfinished_clause_preposition(self):
        # Ends with preposition before punctuation
        text = "This leaves our users with."
        is_comp, reason = is_complete_thought(text)
        self.assertFalse(is_comp)
        self.assertEqual(reason, "dangling_preposition_with")

    def test_output_containing_leaked_special_tokens(self):
        text = "<think>Some reasoning</think> I like building software. <|im_end|>"
        is_comp, reason = is_complete_thought(text)
        self.assertFalse(is_comp)
        self.assertEqual(reason, "leaked_special_tokens")

        # ensure_complete_thoughts must strip them
        cleaned = ensure_complete_thoughts(text)
        self.assertEqual(cleaned, "I like building software.")
        is_clean_comp, _ = is_complete_thought(cleaned)
        self.assertTrue(is_clean_comp)

    def test_ensure_complete_thoughts_trims_dangling_fragment_cleanly(self):
        incomplete = (
            "I have been learning React and TypeScript for several months. "
            "These tools help me build reliable applications because they provide"
        )
        cleaned = ensure_complete_thoughts(incomplete)
        self.assertEqual(
            cleaned,
            "I have been learning React and TypeScript for several months.",
        )
        is_comp, _ = is_complete_thought(cleaned)
        self.assertTrue(is_comp)

    def test_ensure_complete_thoughts_fixes_unclosed_quotes(self):
        text = 'Recently I built an app called "Match Point which is a betting platform for tennis fans.'
        cleaned = ensure_complete_thoughts(text)
        self.assertTrue(cleaned.endswith('"') or cleaned.endswith('."'))
        is_comp, _ = is_complete_thought(cleaned)
        self.assertTrue(is_comp)


class TestUnsupportedClaimsAndStrictFidelity(unittest.TestCase):
    """Step 2C regression tests for closing unsupported-claim gaps and verifying conservative grounding."""

    def test_opinion_preserved_but_invents_reasoning_flagged(self):
        """Preserves author's stance on local storage, but invents reasoning and user motivations."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in the browser's IndexedDB"

        # Preserves local storage opinion, but hallucinates OpenAI APIs and richer insights reasoning
        hallucinated_draft = (
            "I believe chat data should stay stored in local IndexedDB so users can get richer insights "
            "into what people want when they ask questions."
        )
        res = validate_grounding(topic, facts, hallucinated_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("unsupported_reasoning_or_motivation" in i for i in res["flagged_issues"]))

    def test_unsupported_technical_capabilities_flagged(self):
        """Draft introduces external tools or APIs not mentioned in verified facts."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in IndexedDB"

        # Hallucinates third-party OpenAI APIs and scratch model training
        tech_draft = (
            "Our users are building AI models from scratch using web APIs like OpenAI, "
            "so storing chats in IndexedDB gives them control."
        )
        res = validate_grounding(topic, facts, tech_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("unsupported_technical_capability" in i for i in res["flagged_issues"]))

    def test_unsupported_privacy_guarantees_flagged(self):
        """Draft introduces unverified privacy guarantees, data sovereignty, or sensitive data claims."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in IndexedDB"

        # Hallucinates sensitive data protection and complete data sovereignty
        privacy_draft = (
            "Keeping data in IndexedDB ensures that no one else has access to sensitive conversations "
            "and guarantees complete data sovereignty."
        )
        res = validate_grounding(topic, facts, privacy_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("unsupported_privacy_guarantee" in i for i in res["flagged_issues"]))

    def test_unsupported_project_outcomes_or_capability_embellishment_flagged(self):
        """Draft turns an implementation limitation into an embellished positive capability or reliability claim."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point currently uses static match data and simulated match progression. It does not have live match connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        # Embellishes static limitation into a positive reliability feature
        embellished_draft = (
            "Match Point uses static data because rather than relying on live connectivity, "
            "the platform provides a complete and consistent simulation that keeps the user experience dependable."
        )
        res = validate_grounding(topic, facts, embellished_draft)
        self.assertFalse(res["is_grounded"])
        self.assertTrue(any("unsupported_project_outcome_or_capability" in i for i in res["flagged_issues"]))

    def test_strictly_verified_grounded_fallback_match_point(self):
        """Match Point conservative fallback across short, medium, and long modes uses only verified facts."""
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "When building Match Point, I experienced the limitation: could not get live match data, so built a static version where matches never end.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        for mode in ["short", "medium", "long"]:
            draft = build_grounded_fallback_draft(topic, facts, length_mode=mode, lead_in="Recently I built")
            res = validate_grounding(topic, facts, draft)
            self.assertTrue(res["is_grounded"], f"Match Point fallback in {mode} mode must be fully grounded")
            self.assertEqual(len(res["flagged_issues"]), 0)

            # Must NOT contain embellishments
            self.assertNotIn("complete and consistent simulation", draft)
            self.assertNotIn("dependable", draft)
            self.assertNotIn("explore betting mechanics reliably", draft)
            self.assertNotIn("websockets", draft.lower())

            # Must contain core verified facts
            self.assertIn("Match Point", draft)
            self.assertIn("tennis", draft)
            self.assertIn("static", draft)
            is_comp, _ = is_complete_thought(draft)
            self.assertTrue(is_comp, f"Match Point fallback in {mode} mode must be a complete thought")

    def test_strictly_verified_grounded_fallback_indexeddb(self):
        """IndexedDB conservative fallback across short, medium, and long modes uses only verified facts."""
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in the browser's IndexedDB"

        for mode in ["short", "medium", "long"]:
            draft = build_grounded_fallback_draft(topic, facts, length_mode=mode)
            res = validate_grounding(topic, facts, draft)
            self.assertTrue(res["is_grounded"], f"IndexedDB fallback in {mode} mode must be fully grounded")
            self.assertEqual(len(res["flagged_issues"]), 0)

            # Must NOT contain unverified additions
            self.assertNotIn("openai", draft.lower())
            self.assertNotIn("sensitive", draft.lower())
            self.assertNotIn("sovereignty", draft.lower())
            self.assertNotIn("richer insights", draft.lower())
            self.assertNotIn("what people want", draft.lower())

            # Must contain core verified facts
            self.assertIn("IndexedDB", draft)
            self.assertIn("train their own model", draft)
            is_comp, _ = is_complete_thought(draft)
            self.assertTrue(is_comp, f"IndexedDB fallback in {mode} mode must be a complete thought")

    def test_general_conceptual_topic_unaffected(self):
        """General conceptual topic with zero personal facts remains unblocked and fully passes grounding."""
        topic = "Why I prefer learning by building projects rather than just reading documentation"
        self.assertFalse(is_autobiographical_topic(topic))

        draft = (
            "I feel that 90% of what we learn today comes from passive consumption, "
            "which leaves us with nothing but knowledge gaps and never really understand why it works or how to apply it."
        )
        res = validate_grounding(topic, [], draft)
        self.assertTrue(res["is_grounded"])
        self.assertEqual(len(res["flagged_issues"]), 0)


if __name__ == "__main__":
    unittest.main()
