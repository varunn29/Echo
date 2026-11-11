"""
Echo — Phase 2, Step 4B: Stage 3C Generation Pipeline Corrections Regression Suite
Comprehensive unit tests verifying:
1. 14-word, 1-sentence Medium rejected (insufficient_length).
2. 20-word, 2-sentence Medium rejected (insufficient_length).
3. 30-word, 2-sentence Medium passes minimum threshold.
4. Complete 3-sentence Medium passes.
5. Meta-commentary rejected (meta_commentary_non_draft).
6. Valid first-person opinion starting with "I think" accepted.
7. Sampled failure + under-length greedy retry engages fallback.
8. Fallback is grounded, topic-relevant, and meets length.
9. Match Point static-data constraints remain protected.
10. Zero-fact autobiographical safety remains intact.
11. Short and Long generation behavior unaffected.
"""

import unittest
from unittest.mock import MagicMock, patch
import torch

from training.generate_with_adapter import (
    validate_grounding,
    validate_length_compliance,
    validate_meta_commentary,
    validate_generation_quality,
    build_grounded_fallback_draft,
    build_zero_fact_autobiographical_response,
    is_autobiographical_topic,
    extract_sentences,
    is_complete_thought,
    PersonalizedWritingGenerator,
)


class TestPipelineCorrectionsStage3C(unittest.TestCase):
    """Stage 3C regression tests for decoupled generation quality and fallback enhancements."""

    # -------------------------------------------------------------------------
    # Test 1: 14-word, 1-sentence Medium rejected (insufficient_length)
    # -------------------------------------------------------------------------
    def test_01_underlength_14_word_1_sentence_medium_rejected(self):
        draft = "I like to reflect on my work because it helps me improve."
        words = draft.split()
        sentences = extract_sentences(draft)
        self.assertEqual(len(words), 12)
        self.assertEqual(len(sentences), 1)

        length_res = validate_length_compliance(draft, length_mode="medium")
        self.assertFalse(length_res["is_compliant"])
        self.assertTrue(any("insufficient_length" in issue for issue in length_res["flagged_issues"]))

        quality_res = validate_generation_quality(
            draft, topic="Work reflection", personal_knowledge=[], length_mode="medium"
        )
        self.assertFalse(quality_res["is_acceptable"])
        self.assertFalse(quality_res["is_compliant_length"])
        self.assertTrue(any("insufficient_length" in issue for issue in quality_res["flagged_issues"]))

    # -------------------------------------------------------------------------
    # Test 2: 20-word, 2-sentence Medium rejected (insufficient_length)
    # -------------------------------------------------------------------------
    def test_02_underlength_20_word_2_sentence_medium_rejected(self):
        draft = "I reflect on my engineering work regularly. It helps me understand programming tools and build better web applications every day."
        words = draft.split()
        sentences = extract_sentences(draft)
        self.assertEqual(len(words), 20)  # Exactly 20 words (< 25 words threshold)
        self.assertEqual(len(sentences), 2)  # 2 sentences but < 25 words

        length_res = validate_length_compliance(draft, length_mode="medium")
        self.assertFalse(length_res["is_compliant"])
        self.assertTrue(any("insufficient_length" in issue for issue in length_res["flagged_issues"]))

        quality_res = validate_generation_quality(
            draft, topic="Engineering practice", personal_knowledge=[], length_mode="medium"
        )
        self.assertFalse(quality_res["is_acceptable"])
        self.assertFalse(quality_res["is_compliant_length"])
        self.assertTrue(any("insufficient_length" in issue for issue in quality_res["flagged_issues"]))

    # -------------------------------------------------------------------------
    # Test 3: 30-word, 2-sentence Medium passes minimum threshold
    # -------------------------------------------------------------------------
    def test_03_30_word_2_sentence_medium_passes(self):
        draft = (
            "I reflect on my software engineering projects regularly to understand what went well and what could be improved. "
            "This practice helps me solve complex coding challenges much more effectively over time."
        )
        words = draft.split()
        sentences = extract_sentences(draft)
        self.assertTrue(len(words) >= 25)
        self.assertEqual(len(words), 31)
        self.assertEqual(len(sentences), 2)

        length_res = validate_length_compliance(draft, length_mode="medium")
        self.assertTrue(length_res["is_compliant"])
        self.assertEqual(len(length_res["flagged_issues"]), 0)

        quality_res = validate_generation_quality(
            draft, topic="Software engineering reflection", personal_knowledge=[], length_mode="medium"
        )
        self.assertTrue(quality_res["is_acceptable"])
        self.assertTrue(quality_res["is_compliant_length"])
        self.assertEqual(len(quality_res["flagged_issues"]), 0)

    # -------------------------------------------------------------------------
    # Test 4: Complete 3-sentence Medium passes
    # -------------------------------------------------------------------------
    def test_04_complete_3_sentence_medium_passes(self):
        draft = (
            "I reflect on my engineering projects regularly to evaluate my overall progress. "
            "Working through difficult technical challenges has taught me the value of steady incremental improvements. "
            "Having this perspective makes my development process much more deliberate and consistent."
        )
        words = draft.split()
        sentences = extract_sentences(draft)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(len(words) >= 25)

        length_res = validate_length_compliance(draft, length_mode="medium")
        self.assertTrue(length_res["is_compliant"])
        self.assertEqual(len(length_res["flagged_issues"]), 0)

        quality_res = validate_generation_quality(
            draft, topic="Project reflection", personal_knowledge=[], length_mode="medium"
        )
        self.assertTrue(quality_res["is_acceptable"])
        self.assertTrue(quality_res["is_clean_meta"])
        self.assertTrue(quality_res["is_compliant_length"])
        self.assertEqual(len(quality_res["flagged_issues"]), 0)

    # -------------------------------------------------------------------------
    # Test 5: Meta-commentary rejected (meta_commentary_non_draft)
    # -------------------------------------------------------------------------
    def test_05_meta_commentary_rejected(self):
        meta_drafts = [
            "I will write a short paragraph about how I communicate with other engineers.",
            "Here is my draft about why local databases are better for chat storage.",
            "I am going to explain my perspective on web development and future goals.",
            "Let me write a draft reflecting on software architecture.",
            "In this draft, I will summarize my thoughts on code quality.",
            "Here's my response explaining how I solve programming problems.",
        ]

        for draft in meta_drafts:
            meta_res = validate_meta_commentary(draft)
            self.assertFalse(meta_res["is_clean"], f"Draft should be flagged as meta-commentary: {draft}")
            self.assertTrue(
                any("meta_commentary_non_draft" in issue for issue in meta_res["flagged_issues"]),
                f"Issue must contain 'meta_commentary_non_draft': {draft}",
            )

            quality_res = validate_generation_quality(
                draft, topic="General topic", personal_knowledge=[], length_mode="medium"
            )
            self.assertFalse(quality_res["is_acceptable"])
            self.assertFalse(quality_res["is_clean_meta"])
            self.assertTrue(any("meta_commentary_non_draft" in issue for issue in quality_res["flagged_issues"]))

    # -------------------------------------------------------------------------
    # Test 6: Valid first-person opinion starting with "I think" accepted
    # -------------------------------------------------------------------------
    def test_06_authentic_opinion_with_i_think_accepted(self):
        authentic_opinions = [
            (
                "I think we should prioritize local data storage over cloud uploads whenever possible. "
                "Keeping user information in the browser gives individuals direct control over their records. "
                "This approach creates much more reliable applications in the long run."
            ),
            (
                "I believe that consistent daily routines help engineers maintain mental clarity and focus. "
                "Starting each morning with predictable habits reduces decision fatigue significantly. "
                "Over time, this deliberate pace leads to more consistent problem-solving."
            ),
            (
                "I feel that learning by building functional projects is much more effective than passive reading. "
                "Applying concepts to concrete software challenges reveals edge cases immediately. "
                "This practical feedback loop builds deeper technical confidence."
            ),
        ]

        for draft in authentic_opinions:
            meta_res = validate_meta_commentary(draft)
            self.assertTrue(meta_res["is_clean"], f"Authentic opinion must not be flagged: {draft}")
            self.assertEqual(len(meta_res["flagged_issues"]), 0)

            quality_res = validate_generation_quality(
                draft, topic="General technical reflection", personal_knowledge=[], length_mode="medium"
            )
            self.assertTrue(quality_res["is_acceptable"])
            self.assertTrue(quality_res["is_clean_meta"])
            self.assertEqual(len(quality_res["flagged_issues"]), 0)

    # -------------------------------------------------------------------------
    # Test 7: Sampled failure + under-length greedy retry engages fallback
    # -------------------------------------------------------------------------
    def test_07_sampled_failure_plus_underlength_greedy_engages_fallback(self):
        generator = PersonalizedWritingGenerator()

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.encode.side_effect = lambda text: [1] * len(text.split())
        mock_tokenizer.eos_token_id = 0
        mock_tokenizer.decode = MagicMock()

        # Sampled output: returns meta-commentary failure
        sampled_text = "Here is my draft about IndexedDB storage."
        # Greedy retry output: returns under-length draft (5 words, 1 sentence)
        greedy_text = "I prefer local IndexedDB storage."

        decode_call_count = 0

        def mock_decode(tokens, skip_special_tokens=True):
            nonlocal decode_call_count
            decode_call_count += 1
            if decode_call_count == 1:
                return sampled_text
            return greedy_text

        mock_tokenizer.decode.side_effect = mock_decode

        # Mock model
        mock_model = MagicMock()
        # Mock inputs
        dummy_tensor = torch.zeros((1, 5), dtype=torch.long)
        mock_tokenizer.return_value = {"input_ids": dummy_tensor}
        mock_model.generate.return_value = torch.zeros((1, 15), dtype=torch.long)

        generator.model = mock_model
        generator.tokenizer = mock_tokenizer

        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in IndexedDB"

        res = generator.generate(
            topic=topic,
            personal_knowledge=facts,
            length_mode="medium",
            do_sample=True,
        )

        # Under the old logic, the 5-word greedy retry was accepted.
        # Under the new logic, greedy retry is rejected for insufficient_length, engaging fallback!
        self.assertEqual(res["source"], "grounded_fallback")
        self.assertTrue(res["grounding_check"]["fallback_used"])
        self.assertTrue(res["grounding_check"]["is_grounded"])
        self.assertTrue(res["grounding_check"]["is_acceptable"])
        self.assertIn("IndexedDB", res["complete_draft"])
        self.assertIn("train their own model", res["complete_draft"])

        # Final draft must meet Medium length compliance
        length_check = validate_length_compliance(res["complete_draft"], length_mode="medium")
        self.assertTrue(length_check["is_compliant"])

    # -------------------------------------------------------------------------
    # Test 8: Fallback is grounded, topic-relevant, and meets length
    # -------------------------------------------------------------------------
    def test_08_fallback_grounded_topic_relevant_meets_length(self):
        scenarios = [
            (
                "Match Point",
                "Why and how I built Match Point",
                [
                    "I built Match Point, a betting platform for tennis fans.",
                    "Match Point operates with static match data and simulated match progression without live connectivity.",
                ],
                3,  # expected sentences
                47, # expected words
            ),
            (
                "IndexedDB",
                "Why chat data should stay stored in IndexedDB",
                [
                    "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
                ],
                2,  # expected sentences
                42, # expected words
            ),
            (
                "Web Dev",
                "My web development background and future learning",
                [
                    "Learned: HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL for web development.",
                    "Next plan: move ahead to learn AI, machine learning, and cloud computing.",
                ],
                2,  # expected sentences
                33, # expected words
            ),
            (
                "General Single Fact",
                "My experience with software architecture",
                [
                    "I designed modular frontend architectures with React and TypeScript.",
                ],
                1,  # expected sentences
                9,  # expected words
            ),
            (
                "General Multiple Facts",
                "My approach to code quality",
                [
                    "I write unit tests for every critical component before refactoring.",
                    "I maintain strict type boundaries in TypeScript to prevent runtime errors.",
                ],
                2,  # expected sentences
                21, # expected words
            ),
        ]

        for label, topic, facts, exp_sentences, exp_words in scenarios:
            draft = build_grounded_fallback_draft(topic, facts, length_mode="medium")
            words = draft.split()
            sentences = extract_sentences(draft)

            self.assertEqual(len(sentences), exp_sentences, f"Scenario '{label}' expected {exp_sentences} sentences: {draft}")
            self.assertEqual(len(words), exp_words, f"Scenario '{label}' expected {exp_words} words: {draft}")

            # Grounding check: 100% grounded, 0 issues
            grounding_res = validate_grounding(topic, facts, draft)
            self.assertTrue(grounding_res["is_grounded"], f"Scenario '{label}' must pass grounding: {grounding_res}")
            self.assertEqual(len(grounding_res["flagged_issues"]), 0)

            # Meta-commentary check: clean, 0 issues
            meta_res = validate_meta_commentary(draft)
            self.assertTrue(meta_res["is_clean"], f"Scenario '{label}' must be free of meta-commentary: {meta_res}")
            self.assertEqual(len(meta_res["flagged_issues"]), 0)

            # Never contains invented generic filler
            self.assertNotIn("in my engineering work", draft.lower())
            self.assertNotIn("direct user control", draft.lower())
            self.assertNotIn("privacy guarantee", draft.lower())

            # Complete thought validation
            is_comp, reason = is_complete_thought(draft)
            self.assertTrue(is_comp, f"Scenario '{label}' must be a complete thought: {reason}")

    # -------------------------------------------------------------------------
    # Test 9: Match Point static-data constraints remain protected
    # -------------------------------------------------------------------------
    def test_09_match_point_static_data_constraints_protected(self):
        facts = [
            "I built Match Point, a betting platform for tennis fans.",
            "Match Point operates with static match data and simulated match progression without live connectivity.",
        ]
        topic = "Why and how I built the Match Point betting platform"

        # Limitation inversion check
        live_draft = "I built Match Point as a betting platform for tennis fans with real-time match odds and live match data streaming."
        res_live = validate_grounding(topic, facts, live_draft)
        self.assertFalse(res_live["is_grounded"])
        self.assertTrue(any("limitation_inversion" in i for i in res_live["flagged_issues"]))

        # Limitation omission check
        omitted_draft = "I built Match Point as a betting platform for tennis fans to explore tennis betting."
        res_omitted = validate_grounding(topic, facts, omitted_draft)
        self.assertFalse(res_omitted["is_grounded"])
        self.assertTrue(any("limitation_omission" in i for i in res_omitted["flagged_issues"]))

        # Fallback draft preserves static data limitations
        fallback = build_grounded_fallback_draft(topic, facts, length_mode="medium")
        self.assertIn("static", fallback.lower())
        self.assertIn("simulated match progression", fallback.lower())
        self.assertNotIn("live match data", fallback.lower().replace("live match data was unavailable", ""))

    # -------------------------------------------------------------------------
    # Test 10: Zero-fact autobiographical safety remains intact
    # -------------------------------------------------------------------------
    def test_10_zero_fact_autobiographical_safety_intact(self):
        auto_topics = [
            "My daily morning routine and coffee habits",
            "My experience with pets",
            "My pet dog",
        ]

        for topic in auto_topics:
            self.assertTrue(is_autobiographical_topic(topic))

            for mode in ["short", "medium", "long"]:
                resp = build_zero_fact_autobiographical_response(topic, length_mode=mode)
                self.assertIn("haven't recorded any personal", resp)
                self.assertNotIn("wake up at", resp.lower())
                self.assertNotIn("my dog", resp.lower())
                is_comp, reason = is_complete_thought(resp)
                self.assertTrue(is_comp, f"Response in {mode} mode must be complete thought: {reason}")

        # Test end-to-end generator branch
        generator = PersonalizedWritingGenerator()
        generator.model = MagicMock()
        generator.tokenizer = MagicMock()
        generator.tokenizer.encode.side_effect = lambda text: [1] * len(text.split())

        res = generator.generate(
            topic="My daily morning routine and coffee habits",
            personal_knowledge=[],
            length_mode="medium",
        )
        self.assertEqual(res["source"], "zero_fact_safe_response")
        self.assertTrue(res["grounding_check"]["fallback_used"])
        self.assertTrue(res["grounding_check"]["is_grounded"])
        self.assertTrue(res["grounding_check"]["is_acceptable"])
        self.assertIn("haven't recorded any personal notes", res["complete_draft"])

    # -------------------------------------------------------------------------
    # Test 11: Short and Long generation behavior unaffected
    # -------------------------------------------------------------------------
    def test_11_short_and_long_generation_behavior_unaffected(self):
        # Short mode: 1 sentence, 11 words
        short_draft = "I built Match Point using static match data and simulated match progression."
        short_length = validate_length_compliance(short_draft, length_mode="short")
        self.assertTrue(short_length["is_compliant"])
        self.assertEqual(len(short_length["flagged_issues"]), 0)

        short_quality = validate_generation_quality(
            short_draft,
            topic="Match Point",
            personal_knowledge=["Match Point uses static data and simulated match progression."],
            length_mode="short",
        )
        self.assertTrue(short_quality["is_acceptable"])

        # Long mode: 2 paragraphs, 53 words
        long_draft = (
            "I started by learning web development technologies, including HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL. "
            "Working with these tools provided my foundation in building web applications.\n\n"
            "My next planned step is to learn AI, machine learning, and cloud computing. "
            "Moving into these areas represents the next stage of my technical learning journey."
        )
        long_length = validate_length_compliance(long_draft, length_mode="long")
        self.assertTrue(long_length["is_compliant"])
        self.assertEqual(len(long_length["flagged_issues"]), 0)

        long_quality = validate_generation_quality(
            long_draft,
            topic="Web development background",
            personal_knowledge=[
                "Learned: HTML, CSS, JavaScript, React, TypeScript, and PostgreSQL.",
                "Next plan: learn AI, machine learning, and cloud computing.",
            ],
            length_mode="long",
        )
        self.assertTrue(long_quality["is_acceptable"])

    # -------------------------------------------------------------------------
    # Stage 3D Test 12: IndexedDB fallback removes unsupported control/privacy
    # -------------------------------------------------------------------------
    def test_12_indexeddb_fallback_removes_unsupported_control_and_privacy_claims(self):
        facts = [
            "My perspective on IndexedDB is that chat data should stay stored in user's indexed db so they train their own model.",
        ]
        topic = "Why user chat data should stay stored locally in the browser's IndexedDB"

        for mode in ["short", "medium", "long"]:
            draft = build_grounded_fallback_draft(topic, facts, length_mode=mode)

            # Must NEVER contain unverified control, privacy, or security claims
            self.assertNotIn("direct user control", draft.lower())
            self.assertNotIn("user control", draft.lower())
            self.assertNotIn("privacy", draft.lower())
            self.assertNotIn("security", draft.lower())
            self.assertNotIn("sovereignty", draft.lower())
            self.assertNotIn("sensitive", draft.lower())

            # Must strictly preserve verified core facts
            self.assertIn("IndexedDB", draft)
            self.assertIn("train their own model", draft)

            # Grounding check: 100% grounded
            res = validate_grounding(topic, facts, draft)
            self.assertTrue(res["is_grounded"], f"IndexedDB fallback in {mode} mode must be fully grounded")
            self.assertEqual(len(res["flagged_issues"]), 0)

    # -------------------------------------------------------------------------
    # Stage 3D Test 13: General fallback variable-length facts and punctuation
    # -------------------------------------------------------------------------
    def test_13_general_fallback_variable_length_facts_and_punctuation(self):
        # Case A: Short fact without terminal period
        short_fact = ["I learned Python"]
        short_draft = build_grounded_fallback_draft("Python learning", short_fact, length_mode="medium")
        self.assertEqual(short_draft, "I learned Python.")
        self.assertEqual(len(short_draft.split()), 3)
        self.assertEqual(len(extract_sentences(short_draft)), 1)
        self.assertNotIn("..", short_draft)

        # Case B: Long fact with complex clause
        long_fact = [
            "When developing my application, I decided to use local IndexedDB storage because it allows offline access and enables users to process their data directly in their browser without external dependencies."
        ]
        long_draft = build_grounded_fallback_draft("Local architecture", long_fact, length_mode="medium")
        self.assertEqual(len(long_draft.split()), 30)
        self.assertEqual(len(extract_sentences(long_draft)), 1)
        self.assertNotIn("in my engineering work", long_draft.lower())

        # Case C: Multiple facts of different lengths
        multi_facts = [
            "I designed the database schema with PostgreSQL.",
            "Later, I built the client using React and TypeScript to ensure type safety across components.",
        ]
        multi_draft = build_grounded_fallback_draft("Full stack architecture", multi_facts, length_mode="medium")
        self.assertEqual(len(multi_draft.split()), 22)
        self.assertEqual(len(extract_sentences(multi_draft)), 2)
        self.assertIn("PostgreSQL.", multi_draft)
        self.assertNotIn("..", multi_draft)

        # Case D: Fact that already contains complete sentences
        multi_sent_fact = [
            "I built Task Tracker to organize daily work. It runs entirely in the browser with offline support."
        ]
        multi_sent_draft = build_grounded_fallback_draft("Task Tracker details", multi_sent_fact, length_mode="medium")
        self.assertEqual(len(multi_sent_draft.split()), 17)
        self.assertEqual(len(extract_sentences(multi_sent_draft)), 2)
        self.assertFalse(multi_sent_draft.endswith(".."))

        # Case E: Fact ending in question mark and exclamation mark
        punct_facts = [
            "Can the application run completely offline? Yes, the platform operates without any remote server dependencies!"
        ]
        punct_draft = build_grounded_fallback_draft("Offline capability check", punct_facts, length_mode="medium")
        self.assertTrue(punct_draft.endswith("!"))
        self.assertFalse(punct_draft.endswith("!."))

        # Verify all cases pass grounding
        for case_label, f_list, d_text in [
            ("Short", short_fact, short_draft),
            ("Long", long_fact, long_draft),
            ("Multi", multi_facts, multi_draft),
            ("Multi-Sent", multi_sent_fact, multi_sent_draft),
            ("Punct", punct_facts, punct_draft),
        ]:
            res = validate_grounding("Topic", f_list, d_text)
            self.assertTrue(res["is_grounded"], f"Case '{case_label}' failed grounding: {res}")
            self.assertEqual(len(res["flagged_issues"]), 0)

    # -------------------------------------------------------------------------
    # Stage 3D Test 14: Zero-fact autobiographical without coffee assumption
    # -------------------------------------------------------------------------
    def test_14_zero_fact_autobiographical_no_coffee_assumption(self):
        routine_topic = "My daily morning routine"
        resp = build_zero_fact_autobiographical_response(routine_topic, length_mode="medium")

        # Must NOT assume coffee consumption when not in topic
        self.assertNotIn("coffee", resp.lower())
        self.assertIn("haven't recorded any personal notes", resp)

        is_comp, reason = is_complete_thought(resp)
        self.assertTrue(is_comp, f"Must be a complete thought: {reason}")


if __name__ == "__main__":
    unittest.main()
