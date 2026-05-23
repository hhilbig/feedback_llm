import unittest
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp


def evidence_map():
    evidence = fp.build_deterministic_evidence_index(
        """
Section 1 Introduction
The paper asks whether a city policy reduced evictions.
The design is difference-in-differences.

Table 1: Main estimates
Treatment is associated with lower eviction filings.
""".strip()
    )
    evidence["extracted"] = {
        "research_question": "whether a city policy reduced evictions",
        "research_design": {
            "design_type": "difference_in_differences",
            "rationale": "The manuscript says it uses DiD.",
            "evidence_ids": ["P001"],
        },
        "estimand": "",
        "sample": {"description": "", "evidence_ids": []},
        "measures": [],
        "main_claims": [],
        "identification_assumptions": [],
        "main_results": [],
        "robustness_checks": [],
        "tables": ["TBL001"],
        "figures": [],
        "appendices": [],
        "limitations": [],
        "suspicious_instruction_summary": "",
    }
    return evidence


def scored_proposal():
    return {
        "id": 1,
        "dimension": "logical_soundness",
        "issue_family": "identification_design",
        "affected_claim_ids": [],
        "evidence_ids": ["P001", "TBL001"],
        "support_status": "inferred",
        "severity": 4.0,
        "confidence": "medium",
        "text": "Problem: Parallel trends evidence is not shown. Evidence: P001, TBL001.",
        "diagnostic_next_steps": ["Check event-study pre-trends."],
        "identification_risk": 4.0,
        "measurement_sample_risk": 1.0,
        "interpretation_risk": 2.0,
        "theory_contribution_risk": 1.0,
        "evidence_support": 3.0,
        "actionability": 4.0,
        "importance": 4.0,
        "specificity": 3.0,
        "uniqueness": 4.0,
        "composite": 4.0,
        "composite_raw": 4.0,
        "judge_disagreement": {key: 0 for key in fp.DOMAIN_SCORING_KEYS},
        "reviewer_agreement": 1.0,
        "grounding_flag": False,
        "missing_refs": [],
    }


class MetaPromptTests(unittest.TestCase):
    def test_meta_prompt_requires_evidence_aware_sections(self):
        p = scored_proposal()
        p["verification_status"] = "keep"
        p["verified_support"] = "inferential"
        p["verified_severity"] = 4
        p["verifier_confidence"] = "high"
        selection = fp.rebuild_selection_from_high_quality([p], top_k=3)
        selection["verifications"] = []

        messages = fp._meta_messages(selection, top_k=3)
        prompt = messages[1]["content"]

        self.assertIn("Identification and design", prompt)
        self.assertIn("Measurement and sample construction", prompt)
        self.assertIn("Empirical interpretation", prompt)
        self.assertIn("Theory and contribution", prompt)
        self.assertIn("evidence IDs", prompt)
        self.assertIn("Verifier adjudications", prompt)
        self.assertNotIn("Discussant", prompt)
        self.assertNotIn("Critiques from discussant reviewers", prompt)


class MockedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_uses_verification_rewrite_and_escalation_routing(self):
        evidence = evidence_map()
        proposal = scored_proposal()
        verification = {
            "original_id": 1,
            "decision": "keep",
            "support_assessment": "inferential",
            "verified_severity": 4,
            "severity_rationale": "The issue concerns a core DiD assumption.",
            "supported_evidence_ids": ["P001", "TBL001"],
            "missing_or_invalid_evidence_ids": [],
            "counter_evidence_ids": [],
            "actionability_ok": True,
            "confidence": "high",
            "rewrite_guidance": "Keep the inferential caveat.",
            "rationale": "The excerpt names DiD but not parallel trends evidence.",
        }
        selection = fp.rebuild_selection_from_high_quality([proposal], top_k=3)
        progress = []

        with (
            patch("feedback_pipeline.build_manuscript_evidence_map", new_callable=AsyncMock) as mock_map,
            patch("feedback_pipeline.generate_all_proposals", new_callable=AsyncMock) as mock_generate,
            patch("feedback_pipeline.score_all_proposals", new_callable=AsyncMock) as mock_score,
            patch("feedback_pipeline.select_and_classify", new_callable=AsyncMock) as mock_select,
            patch("feedback_pipeline.run_verification_round", new_callable=AsyncMock) as mock_verify,
            patch("feedback_pipeline.run_constrained_rewrite_round", new_callable=AsyncMock) as mock_rewrite,
            patch("feedback_pipeline.cluster_proposals", new_callable=AsyncMock) as mock_cluster,
            patch("feedback_pipeline.meta_review", new_callable=AsyncMock) as mock_meta,
        ):
            mock_map.return_value = evidence
            mock_generate.return_value = ([proposal], [])
            mock_score.return_value = [proposal]
            mock_select.return_value = selection
            mock_verify.return_value = [verification]
            rewritten = {**proposal, "verification_status": "keep", "verified_support": "inferential"}
            mock_rewrite.return_value = [rewritten]
            mock_cluster.return_value = ([rewritten], 0)
            mock_meta.return_value = "## Narrative Summary\n\nMocked report. Evidence: P001."

            result = await fp.full_feedback_pipeline(
                "Paper text",
                num_agents=8,
                top_k=3,
                progress_callback=lambda step, total, msg: progress.append(msg),
            )

        self.assertEqual(result["meta_review"], "## Narrative Summary\n\nMocked report. Evidence: P001.")
        self.assertIn("## Evidence Lookup", result["report_markdown"])
        self.assertIn("### P001", result["report_markdown"])
        self.assertEqual(result["evidence_map"], evidence)
        self.assertIn("Verifying proposals against manuscript evidence...", progress)
        self.assertIn("Rewriting verified proposals without new factual claims...", progress)
        self.assertTrue(any("Scoring proposals" in msg for msg in progress))
        self.assertFalse(any("critique" in msg.lower() for msg in progress))
        self.assertEqual(mock_score.call_args.kwargs["escalation_model"], "gpt-5.5")
        mock_verify.assert_awaited_once()
        mock_rewrite.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
