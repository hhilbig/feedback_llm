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
    def test_meta_prompt_uses_editorial_triage_structure(self):
        p = scored_proposal()
        p["verification_status"] = "keep"
        p["verified_support"] = "inferential"
        p["verified_severity"] = 4
        p["verifier_confidence"] = "high"
        selection = fp.rebuild_selection_from_high_quality([p], top_k=3)
        selection["verifications"] = []
        selection["substantive_profile"] = {
            "designs": ["difference_in_differences", "text_as_data"],
            "data_types": ["survey", "news_corpus"],
            "key_risks": ["inference_level", "model_version_reproducibility"],
        }
        selection["substantive_checks"] = [
            {
                "check_id": "did_inference_level",
                "category": "inference",
                "status": "needs_review",
                "severity": "high",
                "evidence_ids": ["P001"],
                "suggested_check": "Justify clustering or aggregation.",
            }
        ]
        selection["editorial_issue_inputs"] = fp.build_editorial_issue_inputs(selection)
        selection["editorial_triage"] = {
            "editorial_diagnosis": "mostly_major_revision_issues",
            "decision_summary": "No clear rejection-level issue is established.",
            "classified_issues": [
                {
                    "issue_id": "I01",
                    "short_label": "Parallel trends diagnostics",
                    "problem": "DD/DDD pre-trend diagnostics are not sufficiently foregrounded.",
                    "rejection_risk": "conditional",
                    "decision_tier": "major_revision_issue",
                    "could_justify_rejection": True,
                    "why_it_matters": "Failed pre-trends would undermine the causal claim.",
                    "what_would_make_rejection_level": "Full leads show meaningful divergence.",
                    "why_not_currently_rejection": "The issue is inferential from missing diagnostics.",
                    "minimum_fix": "Report full lead estimates.",
                    "fixability": "additional analysis/reporting",
                    "core_claim_affected": "causal estimate",
                    "evidence_strength": "partial",
                    "existing_mitigations": ["event-study"],
                    "output_location": "main_report",
                    "recommended_action": "Report full lead estimates.",
                }
            ],
            "main_report_issue_ids": ["I01"],
            "problem_issue_ids": ["I01"],
            "non_blocking_issue_ids": [],
            "dropped_issue_ids": [],
        }

        messages = fp._meta_messages(selection, top_k=3)
        prompt = messages[1]["content"]

        self.assertIn("## Editorial Summary", prompt)
        self.assertIn("## Clear Problems and Rejection Risk", prompt)
        self.assertIn("## Notes on Non-Rejection Issues", prompt)
        self.assertIn("Potential rejection reason", prompt)
        self.assertIn("Major revision issue", prompt)
        self.assertIn("Rejection risk", prompt)
        self.assertIn("Do not use a Markdown table", prompt)
        self.assertNotIn("| # | Problem |", prompt)
        self.assertIn("evidence IDs", prompt)
        self.assertIn("Editorial triage", prompt)
        self.assertIn("I01", prompt)
        self.assertNotIn("[REQUIRED]", prompt)
        self.assertNotIn("[SUGGESTED]", prompt)
        self.assertNotIn("Discussant", prompt)
        self.assertNotIn("Critiques from discussant reviewers", prompt)


class EditorialTriageTests(unittest.TestCase):
    def test_triage_limits_main_report_and_demotes_checklist_diagnostics(self):
        issue_inputs = [
            {
                "issue_id": f"I{i:02d}",
                "source_type": "verified_proposal",
                "verification_status": "keep",
            }
            for i in range(1, 5)
        ]
        issue_inputs.append(
            {
                "issue_id": "I05",
                "source_type": "substantive_checklist",
                "verification_status": "needs_review",
            }
        )
        triage = {
            "editorial_diagnosis": "potential_rejection_issues",
            "decision_summary": "Several issues are decision relevant.",
            "classified_issues": [
                {
                    "issue_id": issue["issue_id"],
                    "short_label": issue["issue_id"],
                    "problem": issue["issue_id"],
                    "rejection_risk": "high",
                    "decision_tier": "potential_rejection_reason",
                    "could_justify_rejection": issue["source_type"] != "substantive_checklist",
                    "why_it_matters": "Could undermine the main claim.",
                    "what_would_make_rejection_level": "Diagnostics fail.",
                    "why_not_currently_rejection": "Diagnostics may mitigate it.",
                    "minimum_fix": "Run the diagnostic.",
                    "fixability": "additional analysis/reporting",
                    "core_claim_affected": "main claim",
                    "evidence_strength": "partial",
                    "existing_mitigations": [],
                    "output_location": "main_report",
                    "recommended_action": "Run the diagnostic.",
                }
                for issue in issue_inputs
            ],
            "main_report_issue_ids": [],
            "problem_issue_ids": [],
            "non_blocking_issue_ids": [],
            "dropped_issue_ids": [],
        }

        normalized = fp.enforce_editorial_triage_limits(triage, issue_inputs)

        self.assertLessEqual(
            sum(
                1
                for item in normalized["classified_issues"]
                if item["decision_tier"] == "potential_rejection_reason"
            ),
            2,
        )
        self.assertLessEqual(len(normalized["problem_issue_ids"]), 8)
        checklist_item = next(
            item for item in normalized["classified_issues"] if item["issue_id"] == "I05"
        )
        self.assertEqual(checklist_item["decision_tier"], "major_revision_issue")
        self.assertEqual(checklist_item["rejection_risk"], "conditional")


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
            patch("feedback_pipeline.editorial_triage", new_callable=AsyncMock) as mock_triage,
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
            mock_triage.return_value = (
                {
                    "editorial_diagnosis": "mostly_major_revision_issues",
                    "decision_summary": "No clear rejection-level issue.",
                    "classified_issues": [],
                    "main_report_issue_ids": [],
                    "problem_issue_ids": [],
                    "non_blocking_issue_ids": [],
                    "dropped_issue_ids": [],
                    "rejection_level_count": 0,
                },
                [],
            )
            mock_meta.return_value = "## Editorial Summary\n\nMocked report. Evidence: P001."

            result = await fp.full_feedback_pipeline(
                "Paper text",
                num_agents=8,
                top_k=3,
                progress_callback=lambda step, total, msg: progress.append(msg),
            )

        self.assertEqual(result["meta_review"], "## Editorial Summary\n\nMocked report. Evidence: P001.")
        self.assertNotIn("## Evidence Lookup", result["report_markdown"])
        self.assertIn("editorial_triage", result)
        self.assertEqual(result["evidence_map"], evidence)
        self.assertIn("Verifying proposals against manuscript evidence...", progress)
        self.assertIn("Rewriting verified proposals without new factual claims...", progress)
        self.assertIn("Triaging issues by editorial decision relevance...", progress)
        self.assertTrue(any("Scoring proposals" in msg for msg in progress))
        self.assertFalse(any("critique" in msg.lower() for msg in progress))
        self.assertEqual(mock_score.call_args.kwargs["escalation_model"], "gpt-5.5")
        mock_verify.assert_awaited_once()
        mock_rewrite.assert_awaited_once()
        mock_triage.assert_awaited_once()


if __name__ == "__main__":
    unittest.main()
