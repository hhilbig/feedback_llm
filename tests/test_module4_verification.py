import unittest
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp


def sample_evidence_map():
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
        "main_claims": [
            {
                "claim_id": "C001",
                "claim": "The policy reduced evictions.",
                "evidence_ids": ["TBL001"],
                "support_status": "partial",
            }
        ],
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


def proposal(proposal_id=1):
    return {
        "id": proposal_id,
        "dimension": "logical_soundness",
        "issue_family": "identification_design",
        "affected_claim_ids": ["C001"],
        "evidence_ids": ["P001", "TBL001"],
        "support_status": "inferred",
        "severity": 4,
        "confidence": "medium",
        "text": "Problem: Parallel trends evidence is unclear. Evidence: P001, TBL001.",
        "diagnostic_next_steps": ["Check pre-trends.", "Clarify comparison group."],
        "importance": 4.0,
        "specificity": 4.0,
        "actionability": 4.0,
        "uniqueness": 3.0,
        "composite": 4.0,
        "composite_raw": 4.0,
    }


def verification(original_id, decision, support, severity=3):
    return {
        "original_id": original_id,
        "decision": decision,
        "support_assessment": support,
        "verified_severity": severity,
        "severity_rationale": "Severity checked against evidence.",
        "supported_evidence_ids": ["P001"],
        "missing_or_invalid_evidence_ids": [],
        "counter_evidence_ids": [],
        "actionability_ok": True,
        "confidence": "high",
        "rewrite_guidance": "Keep the inferential caveat.",
        "rationale": "The issue follows from the design description.",
    }


class VerificationSchemaTests(unittest.TestCase):
    def test_verification_schema_is_strict(self):
        schema = fp.VERIFICATION_SCHEMA

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"].keys()))
        self.assertIn("contradicted", schema["properties"]["support_assessment"]["enum"])


class VerificationDecisionTests(unittest.TestCase):
    def test_apply_decisions_keeps_demotes_and_removes(self):
        proposals = [proposal(1), proposal(2), proposal(3)]
        verifications = [
            verification(1, "keep", "supported", severity=4),
            verification(2, "demote", "partially_supported", severity=2),
            verification(3, "remove", "unsupported", severity=1),
        ]

        kept, stats = fp.apply_verification_decisions(proposals, verifications)

        self.assertEqual(stats, {"kept": 1, "demoted": 1, "removed": 1})
        self.assertEqual([p["id"] for p in kept], [1, 2])
        self.assertEqual(kept[0]["verification_status"], "keep")
        self.assertEqual(kept[0]["evidence_ids"], ["P001"])
        self.assertEqual(kept[1]["verification_status"], "demote")
        self.assertLess(kept[1]["composite"], 4.0)

    def test_contradicted_keep_is_removed(self):
        kept, stats = fp.apply_verification_decisions(
            [proposal(1)],
            [verification(1, "keep", "contradicted", severity=1)],
        )

        self.assertEqual(kept, [])
        self.assertEqual(stats["removed"], 1)


class VerificationCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_verify_single_uses_strict_schema(self):
        expected = verification(1, "keep", "inferential", severity=4)

        with patch("feedback_pipeline.chat_json_with_retry", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = expected
            result = await fp.verify_single_proposal(
                sample_evidence_map(),
                proposal(1),
                model="gpt-5.4-nano",
            )

        self.assertEqual(result["original_id"], 1)
        self.assertIs(mock_chat.call_args.kwargs["schema"], fp.VERIFICATION_SCHEMA)
        self.assertEqual(mock_chat.call_args.kwargs["schema_name"], "feedback_verification")

    async def test_constrained_rewrite_preserves_scores_and_verification_metadata(self):
        original = proposal(1)
        original["verification"] = verification(1, "keep", "inferential", severity=4)
        original["verification_status"] = "keep"
        rewritten = {
            "id": 1,
            "dimension": "logical_soundness",
            "issue_family": "identification_design",
            "affected_claim_ids": ["C001"],
            "evidence_ids": ["P001"],
            "support_status": "inferred",
            "severity": 4,
            "confidence": "medium",
            "text": "Problem: The identifying assumption needs clearer support. Evidence: P001.",
            "diagnostic_next_steps": ["Check pre-trends.", "Clarify comparison group."],
        }

        with patch("feedback_pipeline.chat_json_with_retry", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = rewritten
            result = await fp.rewrite_single_verified_proposal(
                original,
                model="gpt-5.4-nano",
            )

        self.assertEqual(result["composite"], 4.0)
        self.assertEqual(result["verification_status"], "keep")
        self.assertEqual(result["original_text"], original["text"])
        self.assertIs(mock_chat.call_args.kwargs["schema"], fp.FEEDBACK_PROPOSAL_SCHEMA)
        self.assertEqual(mock_chat.call_args.kwargs["schema_name"], "verified_feedback_rewrite")


class CostStageTests(unittest.TestCase):
    def test_estimate_has_verification_and_rewrite_not_old_loop(self):
        estimate = fp.estimate_cost_before_run(
            "Section 1 Introduction\nThis is a short paper.",
            num_agents=8,
            top_k=3,
        )
        stages = estimate["stages"]

        self.assertIn("verification", stages)
        self.assertIn("rewrite", stages)
        self.assertNotIn("critique", stages)
        self.assertNotIn("revision", stages)
        self.assertNotIn("re_scoring", stages)


if __name__ == "__main__":
    unittest.main()
