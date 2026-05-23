import unittest
from collections import Counter
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp


def sample_evidence_map():
    evidence = fp.build_deterministic_evidence_index(
        """
Section 1 Introduction
We ask whether a city policy reduced evictions.
The paper uses a difference-in-differences design.

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


class DesignAwareAssignmentTests(unittest.TestCase):
    def test_assignments_preserve_role_diversity_and_add_design_specialist(self):
        workers = fp.create_worker_assignments(8, design_type="difference_in_differences")
        role_counts = Counter(worker["role"] for worker in workers)

        self.assertEqual(role_counts["theorist"], 2)
        self.assertEqual(role_counts["rival"], 2)
        self.assertEqual(role_counts["methodologist"], 2)
        self.assertEqual(role_counts["design_specialist"], 1)
        self.assertEqual(role_counts["editor"], 1)
        self.assertTrue(all(worker["design_type"] == "difference_in_differences" for worker in workers))
        self.assertIn("parallel trends", " ".join(worker["persona"] for worker in workers))

    def test_unknown_design_falls_back_to_unclear(self):
        workers = fp.create_worker_assignments(8, design_type="made_up_design")

        self.assertTrue(all(worker["design_type"] == "unclear" for worker in workers))
        self.assertIn("First identify what empirical design", workers[4]["persona"])


class EvidenceAwarePromptTests(unittest.TestCase):
    def test_generation_prompt_uses_evidence_index_and_required_evidence_fields(self):
        evidence = sample_evidence_map()
        prompt = fp._generation_user_prompt(
            evidence["safe_text"],
            worker_id=1,
            evidence_map=evidence,
        )

        self.assertIn("Evidence-indexed manuscript context", prompt)
        self.assertIn("[P001]", prompt)
        self.assertIn("[TBL001]", prompt)
        self.assertIn('"evidence_ids"', prompt)
        self.assertIn('"support_status"', prompt)
        self.assertIn('"affected_claim_ids"', prompt)

    def test_feedback_proposal_schema_is_strict(self):
        schema = fp.FEEDBACK_PROPOSAL_SCHEMA

        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"].keys()))
        self.assertIn("identification_design", schema["properties"]["issue_family"]["enum"])


class GenerationCallTests(unittest.IsolatedAsyncioTestCase):
    async def test_generate_all_uses_schema_and_preserves_evidence_metadata(self):
        evidence = sample_evidence_map()
        worker = fp.create_worker_assignments(8, design_type="difference_in_differences")[6]
        raw_result = {
            "id": 999,
            "dimension": "logical_soundness",
            "issue_family": "identification_design",
            "affected_claim_ids": ["C001"],
            "evidence_ids": ["P001", "TBL001"],
            "support_status": "partial",
            "severity": 5,
            "confidence": "high",
            "text": "Problem: Parallel trends evidence is underspecified. Evidence: P001, TBL001.",
            "diagnostic_next_steps": ["Check event-study pre-trends.", "Clarify comparison group."],
        }

        with patch("feedback_pipeline.chat_json_with_retry", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = raw_result
            proposals, failures = await fp.generate_all_proposals(
                evidence["safe_text"],
                [worker],
                model="gpt-5.4-nano",
                evidence_map=evidence,
            )

        self.assertEqual(failures, [])
        self.assertEqual(proposals[0]["id"], worker["id"])
        self.assertEqual(proposals[0]["role"], "design_specialist")
        self.assertEqual(proposals[0]["evidence_ids"], ["P001", "TBL001"])
        self.assertEqual(proposals[0]["affected_claim_ids"], ["C001"])
        self.assertIs(mock_chat.call_args.kwargs["schema"], fp.FEEDBACK_PROPOSAL_SCHEMA)
        self.assertEqual(mock_chat.call_args.kwargs["schema_name"], "feedback_proposal")


if __name__ == "__main__":
    unittest.main()
