import unittest
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp


def proposal(proposal_id=1, issue_family="identification_design", evidence_ids=None):
    return {
        "id": proposal_id,
        "dimension": "logical_soundness",
        "issue_family": issue_family,
        "affected_claim_ids": [f"C{proposal_id:03d}"],
        "evidence_ids": evidence_ids or [f"P{proposal_id:03d}"],
        "support_status": "inferred",
        "severity": 4,
        "confidence": "medium",
        "text": "Problem: The identifying assumption needs clearer support.",
        "diagnostic_next_steps": ["Check the design assumption."],
        "importance": 4,
        "specificity": 4,
        "actionability": 4,
        "uniqueness": 3,
        "composite": 4,
    }


def score_payload(
    severity=4,
    evidence_support=4,
    actionability=4,
    confidence="medium",
    identification=4,
    measurement=1,
    interpretation=2,
    theory=1,
):
    return {
        "identification_risk": identification,
        "measurement_sample_risk": measurement,
        "interpretation_risk": interpretation,
        "theory_contribution_risk": theory,
        "evidence_support": evidence_support,
        "actionability": actionability,
        "severity": severity,
        "confidence": confidence,
        "rationale": "Domain score rationale.",
    }


class DomainScoringTests(unittest.IsolatedAsyncioTestCase):
    async def test_score_single_uses_domain_schema_and_compatibility_fields(self):
        with patch("feedback_pipeline.chat_json_with_retry", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = [
                score_payload(severity=5, evidence_support=4, actionability=4),
                score_payload(severity=3, evidence_support=2, actionability=4),
            ]
            scored = await fp.score_single_proposal_two_pass(
                "Paper text",
                proposal(1),
                model="gpt-5.4-nano",
            )

        self.assertEqual(scored["severity"], 4.0)
        self.assertEqual(scored["evidence_support"], 3.0)
        self.assertEqual(scored["importance"], scored["severity"])
        self.assertEqual(scored["specificity"], scored["evidence_support"])
        self.assertIn("identification_risk", scored)
        self.assertIn("risk_max", scored)
        self.assertIs(mock_chat.call_args.kwargs["schema"], fp.SCORING_SCHEMA)
        self.assertEqual(mock_chat.call_args.kwargs["schema_name"], "domain_feedback_score")

    async def test_score_all_escalates_severe_ambiguous_cases(self):
        with patch("feedback_pipeline.chat_json_with_retry", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = [
                score_payload(severity=5, evidence_support=2, confidence="low"),
                score_payload(severity=5, evidence_support=2, confidence="low"),
                score_payload(severity=4, evidence_support=4, confidence="high"),
            ]
            scored = await fp.score_all_proposals(
                "Paper text",
                [proposal(1)],
                model="gpt-5.4-nano",
                escalation_model="gpt-5.5",
            )

        self.assertTrue(scored[0]["score_escalated"])
        self.assertEqual(scored[0]["score_escalation_model"], "gpt-5.5")
        self.assertEqual(mock_chat.call_count, 3)

    def test_should_not_escalate_clear_supported_low_risk_case(self):
        scored = {
            **proposal(1, issue_family="writing_structure"),
            **score_payload(severity=2, evidence_support=5, confidence="high", identification=1),
            "scorer_confidence": "high",
            "judge_disagreement": {key: 0 for key in fp.DOMAIN_SCORING_KEYS},
        }

        self.assertFalse(fp.should_escalate_scoring(scored, scored))


class DeduplicationProtectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_dedup_preserves_severe_evidence_distinct_minority_issue(self):
        p1 = proposal(1, evidence_ids=["P001"])
        p2 = proposal(2, evidence_ids=["TBL001"])
        p1["composite"] = 4.5
        p2["composite"] = 4.0

        with patch("feedback_pipeline.embed_texts", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
            kept, removed = await fp.deduplicate_proposals(
                [p1, p2],
                similarity_threshold=0.82,
            )

        self.assertEqual(removed, 0)
        self.assertEqual({p["id"] for p in kept}, {1, 2})

    async def test_dedup_removes_nonprotected_duplicate(self):
        p1 = proposal(1, issue_family="writing_structure", evidence_ids=["P001"])
        p2 = proposal(2, issue_family="writing_structure", evidence_ids=["P001"])
        p1["severity"] = 2
        p2["severity"] = 2

        with patch("feedback_pipeline.embed_texts", new_callable=AsyncMock) as mock_embed:
            mock_embed.return_value = [[1.0, 0.0], [1.0, 0.0]]
            kept, removed = await fp.deduplicate_proposals([p1, p2])

        self.assertEqual(len(kept), 1)
        self.assertEqual(removed, 1)


class PresentationClusteringTests(unittest.IsolatedAsyncioTestCase):
    async def test_cluster_proposals_annotates_without_synthesizing_or_dropping(self):
        p1 = proposal(1)
        p2 = proposal(2)
        p1["_embedding"] = [1.0, 0.0]
        p2["_embedding"] = [1.0, 0.0]

        clustered, num_clusters = await fp.cluster_proposals([p1, p2])

        self.assertEqual(len(clustered), 2)
        self.assertEqual(num_clusters, 1)
        self.assertEqual({p["cluster_size"] for p in clustered}, {2})
        self.assertEqual({tuple(p["source_ids"]) for p in clustered}, {(1, 2)})
        self.assertTrue(all("Problem:" in p["text"] for p in clustered))


class CostEstimateTests(unittest.TestCase):
    def test_estimate_includes_score_escalation(self):
        estimate = fp.estimate_cost_before_run(
            "Section 1 Introduction\nThis paper uses difference-in-differences.",
            num_agents=8,
            top_k=3,
        )

        self.assertIn("score_escalation", estimate["stages"])
        self.assertEqual(estimate["stages"]["score_escalation"]["model"], "gpt-5.6-sol")


if __name__ == "__main__":
    unittest.main()
