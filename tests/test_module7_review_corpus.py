import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp


REVIEW_MD = """# AJPS - Example Difference-in-Differences Paper

- Date: 2026-01-03
- Decision: Revise and resubmit
- Source: Editorial decision
- Gmail message id: `abc123`

## Editor
The editor invited a revision and asked for clearer pre-trend diagnostics.

## Reviewer 1
The reviewer was not convinced by the identification strategy for Example Difference-in-Differences Paper because the difference-in-differences design did not report full pre-treatment leads. They wanted an event-study plot and a joint pre-trend test for AJPS-12345.

## Reviewer 2
The reviewer liked the paper but thought the contribution relative to the existing literature needed sharper framing.
"""


PAPER_MATCHES_MD = """# Paper Matches

## Main Review Records

| Journal | Date | Manuscript | Review file | Matched paper file | Match status | Source notes |
|---|---:|---|---|---|---|---|
| AJPS | 2026-01-03 | Example Difference-in-Differences Paper | `ajps/example.md` | `papers/files/example.pdf` | Exact | Gmail attachment |
"""


RAW_REVIEW_MD = """# Raw Gmail Export - Example Difference-in-Differences Paper

- Review file: `ajps/example.md`
- Gmail message id: `abc123`
- Source: Gmail raw body

## Editor
The editor invited a revision based on the raw reports.

## Reviewer #1
This is the raw referee report. The difference-in-differences design needs full pre-treatment leads, a joint pre-trend test, and clearer timing assumptions.

## Reviewer #2
This raw report says the literature contribution is not sharp enough and should explain the novelty over prior work.
"""


def make_archive():
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    (root / "ajps").mkdir()
    (root / "forwarded_or_low_confidence").mkdir()
    (root / "papers" / "files").mkdir(parents=True)
    (root / "papers" / "PAPER_MATCHES.md").write_text(PAPER_MATCHES_MD, encoding="utf-8")
    (root / "ajps" / "example.md").write_text(REVIEW_MD, encoding="utf-8")
    (root / "forwarded_or_low_confidence" / "unclear.md").write_text(REVIEW_MD, encoding="utf-8")
    (root / "papers" / "files" / "example.pdf").write_bytes(b"%PDF-1.4\n")
    return tmp, root


def add_raw_export(root):
    (root / fp.RAW_REVIEW_EXPORT_DIR).mkdir()
    (root / fp.RAW_REVIEW_EXPORT_DIR / "abc123.md").write_text(RAW_REVIEW_MD, encoding="utf-8")


class ReviewCorpusParsingTests(unittest.TestCase):
    def test_parse_review_markdown_extracts_sections_and_metadata(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        record = fp.parse_review_markdown(root / "ajps" / "example.md", archive_root=root)

        self.assertEqual(record["journal"], "AJPS")
        self.assertEqual(record["decision"], "Revise and resubmit")
        self.assertEqual(record["gmail_id"], "abc123")
        self.assertEqual(record["review_file"], "ajps/example.md")
        self.assertEqual(record["source_kind"], fp.REVIEW_MEMORY_SOURCE_KIND)
        self.assertEqual({section["reviewer_id"] for section in record["sections"]}, {"Editor", "Reviewer 1", "Reviewer 2"})

    def test_load_review_corpus_attaches_paper_matches_and_atomizes_issues(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        corpus = fp.load_review_corpus(root)

        self.assertEqual(corpus["stats"]["records"], 1)
        self.assertEqual(corpus["stats"]["excluded_low_confidence_records"], 1)
        self.assertGreaterEqual(corpus["stats"]["issues"], 3)
        self.assertEqual(corpus["stats"]["records_with_papers"], 1)
        self.assertEqual(corpus["stats"]["matched_pdf_files"], 1)
        first_record = corpus["records"][0]
        self.assertTrue(first_record["paper_id"].startswith("paper_"))
        self.assertIn("papers/files/example.pdf", first_record["matched_paper_files"][0])
        first_issue = corpus["issues"][0]
        self.assertTrue(first_issue["paper_id"].startswith("paper_"))
        self.assertNotIn("example_difference", first_issue["paper_id"])
        self.assertEqual(first_issue["source_kind"], fp.REVIEW_MEMORY_SOURCE_KIND)
        self.assertTrue(first_issue["paper_section"])
        self.assertIn(first_issue["reviewer_confidence"], {"low", "medium", "high"})
        issue_text_blob = "\n".join(issue["issue_text"] for issue in corpus["issues"])
        self.assertNotIn("Example Difference-in-Differences Paper", issue_text_blob)
        self.assertNotIn("AJPS-12345", issue_text_blob)
        issue_types = {issue["issue_type"] for issue in corpus["issues"]}
        self.assertIn("identification", issue_types)
        self.assertIn("theory", issue_types)

    def test_low_confidence_records_are_explicit_opt_in(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        default_corpus = fp.load_review_corpus(root)
        inclusive_corpus = fp.load_review_corpus(root, include_low_confidence=True)

        self.assertEqual(default_corpus["stats"]["records"], 1)
        self.assertEqual(inclusive_corpus["stats"]["records"], 2)
        self.assertTrue(any(issue["quality_flag"] == "low_confidence" for issue in inclusive_corpus["issues"]))

    def test_raw_gmail_export_replaces_digest_sections_when_present(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        add_raw_export(root)

        corpus = fp.load_review_corpus(root)

        self.assertEqual(corpus["stats"]["raw_review_records"], 1)
        self.assertEqual(corpus["records"][0]["source_kind"], fp.RAW_REVIEW_SOURCE_KIND)
        issue_text = "\n".join(issue["issue_text"] for issue in corpus["issues"])
        self.assertIn("full pre-treatment leads", issue_text)
        self.assertNotIn("Example Difference-in-Differences Paper", issue_text)


class ReviewMemoryTests(unittest.TestCase):
    def test_retrieve_similar_issues_and_build_context(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        corpus = fp.load_review_corpus(root)

        results = fp.retrieve_similar_review_issues(
            "parallel trends and pre-treatment leads are missing",
            corpus,
            top_k=2,
            design_type="difference_in_differences",
        )
        context = fp.build_review_memory_context(
            "parallel trends and pre-treatment leads are missing",
            corpus,
            top_k=2,
            design_type="difference_in_differences",
        )
        unrelated = fp.retrieve_similar_review_issues(
            "galaxy spectra and supernova brightness",
            corpus,
            top_k=2,
        )
        unrelated_context = fp.build_review_memory_context(
            "galaxy spectra and supernova brightness",
            corpus,
            top_k=2,
        )

        self.assertTrue(results)
        self.assertIn("Historical reviewer examples", context)
        self.assertIn("digest-derived anonymized issue pattern", context)
        self.assertIn("Do not import facts", context)
        self.assertNotIn("Example Difference-in-Differences Paper", context)
        self.assertNotIn("AJPS-12345", context)
        self.assertEqual(unrelated, [])
        self.assertEqual(unrelated_context, "")

    def test_reviewer_likelihood_and_eval_metrics_are_local(self):
        human = [
            {
                "atomic_issue_id": "h1",
                "issue_text": "The paper does not report full pre-treatment leads for the difference-in-differences design.",
                "decision_tier": "major_revision_issue",
            },
            {
                "atomic_issue_id": "h2",
                "issue_text": "The contribution relative to the existing literature is unclear.",
                "decision_tier": "minor_revision_issue",
            },
        ]
        generated = [
            {
                "id": 1,
                "text": "Problem: The DD design needs full pre-treatment leads and a joint pre-trend test.",
            }
        ]
        metrics = fp.compare_generated_to_human_issues(generated, human, top_k=1)

        self.assertGreater(metrics["human_issue_recall_at_k"], 0)
        self.assertGreater(metrics["major_issue_recall_at_k"], 0)
        self.assertEqual(metrics["matches"][0]["label"], "matched")

    def test_semantic_eval_matches_paraphrased_reviewer_concern(self):
        human = [
            {
                "atomic_issue_id": "h1",
                "issue_text": "The paper needs a theory of the conditions under which authoritarian governments are meaningfully responsive, and should better differentiate the contribution from prior GDR petition work.",
                "issue_type": "theory",
                "decision_tier": "major_revision_issue",
                "paper_section": "theory_framing",
            }
        ]
        generated = [
            {
                "id": 1,
                "text": "Problem: The theory and contribution are underspecified because the manuscript does not explain when petitions lead autocrats to provide substantive responsiveness rather than merely replicate prior work.",
                "issue_type": "theory",
            }
        ]

        metrics = fp.compare_generated_to_human_issues(generated, human, top_k=1)

        self.assertIn(metrics["matches"][0]["label"], {"matched", "partially_matched"})
        self.assertIn("theory_development", metrics["matches"][0]["shared_concepts"])
        self.assertGreater(metrics["reviewer_likelihood_precision_at_k"], 0)

    def test_semantic_eval_does_not_match_unrelated_concern(self):
        human = [
            {
                "atomic_issue_id": "h1",
                "issue_text": "The difference-in-differences design needs full pre-treatment leads and a joint pre-trend test.",
                "issue_type": "identification",
                "decision_tier": "major_revision_issue",
            }
        ]
        generated = [
            {
                "id": 1,
                "text": "Problem: The figure titles and legends should be clearer for readers.",
                "issue_type": "presentation",
            }
        ]

        metrics = fp.compare_generated_to_human_issues(generated, human, top_k=1)

        self.assertEqual(metrics["matches"][0]["label"], "novel_or_unmatched")
        self.assertEqual(metrics["reviewer_likelihood_precision_at_k"], 0)

    def test_annotation_keeps_reviewer_likelihood_separate(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        corpus = fp.load_review_corpus(root)
        proposals = [
            {
                "id": 1,
                "issue_family": "identification_design",
                "dimension": "logical_soundness",
                "text": "Problem: The paper does not report full pre-treatment leads for the DiD design.",
                "severity": 4,
                "evidence_support": 3,
            }
        ]

        annotated = fp.annotate_reviewer_calibration(proposals, corpus)

        self.assertIn("reviewer_likelihood_score", annotated[0])
        self.assertIn("decision_risk_score", annotated[0])
        self.assertIn("similar_issue_ids", annotated[0])
        self.assertEqual(proposals[0]["severity"], annotated[0]["severity"])

    def test_rewrite_messages_forbid_new_facts(self):
        messages = fp.build_reviewer_style_rewrite_messages(
            {
                "problem": "Parallel trends are not shown.",
                "minimum_fix": "Report full leads.",
            },
            examples=[{"issue_text": "The design would be stronger with clearer diagnostics."}],
        )
        prompt = messages[1]["content"]

        self.assertIn("Do not add facts", prompt)
        self.assertIn("Historical style examples", prompt)


class ReviewMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_passes_review_memory_to_generation_and_triage(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        evidence = fp.build_deterministic_evidence_index("Section 1\nThe paper uses difference-in-differences.")
        evidence["extracted"] = {
            "research_question": "",
            "research_design": {"design_type": "difference_in_differences", "evidence_ids": ["P001"]},
            "main_claims": [],
            "tables": [],
            "figures": [],
            "appendices": [],
            "limitations": [],
        }
        proposal = {
            "id": 1,
            "dimension": "logical_soundness",
            "issue_family": "identification_design",
            "affected_claim_ids": [],
            "evidence_ids": ["P001"],
            "support_status": "inferred",
            "severity": 4,
            "confidence": "medium",
            "text": "Problem: Parallel trends are not shown. Evidence: P001.",
            "diagnostic_next_steps": ["Report leads."],
            "composite": 4,
            "importance": 4,
            "specificity": 3,
            "actionability": 4,
        }
        selection = fp.rebuild_selection_from_high_quality([proposal], top_k=1)

        with (
            patch("feedback_pipeline.build_manuscript_evidence_map", new_callable=AsyncMock) as mock_map,
            patch("feedback_pipeline.generate_all_proposals", new_callable=AsyncMock) as mock_generate,
            patch("feedback_pipeline.score_all_proposals", new_callable=AsyncMock) as mock_score,
            patch("feedback_pipeline.select_and_classify", new_callable=AsyncMock) as mock_select,
            patch("feedback_pipeline.run_verification_round", new_callable=AsyncMock) as mock_verify,
            patch("feedback_pipeline.run_constrained_rewrite_round", new_callable=AsyncMock) as mock_rewrite,
            patch("feedback_pipeline.editorial_triage", new_callable=AsyncMock) as mock_triage,
            patch("feedback_pipeline.meta_review", new_callable=AsyncMock) as mock_meta,
        ):
            mock_map.return_value = evidence
            mock_generate.return_value = ([proposal], [])
            mock_score.return_value = [proposal]
            async def select_side_effect(scored, top_k, tracker=None):
                return fp.rebuild_selection_from_high_quality(scored, top_k)

            async def rewrite_side_effect(items, model=None, tracker=None):
                return items

            mock_select.side_effect = select_side_effect
            mock_verify.return_value = []
            mock_rewrite.side_effect = rewrite_side_effect
            mock_triage.return_value = (
                {
                    "editorial_diagnosis": "mostly_major_revision_issues",
                    "decision_summary": "Mock triage.",
                    "classified_issues": [],
                    "main_report_issue_ids": [],
                    "problem_issue_ids": [],
                    "non_blocking_issue_ids": [],
                    "dropped_issue_ids": [],
                    "rejection_level_count": 0,
                },
                [],
            )
            mock_meta.return_value = "## Editorial Summary\n\nMock."

            result = await fp.full_feedback_pipeline(
                "Section 1\nThe paper uses difference-in-differences.",
                review_corpus_path=str(root),
            )

        self.assertTrue(mock_generate.call_args.kwargs["review_memory_context"])
        self.assertTrue(mock_triage.call_args.kwargs["review_memory_context"])
        triage_selection = mock_triage.call_args.args[0]
        self.assertIn("reviewer_likelihood_score", triage_selection["high_quality"][0])
        self.assertEqual(result["review_corpus"]["stats"]["records"], 1)
        self.assertIn("reviewer_likelihood_score", result["scored"][0])


class HistoricalReviewEvalTests(unittest.IsolatedAsyncioTestCase):
    async def test_holdout_splits_exclude_heldout_paper_from_training_corpus(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        corpus = fp.load_review_corpus(root)

        splits = fp.build_review_holdout_splits(corpus, require_existing_pdf=False)

        self.assertEqual(len(splits), 1)
        heldout_id = splits[0]["paper_id"]
        train = fp.filter_review_corpus_for_holdout(corpus, heldout_id)
        self.assertEqual(train["holdout_paper_id"], heldout_id)
        self.assertEqual(train["stats"]["heldout_records"], 1)
        self.assertFalse(any(record["paper_id"] == heldout_id for record in train["records"]))
        self.assertFalse(any(issue["paper_id"] == heldout_id for issue in train["issues"]))

    async def test_eval_dry_run_estimates_cost_and_does_not_call_pipeline(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        output_path = root / "eval_plan.json"

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch("feedback_pipeline.full_feedback_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Section 1\nThe paper uses difference-in-differences and reports treatment effects.",
                "ok",
            )
            result = await fp.run_historical_review_eval(
                root,
                output_path=output_path,
                max_splits=1,
                require_existing_pdf=False,
                run_api=False,
            )

        self.assertTrue(output_path.exists())
        self.assertEqual(result["summary"]["mode"], "dry_run")
        self.assertEqual(result["summary"]["splits"], 1)
        self.assertGreater(result["summary"]["total_estimated_cost_usd"], 0)
        self.assertEqual(result["splits"][0]["status"], "dry_run_estimated")
        self.assertFalse(mock_pipeline.called)
        rendered = fp.render_historical_review_eval_summary(result)
        self.assertIn("Historical Review Evaluation", rendered)

    async def test_eval_api_mode_uses_filtered_corpus_and_compares_to_human_issues(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        async def fake_pipeline(*args, **kwargs):
            train_corpus = kwargs["review_corpus"]
            heldout_id = train_corpus["holdout_paper_id"]
            self.assertFalse(any(issue["paper_id"] == heldout_id for issue in train_corpus["issues"]))
            return {
                "selection": {
                    "high_quality": [
                        {
                            "id": 1,
                            "text": "Problem: The difference-in-differences design lacks full pre-treatment leads and a joint pre-trend test.",
                        }
                    ]
                },
                "actual_usage": {"total_cost_usd": 0.01},
            }

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch("feedback_pipeline.full_feedback_pipeline", new=AsyncMock(side_effect=fake_pipeline)) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Section 1\nThe paper uses difference-in-differences and discusses pre-treatment trends.",
                "ok",
            )
            result = await fp.run_historical_review_eval(
                root,
                max_splits=1,
                require_existing_pdf=False,
                run_api=True,
            )

        self.assertTrue(mock_pipeline.called)
        self.assertEqual(result["summary"]["mode"], "api")
        self.assertEqual(result["summary"]["api_evaluated_splits"], 1)
        self.assertEqual(result["splits"][0]["status"], "api_evaluated")
        self.assertEqual(result["splits"][0]["generated_issue_count"], 1)
        self.assertIn("generated_issue_summaries", result["splits"][0])
        self.assertGreater(result["splits"][0]["metrics"]["human_issue_recall_at_k"], 0)


if __name__ == "__main__":
    unittest.main()
