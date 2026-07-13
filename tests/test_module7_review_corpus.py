import json
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


REVIEW_PRIOR_ARTIFACT = {
    "artifact_version": "v1",
    "privacy_audit": {
        "passed": True,
        "max_raw_review_ngram_overlap": 6,
        "identifiable_setting_flags": 0,
    },
    "heldout_eval": {
        "major_issue_recall_at_8_delta": 0.05,
        "unsupported_claim_rate_delta": 0.0,
        "duplicate_laundry_list_rate_delta": -0.01,
    },
    "priors": [
        {
            "prior_id": "did_pretrend_diagnostics",
            "use_for": {
                "generation_checklist": True,
                "triage_calibration": True,
                "style_rewrite": False,
            },
            "applies_when": {
                "design": ["difference_in_differences", "event_study"],
                "claim_type": ["causal"],
            },
            "reviewer_concern": (
                "Reviewers scrutinize whether pre-treatment trend diagnostics are transparent "
                "enough to support a causal DD claim."
            ),
            "raise_if_missing": [
                "no full pre-treatment lead table",
                "no inference-level justification",
            ],
            "demote_if_present": [
                "full lead table reported",
                "raw group trends shown",
                "joint pre-trend test reported",
            ],
            "suppress_if": [
                "claim is descriptive",
                "design does not make a causal claim",
            ],
            "decision_tier_prior": {
                "potential_rejection": 0.20,
                "major_revision": 0.55,
                "minor_revision": 0.20,
                "nice_to_have": 0.05,
            },
            "rejection_trigger": (
                "Becomes rejection-relevant if leads diverge or inference is not credible."
            ),
            "minimum_fix": [
                "show full pre-treatment leads",
                "justify inference level",
            ],
            "reviewer_agreement": "medium",
            "known_disagreement": (
                "Some reviewers accept this if robustness checks and raw trends are clear."
            ),
            "support": {
                "paper_support": "medium",
                "comment_support": "high",
                "editor_signal": "some",
            },
            "privacy_status": "safe_abstracted",
        }
    ],
}


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


def add_second_review_record(root):
    second = REVIEW_MD.replace(
        "Example Difference-in-Differences Paper",
        "Second Difference-in-Differences Paper",
    ).replace("abc123", "def456")
    (root / "ajps" / "second.md").write_text(second, encoding="utf-8")
    (root / "papers" / "files" / "second.pdf").write_bytes(b"%PDF-1.4\n")
    matches_path = root / "papers" / "PAPER_MATCHES.md"
    matches = matches_path.read_text(encoding="utf-8")
    matches += (
        "| AJPS | 2026-01-04 | Second Difference-in-Differences Paper | "
        "`ajps/second.md` | `papers/files/second.pdf` | Exact | Gmail attachment |\n"
    )
    matches_path.write_text(matches, encoding="utf-8")


def json_clone(value):
    return json.loads(json.dumps(value))


def evidence_map_for(text, design_type="difference_in_differences"):
    evidence = fp.build_deterministic_evidence_index(text)
    evidence["extracted"] = {
        "research_design": {
            "design_type": design_type,
            "rationale": "",
            "evidence_ids": [],
        }
    }
    profile = fp.build_substantive_design_profile(evidence)
    evidence["substantive_profile"] = profile
    evidence["substantive_checks"] = fp.build_substantive_checklist_findings(evidence, profile)
    return evidence


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

    def test_parse_review_markdown_accepts_indented_referee_headings(self):
        tmp = tempfile.TemporaryDirectory()
        root = Path(tmp.name)
        self.addCleanup(tmp.cleanup)
        (root / "other_journals").mkdir()
        path = root / "other_journals" / "jrssa.md"
        path.write_text(
            """    # JRSSA - Example Paper

    - Date: 2026-01-03
    - Decision: Revise and resubmit

    ## Associate Editor
The associate editor summarized the main methodological concern.

## Referee 1
The referee asked for more simulation evidence and clearer derivations.

## Referee 2 attachment
The attached report raised concerns about confidence interval coverage.
""",
            encoding="utf-8",
        )

        record = fp.parse_review_markdown(path, archive_root=root)

        self.assertEqual(record["journal"], "JRSSA")
        self.assertEqual(record["manuscript"], "Example Paper")
        self.assertEqual(
            [section["reviewer_id"] for section in record["sections"]],
            ["Associate Editor", "Referee 1", "Referee 2 attachment"],
        )

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

    def test_human_target_cleanup_excludes_non_substantive_atoms(self):
        human = [
            {
                "atomic_issue_id": "drop_title",
                "issue_text": "Petitions and Housing Construction in the GDR. They are attached below.",
                "issue_type": "other",
                "quality_flag": "use_for_style_only",
            },
            {
                "atomic_issue_id": "drop_decision",
                "issue_text": "Based on these reviews, I am afraid that I cannot accept this paper for publication.",
                "issue_type": "other",
                "quality_flag": "use",
            },
            {
                "atomic_issue_id": "drop_praise",
                "issue_text": "The paper is well-written, clear, and uses appropriate methods within the boundaries of the available data.",
                "issue_type": "measurement",
                "quality_flag": "use",
            },
            {
                "atomic_issue_id": "drop_citation",
                "issue_text": "See either Dimitrov 2023 or the work on Chinese air pollution.",
                "issue_type": "other",
                "quality_flag": "use_for_style_only",
            },
            {
                "atomic_issue_id": "keep_novelty",
                "issue_text": "The contribution remains insufficiently novel relative to the existing literature.",
                "issue_type": "theory",
                "quality_flag": "use",
            },
            {
                "atomic_issue_id": "keep_figure",
                "issue_text": "Figure 1 would benefit from the inclusion of confidence intervals.",
                "issue_type": "presentation",
                "quality_flag": "use_for_style_only",
            },
            {
                "atomic_issue_id": "keep_question",
                "issue_text": "Were the units the same across the country or did they vary over geography?",
                "issue_type": "other",
                "quality_flag": "use_for_style_only",
            },
        ]

        targets, excluded = fp.filter_human_review_target_issues(human)

        self.assertEqual(
            {issue["atomic_issue_id"] for issue in targets},
            {"keep_novelty", "keep_figure", "keep_question"},
        )
        self.assertEqual(len(excluded), 4)
        reasons = {item["reason"] for item in excluded}
        self.assertIn("title_or_attachment_fragment", reasons)
        self.assertIn("editor_or_decision_boilerplate", reasons)
        self.assertIn("generic_praise", reasons)
        self.assertIn("citation_fragment", reasons)

    def test_human_issue_clusters_deduplicate_repeated_concerns(self):
        human = [
            {
                "atomic_issue_id": "h1",
                "issue_text": "The contribution relative to the existing literature is unclear and needs sharper framing.",
                "issue_type": "theory",
                "decision_tier": "major_revision_issue",
                "paper_section": "theory_framing",
                "reviewer_id": "Reviewer 1",
                "review_file": "ajps/example.md",
            },
            {
                "atomic_issue_id": "h2",
                "issue_text": "The paper should better differentiate its contribution from prior work and explain what is new.",
                "issue_type": "theory",
                "decision_tier": "major_revision_issue",
                "paper_section": "theory_framing",
                "reviewer_id": "Reviewer 2",
                "review_file": "ajps/example.md",
            },
            {
                "atomic_issue_id": "h3",
                "issue_text": "The difference-in-differences design needs full pre-treatment leads and a joint pre-trend test.",
                "issue_type": "identification",
                "decision_tier": "major_revision_issue",
                "paper_section": "identification",
                "reviewer_id": "Reviewer 1",
                "review_file": "ajps/example.md",
            },
        ]
        generated = [
            {
                "id": 1,
                "text": "Problem: The novelty and contribution are still underspecified relative to prior work.",
                "issue_type": "theory",
            },
            {
                "id": 2,
                "text": "Problem: The manuscript needs a sharper account of what is new in the existing literature.",
                "issue_type": "theory",
            },
        ]

        clusters = fp.cluster_human_review_issues(human)
        metrics = fp.compare_generated_to_human_issues(generated, human, top_k=2)

        self.assertEqual(len(clusters), 2)
        self.assertEqual(metrics["human_issue_cluster_count"], 2)
        self.assertEqual(metrics["matched_human_issue_cluster_count"], 1)
        self.assertEqual(metrics["human_issue_cluster_recall_at_k"], 0.5)
        self.assertEqual(metrics["human_issue_candidate_count"], 3)
        self.assertEqual(metrics["human_issue_target_count"], 3)
        self.assertEqual(metrics["reviewer_likelihood_precision_at_k"], 1.0)
        self.assertEqual(metrics["deduplicated_reviewer_likelihood_precision_at_k"], 0.5)
        self.assertEqual(metrics["duplicate_generated_cluster_matches"], 1)
        self.assertEqual(metrics["matches"][0]["best_human_cluster_size"], 2)

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


class ReviewPriorArtifactTests(unittest.TestCase):
    def test_review_prior_audit_accepts_bucketed_safe_artifact(self):
        audit = fp.audit_review_prior_artifact(
            REVIEW_PRIOR_ARTIFACT,
            require_deployment_gate=True,
        )

        self.assertTrue(audit["passed"], audit)
        self.assertEqual(audit["errors"], [])

    def test_review_prior_audit_rejects_exact_counts_and_identifiers(self):
        artifact = json_clone(REVIEW_PRIOR_ARTIFACT)
        prior = artifact["priors"][0]
        prior["support"]["n_papers"] = 6
        prior["reviewer_concern"] = "Hanno's AJPS paper with N=1234 needs this check."

        audit = fp.audit_review_prior_artifact(artifact)

        self.assertFalse(audit["passed"])
        errors = "\n".join(audit["errors"])
        self.assertIn("support.n_papers must be bucketed", errors)
        self.assertIn("author-identifying name", errors)
        self.assertIn("journal-specific label", errors)
        self.assertIn("sample-size-like numeric fingerprint", errors)

    def test_review_prior_gap_context_selects_unsatisfied_post_cold_pass_check(self):
        evidence = evidence_map_for(
            """
            The paper estimates a difference-in-differences design and makes a causal claim.
            The treatment timing varies across groups, but the draft does not discuss pre-period diagnostics.
            """,
        )

        context = fp.build_review_prior_gap_context(
            evidence,
            REVIEW_PRIOR_ARTIFACT,
            existing_issues=[{"text": "Problem: The literature contribution is underspecified."}],
            top_k=3,
        )

        self.assertIn("Structured reviewer-prior gap checks", context)
        self.assertIn("did_pretrend_diagnostics", context)
        self.assertIn("missing_checks_to_inspect", context)
        self.assertIn("Only manuscript evidence IDs can support a critique", context)

    def test_review_prior_gap_context_skips_prior_covered_by_cold_pass(self):
        evidence = evidence_map_for(
            """
            The paper estimates a difference-in-differences design and makes a causal claim.
            The treatment timing varies across groups.
            """,
        )

        context = fp.build_review_prior_gap_context(
            evidence,
            REVIEW_PRIOR_ARTIFACT,
            existing_issues=[
                {
                    "text": (
                        "Problem: Evidence P001 leaves the DD pre-treatment leads and "
                        "inference-level justification unclear."
                    )
                }
            ],
        )

        self.assertEqual(context, "")

    def test_review_prior_assessment_demotes_or_suppresses_when_conditions_apply(self):
        addressed = evidence_map_for(
            """
            The paper estimates a difference-in-differences design and makes a causal claim.
            A full pre-treatment lead table is reported with raw group trends and a joint pre-trend test.
            The paper also justifies the inference level.
            """,
        )
        prior = REVIEW_PRIOR_ARTIFACT["priors"][0]

        addressed_assessment = fp.assess_review_prior_for_evidence(prior, addressed)

        self.assertEqual(addressed_assessment["status"], "demoted")
        self.assertTrue(addressed_assessment["demoted_by"])

        descriptive_prior = json_clone(prior)
        descriptive_prior["applies_when"] = {}
        descriptive = evidence_map_for(
            "This is a descriptive paper and does not make a causal claim.",
            design_type="descriptive",
        )

        descriptive_assessment = fp.assess_review_prior_for_evidence(descriptive_prior, descriptive)

        self.assertEqual(descriptive_assessment["status"], "suppressed")
        self.assertTrue(descriptive_assessment["suppressed_by"])

    def test_distill_review_prior_from_corpus_produces_bucketed_safe_artifact(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        corpus = fp.load_review_corpus(root)

        result = fp.distill_review_prior_from_corpus(
            corpus,
            min_support_papers=1,
            min_support_comments=1,
            artifact_version="test_v1",
        )

        artifact = result["artifact"]
        audit = fp.audit_review_prior_artifact(artifact)

        self.assertTrue(audit["passed"], audit)
        self.assertTrue(artifact["privacy_audit"]["passed"])
        self.assertGreaterEqual(result["summary"]["priors"], 1)
        self.assertIn("source_summary_exact", result["local_audit"])
        self.assertGreaterEqual(result["local_audit"]["source_summary_exact"]["n_target_issues"], 1)
        for prior in artifact["priors"]:
            self.assertEqual(prior["privacy_status"], "safe_abstracted")
            self.assertIn("raise_if_missing", prior)
            self.assertIn("demote_if_present", prior)
            self.assertIn("suppress_if", prior)
            for value in prior["support"].values():
                self.assertIsInstance(value, str)
        artifact_text = json.dumps(artifact)
        self.assertNotIn("Example Difference-in-Differences Paper", artifact_text)
        self.assertNotIn("AJPS-12345", artifact_text)

    def test_write_and_load_distilled_review_prior_outputs(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        corpus = fp.load_review_corpus(root)
        result = fp.distill_review_prior_from_corpus(
            corpus,
            min_support_papers=1,
            min_support_comments=1,
            artifact_version="test_v1",
        )
        artifact_path = root / "private_review_prior.json"
        audit_path = root / "private_review_prior.local_audit.json"

        paths = fp.write_review_prior_distillation(result, artifact_path, audit_path)
        loaded = fp.load_review_prior(artifact_path)

        self.assertTrue(Path(paths["artifact_output"]).exists())
        self.assertTrue(Path(paths["audit_output"]).exists())
        self.assertIn("runtime_audit", loaded)
        self.assertTrue(loaded["runtime_audit"]["passed"])
        local_audit = json.loads(audit_path.read_text(encoding="utf-8"))
        self.assertIn("prior_support_exact", local_audit)


class ReviewMemoryIntegrationTests(unittest.IsolatedAsyncioTestCase):
    async def test_full_pipeline_uses_review_prior_after_cold_pass_only(self):
        evidence = evidence_map_for(
            """
            The paper estimates a difference-in-differences design and makes a causal claim.
            The treatment timing varies across groups, but pre-period diagnostics are not discussed.
            """,
        )
        cold_proposal = {
            "id": 1,
            "dimension": "contribution",
            "issue_family": "theory_contribution",
            "affected_claim_ids": [],
            "evidence_ids": ["P001"],
            "support_status": "partial",
            "severity": 3,
            "confidence": "medium",
            "text": "Problem: The contribution is underspecified. Evidence: P001.",
            "diagnostic_next_steps": ["Clarify the contribution."],
            "composite": 3,
            "importance": 3,
            "specificity": 3,
            "actionability": 3,
        }
        gap_proposal = {
            "id": 2,
            "dimension": "logical_soundness",
            "issue_family": "identification_design",
            "affected_claim_ids": [],
            "evidence_ids": ["P001"],
            "support_status": "inferred",
            "severity": 4,
            "confidence": "medium",
            "text": "Problem: The DD pre-trend diagnostics are missing. Evidence: P001.",
            "diagnostic_next_steps": ["Report full pre-treatment leads."],
            "composite": 4,
            "importance": 4,
            "specificity": 3,
            "actionability": 4,
        }

        async def select_side_effect(scored, top_k, tracker=None):
            return fp.rebuild_selection_from_high_quality(scored, top_k)

        async def rewrite_side_effect(items, model=None, tracker=None):
            return items

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
            mock_generate.side_effect = [([cold_proposal], []), ([gap_proposal], [])]
            mock_score.return_value = [cold_proposal, gap_proposal]
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
                "Paper text",
                review_prior=json_clone(REVIEW_PRIOR_ARTIFACT),
                review_prior_top_k=1,
            )

        self.assertEqual(mock_generate.await_count, 2)
        cold_call = mock_generate.await_args_list[0]
        gap_call = mock_generate.await_args_list[1]
        self.assertEqual(cold_call.kwargs.get("review_prior_gap_context", ""), "")
        self.assertEqual(cold_call.kwargs.get("review_memory_context", ""), "")
        self.assertIn("Structured reviewer-prior gap checks", gap_call.kwargs["review_prior_gap_context"])
        self.assertEqual(gap_call.kwargs.get("review_memory_context", ""), "")
        self.assertTrue(mock_triage.call_args.kwargs["review_prior_context"])
        self.assertEqual(result["review_prior"]["gap_checks_selected"], 1)
        self.assertEqual(result["review_prior"]["gap_proposals_generated"], 1)
        self.assertEqual(result["proposals"][1]["generation_source"], "review_prior_gap")
        self.assertEqual(result["proposals"][1]["review_prior_id"], "did_pretrend_diagnostics")

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
                "Title\nHanno Hilbig\nhanno@example.edu\n\nAbstract\nThe paper uses difference-in-differences and reports treatment effects.",
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
        self.assertTrue(result["splits"][0]["api_redaction"]["enabled"])
        self.assertGreater(result["splits"][0]["api_redaction"]["redactions"]["title_page_blocks"], 0)
        self.assertGreater(result["splits"][0]["human_issue_candidate_count"], 0)
        self.assertGreater(result["splits"][0]["human_issue_cluster_count"], 0)
        self.assertFalse(mock_pipeline.called)
        rendered = fp.render_historical_review_eval_summary(result)
        self.assertIn("Historical Review Evaluation", rendered)
        self.assertIn("Targets", rendered)
        self.assertIn("Clusters", rendered)

    async def test_eval_api_mode_uses_filtered_corpus_and_compares_to_human_issues(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        async def fake_pipeline(*args, **kwargs):
            self.assertNotIn("Hanno Hilbig", args[0])
            self.assertNotIn("hanno@example.edu", args[0])
            self.assertIn("[title page and author metadata redacted]", args[0])
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
                "Title\nHanno Hilbig\nhanno@example.edu\n\nAbstract\nThe paper uses difference-in-differences and discusses pre-treatment trends.",
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
        self.assertGreater(result["splits"][0]["metrics"]["human_issue_target_count"], 0)
        self.assertGreater(result["splits"][0]["metrics"]["human_issue_cluster_recall_at_k"], 0)
        self.assertIn("mean_human_issue_cluster_recall_at_k", result["summary"])
        self.assertIn("mean_deduplicated_reviewer_likelihood_precision_at_k", result["summary"])

    async def test_review_prior_eval_gate_dry_run_estimates_three_modes(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        add_second_review_record(root)
        output_path = root / "eval_gate_plan.json"

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch("feedback_pipeline.full_feedback_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Title\nHanno Hilbig\nhanno@example.edu\n\nAbstract\nThe paper uses difference-in-differences and reports treatment effects.",
                "ok",
            )
            result = await fp.run_review_prior_eval_gate(
                root,
                output_path=output_path,
                max_splits=1,
                require_existing_pdf=False,
                run_api=False,
                review_prior_min_papers=1,
                review_prior_min_comments=1,
            )

        self.assertTrue(output_path.exists())
        self.assertEqual(result["summary"]["run_mode"], "dry_run")
        self.assertEqual(set(result["modes"]), {"baseline", "safe_prior", "local_raw_memory"})
        for mode in result["modes"].values():
            self.assertEqual(mode["summary"]["splits"], 1)
            self.assertGreaterEqual(mode["summary"]["total_estimated_cost_usd"], 0)
        safe_split = result["modes"]["safe_prior"]["splits"][0]
        self.assertIn(safe_split["status"], {"dry_run_estimated", "skipped_no_train_prior"})
        self.assertFalse(mock_pipeline.called)
        rendered = fp.render_review_prior_eval_gate_summary(result)
        self.assertIn("Review Prior Evaluation Gate", rendered)
        self.assertIn("baseline", rendered)
        self.assertIn("safe_prior", rendered)

    async def test_review_prior_eval_gate_batch_dry_run_discounts_estimates(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        add_second_review_record(root)

        with patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract:
            mock_extract.return_value = (
                "Title\nHanno Hilbig\nhanno@example.edu\n\nAbstract\nThe paper uses difference-in-differences and reports treatment effects.",
                "ok",
            )
            result = await fp.run_review_prior_eval_gate(
                root,
                max_splits=1,
                require_existing_pdf=False,
                run_api=False,
                batch_api=True,
                review_prior_min_papers=1,
                review_prior_min_comments=1,
            )

        self.assertTrue(result["summary"]["batch_api"])
        baseline_split = result["modes"]["baseline"]["splits"][0]
        self.assertLess(
            baseline_split["estimated_cost_usd"],
            baseline_split["estimated_cost_usd_without_batch_discount"],
        )
        self.assertEqual(baseline_split["batch_api_chat_price_multiplier"], 0.5)

    async def test_review_prior_eval_gate_batch_api_requires_batch_context(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        with self.assertRaisesRegex(ValueError, "openai_batch_chat_context"):
            await fp.run_review_prior_eval_gate(
                root,
                require_existing_pdf=False,
                run_api=True,
                batch_api=True,
            )

    async def test_review_prior_eval_gate_api_mode_compares_modes_without_leakage(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        add_second_review_record(root)

        async def fake_pipeline(paper_text, *args, **kwargs):
            self.assertNotIn("Hanno Hilbig", paper_text)
            self.assertNotIn("hanno@example.edu", paper_text)
            self.assertIn("[title page and author metadata redacted]", paper_text)
            if kwargs.get("review_prior"):
                generated = [
                    {
                        "id": 1,
                        "text": "Problem: The difference-in-differences design lacks full pre-treatment leads and a joint pre-trend test.",
                        "verification_status": "keep",
                        "verified_support": "inferential",
                    }
                ]
            elif kwargs.get("review_corpus"):
                train_corpus = kwargs["review_corpus"]
                heldout_id = train_corpus["holdout_paper_id"]
                self.assertFalse(any(issue["paper_id"] == heldout_id for issue in train_corpus["issues"]))
                generated = [
                    {
                        "id": 1,
                        "text": "Problem: The difference-in-differences design lacks full pre-treatment leads and a joint pre-trend test.",
                        "verification_status": "keep",
                        "verified_support": "inferential",
                    }
                ]
            else:
                generated = [
                    {
                        "id": 1,
                        "text": "Problem: The bibliography contains inconsistent capitalization.",
                        "verification_status": "keep",
                        "verified_support": "direct",
                    }
                ]
            return {
                "selection": {"high_quality": generated},
                "actual_usage": {"total_cost_usd": 0.01},
            }

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch("feedback_pipeline.full_feedback_pipeline", new=AsyncMock(side_effect=fake_pipeline)) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Title\nHanno Hilbig\nhanno@example.edu\n\nAbstract\nThe paper uses difference-in-differences and discusses pre-treatment trends.",
                "ok",
            )
            result = await fp.run_review_prior_eval_gate(
                root,
                max_splits=1,
                require_existing_pdf=False,
                run_api=True,
                review_prior_min_papers=1,
                review_prior_min_comments=1,
            )

        self.assertEqual(mock_pipeline.await_count, 3)
        self.assertEqual(result["summary"]["run_mode"], "api")
        self.assertEqual(result["modes"]["baseline"]["summary"]["api_evaluated_splits"], 1)
        self.assertEqual(result["modes"]["safe_prior"]["summary"]["api_evaluated_splits"], 1)
        self.assertEqual(result["modes"]["local_raw_memory"]["summary"]["api_evaluated_splits"], 1)
        self.assertEqual(result["gate"]["status"], "evaluated")
        self.assertTrue(result["gate"]["major_issue_cluster_recall_at_k"]["safe_prior_delta"] >= 0)
        self.assertGreater(
            result["modes"]["safe_prior"]["summary"]["mean_major_issue_cluster_recall_at_k"],
            result["modes"]["baseline"]["summary"]["mean_major_issue_cluster_recall_at_k"],
        )


if __name__ == "__main__":
    unittest.main()
