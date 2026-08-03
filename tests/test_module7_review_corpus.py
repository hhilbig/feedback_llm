import csv
import json
import os
import stat
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch

import feedback_pipeline as fp
import review_corpus_manifest as rcm


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


def write_manifest_case_files(root, family_id, case_id, suffix=""):
    manuscript = root / f"manuscript{suffix}.txt"
    feedback = root / f"feedback{suffix}.txt"
    manuscript.write_text(
        f"Abstract\nManuscript evidence for {case_id} uses a panel design.",
        encoding="utf-8",
    )
    feedback.write_text(
        "The reviewer asks for stronger identifying-assumption diagnostics and a falsification test.",
        encoding="utf-8",
    )
    return {
        "family_id": family_id,
        "case_id": case_id,
        "benchmark_tier": "primary",
        "classification": "journal_review",
        "disposition": "evaluation",
        "manuscript_files": [
            {
                "path": manuscript.name,
                "sha256": rcm.sha256_file(manuscript),
                "role": "main",
            }
        ],
        "sources": [
            {
                "source_id": f"source_{case_id}",
                "path": feedback.name,
                "sha256": rcm.sha256_file(feedback),
                "extractor": "text",
                "reviewer_id": f"reviewer_{case_id}",
                "provenance": "human_feedback",
                "source_type": "journal_review",
                "version_match": "exact_submission",
                "disposition": "evaluation",
            }
        ],
    }


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


class PrivateJsonWriterTests(unittest.TestCase):
    def test_private_json_is_atomic_and_private_before_publish(self):
        with tempfile.TemporaryDirectory() as temp:
            output = Path(temp) / "private" / "result.json"
            real_replace = os.replace
            observed = {}

            def inspect_replace(source, destination):
                source_path = Path(source)
                destination_path = Path(destination)
                observed["temporary_mode"] = stat.S_IMODE(source_path.stat().st_mode)
                observed["parent_mode"] = stat.S_IMODE(
                    destination_path.parent.stat().st_mode
                )
                observed["destination_existed"] = destination_path.exists()
                real_replace(source, destination)

            with patch.object(fp.os, "replace", side_effect=inspect_replace):
                written = fp._write_private_json(output, {"private": "payload"})

            self.assertEqual(observed["temporary_mode"], 0o600)
            self.assertEqual(observed["parent_mode"], 0o700)
            self.assertFalse(observed["destination_existed"])
            self.assertEqual(stat.S_IMODE(written.stat().st_mode), 0o600)
            self.assertEqual(
                json.loads(written.read_text(encoding="utf-8")),
                {"private": "payload"},
            )


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

    async def test_manifest_loader_holds_out_every_case_in_same_family(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        manifest = {
            "manifest_version": "v1",
            "corpus_id": "integration_corpus",
            "cases": [
                write_manifest_case_files(root, "family_a", "case_a1", "_a1"),
                write_manifest_case_files(root, "family_a", "case_a2", "_a2"),
                write_manifest_case_files(root, "family_b", "case_b1", "_b1"),
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        root.chmod(0o700)
        for private_file in root.iterdir():
            if private_file.is_file():
                private_file.chmod(0o600)

        with patch.object(rcm, "DEFAULT_PRIVATE_ROOT", root):
            corpus = fp.load_review_corpus(manifest_path)
        first_family_record = next(
            record for record in corpus["records"] if record["family_id"] == "family_a"
        )
        train = fp.filter_review_corpus_for_holdout(
            corpus,
            first_family_record["paper_id"],
            heldout_family_id="family_a",
        )

        self.assertTrue(corpus["manifest_version"])
        self.assertTrue(all(issue["issue_type"] != "unclassified" for issue in corpus["issues"]))
        self.assertFalse(any(record["family_id"] == "family_a" for record in train["records"]))
        self.assertFalse(any(issue["family_id"] == "family_a" for issue in train["issues"]))
        self.assertTrue(any(record["family_id"] == "family_b" for record in train["records"]))

    async def test_manifest_enrichment_is_resealed_and_can_be_written_as_snapshot(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        root = Path(temp.name)
        manifest = {
            "manifest_version": "v1",
            "corpus_id": "enrichment_regression_corpus",
            "cases": [
                write_manifest_case_files(root, "family_a", "case_a1", "_a1")
            ],
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        root.chmod(0o700)
        for private_file in root.iterdir():
            if private_file.is_file():
                private_file.chmod(0o600)

        with patch.object(rcm, "DEFAULT_PRIVATE_ROOT", root):
            corpus = fp.load_review_corpus(manifest_path)
            snapshot = rcm.write_private_corpus(
                corpus,
                "snapshots/enriched.json",
                private_root=root,
            )
            reloaded = fp.load_review_corpus(snapshot)

        self.assertTrue(all(issue["issue_type"] != "unclassified" for issue in corpus["issues"]))
        self.assertEqual(reloaded, corpus)
        self.assertEqual(stat.S_IMODE(snapshot.stat().st_mode), 0o600)

        tampered = json.loads(snapshot.read_text(encoding="utf-8"))
        tampered["issues"][0]["issue_text"] = "tampered after snapshot creation"
        snapshot.write_text(json.dumps(tampered), encoding="utf-8")
        snapshot.chmod(0o600)
        with patch.object(rcm, "DEFAULT_PRIVATE_ROOT", root):
            with self.assertRaises(rcm.ManifestValidationError):
                fp.load_review_corpus(snapshot)

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
        self.assertEqual(result["summary"]["memory_mode"], "none")
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
        portable = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(portable["schema_version"], "portable_review_eval_v1")
        self.assertNotIn(str(root), json.dumps(portable))
        self.assertNotIn("Example Difference-in-Differences Paper", json.dumps(portable))

    async def test_manifest_dry_summary_keeps_all_selected_human_issues(self):
        corpus = {
            "manifest_version": "v1",
            "records": [
                {
                    "paper_id": "paper_short",
                    "family_id": "family_short",
                    "case_id": "case_short",
                    "manifest_case": True,
                    "review_file": "source_short",
                    "disposition": "evaluation",
                    "benchmark_tier": "primary",
                    "classification": "informal_feedback",
                    "source_type": "human_feedback",
                    "matched_paper_files": ["manuscript.txt"],
                    "manuscript_files": ["manuscript.txt"],
                }
            ],
            "issues": [
                {
                    "paper_id": "paper_short",
                    "family_id": "family_short",
                    "case_id": "case_short",
                    "atomic_issue_id": "short",
                    "issue_text": "Why?",
                    "issue_type": "other",
                    "quality_flag": "use_for_style_only",
                    "disposition": "evaluation",
                }
            ],
            "paper_matches": {},
            "excluded_records": [],
            "stats": {
                "records": 1,
                "issues": 1,
                "excluded_low_confidence_records": 0,
            },
        }
        with (
            patch("feedback_pipeline.load_review_corpus", return_value=corpus),
            patch(
                "feedback_pipeline.extract_text_from_paper_file",
                return_value=("Abstract\nA panel design.", "ok"),
            ),
            patch(
                "feedback_pipeline.estimate_cost_before_run",
                return_value={"estimated_total_cost_usd": 1.0, "stages": {}},
            ),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            result = await fp.run_historical_review_eval(
                "private_manifest.json",
                run_api=False,
                require_existing_pdf=False,
            )

        self.assertFalse(mock_pipeline.called)
        self.assertEqual(result["summary"]["corpus_issue_targets"], 1)
        self.assertEqual(result["summary"]["corpus_issue_targets_excluded"], 0)
        self.assertEqual(result["splits"][0]["human_issue_count"], 1)

    async def test_partial_gold_dry_run_uses_projection_and_makes_no_api_calls(self):
        corpus = {
            "manifest_version": "v1",
            "records": [
                {
                    "paper_id": "paper_1",
                    "family_id": "family_1",
                    "case_id": "case_1",
                    "manifest_case": True,
                    "review_file": "source_1",
                    "disposition": "evaluation",
                    "benchmark_tier": "primary",
                    "classification": "informal_feedback",
                    "source_type": "human_feedback",
                    "matched_paper_files": ["manuscript.txt"],
                    "manuscript_files": ["manuscript.txt"],
                }
            ],
            "issues": [
                {
                    "paper_id": "paper_1",
                    "family_id": "family_1",
                    "case_id": "case_1",
                    "atomic_issue_id": "issue_1",
                    "issue_text": "The comparison group is not credible.",
                    "disposition": "evaluation",
                }
            ],
            "paper_matches": {},
            "excluded_records": [],
            "stats": {
                "records": 1,
                "issues": 1,
                "excluded_low_confidence_records": 0,
            },
        }
        raw_gold = {
            "status": "pending_human_adjudication",
            "binding_hash": "source-binding",
            "rows": [
                {
                    "packet_version": "feedback-llm-adjudication-v1",
                    "binding_hash": "source-binding",
                    "family_id": "family_1",
                    "cluster_id": "major_1",
                    "full_adjudication_required": "yes",
                    "tier_screen": "major",
                    "include": "yes",
                    "canonical_issue": "The comparison group is not credible.",
                    "severity": "major_revision_issue",
                    "evidentiary_support": "supported",
                    "duplicate_cluster_ids": "[]",
                },
                {
                    "packet_version": "feedback-llm-adjudication-v1",
                    "binding_hash": "source-binding",
                    "family_id": "family_1",
                    "cluster_id": "pending_1",
                    "full_adjudication_required": "no",
                    "tier_screen": "",
                    "duplicate_cluster_ids": "[]",
                },
            ],
        }
        with (
            patch("feedback_pipeline.load_review_corpus", return_value=corpus),
            patch(
                "feedback_pipeline._load_current_gold_adjudication",
                return_value=raw_gold,
            ),
            patch(
                "feedback_pipeline.extract_text_from_paper_file",
                return_value=("Abstract\nA panel design.", "ok"),
            ),
            patch(
                "feedback_pipeline.estimate_cost_before_run",
                return_value={"estimated_total_cost_usd": 1.25, "stages": {}},
            ),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
            patch(
                "review_adjudication.write_generated_adjudication_packet"
            ) as mock_generated_packet,
        ):
            result = await fp.run_historical_review_eval(
                "private_manifest.json",
                run_api=False,
                require_existing_pdf=False,
                adjudication_path="gold.csv",
                gold_mode="partial",
            )

        self.assertFalse(mock_pipeline.called)
        self.assertFalse(mock_generated_packet.called)
        self.assertEqual(result["status"], "dry_run")
        self.assertEqual(result["summary"]["gold_mode"], "partial")
        self.assertEqual(result["summary"]["evaluation_scope"], "partial_gold_pilot")
        self.assertEqual(
            result["summary"]["gold_coverage"]["adjudicated_cluster_count"], 1
        )
        self.assertEqual(result["summary"]["api_evaluated_splits"], 0)

    async def test_paid_manifest_preserves_full_checkpoint_before_failed_audit(self):
        order = ["family-z", "family-a", "family-y", "family-b", "family-x"]
        counts = [5, 5, 4, 4, 5]
        records = [
            {
                "paper_id": f"paper-{index}",
                "family_id": family,
                "case_id": f"case-{index}",
                "manifest_case": True,
                "disposition": "evaluation",
            }
            for index, family in enumerate(order)
        ]
        corpus = {
            "manifest_version": "v1",
            "records": records,
            "issues": [],
            "paper_matches": {},
            "excluded_records": [],
            "stats": {
                "records": 5,
                "issues": 0,
                "excluded_low_confidence_records": 0,
            },
        }
        splits = [
            {
                "paper_id": record["paper_id"],
                "family_id": record["family_id"],
                "case_id": record["case_id"],
                "benchmark_tier": "secondary" if index == 4 else "primary",
                "classification": (
                    "exact_journal_review"
                    if index in {1, 2, 3}
                    else "version_matched_informal_feedback"
                ),
                "journals": ["journal"] if index in {1, 2, 3} else [],
                "source_types": [
                    "journal_review"
                    if index in {1, 2, 3}
                    else "informal_feedback"
                ],
            }
            for index, record in enumerate(records)
        ]
        full_text = "Full generated issue. " + ("x" * 1400) + " CHECKPOINT_TAIL"
        pipeline_results = []
        for case_index, count in enumerate(counts):
            proposals = [
                {
                    "id": f"pipeline-{case_index}-{rank}",
                    "text": (
                        full_text
                        if case_index == 0 and rank == 1
                        else f"Concern {rank} for case {case_index}."
                    ),
                    "evidence_ids": [f"P{case_index:02d}{rank:02d}"],
                }
                for rank in range(1, count + 1)
            ]
            pipeline_results.append(
                {
                    "selection": {"top_proposals": proposals},
                    "actual_usage": {"total_cost_usd": 0.1},
                }
            )
        gold = {
            "status": "ready",
            "binding_hash": "gold-binding",
            "rows": [],
            "evaluation_scope": "partial_gold_pilot",
            "coverage": {},
        }

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            adjudication_path = root / "final" / "gold.csv"
            portable_path = root / "portable.json"
            with (
                patch("review_adjudication.DEFAULT_PRIVATE_ROOT", root),
                patch("feedback_pipeline.load_review_corpus", return_value=corpus),
                patch(
                    "feedback_pipeline._load_current_gold_adjudication",
                    return_value=gold,
                ),
                patch(
                    "feedback_pipeline._gold_adjudication_for_evaluation",
                    return_value=gold,
                ),
                patch(
                    "feedback_pipeline.build_review_holdout_splits",
                    return_value=splits,
                ),
                patch(
                    "feedback_pipeline._extract_holdout_manuscript_bundle",
                    return_value=("Abstract\nA panel design.", "ok", ["paper.pdf"]),
                ),
                patch(
                    "feedback_pipeline.estimate_cost_before_run",
                    return_value={"estimated_total_cost_usd": 0.1, "stages": {}},
                ),
                patch(
                    "feedback_pipeline.full_feedback_pipeline",
                    new_callable=AsyncMock,
                    side_effect=pipeline_results,
                ) as mock_pipeline,
                patch(
                    "feedback_pipeline._git_run_state",
                    return_value={"commit": "abc123", "dirty": False},
                ),
                patch("feedback_pipeline._manifest_pilot_structure_errors", return_value=[]),
                patch(
                    "feedback_pipeline._manifest_baseline_audit_errors",
                    return_value=["forced post-run audit failure"],
                ),
            ):
                result = await fp.run_historical_review_eval(
                    "private_manifest.json",
                    output_path=portable_path,
                    run_api=True,
                    require_existing_pdf=False,
                    adjudication_path=adjudication_path,
                    gold_mode="partial",
                    max_cost_usd=10.0,
                )

            self.assertEqual(mock_pipeline.await_count, 5)
            self.assertEqual(result["status"], "incomplete_run")
            self.assertTrue(result["generated_adjudication"]["checkpoint_only"])
            self.assertEqual(result["generated_checkpoint"]["issue_count"], 23)
            checkpoint_path = Path(result["generated_checkpoint"]["csv_path"])
            self.assertTrue(checkpoint_path.exists())
            self.assertEqual(stat.S_IMODE(checkpoint_path.stat().st_mode), 0o600)
            with checkpoint_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 23)
            self.assertEqual(
                {row["binding_hash"] for row in rows},
                {result["generated_checkpoint"]["binding_hash"]},
            )
            preserved = next(
                row
                for row in rows
                if row["family_id"] == "family-z" and row["rank"] == "1"
            )
            self.assertEqual(preserved["generated_text"], full_text)
            self.assertEqual(json.loads(preserved["evidence_ids"]), ["P0001"])
            self.assertTrue((root / "final" / "generated_adjudication.in_progress.csv").exists())
            portable = json.loads(portable_path.read_text(encoding="utf-8"))
            portable_text = json.dumps(portable)
            self.assertNotIn("CHECKPOINT_TAIL", portable_text)
            self.assertNotIn(str(root), portable_text)
            for call in mock_pipeline.await_args_list:
                self.assertIsNone(call.kwargs["review_corpus"])
                self.assertIsNone(call.kwargs["review_prior"])

    async def test_paid_manifest_pending_gold_is_rejected_by_default_before_api(self):
        corpus = {
            "manifest_version": "v1",
            "records": [],
            "issues": [],
            "paper_matches": {},
            "stats": {
                "records": 0,
                "issues": 0,
                "excluded_low_confidence_records": 0,
            },
        }
        with (
            patch("feedback_pipeline.load_review_corpus", return_value=corpus),
            patch(
                "feedback_pipeline._load_current_gold_adjudication",
                return_value={
                    "status": "pending_human_adjudication",
                    "binding_hash": "source-binding",
                    "pending_fields": ["cluster: tier_screen"],
                    "rows": [],
                },
            ),
            patch(
                "feedback_pipeline._git_run_state",
                return_value={"commit": "abc123", "dirty": False},
            ),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            with self.assertRaisesRegex(ValueError, "not ready"):
                await fp.run_historical_review_eval(
                    "private_manifest.json",
                    run_api=True,
                    max_cost_usd=10.0,
                    adjudication_path="gold.csv",
                )

        self.assertFalse(mock_pipeline.called)

    async def test_paid_eval_rejects_nonfinite_or_nonpositive_ceiling_before_api(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        for invalid in (0.0, -1.0, float("nan"), float("inf"), float("-inf")):
            with self.subTest(invalid=invalid):
                with patch(
                    "feedback_pipeline.full_feedback_pipeline",
                    new_callable=AsyncMock,
                ) as mock_pipeline:
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        await fp.run_historical_review_eval(
                            root,
                            run_api=True,
                            max_cost_usd=invalid,
                        )
                self.assertFalse(mock_pipeline.called)

        with self.assertRaisesRegex(ValueError, "positive finite"):
            await fp.run_historical_review_eval(
                root,
                run_api=False,
                max_cost_usd=float("nan"),
            )

    async def test_manifest_eval_rejects_non_five_top_k(self):
        corpus = {
            "manifest_version": "v1",
            "records": [],
            "issues": [],
            "stats": {},
        }
        with patch("feedback_pipeline.load_review_corpus", return_value=corpus):
            with self.assertRaisesRegex(ValueError, "top_k=5"):
                await fp.run_historical_review_eval(
                    "unused.json",
                    run_api=False,
                    top_k=6,
                )

    async def test_paid_manifest_eval_requires_clean_committed_worktree(self):
        corpus = {
            "manifest_version": "v1",
            "records": [],
            "issues": [],
            "paper_matches": {},
            "stats": {
                "records": 0,
                "issues": 0,
                "excluded_low_confidence_records": 0,
            },
        }
        with (
            patch("feedback_pipeline.load_review_corpus", return_value=corpus),
            patch("feedback_pipeline.build_review_holdout_splits", return_value=[]),
            patch(
                "feedback_pipeline._git_run_state",
                return_value={"commit": "abc123", "dirty": True},
            ),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            with self.assertRaisesRegex(ValueError, "clean, committed"):
                await fp.run_historical_review_eval(
                    "private_manifest.json",
                    run_api=True,
                    memory_mode="none",
                    max_cost_usd=1.0,
                )

        self.assertFalse(mock_pipeline.called)

    async def test_paid_manifest_eval_rejects_partial_selection_before_api(self):
        records = []
        issues = []
        for index in range(5):
            is_secondary = index == 4
            is_journal = index in {1, 2, 3}
            records.append(
                {
                    "manifest_case": True,
                    "disposition": "evaluation",
                    "paper_id": f"paper_{index}",
                    "family_id": f"family_{index}",
                    "case_id": f"case_{index}",
                    "review_file": f"source_{index}",
                    "benchmark_tier": "secondary" if is_secondary else "primary",
                    "classification": (
                        "exact_journal_review" if is_journal else "informal_feedback"
                    ),
                    "journal": "journal" if is_journal else "",
                    "source_type": "journal_review" if is_journal else "human_feedback",
                    "matched_paper_files": [f"manuscript_{index}.txt"],
                    "manuscript_files": [f"manuscript_{index}.txt"],
                }
            )
            issues.append(
                {
                    "paper_id": f"paper_{index}",
                    "family_id": f"family_{index}",
                    "case_id": f"case_{index}",
                    "atomic_issue_id": f"issue_{index}",
                    "issue_text": "Why?",
                    "issue_type": "other",
                    "decision_tier": "unadjudicated",
                    "quality_flag": "use_for_style_only",
                    "disposition": "evaluation",
                }
            )
        corpus = {
            "manifest_version": "v1",
            "records": records,
            "issues": issues,
            "paper_matches": {},
            "excluded_records": [],
            "stats": {
                "records": 5,
                "issues": 5,
                "excluded_low_confidence_records": 0,
            },
        }

        with (
            patch("feedback_pipeline.load_review_corpus", return_value=corpus),
            patch(
                "feedback_pipeline._load_current_gold_adjudication",
                return_value={"status": "ready", "pending_fields": []},
            ),
            patch(
                "feedback_pipeline._git_run_state",
                return_value={"commit": "abc123", "dirty": False},
            ),
            patch(
                "feedback_pipeline.extract_text_from_paper_file",
                return_value=("Abstract\nA panel design.", "ok"),
            ),
            patch(
                "feedback_pipeline.estimate_cost_before_run",
                return_value={"estimated_total_cost_usd": 1.0, "stages": {}},
            ),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new_callable=AsyncMock,
            ) as mock_pipeline,
        ):
            with self.assertRaisesRegex(ValueError, "complete fixed pilot"):
                await fp.run_historical_review_eval(
                    "private_manifest.json",
                    run_api=True,
                    max_splits=1,
                    require_existing_pdf=False,
                    memory_mode="none",
                    max_cost_usd=100.0,
                    adjudication_path="gold.csv",
                )

        self.assertFalse(mock_pipeline.called)

    async def test_cold_eval_passes_no_feedback_or_prior_and_scores_final_top_k(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        async def fake_pipeline(*args, **kwargs):
            self.assertIsNone(kwargs.get("review_corpus"))
            self.assertIsNone(kwargs.get("review_prior"))
            serialized_call = json.dumps(
                {"paper_text": args[0], "kwargs": kwargs},
                default=str,
            )
            self.assertNotIn("AJPS-12345", serialized_call)
            self.assertNotIn("Reviewer 1", serialized_call)
            self.assertNotIn("The editor invited a revision", serialized_call)
            self.assertNotIn(str(root), serialized_call)
            return {
                "selection": {
                    "top_proposals": [
                        {
                            "id": 1,
                            "text": "Problem: The DD design lacks full pre-treatment leads and a joint pre-trend test.",
                        }
                    ],
                    "high_quality": [
                        {"id": 2, "text": "Problem: This unrelated issue must not be scored."},
                        {"id": 3, "text": "Problem: Nor should this lower-ranked issue."},
                    ],
                },
                "actual_usage": {"total_cost_usd": 0.01},
            }

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new=AsyncMock(side_effect=fake_pipeline),
            ) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Abstract\nThe paper uses difference-in-differences and discusses pre-treatment trends.",
                "ok",
            )
            result = await fp.run_historical_review_eval(
                root,
                max_splits=1,
                require_existing_pdf=False,
                run_api=True,
                memory_mode="none",
                max_cost_usd=100.0,
            )

        self.assertEqual(mock_pipeline.await_count, 1)
        self.assertEqual(result["summary"]["memory_mode"], "none")
        self.assertEqual(result["splits"][0]["memory_record_count"], 0)
        self.assertEqual(result["splits"][0]["generated_issue_count"], 1)
        self.assertGreater(result["splits"][0]["metrics"]["major_issue_recall_at_k"], 0)

    async def test_paid_eval_cost_preflight_blocks_before_pipeline_call(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)

        with (
            patch("feedback_pipeline.extract_text_from_paper_file") as mock_extract,
            patch("feedback_pipeline.full_feedback_pipeline", new_callable=AsyncMock) as mock_pipeline,
        ):
            mock_extract.return_value = (
                "Abstract\nThe paper uses difference-in-differences and reports treatment effects.",
                "ok",
            )
            with self.assertRaisesRegex(ValueError, "exceeds max_cost_usd"):
                await fp.run_historical_review_eval(
                    root,
                    max_splits=1,
                    require_existing_pdf=False,
                    run_api=True,
                    memory_mode="none",
                    max_cost_usd=0.000001,
                )

        self.assertFalse(mock_pipeline.called)

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
                memory_mode="local_raw",
                max_cost_usd=100.0,
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
        portable = json.loads(output_path.read_text(encoding="utf-8"))
        serialized = json.dumps(portable)
        self.assertEqual(
            portable["schema_version"],
            "portable_review_prior_eval_gate_v1",
        )
        self.assertNotIn(str(root), serialized)
        self.assertNotIn("Example Difference-in-Differences Paper", serialized)
        self.assertNotIn("matched_paper_files", serialized)
        self.assertEqual(output_path.stat().st_mode & 0o777, 0o600)

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
                max_cost_usd=100.0,
            )

    async def test_review_prior_gate_checks_remaining_budget_before_each_request(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        add_second_review_record(root)

        async def fake_pipeline(*args, **kwargs):
            return {
                "selection": {
                    "top_proposals": [
                        {"id": 1, "text": "Problem: missing design diagnostic."}
                    ]
                },
                "actual_usage": {"total_cost_usd": 0.45},
            }

        fake_cost = {
            "estimated_total_cost_usd": 0.1,
            "stages": {},
        }
        with (
            patch(
                "feedback_pipeline.extract_text_from_paper_file",
                return_value=("Abstract\nA panel design.", "ok"),
            ),
            patch("feedback_pipeline.estimate_cost_before_run", return_value=fake_cost),
            patch(
                "feedback_pipeline.full_feedback_pipeline",
                new=AsyncMock(side_effect=fake_pipeline),
            ) as mock_pipeline,
        ):
            result = await fp.run_review_prior_eval_gate(
                root,
                max_splits=1,
                require_existing_pdf=False,
                run_api=True,
                max_cost_usd=0.5,
                review_prior_min_papers=1,
                review_prior_min_comments=1,
            )

        self.assertEqual(mock_pipeline.await_count, 1)
        self.assertEqual(
            result["modes"]["safe_prior"]["splits"][0]["status"],
            "skipped_cost_ceiling",
        )
        self.assertEqual(
            result["modes"]["local_raw_memory"]["splits"][0]["status"],
            "skipped_cost_ceiling",
        )
        self.assertEqual(result["summary"]["total_actual_cost_usd"], 0.45)

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
                max_cost_usd=100.0,
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

    async def test_review_prior_gate_rejects_nonfinite_ceiling_before_api(self):
        tmp, root = make_archive()
        self.addCleanup(tmp.cleanup)
        for invalid in (0.0, float("nan"), float("inf")):
            with self.subTest(invalid=invalid):
                with patch(
                    "feedback_pipeline.full_feedback_pipeline",
                    new_callable=AsyncMock,
                ) as mock_pipeline:
                    with self.assertRaisesRegex(ValueError, "positive finite"):
                        await fp.run_review_prior_eval_gate(
                            root,
                            run_api=True,
                            max_cost_usd=invalid,
                        )
                self.assertFalse(mock_pipeline.called)


class ReviewEvalHardeningTests(unittest.TestCase):
    def test_manifest_gold_screen_keeps_every_evaluation_issue(self):
        corpus = {
            "issues": [
                {
                    "atomic_issue_id": "short",
                    "issue_text": "Why?",
                    "issue_type": "other",
                    "quality_flag": "use_for_style_only",
                    "disposition": "evaluation",
                },
                {
                    "atomic_issue_id": "explicitly-deferred",
                    "issue_text": "Deferred source material.",
                    "disposition": "deferred",
                },
            ]
        }

        screened = fp._manifest_evaluation_issues(corpus)

        self.assertEqual(
            [issue["atomic_issue_id"] for issue in screened],
            ["short"],
        )

    def test_manifest_holdout_keeps_short_style_only_issue(self):
        corpus = {
            "manifest_version": "v1",
            "records": [
                {
                    "paper_id": "paper_short",
                    "family_id": "family_short",
                    "case_id": "case_short",
                    "manifest_case": True,
                    "review_file": "source_short",
                    "disposition": "evaluation",
                    "benchmark_tier": "primary",
                    "matched_paper_files": [],
                    "manuscript_files": [],
                }
            ],
            "issues": [
                {
                    "paper_id": "paper_short",
                    "family_id": "family_short",
                    "case_id": "case_short",
                    "atomic_issue_id": "short",
                    "issue_text": "Why?",
                    "issue_type": "other",
                    "quality_flag": "use_for_style_only",
                    "disposition": "evaluation",
                }
            ],
            "paper_matches": {},
            "excluded_records": [],
        }

        splits = fp.build_review_holdout_splits(
            corpus,
            require_existing_pdf=False,
        )

        self.assertEqual(len(splits), 1)
        self.assertEqual(splits[0]["human_issue_candidate_count"], 1)
        self.assertEqual(splits[0]["human_issue_count"], 1)

    def test_eval_does_not_fall_back_when_final_top_proposals_is_empty(self):
        generated = fp._generated_issues_for_eval(
            {
                "selection": {
                    "top_proposals": [],
                    "high_quality": [{"id": "not-final"}],
                }
            },
            top_k=5,
        )
        self.assertEqual(generated, [])

    def test_portable_status_is_fixed_enum(self):
        statuses = {
            "ok": "ok",
            "missing_file": "unavailable",
            "unsupported_file_type:.rtf": "unsupported",
            "bundle_extract_error:/private/source.pdf:permission denied": "extract_error",
            "unexpected private detail": "unknown",
        }
        for raw, expected in statuses.items():
            with self.subTest(raw=raw):
                portable = fp.portable_review_eval_result(
                    {
                        "status": "pending_human_adjudication",
                        "summary": {},
                        "splits": [
                            {
                                "paper_id": "paper",
                                "paper_text_status": raw,
                            }
                        ],
                    }
                )
                self.assertEqual(
                    portable["splits"][0]["paper_text_status"],
                    expected,
                )
                self.assertEqual(portable["status"], "pending_human_adjudication")
                self.assertNotIn("private", json.dumps(portable))

    def test_manifest_audit_rejects_partial_and_accepts_complete_case_set(self):
        records = [
            {
                "manifest_case": True,
                "disposition": "evaluation",
                "paper_id": f"paper_{index}",
                "family_id": f"family_{index}",
                "case_id": f"case_{index}",
                "benchmark_tier": "secondary" if index == 4 else "primary",
            }
            for index in range(5)
        ]
        corpus = {
            "manifest_version": "v1",
            "records": records,
            "issues": [],
        }

        def split_for(index):
            return {
                "paper_id": f"paper_{index}",
                "family_id": f"family_{index}",
                "case_id": f"case_{index}",
                "benchmark_tier": "secondary" if index == 4 else "primary",
                "classification": (
                    "exact_journal_review"
                    if index in {1, 2, 3}
                    else "version_matched_informal_feedback"
                ),
                "journals": ["journal"] if index in {1, 2, 3} else [],
                "source_types": [
                    "journal_review" if index in {1, 2, 3} else "informal_feedback"
                ],
                "status": "pending_human_adjudication",
                "paper_text_status": "ok",
                "generated_issue_count": 5,
            }

        complete_splits = [split_for(index) for index in range(5)]
        complete = {
            "summary": {"api_evaluated_splits": 5},
            "splits": complete_splits,
            "run_metadata": {
                "corpus_binding_hash": fp._corpus_binding_hash(corpus),
                "top_k": 5,
                "top_k_policy": fp.REVIEW_BASELINE_TOP_K_POLICY,
                "num_agents": 8,
                "memory_mode": "none",
                "benchmark_binding": fp._manifest_benchmark_binding(
                    corpus,
                    complete_splits,
                ),
            },
        }
        self.assertEqual(fp._manifest_baseline_audit_errors(corpus, complete), [])

        short_output = json_clone(complete)
        short_output["splits"][2]["generated_issue_count"] = 4
        short_output["splits"][3]["generated_issue_count"] = 4
        self.assertEqual(
            fp._manifest_baseline_audit_errors(corpus, short_output), []
        )

        partial = json_clone(complete)
        partial["splits"] = partial["splits"][:-1]
        partial["summary"]["api_evaluated_splits"] = 4
        errors = fp._manifest_baseline_audit_errors(corpus, partial)
        self.assertTrue(any("complete expected case set" in error for error in errors))
        self.assertTrue(any("five API-evaluated" in error for error in errors))

        zero_output = json_clone(complete)
        zero_output["splits"][0]["generated_issue_count"] = 0
        errors = fp._manifest_baseline_audit_errors(corpus, zero_output)
        self.assertTrue(any("between one and five" in error for error in errors))

        too_many = json_clone(complete)
        too_many["splits"][0]["generated_issue_count"] = 6
        errors = fp._manifest_baseline_audit_errors(corpus, too_many)
        self.assertTrue(any("between one and five" in error for error in errors))

        wrong_tiers = json_clone(complete)
        wrong_tiers["splits"][0]["benchmark_tier"] = "secondary"
        errors = fp._manifest_baseline_audit_errors(corpus, wrong_tiers)
        self.assertTrue(any("four primary and one secondary" in error for error in errors))

        wrong_journals = json_clone(complete)
        wrong_journals["splits"][1]["journals"] = []
        wrong_journals["splits"][1]["classification"] = "informal_feedback"
        wrong_journals["splits"][1]["source_types"] = ["informal_feedback"]
        errors = fp._manifest_baseline_audit_errors(corpus, wrong_journals)
        self.assertTrue(any("three primary journal" in error for error in errors))

    def test_partial_finalization_uses_frozen_projection_binding_and_status(self):
        records = [
            {
                "manifest_case": True,
                "disposition": "evaluation",
                "paper_id": f"paper_{index}",
                "family_id": f"family_{index}",
                "case_id": f"case_{index}",
                "benchmark_tier": "secondary" if index == 4 else "primary",
            }
            for index in range(5)
        ]
        corpus = {
            "manifest_version": "v1",
            "records": records,
            "issues": [],
            "stats": {"records": 5, "issues": 0},
        }
        raw_gold = {
            "status": "pending_human_adjudication",
            "binding_hash": "source-binding",
            "rows": [
                {
                    "packet_version": "feedback-llm-adjudication-v1",
                    "binding_hash": "source-binding",
                    "family_id": "family_0",
                    "cluster_id": "major_0",
                    "full_adjudication_required": "yes",
                    "sampled_minor": "no",
                    "tier_screen": "major",
                    "include": "yes",
                    "canonical_issue": "The comparison group is not credible.",
                    "severity": "major_revision_issue",
                    "evidentiary_support": "supported",
                    "duplicate_cluster_ids": "[]",
                },
                {
                    "packet_version": "feedback-llm-adjudication-v1",
                    "binding_hash": "source-binding",
                    "family_id": "family_0",
                    "cluster_id": "pending_0",
                    "full_adjudication_required": "no",
                    "tier_screen": "",
                    "duplicate_cluster_ids": "[]",
                },
            ],
        }
        projected = fp._gold_adjudication_for_evaluation(raw_gold, "partial")
        generated_rows = []
        for index in range(5):
            returned_count = 4 if index in {2, 3} else 5
            for rank in range(1, returned_count + 1):
                generated_rows.append(
                    {
                        "family_id": f"family_{index}",
                        "case_id": f"case_{index}",
                        "rank": rank,
                        "generated_issue_id": f"g_{index}_{rank}",
                        "correctness": "correct",
                        "significance": "significant",
                        "evidence_sufficiency": "sufficient",
                        "human_match_status": (
                            "matched" if index == 0 and rank == 1 else "unmatched"
                        ),
                        "confirmed_human_cluster_ids": (
                            '["major_0"]' if index == 0 and rank == 1 else "[]"
                        ),
                        "duplicate_status": "unique",
                        "valid_novelty": "no",
                    }
                )
        splits = [
            {
                "family_id": f"family_{index}",
                "case_id": f"case_{index}",
                "benchmark_tier": "secondary" if index == 4 else "primary",
                "journals": ["journal"] if index in {1, 2, 3} else [],
                "actual_usage": {"total_cost_usd": 1.0},
            }
            for index in range(5)
        ]
        local_audit = {
            "run_metadata": {
                "git": {"commit": "abc123", "dirty": False},
                "routing": {},
                "top_k": 5,
                "top_k_policy": fp.REVIEW_BASELINE_TOP_K_POLICY,
                "num_agents": 8,
                "reviewer_roles": {},
                "memory_mode": "none",
                "gold_mode": "partial",
                "gold_binding_hash": projected["binding_hash"],
                "benchmark_binding": {},
            },
            "generated_adjudication": {"binding_hash": "generated-binding"},
            "splits": splits,
        }
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            audit_path.write_text(json.dumps(local_audit), encoding="utf-8")
            with (
                patch("feedback_pipeline.load_review_corpus", return_value=corpus),
                patch(
                    "feedback_pipeline._load_current_gold_adjudication",
                    return_value=raw_gold,
                ),
                patch(
                    "feedback_pipeline._manifest_baseline_audit_errors",
                    return_value=[],
                ),
                patch(
                    "review_adjudication.load_generated_adjudication",
                    return_value={"status": "ready", "rows": generated_rows},
                ) as mock_load_generated,
            ):
                result = fp.finalize_review_evaluation(
                    "manifest.json",
                    "gold.csv",
                    "generated.csv",
                    audit_path,
                    gold_mode="partial",
                )

        self.assertEqual(result["status"], "partial_gold_pilot")
        self.assertEqual(
            result["schema_version"], "portable_review_eval_partial_metrics_v1"
        )
        self.assertEqual(result["metrics"]["status"], "partial_gold_pilot")
        self.assertEqual(result["metrics"]["total_generated_issue_count"], 23)
        self.assertEqual(result["metrics"]["total_unfilled_issue_slot_count"], 2)
        self.assertEqual(
            mock_load_generated.call_args.kwargs["expected_gold_binding_hash"],
            projected["binding_hash"],
        )
        self.assertEqual(
            mock_load_generated.call_args.kwargs["valid_gold_cluster_ids"],
            ["major_0"],
        )

        changed_audit = json_clone(local_audit)
        changed_audit["run_metadata"]["gold_binding_hash"] = "older-projection"
        with tempfile.TemporaryDirectory() as tmp:
            audit_path = Path(tmp) / "audit.json"
            audit_path.write_text(json.dumps(changed_audit), encoding="utf-8")
            with (
                patch("feedback_pipeline.load_review_corpus", return_value=corpus),
                patch(
                    "feedback_pipeline._load_current_gold_adjudication",
                    return_value=raw_gold,
                ),
                patch(
                    "feedback_pipeline._manifest_baseline_audit_errors",
                    return_value=[],
                ),
            ):
                with self.assertRaisesRegex(ValueError, "changed after"):
                    fp.finalize_review_evaluation(
                        "manifest.json",
                        "gold.csv",
                        "generated.csv",
                        audit_path,
                        gold_mode="partial",
                    )

    def test_cli_rejects_conflicting_modes_and_out_of_scope_flags(self):
        cases = [
            ["--inspect-review-corpus", "--eval-review-corpus", "corpus"],
            ["--eval-batch-api", "--file", "paper.txt"],
            ["--eval-run-api", "--file", "paper.txt"],
            ["--eval-gold-mode", "partial", "--file", "paper.txt"],
            ["--eval-output", "result.json", "--file", "paper.txt"],
            ["--review-corpus-output", "corpus.json", "--file", "paper.txt"],
            ["--eval-review-corpus", "corpus.json", "--file", "paper.txt"],
            [
                "--eval-review-prior-gate",
                "corpus.json",
                "--eval-memory-mode",
                "none",
            ],
        ]
        for argv in cases:
            with self.subTest(argv=argv):
                with patch("sys.stderr"):
                    with self.assertRaises(SystemExit):
                        fp.main(argv)

    def test_cli_rejects_nan_paid_ceiling(self):
        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                fp.main(
                    [
                        "--eval-review-corpus",
                        "corpus.json",
                        "--eval-run-api",
                        "--eval-max-cost-usd",
                        "nan",
                    ]
                )

        with patch("sys.stderr"):
            with self.assertRaises(SystemExit):
                fp.main(
                    [
                        "--eval-review-corpus",
                        "corpus.json",
                        "--eval-run-api",
                    ]
                )


if __name__ == "__main__":
    unittest.main()
