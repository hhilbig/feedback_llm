import json
import os
import stat
import tempfile
import unittest
import zipfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import fitz

import review_corpus_manifest as rcm


DOCUMENT_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:body>
    <w:p><w:r><w:t>1. The design needs a stronger comparison group and diagnostic evidence.</w:t></w:r></w:p>
    <w:p>
      <w:commentRangeStart w:id="0"/>
      <w:r><w:t>parallel trends claim</w:t></w:r>
      <w:commentRangeEnd w:id="0"/>
      <w:r><w:commentReference w:id="0"/></w:r>
    </w:p>
    <w:p><w:r><w:t>2. Clarify the contribution relative to the closest prior study.</w:t></w:r></w:p>
  </w:body>
</w:document>
"""


COMMENTS_XML = """<?xml version="1.0" encoding="UTF-8" standalone="yes"?>
<w:comments xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
  <w:comment w:id="0" w:author="A Real Reviewer">
    <w:p><w:r><w:t>Please show the full set of pre-treatment coefficients.</w:t></w:r></w:p>
  </w:comment>
</w:comments>
"""


def write_docx(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("word/document.xml", DOCUMENT_XML)
        archive.writestr("word/comments.xml", COMMENTS_XML)


def write_annotated_pdf(path: Path) -> None:
    document = fitz.open()
    page = document.new_page()
    page.insert_text((72, 72), "The identifying assumption is parallel trends.")
    rectangles = page.search_for("parallel trends")
    annotation = page.add_highlight_annot(rectangles)
    annotation.set_info(content="Show the complete pre-treatment event study.")
    annotation.update()
    document.save(path)
    document.close()


def write_two_page_pdf(path: Path) -> None:
    document = fitz.open()
    first = document.new_page()
    first.insert_text((72, 72), "First page boilerplate should not become an issue.")
    second = document.new_page()
    second.insert_text((72, 72), "The second-page concern asks for a placebo comparison.")
    document.save(path)
    document.close()


def file_entry(path: Path, role: str | None = None):
    entry = {"path": path.name, "sha256": rcm.sha256_file(path)}
    if role:
        entry["role"] = role
    return entry


def source_entry(
    path: Path,
    *,
    source_id: str,
    extractor: str,
    reviewer_id: str = "reviewer_001",
    selectors=None,
    version_match: str = "exact_submission",
    disposition: str = "evaluation",
    **extra,
):
    source = {
        "source_id": source_id,
        "path": path.name,
        "sha256": rcm.sha256_file(path),
        "extractor": extractor,
        "reviewer_id": reviewer_id,
        "provenance": "human_test_feedback",
        "source_type": "human_test_feedback",
        "feedback_date": "2026-04",
        "version_match": version_match,
        "disposition": disposition,
    }
    if selectors is not None:
        source["selectors"] = selectors
    source.update(extra)
    return source


def manifest_for(
    manuscript_files,
    sources,
    *,
    tier="primary",
    family_id="family_001",
    case_id="case_001",
    disposition="evaluation",
):
    return {
        "manifest_version": "v1",
        "corpus_id": "test_corpus",
        "cases": [
            {
                "family_id": family_id,
                "case_id": case_id,
                "benchmark_tier": tier,
                "classification": "test_feedback",
                "disposition": disposition,
                "manuscript_files": manuscript_files,
                "sources": sources,
            }
        ],
    }


class ManifestImporterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.manuscript = self.root / "manuscript.txt"
        self.manuscript.write_text("Main manuscript evidence.", encoding="utf-8")

    def test_markdown_item_selector_excludes_generated_items(self):
        feedback = self.root / "seminar.md"
        feedback.write_text(
            """# Seminar feedback

## Human feedback

1. First human concern has enough substantive detail to evaluate.
2. Second human concern asks for clearer identifying assumptions.
3. Third human concern requests a measurement validity check.
4. Fourth human concern asks for an alternative-explanation test.
5. Fifth human concern asks what drives treatment variation.
6. Sixth human concern identifies a framing mismatch in the paper.
7. Seventh human concern suggests a broader research implication.

## Additional Anticipated Feedback

8. This generated concern must never enter the corpus.
9. This generated issue must also stay excluded.
""",
            encoding="utf-8",
        )
        manifest = manifest_for(
            [file_entry(self.manuscript, "main")],
            [
                source_entry(
                    feedback,
                    source_id="insurers_human_seminar",
                    extractor="markdown",
                    selectors={
                        "start_heading": "## Human feedback",
                        "end_heading": "## Additional Anticipated Feedback",
                        "include_item_numbers": list(range(1, 8)),
                    },
                )
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)

        self.assertEqual(corpus["stats"]["issues"], 7)
        text = "\n".join(issue["issue_text"] for issue in corpus["issues"])
        self.assertIn("Seventh human concern", text)
        self.assertNotIn("generated", text.lower())
        self.assertEqual(corpus["issues"][0]["source_locator"], "item:1")

    def test_pdf_annotations_preserve_anchor_but_not_comment_in_manuscript(self):
        annotated = self.root / "annotated.pdf"
        write_annotated_pdf(annotated)
        manifest = manifest_for(
            [file_entry(annotated, "main")],
            [
                source_entry(
                    annotated,
                    source_id="annotated_review",
                    extractor="pdf_annotations",
                    version_match="exact_embedded",
                )
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)
        manuscript_text = rcm.extract_manuscript_text(annotated)

        self.assertEqual(corpus["stats"]["issues"], 1)
        issue = corpus["issues"][0]
        self.assertEqual(
            issue["issue_text"], "Show the complete pre-treatment event study."
        )
        self.assertIn("parallel trends", issue["anchor_text"])
        self.assertIn("page:1;annotation:1", issue["source_locator"])
        self.assertIn("parallel trends", manuscript_text)
        self.assertNotIn("complete pre-treatment", manuscript_text)

    def test_docx_comments_and_body_use_standard_library_xml(self):
        document = self.root / "comments.docx"
        write_docx(document)
        manifest = manifest_for(
            [file_entry(self.manuscript, "main")],
            [
                source_entry(
                    document,
                    source_id="word_comment",
                    extractor="docx_comments",
                    reviewer_id="reviewer_001",
                ),
                source_entry(
                    document,
                    source_id="word_body",
                    extractor="docx_body",
                    reviewer_id="reviewer_002",
                    selectors={"include_item_numbers": [1, 2]},
                ),
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)

        self.assertEqual(corpus["stats"]["issues"], 3)
        comments = [
            issue for issue in corpus["issues"] if issue["source_id"] == "word_comment"
        ]
        self.assertEqual(comments[0]["source_locator"], "comment:0")
        self.assertEqual(comments[0]["anchor_text"], "parallel trends claim")
        self.assertNotIn("A Real Reviewer", json.dumps(corpus))

    def test_pdf_page_selector_and_ordered_manuscript_bundle(self):
        review = self.root / "review.pdf"
        write_two_page_pdf(review)
        appendix = self.root / "appendix.txt"
        appendix.write_text("Supplement evidence follows the main text.", encoding="utf-8")
        manifest = manifest_for(
            [file_entry(self.manuscript, "main"), file_entry(appendix, "appendix")],
            [
                source_entry(
                    review,
                    source_id="review_page_two",
                    extractor="pdf_text",
                    selectors={"pages": [2], "min_words": 2},
                )
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)
        bundle = rcm.extract_ordered_manuscript_bundle(
            manifest["cases"][0]["manuscript_files"], base_dir=self.root
        )

        self.assertEqual(corpus["stats"]["issues"], 1)
        self.assertIn("second-page concern", corpus["issues"][0]["issue_text"])
        self.assertNotIn("boilerplate", corpus["issues"][0]["issue_text"])
        self.assertLess(bundle.index("Main manuscript"), bundle.index("Supplement evidence"))
        self.assertIn("ROLE:appendix", bundle)

    def test_duplicate_sources_deduplicate_but_independent_reviewers_survive(self):
        first = self.root / "first.txt"
        second = self.root / "second.txt"
        content = "The report requests a stronger falsification test for the central design."
        first.write_text(content, encoding="utf-8")
        second.write_text(content, encoding="utf-8")
        manifest = manifest_for(
            [file_entry(self.manuscript, "main")],
            [
                source_entry(first, source_id="copy_a", extractor="text"),
                source_entry(second, source_id="copy_b", extractor="text"),
                source_entry(
                    second,
                    source_id="independent_identical",
                    extractor="text",
                    reviewer_id="reviewer_002",
                ),
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)

        self.assertEqual(corpus["stats"]["duplicate_sources"], 1)
        self.assertEqual(corpus["stats"]["issues"], 2)
        self.assertEqual(
            {issue["reviewer_id"] for issue in corpus["issues"]},
            {"reviewer_001", "reviewer_002"},
        )

    def test_declared_superseded_representation_is_not_imported(self):
        digest = self.root / "digest.txt"
        raw = self.root / "raw.txt"
        digest.write_text("A derivative digest of the underlying human concern.", encoding="utf-8")
        raw.write_text("The raw report asks for an explicit robustness comparison.", encoding="utf-8")
        manifest = manifest_for(
            [file_entry(self.manuscript, "main")],
            [
                source_entry(digest, source_id="old_digest", extractor="text"),
                source_entry(
                    raw,
                    source_id="raw_report",
                    extractor="text",
                    supersedes=["old_digest"],
                ),
            ],
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)

        self.assertEqual([record["source_id"] for record in corpus["records"]], ["raw_report"])
        self.assertIn("old_digest", corpus["excluded_records"])

    def test_evaluation_sources_reject_ai_derivative_and_response_provenance(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "A substantive issue that would otherwise be eligible for extraction.",
            encoding="utf-8",
        )
        forbidden_labels = [
            ("ai_generated", "reviewer_report"),
            ("human_feedback", "derivative_summary"),
            ("author_response", "reviewer_report"),
        ]
        for provenance, source_type in forbidden_labels:
            with self.subTest(provenance=provenance, source_type=source_type):
                source = source_entry(
                    feedback,
                    source_id="forbidden_source",
                    extractor="text",
                )
                source["provenance"] = provenance
                source["source_type"] = source_type
                with self.assertRaisesRegex(
                    rcm.ManifestValidationError,
                    "AI-generated, derivative, or response material",
                ):
                    rcm.build_review_corpus_from_manifest(
                        manifest_for([file_entry(self.manuscript)], [source]),
                        base_dir=self.root,
                    )

        missing_type = source_entry(
            feedback,
            source_id="missing_type",
            extractor="text",
        )
        missing_type.pop("source_type")
        with self.assertRaisesRegex(rcm.ManifestValidationError, "source_type is required"):
            rcm.build_review_corpus_from_manifest(
                manifest_for([file_entry(self.manuscript)], [missing_type]),
                base_dir=self.root,
            )

    def test_stale_hash_missing_file_and_unsupported_extractor_fail_closed(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text("A substantive human issue that should be extracted.", encoding="utf-8")
        valid_source = source_entry(feedback, source_id="source_001", extractor="text")

        stale = manifest_for([file_entry(self.manuscript)], [dict(valid_source)])
        stale["cases"][0]["sources"][0]["sha256"] = "0" * 64
        with self.assertRaisesRegex(rcm.ManifestValidationError, "SHA-256 is stale"):
            rcm.build_review_corpus_from_manifest(stale, base_dir=self.root)

        missing = manifest_for([file_entry(self.manuscript)], [dict(valid_source)])
        missing["cases"][0]["sources"][0]["path"] = "missing.txt"
        with self.assertRaisesRegex(rcm.ManifestValidationError, "is missing"):
            rcm.build_review_corpus_from_manifest(missing, base_dir=self.root)

        unsupported = manifest_for([file_entry(self.manuscript)], [dict(valid_source)])
        unsupported["cases"][0]["sources"][0]["extractor"] = "rtf"
        with self.assertRaisesRegex(rcm.ManifestValidationError, "extractor must be"):
            rcm.build_review_corpus_from_manifest(unsupported, base_dir=self.root)

        placeholder = self.root / "placeholder.txt"
        placeholder.write_bytes(b"")
        placeholder_manifest = manifest_for(
            [file_entry(self.manuscript)],
            [
                source_entry(
                    placeholder,
                    source_id="cloud_placeholder",
                    extractor="text",
                )
            ],
        )
        with self.assertRaisesRegex(rcm.ManifestValidationError, "cloud placeholder"):
            rcm.build_review_corpus_from_manifest(
                placeholder_manifest, base_dir=self.root
            )

    @unittest.skipUnless(hasattr(stat, "SF_DATALESS"), "macOS dataless flag")
    def test_dataless_cloud_placeholder_is_detected_without_reading(self):
        with patch.object(
            Path,
            "stat",
            return_value=SimpleNamespace(st_flags=stat.SF_DATALESS),
        ):
            self.assertTrue(rcm._file_is_offline(Path("cloud-placeholder.pdf")))

    def test_primary_near_exact_and_primary_empty_feedback_fail_closed(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text("tiny", encoding="utf-8")
        near_exact = manifest_for(
            [file_entry(self.manuscript)],
            [
                source_entry(
                    feedback,
                    source_id="near_exact",
                    extractor="text",
                    version_match="near_exact",
                )
            ],
        )
        with self.assertRaisesRegex(rcm.ManifestValidationError, "not version matched"):
            rcm.build_review_corpus_from_manifest(near_exact, base_dir=self.root)

        empty = manifest_for(
            [file_entry(self.manuscript)],
            [
                source_entry(
                    feedback,
                    source_id="empty_source",
                    extractor="text",
                    selectors={"min_words": 2},
                )
            ],
        )
        with self.assertRaisesRegex(rcm.ManifestValidationError, "produced no eligible"):
            rcm.build_review_corpus_from_manifest(empty, base_dir=self.root)

    def test_manifest_detection_audit_privacy_and_determinism(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "The reviewer requests a transparent account of the identifying variation.",
            encoding="utf-8",
        )
        manifest = manifest_for(
            [file_entry(self.manuscript)],
            [source_entry(feedback, source_id="source_001", extractor="text")],
        )
        manifest["cases"][0]["manuscript_files"][0]["path"] = str(self.manuscript)
        manifest["cases"][0]["sources"][0]["path"] = str(feedback)
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(manifest_path, 0o600)

        first = rcm.build_review_corpus_from_manifest(
            manifest_path, private_root=self.root
        )
        second = rcm.build_review_corpus_from_manifest(
            manifest_path, private_root=self.root
        )

        self.assertTrue(rcm.is_review_manifest(manifest_path))
        self.assertEqual(first["corpus_fingerprint"], second["corpus_fingerprint"])
        self.assertEqual(first["binding_hash"], first["corpus_fingerprint"])
        self.assertEqual(first["issues"], second["issues"])
        record = first["records"][0]
        issue = first["issues"][0]
        self.assertTrue(record["manifest_case"])
        self.assertEqual(record["family_id"], "family_001")
        self.assertTrue(record["paper_id"].startswith("paper_"))
        self.assertEqual(record["manuscript_files"], [str(self.manuscript.resolve())])
        self.assertEqual(record["manuscript_hashes"], [rcm.sha256_file(self.manuscript)])
        self.assertEqual(record["source_hash"], record["source_sha256"])
        self.assertEqual(issue["paper_id"], record["paper_id"])
        self.assertEqual(issue["match_confidence"], "unreviewed")
        self.assertEqual(first["stats"]["excluded_low_confidence_records"], 0)
        self.assertEqual(first["stats"]["raw_review_records"], 0)
        audit_json = json.dumps(first["audit"])
        self.assertNotIn(str(self.root), audit_json)
        self.assertNotIn("identifying variation", audit_json)

    def test_same_family_cases_have_distinct_case_scoped_paper_ids(self):
        first_feedback = self.root / "first_round.txt"
        second_feedback = self.root / "second_round.txt"
        second_manuscript = self.root / "manuscript_revision.txt"
        first_feedback.write_text(
            "The first review asks for evidence about the identifying assumption.",
            encoding="utf-8",
        )
        second_feedback.write_text(
            "The second review asks for a clearer measurement validation exercise.",
            encoding="utf-8",
        )
        second_manuscript.write_text("Revised manuscript evidence.", encoding="utf-8")
        manifest = manifest_for(
            [file_entry(self.manuscript)],
            [source_entry(first_feedback, source_id="round_one", extractor="text")],
        )
        manifest["cases"].append(
            {
                "family_id": "family_001",
                "case_id": "case_002",
                "benchmark_tier": "secondary",
                "classification": "later_round",
                "disposition": "evaluation",
                "manuscript_files": [file_entry(second_manuscript)],
                "sources": [
                    source_entry(
                        second_feedback,
                        source_id="round_two",
                        extractor="text",
                        version_match="near_exact",
                    )
                ],
            }
        )

        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)

        self.assertEqual(corpus["stats"]["families"], 1)
        self.assertEqual(corpus["stats"]["cases"], 2)
        self.assertEqual(len({record["paper_id"] for record in corpus["records"]}), 2)
        self.assertEqual({record["family_id"] for record in corpus["records"]}, {"family_001"})
        self.assertEqual(corpus["audit"]["family_tiers"], {"family_001": "primary"})

    def test_private_writer_enforces_location_and_permissions(self):
        private_root = self.root / "private"
        target = rcm.write_private_corpus(
            {"private": "human feedback"},
            "corpora/corpus.json",
            private_root=private_root,
        )

        self.assertEqual(target.read_text(encoding="utf-8").strip(), '{\n  "private": "human feedback"\n}')
        self.assertEqual(stat_mode(private_root), 0o700)
        self.assertEqual(stat_mode(target.parent), 0o700)
        self.assertEqual(stat_mode(target), 0o600)
        with self.assertRaisesRegex(rcm.ManifestValidationError, "must stay below"):
            rcm.write_private_corpus(
                {"private": True},
                self.root / "outside.json",
                private_root=private_root,
            )

    def test_private_snapshot_is_distinct_from_manifest_and_can_reload(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "The report asks for a transparent robustness exercise and interpretation.",
            encoding="utf-8",
        )
        manifest = manifest_for(
            [file_entry(self.manuscript)],
            [source_entry(feedback, source_id="source_001", extractor="text")],
        )
        corpus = rcm.build_review_corpus_from_manifest(manifest, base_dir=self.root)
        private_root = self.root / "private"
        target = rcm.write_private_corpus(
            corpus,
            "corpora/corpus.json",
            private_root=private_root,
        )

        self.assertFalse(rcm.is_review_manifest(target))
        self.assertTrue(rcm.is_normalized_review_corpus(target))
        self.assertEqual(
            rcm.load_private_corpus(target, private_root=private_root), corpus
        )
        self.assertEqual(
            rcm.build_review_corpus_from_manifest(
                target, private_root=private_root
            ),
            corpus,
        )

        tampered = json.loads(target.read_text(encoding="utf-8"))
        tampered["binding_hash"] = "0" * 64
        target.write_text(json.dumps(tampered), encoding="utf-8")
        os.chmod(target, 0o600)
        with self.assertRaisesRegex(rcm.ManifestValidationError, "does not match"):
            rcm.load_private_corpus(target, private_root=private_root)

    def test_snapshot_integrity_rejects_issue_family_and_path_tampering(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "The report asks for an explicit falsification test.", encoding="utf-8"
        )
        corpus = rcm.build_review_corpus_from_manifest(
            manifest_for(
                [file_entry(self.manuscript)],
                [source_entry(feedback, source_id="source_001", extractor="text")],
            ),
            base_dir=self.root,
        )
        private_root = self.root / "private"
        target = rcm.write_private_corpus(
            corpus, "corpora/corpus.json", private_root=private_root
        )

        mutations = [
            lambda value: value["issues"][0].update({"issue_text": "changed issue"}),
            lambda value: value["issues"][0].update({"family_id": "changed_family"}),
            lambda value: value["records"][0]["manuscript_files"].__setitem__(
                0, "/tmp/changed-manuscript.pdf"
            ),
        ]
        original = json.loads(target.read_text(encoding="utf-8"))
        for mutate in mutations:
            tampered = json.loads(json.dumps(original))
            mutate(tampered)
            target.write_text(json.dumps(tampered), encoding="utf-8")
            os.chmod(target, 0o600)
            with self.assertRaisesRegex(
                rcm.ManifestValidationError, "integrity hash"
            ):
                rcm.load_private_corpus(target, private_root=private_root)

    def test_trusted_in_memory_enrichment_requires_explicit_reseal(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "The report asks for a full pre-treatment event-study diagnostic.",
            encoding="utf-8",
        )
        corpus = rcm.build_review_corpus_from_manifest(
            manifest_for(
                [file_entry(self.manuscript)],
                [source_entry(feedback, source_id="source_001", extractor="text")],
            ),
            base_dir=self.root,
        )
        corpus["issues"][0]["issue_type"] = "identification"
        private_root = self.root / "private"
        with self.assertRaisesRegex(rcm.ManifestValidationError, "integrity hash"):
            rcm.write_private_corpus(
                corpus, "stale.json", private_root=private_root
            )

        sealed = rcm.reseal_normalized_corpus(corpus)
        target = rcm.write_private_corpus(
            sealed, "sealed.json", private_root=private_root
        )
        self.assertEqual(
            rcm.load_private_corpus(target, private_root=private_root), sealed
        )

    def test_private_manifest_path_and_permission_guards_keep_template_detectable(self):
        feedback = self.root / "feedback.txt"
        feedback.write_text(
            "The report requests a substantive robustness test.", encoding="utf-8"
        )
        manifest = manifest_for(
            [file_entry(self.manuscript)],
            [source_entry(feedback, source_id="source_001", extractor="text")],
        )
        manifest["cases"][0]["manuscript_files"][0]["path"] = str(self.manuscript)
        manifest["cases"][0]["sources"][0]["path"] = str(feedback)
        outside = self.root / "outside.json"
        outside.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(outside, 0o600)
        private_root = self.root / "private"
        private_root.mkdir(mode=0o700)
        with self.assertRaisesRegex(rcm.ManifestValidationError, "must stay below"):
            rcm.load_review_manifest(outside, private_root=private_root)

        inside = private_root / "manifest.json"
        inside.write_text(json.dumps(manifest), encoding="utf-8")
        os.chmod(inside, 0o644)
        with self.assertRaisesRegex(rcm.ManifestValidationError, "must be 0600"):
            rcm.load_review_manifest(inside, private_root=private_root)
        os.chmod(inside, 0o600)
        self.assertEqual(
            rcm.load_review_manifest(inside, private_root=private_root), manifest
        )

        template = Path(__file__).resolve().parents[1] / "review_manifest.example.json"
        self.assertTrue(rcm.is_review_manifest(template))


def stat_mode(path: Path) -> int:
    return os.stat(path).st_mode & 0o777


if __name__ == "__main__":
    unittest.main()
