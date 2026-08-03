import json
import os
import stat
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

import adjudication_app as app
import feedback_pipeline as fp
import review_adjudication as ra


def issue(
    family,
    issue_id,
    text,
    tier="minor_revision_issue",
    *,
    cluster_id=None,
):
    return {
        "family_id": family,
        "case_id": "v1",
        "atomic_issue_id": issue_id,
        "issue_text": text,
        "decision_tier": tier,
        "issue_type": "identification",
        "source_id": f"source-{family}",
        "source_hash": f"hash-{family}",
        "reviewer_id": f"reviewer-{family}",
        "source_locator": "p. 1",
        "extraction_rule_hash": "rules-v1",
        "cluster_id": cluster_id or f"cluster-{issue_id}",
    }


def complete_row(row):
    row["tier_screen"] = row["provisional_tier"]
    if row["full_adjudication_required"] == "yes":
        row["include"] = "yes"
        row["canonical_issue"] = row["representative_text"]
        row["severity"] = (
            "major_revision_issue"
            if row["tier_screen"] == "major"
            else "minor_revision_issue"
        )
        row["evidentiary_support"] = "supported"


class GoldEditorHelperTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.packet = ra.write_gold_adjudication_packet(
            [
                issue(
                    "family-a",
                    "major",
                    "The identifying assumption is not defended.",
                    "major_revision_issue",
                ),
                issue("family-a", "minor-a", "Clarify the outcome definition."),
                issue("family-a", "minor-b", "Clarify the appendix table."),
                issue("family-b", "minor-c", "Define the comparison group."),
            ],
            self.tmp.name,
            minor_sample_size=0,
            private_root=self.tmp.name,
        )

    def test_requirements_progress_and_promoted_minor(self):
        rows = deepcopy(self.packet["rows"])
        progress = ra.gold_adjudication_progress(rows)
        self.assertEqual(progress["total"], 4)
        self.assertEqual(progress["tier_done"], 0)
        self.assertEqual(progress["full_total"], 1)

        major = next(row for row in rows if row["provisional_tier"] == "major")
        complete_row(major)
        quick_minor = next(row for row in rows if row["representative_issue_id"] == "minor-a")
        quick_minor["tier_screen"] = "minor"
        promoted = next(row for row in rows if row["representative_issue_id"] == "minor-b")
        promoted["tier_screen"] = "major"

        progress = ra.gold_adjudication_progress(rows)
        self.assertEqual(progress["tier_done"], 3)
        self.assertEqual(progress["full_total"], 2)
        self.assertEqual(progress["full_done"], 1)
        promoted_state = ra.gold_row_requirements(
            promoted,
            family_by_cluster={row["cluster_id"]: row["family_id"] for row in rows},
        )
        self.assertTrue(promoted_state["full_required"])
        self.assertIn("include", promoted_state["pending_fields"])

        tier_queue = app._filter_rows(
            rows,
            queue="Tier screen",
            family="All families",
            group="All selected rows",
            unfinished_only=True,
        )
        self.assertIn(promoted["cluster_id"], {row["cluster_id"] for row in tier_queue})
        self.assertNotIn(major["cluster_id"], {row["cluster_id"] for row in tier_queue})

    def test_save_is_atomic_private_and_preserves_bound_fields(self):
        state = ra.load_gold_editor_state(
            self.packet["csv_path"], private_root=self.tmp.name
        )
        rows = deepcopy(state["rows"])
        target = next(row for row in rows if row["provisional_tier"] == "major")
        immutable_before = {column: target[column] for column in ra.GOLD_IMMUTABLE_COLUMNS}
        complete_row(target)

        saved = ra.save_gold_editor_rows(
            self.packet["csv_path"],
            rows,
            expected_revision=state["revision"],
            private_root=self.tmp.name,
        )

        saved_target = next(
            row for row in saved["rows"] if row["cluster_id"] == target["cluster_id"]
        )
        self.assertEqual(
            {column: saved_target[column] for column in ra.GOLD_IMMUTABLE_COLUMNS},
            immutable_before,
        )
        self.assertEqual(
            stat.S_IMODE(Path(self.packet["csv_path"]).stat().st_mode), 0o600
        )
        self.assertNotEqual(saved["revision"], state["revision"])

    def test_stale_browser_revision_cannot_overwrite_newer_save(self):
        original = ra.load_gold_editor_state(
            self.packet["csv_path"], private_root=self.tmp.name
        )
        first_rows = deepcopy(original["rows"])
        first_rows[0]["tier_screen"] = "minor"
        first = ra.save_gold_editor_rows(
            self.packet["csv_path"],
            first_rows,
            expected_revision=original["revision"],
            private_root=self.tmp.name,
        )

        stale_rows = deepcopy(original["rows"])
        stale_rows[0]["tier_screen"] = "exclude"
        with self.assertRaisesRegex(RuntimeError, "changed since this page loaded"):
            ra.save_gold_editor_rows(
                self.packet["csv_path"],
                stale_rows,
                expected_revision=original["revision"],
                private_root=self.tmp.name,
            )
        current = ra.load_gold_editor_state(
            self.packet["csv_path"], private_root=self.tmp.name
        )
        self.assertEqual(current["revision"], first["revision"])
        self.assertEqual(current["rows"][0]["tier_screen"], "minor")

    def test_simultaneous_saves_are_serialized(self):
        original = ra.load_gold_editor_state(
            self.packet["csv_path"], private_root=self.tmp.name
        )
        first_rows = deepcopy(original["rows"])
        second_rows = deepcopy(original["rows"])
        first_rows[0]["tier_screen"] = "minor"
        second_rows[0]["tier_screen"] = "exclude"
        entered_write = threading.Event()
        release_write = threading.Event()
        original_write = ra._write_csv
        outcomes = []

        def delayed_write(*args, **kwargs):
            if not entered_write.is_set():
                entered_write.set()
                self.assertTrue(release_write.wait(timeout=5))
            return original_write(*args, **kwargs)

        def save(rows):
            try:
                result = ra.save_gold_editor_rows(
                    self.packet["csv_path"],
                    rows,
                    expected_revision=original["revision"],
                    private_root=self.tmp.name,
                )
                outcomes.append(("saved", result["revision"]))
            except Exception as exc:  # capture the background-thread result
                outcomes.append(("error", exc))

        with patch.object(ra, "_write_csv", side_effect=delayed_write):
            first = threading.Thread(target=save, args=(first_rows,))
            second = threading.Thread(target=save, args=(second_rows,))
            first.start()
            self.assertTrue(entered_write.wait(timeout=5))
            second.start()
            time.sleep(0.05)
            self.assertTrue(second.is_alive())
            release_write.set()
            first.join(timeout=5)
            second.join(timeout=5)

        self.assertFalse(first.is_alive())
        self.assertFalse(second.is_alive())
        self.assertEqual(sum(kind == "saved" for kind, _ in outcomes), 1)
        errors = [value for kind, value in outcomes if kind == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        lock_path = Path(self.packet["csv_path"]).with_name(
            ".gold_adjudication.csv.lock"
        )
        self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_invalid_or_immutable_edits_do_not_replace_file(self):
        state = ra.load_gold_editor_state(
            self.packet["csv_path"], private_root=self.tmp.name
        )
        path = Path(self.packet["csv_path"])
        before = path.read_bytes()

        immutable_edit = deepcopy(state["rows"])
        immutable_edit[0]["representative_text"] = "Changed bound text"
        with self.assertRaisesRegex(ValueError, "bound packet fields"):
            ra.save_gold_editor_rows(
                path,
                immutable_edit,
                expected_revision=state["revision"],
                private_root=self.tmp.name,
            )
        self.assertEqual(path.read_bytes(), before)

        invalid_edit = deepcopy(state["rows"])
        invalid_edit[0]["duplicate_cluster_ids"] = "not-json"
        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            ra.save_gold_editor_rows(
                path,
                invalid_edit,
                expected_revision=state["revision"],
                private_root=self.tmp.name,
            )
        self.assertEqual(path.read_bytes(), before)

    def test_duplicate_choices_are_family_scoped(self):
        rows = self.packet["rows"]
        current = next(row for row in rows if row["family_id"] == "family-a")
        choices = ra.eligible_duplicate_clusters(rows, current["cluster_id"])
        self.assertTrue(choices)
        self.assertTrue(all(row["family_id"] == "family-a" for row in choices))
        self.assertNotIn(current["cluster_id"], {row["cluster_id"] for row in choices})

    def test_binding_source_prefers_manifest_and_falls_back_to_snapshot(self):
        root = Path(self.tmp.name)
        packet_dir = root / "pilot" / "adjudication" / "final"
        packet_dir.mkdir(parents=True)
        packet = packet_dir / "gold_adjudication.csv"
        packet.touch()
        corpus = root / "pilot" / "corpus.json"
        corpus.write_text("{}", encoding="utf-8")
        self.assertEqual(app._binding_source(packet, root), corpus.resolve())
        manifest = root / "pilot" / "review_manifest.json"
        manifest.write_text("{}", encoding="utf-8")
        self.assertEqual(app._binding_source(packet, root), manifest.resolve())

    def test_row_edit_clears_irrelevant_values(self):
        row = deepcopy(self.packet["rows"][1])
        updated = app._build_row_edit(
            row,
            tier="minor",
            include="yes",
            canonical_issue="Should be cleared",
            severity="minor_revision_issue",
            evidentiary_support="supported",
            duplicate_cluster_ids=[self.packet["rows"][2]["cluster_id"]],
            exclusion_reason="Should be cleared",
            adjudicator_notes="Keep this note",
        )
        self.assertEqual(updated["include"], "")
        self.assertEqual(updated["canonical_issue"], "")
        self.assertEqual(updated["duplicate_cluster_ids"], "[]")
        self.assertEqual(updated["exclusion_reason"], "")
        self.assertEqual(updated["adjudicator_notes"], "Keep this note")

    def test_malformed_json_list_cells_are_invalid(self):
        gold_rows = deepcopy(self.packet["rows"])
        gold_rows[0]["duplicate_cluster_ids"] = "not-json"
        self.assertEqual(ra.validate_gold_rows(gold_rows)["status"], "invalid")

        generated_row = {column: "" for column in ra.GENERATED_COLUMNS}
        generated_row.update(
            {
                "packet_version": ra.PACKET_VERSION,
                "binding_hash": "baseline-binding",
                "gold_binding_hash": "gold-binding",
                "family_id": "family-a",
                "case_id": "v1",
                "rank": "1",
                "generated_issue_id": "generated-1",
                "correctness": "correct",
                "significance": "significant",
                "evidence_sufficiency": "sufficient",
                "human_match_status": "unmatched",
                "confirmed_human_cluster_ids": "not-json",
                "duplicate_status": "unique",
                "valid_novelty": "no",
            }
        )
        generated = ra.validate_generated_rows([generated_row])
        self.assertEqual(generated["status"], "invalid")
        self.assertTrue(
            any("must be a JSON list" in error for error in generated["errors"])
        )

    def test_generated_edit_clears_conditional_fields(self):
        row = {column: "" for column in ra.GENERATED_COLUMNS}
        row.update(
            {
                "confirmed_human_cluster_ids": json.dumps(["cluster-a"]),
                "duplicate_of_generated_id": "generated-2",
                "valid_novelty": "yes",
            }
        )
        updated = app._build_generated_row_edit(
            row,
            correctness="correct",
            significance="significant",
            evidence_sufficiency="sufficient",
            human_match_status="unmatched",
            confirmed_human_cluster_ids=["cluster-a"],
            duplicate_status="duplicate",
            duplicate_of_generated_id="generated-2",
            valid_novelty="yes",
            adjudicator_notes="  checked  ",
        )
        self.assertEqual(updated["confirmed_human_cluster_ids"], "[]")
        self.assertEqual(updated["duplicate_of_generated_id"], "generated-2")
        self.assertEqual(updated["valid_novelty"], "no")
        self.assertEqual(updated["adjudicator_notes"], "checked")

    def test_generated_candidates_are_scoped_to_family_and_case(self):
        gold_rows = [
            {
                "cluster_id": "a-included",
                "family_id": "family-a",
                "include": "yes",
            },
            {
                "cluster_id": "a-excluded",
                "family_id": "family-a",
                "include": "no",
            },
            {
                "cluster_id": "b-included",
                "family_id": "family-b",
                "include": "yes",
            },
        ]
        self.assertEqual(
            [
                row["cluster_id"]
                for row in app._generated_match_candidates(gold_rows, "family-a")
            ],
            ["a-included"],
        )

        generated_rows = [
            {
                "generated_issue_id": "a1",
                "family_id": "family-a",
                "case_id": "case-1",
                "rank": "1",
            },
            {
                "generated_issue_id": "a2",
                "family_id": "family-a",
                "case_id": "case-1",
                "rank": "2",
            },
            {
                "generated_issue_id": "a3",
                "family_id": "family-a",
                "case_id": "case-2",
                "rank": "1",
            },
            {
                "generated_issue_id": "b1",
                "family_id": "family-b",
                "case_id": "case-1",
                "rank": "1",
            },
        ]
        self.assertEqual(
            [
                row["generated_issue_id"]
                for row in app._generated_duplicate_candidates(
                    generated_rows, "a1"
                )
            ],
            ["a2"],
        )


class AdjudicationAppSmokeTests(unittest.TestCase):
    def test_app_opens_and_saves_a_complete_full_row(self):
        with tempfile.TemporaryDirectory() as tmp:
            packet = ra.write_gold_adjudication_packet(
                [
                    issue(
                        "family-a",
                        "major",
                        "The identifying assumption is not defended.",
                        "major_revision_issue",
                    )
                ],
                tmp,
                private_root=tmp,
            )
            environment = {
                "FEEDBACK_LLM_PRIVATE_ROOT": tmp,
                "FEEDBACK_LLM_ADJUDICATION_PATH": packet["csv_path"],
            }
            with patch.dict(os.environ, environment):
                test_app = AppTest.from_file(
                    str(Path(app.__file__)), default_timeout=10
                ).run()
                self.assertEqual(len(test_app.exception), 0)
                self.assertEqual(test_app.title[0].value, "Adjudicate historical feedback")

                next(
                    field
                    for field in test_app.radio
                    if field.label == "How consequential is this concern?"
                ).set_value("major")
                next(
                    field
                    for field in test_app.radio
                    if field.label == "Keep this as a valid human concern?"
                ).set_value("yes")
                next(
                    field
                    for field in test_app.text_area
                    if field.label == "Canonical issue wording"
                ).set_value("The paper must defend its identifying assumption.")
                next(
                    field for field in test_app.selectbox if field.label == "Severity"
                ).set_value("major_revision_issue")
                next(
                    field
                    for field in test_app.selectbox
                    if field.label == "Evidentiary support"
                ).set_value("supported")
                next(button for button in test_app.button if button.label == "Save").click()
                test_app.run()
                self.assertEqual(len(test_app.exception), 0)

            saved = ra.load_gold_editor_state(packet["csv_path"], private_root=tmp)
            self.assertEqual(saved["validation"]["status"], "ready")
            self.assertEqual(saved["rows"][0]["tier_screen"], "major")

    def test_generated_workflow_opens_and_saves_one_complete_label(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            human_packet = ra.write_gold_adjudication_packet(
                [
                    issue(
                        "family-a",
                        "major",
                        "The identifying assumption is not defended.",
                        "major_revision_issue",
                    )
                ],
                root,
                private_root=root,
            )
            gold_rows = deepcopy(human_packet["rows"])
            complete_row(gold_rows[0])
            gold = ra.validate_gold_rows(
                gold_rows,
                expected_binding_hash=human_packet["binding_hash"],
            )
            self.assertEqual(gold["status"], "ready")
            ra._write_csv(
                root / "gold_adjudication.csv",
                ra.GOLD_COLUMNS,
                gold_rows,
            )

            corpus = {"binding_hash": "test-corpus-binding"}
            run_metadata = {
                "git": {"commit": "test-commit"},
                "routing": {"generation": "test-model"},
                "num_agents": 8,
                "reviewer_roles": {"editor": 1},
                "top_k": 5,
                "top_k_policy": "up_to_k_missing_slots_count_as_misses",
                "memory_mode": "none",
                "gold_mode": "complete",
                "gold_binding_hash": gold["binding_hash"],
                "benchmark_binding": {"cases": ["case-a"]},
            }
            run_context = fp._review_eval_generated_binding_context(
                corpus,
                run_metadata,
            )
            generated_packet = ra.write_generated_adjudication_packet(
                [
                    {
                        "family_id": "family-a",
                        "case_id": "case-a",
                        "id": "pipeline-1",
                        "rank": 1,
                        "text": "The identifying assumption needs a direct defense.",
                        "evidence_ids": ["P001"],
                    }
                ],
                gold,
                root,
                run_binding_context=run_context,
                private_root=root,
            )
            manifest = root / "review_manifest.json"
            manifest.write_text("{}", encoding="utf-8")
            manifest.chmod(0o600)
            audit_path = root / "baseline_run.local_audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "run_metadata": run_metadata,
                        "generated_adjudication": {
                            "binding_hash": generated_packet["binding_hash"]
                        },
                    }
                ),
                encoding="utf-8",
            )
            audit_path.chmod(0o600)

            environment = {
                "FEEDBACK_LLM_PRIVATE_ROOT": str(root),
                "FEEDBACK_LLM_GENERATED_ADJUDICATION_PATH": generated_packet[
                    "csv_path"
                ],
            }
            with (
                patch.dict(os.environ, environment),
                patch.object(fp, "load_review_corpus", return_value=corpus),
                patch.object(fp, "_manifest_baseline_audit_errors", return_value=[]),
                patch.object(
                    fp,
                    "_load_current_gold_adjudication",
                    return_value=gold,
                ),
                patch.object(
                    fp,
                    "_gold_adjudication_for_evaluation",
                    return_value=gold,
                ),
            ):
                test_app = AppTest.from_file(
                    str(Path(app.__file__)), default_timeout=10
                ).run()
                self.assertEqual(len(test_app.exception), 0)
                self.assertEqual(test_app.title[0].value, "Label generated feedback")
                self.assertEqual(
                    next(
                        field for field in test_app.radio if field.label == "Task"
                    ).value,
                    "Generated issues",
                )

                next(
                    field
                    for field in test_app.radio
                    if field.label == "Is the critique correct?"
                ).set_value("correct")
                next(
                    field
                    for field in test_app.radio
                    if field.label == "How significant is it?"
                ).set_value("significant")
                next(
                    field
                    for field in test_app.radio
                    if field.label == "Is the cited evidence sufficient?"
                ).set_value("sufficient")
                next(
                    field
                    for field in test_app.radio
                    if field.label
                    == "Does it match an adjudicated human concern?"
                ).set_value("unmatched")
                next(
                    field
                    for field in test_app.radio
                    if field.label
                    == "Is this a duplicate of another final critique for this manuscript?"
                ).set_value("unique")
                test_app.run()
                self.assertEqual(len(test_app.exception), 0)
                novelty = next(
                    field
                    for field in test_app.radio
                    if field.label
                    == "Is this a valid novel contribution from the pipeline?"
                )
                self.assertFalse(novelty.disabled)
                novelty.set_value("no")
                next(
                    button for button in test_app.button if button.label == "Save"
                ).click()
                test_app.run()
                self.assertEqual(len(test_app.exception), 0)

            saved = ra.load_generated_editor_state(
                generated_packet["csv_path"],
                expected_binding_hash=generated_packet["binding_hash"],
                expected_gold_binding_hash=gold["binding_hash"],
                gold_cluster_families={
                    gold_rows[0]["cluster_id"]: "family-a"
                },
                run_binding_context=run_context,
                private_root=root,
            )
            self.assertEqual(saved["validation"]["status"], "ready")
            self.assertEqual(saved["progress"]["complete"], 1)
            self.assertEqual(saved["rows"][0]["correctness"], "correct")


if __name__ == "__main__":
    unittest.main()
