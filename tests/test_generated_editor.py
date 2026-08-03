import csv
import json
import stat
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import feedback_pipeline as fp
import review_adjudication as ra


def _human_issue(family: str, issue_id: str) -> dict:
    return {
        "family_id": family,
        "case_id": f"{family}-case",
        "atomic_issue_id": issue_id,
        "issue_text": f"Human concern {issue_id} for {family}.",
        "decision_tier": "major_revision_issue",
        "issue_type": "identification",
        "source_id": f"source-{family}",
        "source_hash": f"hash-{family}",
        "reviewer_id": f"reviewer-{family}",
        "source_locator": "p. 1",
        "extraction_rule_hash": "rules-v1",
        "cluster_id": f"cluster-{issue_id}",
    }


def _complete_generated_row(row: dict, *, human_cluster: str | None = None) -> None:
    row.update(
        {
            "correctness": "correct",
            "significance": "significant",
            "evidence_sufficiency": "sufficient",
            "human_match_status": "matched" if human_cluster else "unmatched",
            "confirmed_human_cluster_ids": json.dumps(
                [human_cluster] if human_cluster else []
            ),
            "duplicate_status": "unique",
            "duplicate_of_generated_id": "",
            "valid_novelty": "no",
        }
    )


def _rewrite_csv(path: str | Path, update) -> None:
    csv_path = Path(path)
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        update(row)
    ra._write_csv(csv_path, ra.GENERATED_COLUMNS, rows)


class GeneratedEditorBackendTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        gold_packet = ra.write_gold_adjudication_packet(
            [
                _human_issue("family-a", "a"),
                _human_issue("family-b", "b"),
            ],
            self.tmp.name,
            private_root=self.tmp.name,
        )
        gold_rows = deepcopy(gold_packet["rows"])
        for row in gold_rows:
            row.update(
                {
                    "tier_screen": "major",
                    "include": "yes",
                    "canonical_issue": row["representative_text"],
                    "severity": "major_revision_issue",
                    "evidentiary_support": "supported",
                }
            )
        self.gold = ra.validate_gold_rows(
            gold_rows,
            expected_binding_hash=gold_packet["binding_hash"],
        )
        self.gold_families = {
            row["cluster_id"]: row["family_id"] for row in self.gold["rows"]
        }
        self.family_a_cluster = next(
            cluster_id
            for cluster_id, family_id in self.gold_families.items()
            if family_id == "family-a"
        )
        generated = [
            {
                "family_id": "family-a",
                "case_id": "case-a-1",
                "id": "a1",
                "rank": 1,
                "text": "First generated concern for family A.",
                "evidence_ids": ["P001"],
            },
            {
                "family_id": "family-a",
                "case_id": "case-a-1",
                "id": "a2",
                "rank": 2,
                "text": "Second generated concern for family A.",
                "evidence_ids": ["P002"],
            },
            {
                "family_id": "family-a",
                "case_id": "case-a-2",
                "id": "a3",
                "rank": 1,
                "text": "Generated concern for another family A case.",
                "evidence_ids": ["P003"],
            },
            {
                "family_id": "family-b",
                "case_id": "case-b-1",
                "id": "b1",
                "rank": 1,
                "text": "Generated concern for family B.",
                "evidence_ids": ["P004"],
            },
        ]
        self.context = {"commit": "abc", "memory_mode": "none"}
        self.packet = ra.write_generated_adjudication_packet(
            generated,
            self.gold,
            self.tmp.name,
            run_binding_context=self.context,
            private_root=self.tmp.name,
        )

    def _load(self):
        return ra.load_generated_editor_state(
            self.packet["csv_path"],
            expected_binding_hash=self.packet["binding_hash"],
            expected_gold_binding_hash=self.gold["binding_hash"],
            gold_cluster_families=self.gold_families,
            run_binding_context=self.context,
            top_k=5,
            private_root=self.tmp.name,
        )

    def _save(self, state, rows):
        return ra.save_generated_editor_rows(
            self.packet["csv_path"],
            rows,
            expected_revision=state["revision"],
            expected_binding_hash=self.packet["binding_hash"],
            expected_gold_binding_hash=self.gold["binding_hash"],
            gold_cluster_families=self.gold_families,
            run_binding_context=self.context,
            top_k=5,
            private_root=self.tmp.name,
        )

    def test_column_partition_and_progress_helpers(self):
        editable = set(ra.GENERATED_EDITABLE_COLUMNS)
        immutable = set(ra.GENERATED_IMMUTABLE_COLUMNS)
        self.assertFalse(editable & immutable)
        self.assertEqual(editable | immutable, set(ra.GENERATED_COLUMNS))
        self.assertTrue(
            {
                "proposed_human_cluster_id",
                "proposed_match_score",
                "proposed_shared_terms",
            }
            <= immutable
        )

        state = self._load()
        self.assertEqual(state["validation"]["status"], "pending_human_adjudication")
        self.assertEqual(state["progress"]["total"], 4)
        self.assertEqual(state["progress"]["complete"], 0)
        requirements = ra.generated_row_requirements(
            state["rows"][0],
            generated_rows=state["rows"],
            gold_cluster_families=self.gold_families,
            top_k=5,
        )
        self.assertFalse(requirements["complete"])
        self.assertIn("correctness", requirements["pending_fields"])

    def test_valid_pending_draft_and_complete_save(self):
        state = self._load()
        immutable_before = [
            {column: row[column] for column in ra.GENERATED_IMMUTABLE_COLUMNS}
            for row in state["rows"]
        ]
        draft = deepcopy(state["rows"])
        draft[0]["correctness"] = "correct"
        saved_draft = self._save(state, draft)
        self.assertEqual(
            saved_draft["validation"]["status"], "pending_human_adjudication"
        )
        self.assertNotEqual(saved_draft["revision"], state["revision"])
        self.assertEqual(
            [
                {column: row[column] for column in ra.GENERATED_IMMUTABLE_COLUMNS}
                for row in saved_draft["rows"]
            ],
            immutable_before,
        )

        completed = deepcopy(saved_draft["rows"])
        for row in completed:
            _complete_generated_row(row)
        completed[0]["human_match_status"] = "matched"
        completed[0]["confirmed_human_cluster_ids"] = json.dumps(
            [self.family_a_cluster]
        )
        saved = self._save(saved_draft, completed)
        self.assertEqual(saved["validation"]["status"], "ready")
        self.assertEqual(saved["progress"]["complete"], 4)
        self.assertEqual(
            stat.S_IMODE(Path(self.packet["csv_path"]).stat().st_mode), 0o600
        )

    def test_stale_revision_cannot_overwrite(self):
        original = self._load()
        first_rows = deepcopy(original["rows"])
        first_rows[0]["correctness"] = "correct"
        first = self._save(original, first_rows)

        stale_rows = deepcopy(original["rows"])
        stale_rows[0]["correctness"] = "incorrect"
        with self.assertRaisesRegex(RuntimeError, "changed since this page loaded"):
            self._save(original, stale_rows)
        current = self._load()
        self.assertEqual(current["revision"], first["revision"])
        self.assertEqual(current["rows"][0]["correctness"], "correct")

    def test_simultaneous_saves_are_serialized(self):
        original = self._load()
        first_rows = deepcopy(original["rows"])
        second_rows = deepcopy(original["rows"])
        first_rows[0]["correctness"] = "correct"
        second_rows[0]["correctness"] = "incorrect"
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
                outcomes.append(("saved", self._save(original, rows)["revision"]))
            except Exception as exc:
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

        self.assertEqual(sum(kind == "saved" for kind, _ in outcomes), 1)
        errors = [value for kind, value in outcomes if kind == "error"]
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], RuntimeError)
        lock_path = Path(self.packet["csv_path"]).with_name(
            ".generated_adjudication.csv.lock"
        )
        self.assertEqual(stat.S_IMODE(lock_path.stat().st_mode), 0o600)

    def test_every_immutable_field_and_row_shape_are_protected(self):
        state = self._load()
        path = Path(self.packet["csv_path"])
        before = path.read_bytes()
        for column in ra.GENERATED_IMMUTABLE_COLUMNS:
            changed = deepcopy(state["rows"])
            changed[0][column] = f"changed::{column}"
            with self.subTest(column=column):
                with self.assertRaises(ValueError):
                    self._save(state, changed)
                self.assertEqual(path.read_bytes(), before)

        with self.assertRaisesRegex(ValueError, "row count"):
            self._save(state, state["rows"][:-1])
        reordered = deepcopy(state["rows"])
        reordered[0], reordered[1] = reordered[1], reordered[0]
        with self.assertRaisesRegex(ValueError, "row order"):
            self._save(state, reordered)
        missing = deepcopy(state["rows"])
        del missing[0]["correctness"]
        with self.assertRaisesRegex(ValueError, "missing required columns"):
            self._save(state, missing)
        self.assertEqual(path.read_bytes(), before)

    def test_invalid_edits_do_not_replace_file(self):
        state = self._load()
        path = Path(self.packet["csv_path"])
        before = path.read_bytes()
        invalid = deepcopy(state["rows"])
        invalid[0]["correctness"] = "not-a-label"
        with self.assertRaisesRegex(ValueError, "correctness must be one of"):
            self._save(state, invalid)
        invalid = deepcopy(state["rows"])
        invalid[0]["confirmed_human_cluster_ids"] = "not-json"
        with self.assertRaisesRegex(ValueError, "must be a JSON list"):
            self._save(state, invalid)
        self.assertEqual(path.read_bytes(), before)

    def test_family_case_and_novelty_constraints(self):
        rows = deepcopy(self._load()["rows"])
        for row in rows:
            _complete_generated_row(row)
        family_b_cluster = next(
            cluster_id
            for cluster_id, family_id in self.gold_families.items()
            if family_id == "family-b"
        )
        rows[0]["human_match_status"] = "matched"
        rows[0]["confirmed_human_cluster_ids"] = json.dumps([family_b_cluster])
        validation = ra.validate_generated_rows(
            rows,
            gold_cluster_families=self.gold_families,
            top_k=5,
        )
        self.assertEqual(validation["status"], "invalid")
        self.assertTrue(any("same family" in error for error in validation["errors"]))

        rows = deepcopy(self._load()["rows"])
        for row in rows:
            _complete_generated_row(row)
        rows[0]["duplicate_status"] = "duplicate"
        rows[0]["duplicate_of_generated_id"] = rows[2]["generated_issue_id"]
        validation = ra.validate_generated_rows(rows, top_k=5)
        self.assertTrue(
            any("same family and case" in error for error in validation["errors"])
        )

        rows[0]["duplicate_of_generated_id"] = rows[1]["generated_issue_id"]
        rows[0]["valid_novelty"] = "yes"
        validation = ra.validate_generated_rows(rows, top_k=5)
        self.assertTrue(any("must be unique" in error for error in validation["errors"]))

    def test_external_bindings_and_immutable_content_are_rechecked(self):
        with self.assertRaisesRegex(ValueError, "immutable fields"):
            ra.load_generated_editor_state(
                self.packet["csv_path"],
                expected_binding_hash="wrong-binding",
                expected_gold_binding_hash=self.gold["binding_hash"],
                gold_cluster_families=self.gold_families,
                run_binding_context=self.context,
                private_root=self.tmp.name,
            )

        _rewrite_csv(
            self.packet["csv_path"],
            lambda row: row.update({"generated_text": "tampered"})
            if row["rank"] == "1" and row["case_id"] == "case-a-1"
            else None,
        )
        with self.assertRaisesRegex(ValueError, "immutable fields"):
            self._load()


class GeneratedBindingContextTests(unittest.TestCase):
    def test_shared_context_uses_frozen_run_metadata(self):
        corpus = {"binding_hash": "corpus-binding"}
        metadata = {
            "git": {"commit": "abc123"},
            "routing": {"generation": "model-a"},
            "num_agents": 8,
            "reviewer_roles": {"editor": 1},
            "top_k": 5,
            "top_k_policy": "up_to_k_missing_slots_count_as_misses",
            "memory_mode": "none",
            "gold_mode": "partial",
            "benchmark_binding": {"cases": ["one"]},
        }
        context = fp._review_eval_generated_binding_context(corpus, metadata)
        self.assertEqual(
            context,
            {
                "corpus_binding_hash": "corpus-binding",
                "git_commit": "abc123",
                "routing": {"generation": "model-a"},
                "num_agents": 8,
                "reviewer_roles": {"editor": 1},
                "top_k": 5,
                "top_k_policy": "up_to_k_missing_slots_count_as_misses",
                "memory_mode": "none",
                "gold_mode": "partial",
                "benchmark_binding": {"cases": ["one"]},
            },
        )
        self.assertEqual(
            fp._review_eval_generated_binding_context(
                corpus, metadata, gold_mode="complete"
            )["gold_mode"],
            "complete",
        )


if __name__ == "__main__":
    unittest.main()
