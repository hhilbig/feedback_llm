import csv
import json
import os
import stat
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import review_adjudication as ra


def issue(
    family,
    issue_id,
    text,
    tier="minor_revision_issue",
    *,
    case="v1",
    source="review-1",
    reviewer="reviewer-a",
    locator="p. 1",
):
    return {
        "family_id": family,
        "case_id": case,
        "atomic_issue_id": issue_id,
        "issue_text": text,
        "decision_tier": tier,
        "issue_type": "identification",
        "source_id": source,
        "source_hash": "sha256-" + source,
        "reviewer_id": reviewer,
        "source_locator": locator,
        "extraction_rule_hash": "rules-v1",
    }


def rewrite_csv(path, update):
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        columns = reader.fieldnames
        rows = [dict(row) for row in reader]
    for row in rows:
        update(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    return rows


class ClusterAndSelectionTests(unittest.TestCase):
    def test_clusters_are_stable_family_scoped_and_preserve_provenance(self):
        text = "The identification strategy needs full pretreatment leads and a joint pretrend test."
        issues = [
            issue("family-a", "a1", text, "major_revision_issue"),
            issue("family-a", "a2", text, source="review-2", reviewer="reviewer-b"),
            issue("family-b", "b1", text, "major_revision_issue"),
        ]

        forward = ra.cluster_normalized_issues(issues)
        reverse = ra.cluster_normalized_issues(list(reversed(issues)))

        self.assertEqual(forward, reverse)
        self.assertEqual(len(forward), 2)
        family_a = next(row for row in forward if row["family_id"] == "family-a")
        self.assertEqual(family_a["issue_count"], 2)
        self.assertEqual(family_a["provisional_tier"], "major")
        self.assertEqual(family_a["reviewer_ids"], ["reviewer-a", "reviewer-b"])
        self.assertEqual(family_a["source_ids"], ["review-1", "review-2"])

    def test_selection_uses_all_majors_and_five_deterministic_minors_per_family(self):
        clusters = []
        for family, major_count, minor_count in [("family-a", 2, 8), ("family-b", 1, 3)]:
            for index in range(major_count):
                clusters.append(
                    {"family_id": family, "cluster_id": f"{family}-major-{index}", "provisional_tier": "major"}
                )
            for index in range(minor_count):
                clusters.append(
                    {"family_id": family, "cluster_id": f"{family}-minor-{index}", "provisional_tier": "minor"}
                )

        selected = ra.select_full_adjudication_clusters(clusters)
        selected_again = ra.select_full_adjudication_clusters(list(reversed(clusters)))

        self.assertEqual(selected, selected_again)
        self.assertTrue({"family-a-major-0", "family-a-major-1", "family-b-major-0"} <= selected)
        self.assertEqual(len([item for item in selected if item.startswith("family-a-minor")]), 5)
        self.assertEqual(len([item for item in selected if item.startswith("family-b-minor")]), 3)


class GoldPacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.issues = [
            issue("family-a", "major-1", "The causal design lacks a credible identifying assumption.", "major_revision_issue")
        ] + [
            issue("family-a", f"minor-{index}", f"Clarify robustness diagnostic number {index} in the appendix.")
            for index in range(7)
        ]
        for item in self.issues:
            item["cluster_id"] = "declared-" + item["atomic_issue_id"]

    def test_packet_contains_every_cluster_is_private_and_detects_staleness(self):
        packet = ra.write_gold_adjudication_packet(
            self.issues,
            self.tmp.name,
            binding_context={"manifest_hash": "manifest-v1"},
            private_root=self.tmp.name,
        )

        self.assertEqual(packet["cluster_count"], 8)
        self.assertEqual(packet["full_adjudication_count"], 6)
        self.assertEqual(stat.S_IMODE(Path(self.tmp.name).stat().st_mode), 0o700)
        self.assertEqual(stat.S_IMODE(Path(packet["csv_path"]).stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(Path(packet["markdown_path"]).stat().st_mode), 0o600)
        self.assertTrue(all(row["tier_screen"] == "" for row in packet["rows"]))
        self.assertEqual(
            ra.load_gold_adjudication(
                packet["csv_path"],
                expected_binding_hash=packet["binding_hash"],
                private_root=self.tmp.name,
            )["status"],
            "pending_human_adjudication",
        )

        def complete(row):
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

        rewrite_csv(packet["csv_path"], complete)
        ready = ra.load_gold_adjudication(
            packet["csv_path"],
            clusters=packet["clusters"],
            binding_context={"manifest_hash": "manifest-v1"},
            private_root=self.tmp.name,
        )
        self.assertEqual(ready["status"], "ready")

        stale = ra.load_gold_adjudication(
            packet["csv_path"],
            clusters=packet["clusters"],
            binding_context={"manifest_hash": "changed-manifest"},
            private_root=self.tmp.name,
        )
        self.assertEqual(stale["status"], "stale")
        self.assertTrue(any("binding hash" in item for item in stale["errors"]))

        tampered = rewrite_csv(
            packet["csv_path"],
            lambda row: row.update({"representative_text": "changed"})
            if row["cluster_id"] == ready["rows"][0]["cluster_id"]
            else None,
        )
        self.assertTrue(tampered)
        text_stale = ra.load_gold_adjudication(
            packet["csv_path"],
            clusters=packet["clusters"],
            binding_context={"manifest_hash": "manifest-v1"},
            private_root=self.tmp.name,
        )
        self.assertEqual(text_stale["status"], "stale")
        self.assertTrue(any("representative_text" in item for item in text_stale["errors"]))

    def test_promoted_minor_requires_full_adjudication(self):
        packet = ra.write_gold_adjudication_packet(
            [issue("family-a", "minor-1", "Clarify the outcome definition.")],
            self.tmp.name,
            minor_sample_size=0,
            private_root=self.tmp.name,
        )
        rows = deepcopy(packet["rows"])
        rows[0]["tier_screen"] = "major"
        validation = ra.validate_gold_rows(rows, expected_binding_hash=packet["binding_hash"])
        self.assertEqual(validation["status"], "pending_human_adjudication")
        self.assertIn(f"{rows[0]['cluster_id']}: include", validation["pending_fields"])

    def test_packet_path_permissions_and_atomic_creation_are_enforced(self):
        private_root = Path(self.tmp.name) / "private"
        outside = Path(self.tmp.name) / "outside"
        with self.assertRaisesRegex(ValueError, "must stay below"):
            ra.write_gold_adjudication_packet(
                self.issues,
                outside,
                private_root=private_root,
            )

        real_replace = os.replace
        observed = {}

        def inspect_replace(source, destination):
            source_path = Path(source)
            destination_path = Path(destination)
            observed.setdefault("temporary_modes", []).append(
                stat.S_IMODE(source_path.stat().st_mode)
            )
            observed.setdefault("destination_existed", []).append(
                destination_path.exists()
            )
            observed.setdefault("directory_modes", []).append(
                stat.S_IMODE(destination_path.parent.stat().st_mode)
            )
            real_replace(source, destination)

        with patch.object(ra.os, "replace", side_effect=inspect_replace):
            packet = ra.write_gold_adjudication_packet(
                self.issues,
                "packets",
                private_root=private_root,
            )

        self.assertEqual(observed["temporary_modes"], [0o600, 0o600])
        self.assertEqual(observed["destination_existed"], [False, False])
        self.assertEqual(observed["directory_modes"], [0o700, 0o700])
        self.assertEqual(stat.S_IMODE(private_root.stat().st_mode), 0o700)

        os.chmod(packet["csv_path"], 0o644)
        with self.assertRaisesRegex(ValueError, "must be 0600"):
            ra.load_gold_adjudication(
                packet["csv_path"],
                private_root=private_root,
            )


class GeneratedPacketTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        human = [
            issue(
                "family-a",
                "h1",
                "The identification strategy needs full pretreatment leads and a joint pretrend test.",
                "major_revision_issue",
            )
        ]
        gold_packet = ra.write_gold_adjudication_packet(
            human, self.tmp.name, private_root=self.tmp.name
        )
        gold_rows = deepcopy(gold_packet["rows"])
        gold_rows[0].update(
            {
                "tier_screen": "major",
                "include": "yes",
                "canonical_issue": gold_rows[0]["representative_text"],
                "severity": "major_revision_issue",
                "evidentiary_support": "supported",
            }
        )
        self.gold = ra.validate_gold_rows(
            gold_rows, expected_binding_hash=gold_packet["binding_hash"]
        )

    def test_top_five_packet_proposes_matches_and_requires_all_labels(self):
        generated = [
            {
                "family_id": "family-a",
                "case_id": "v1",
                "id": f"g{index}",
                "rank": index,
                "text": (
                    "The identification strategy needs pretreatment leads and a joint pretrend test."
                    if index == 1
                    else f"Generated concern number {index}."
                ),
                "evidence_ids": [f"E{index}"],
            }
            for index in range(1, 7)
        ]
        packet = ra.write_generated_adjudication_packet(
            generated,
            self.gold,
            self.tmp.name,
            run_binding_context={"commit": "abc", "memory_mode": "none"},
            private_root=self.tmp.name,
        )

        self.assertEqual(packet["row_count"], 5)
        self.assertEqual([row["rank"] for row in packet["rows"]], [1, 2, 3, 4, 5])
        self.assertEqual(
            packet["rows"][0]["proposed_human_cluster_id"],
            self.gold["rows"][0]["cluster_id"],
        )
        self.assertEqual(
            ra.load_generated_adjudication(
                packet["csv_path"],
                expected_binding_hash=packet["binding_hash"],
                private_root=self.tmp.name,
            )["status"],
            "pending_human_adjudication",
        )

        human_cluster = self.gold["rows"][0]["cluster_id"]

        def complete(row):
            row["correctness"] = "correct"
            row["significance"] = "significant"
            row["evidence_sufficiency"] = "sufficient"
            row["duplicate_status"] = "unique"
            row["valid_novelty"] = "no"
            if row["rank"] == "1":
                row["human_match_status"] = "matched"
                row["confirmed_human_cluster_ids"] = json.dumps([human_cluster])
            else:
                row["human_match_status"] = "unmatched"
                row["confirmed_human_cluster_ids"] = "[]"

        rewrite_csv(packet["csv_path"], complete)
        ready = ra.load_generated_adjudication(
            packet["csv_path"],
            expected_binding_hash=packet["binding_hash"],
            expected_gold_binding_hash=self.gold["binding_hash"],
            valid_gold_cluster_ids=[human_cluster],
            private_root=self.tmp.name,
        )
        self.assertEqual(ready["status"], "ready")
        stale = ra.load_generated_adjudication(
            packet["csv_path"],
            expected_binding_hash="different-baseline",
            private_root=self.tmp.name,
        )
        self.assertEqual(stale["status"], "stale")

        context_stale = ra.load_generated_adjudication(
            packet["csv_path"],
            generated_issues=generated,
            expected_gold_binding_hash=self.gold["binding_hash"],
            run_binding_context={"commit": "different", "memory_mode": "none"},
            private_root=self.tmp.name,
        )
        self.assertEqual(context_stale["status"], "stale")

        rewrite_csv(
            packet["csv_path"],
            lambda row: row.update({"generated_text": "tampered generated issue"})
            if row["rank"] == "1"
            else None,
        )
        with self.assertRaisesRegex(ValueError, "immutable fields"):
            ra.load_generated_adjudication(
                packet["csv_path"],
                expected_binding_hash=packet["binding_hash"],
                expected_gold_binding_hash=self.gold["binding_hash"],
                valid_gold_cluster_ids=[human_cluster],
                run_binding_context={"commit": "abc", "memory_mode": "none"},
                private_root=self.tmp.name,
            )

    def test_pipeline_issue_ids_are_scoped_across_cases(self):
        selected = ra.select_generated_top_k(
            [
                {"family_id": "family-a", "case_id": "v1", "id": "P001", "text": "First concern."},
                {"family_id": "family-b", "case_id": "v1", "id": "P001", "text": "Second concern."},
            ]
        )
        self.assertEqual(len({row["generated_issue_id"] for row in selected}), 2)
        self.assertTrue(all(row["generated_issue_id"].startswith("GI_") for row in selected))


class AggregateMetricTests(unittest.TestCase):
    def test_metrics_are_primary_macro_separate_secondary_and_privacy_safe(self):
        def gold(family, cluster, tier, sampled="no"):
            return {
                "family_id": family,
                "cluster_id": cluster,
                "tier_screen": tier,
                "sampled_minor": sampled,
                "include": "yes",
            }

        gold_validation = {
            "status": "ready",
            "rows": [
                gold("secret-family-a", "a-major-1", "major"),
                gold("secret-family-a", "a-major-2", "major"),
                gold("secret-family-a", "a-minor-1", "minor", "yes"),
                gold("secret-family-b", "b-major-1", "major"),
                gold("secret-family-c", "c-major-1", "major"),
            ],
        }

        def generated(
            family,
            issue_id,
            *,
            cluster=None,
            correct="correct",
            significance="significant",
            evidence="sufficient",
            novelty="no",
            duplicate=None,
            rank=1,
        ):
            return {
                "family_id": family,
                "case_id": "case",
                "rank": rank,
                "generated_issue_id": issue_id,
                "correctness": correct,
                "significance": significance,
                "evidence_sufficiency": evidence,
                "human_match_status": "matched" if cluster else "unmatched",
                "confirmed_human_cluster_ids": json.dumps([cluster] if cluster else []),
                "duplicate_status": "duplicate" if duplicate else "unique",
                "duplicate_of_generated_id": duplicate or "",
                "valid_novelty": novelty,
            }

        generated_validation = {
            "status": "ready",
            "rows": [
                generated("secret-family-a", "a-g1", cluster="a-major-1", rank=1),
                generated("secret-family-a", "a-g2", cluster="a-minor-1", significance="minor", rank=2),
                generated("secret-family-a", "a-g3", novelty="yes", rank=3),
                generated(
                    "secret-family-a",
                    "a-g4",
                    correct="incorrect",
                    significance="not_significant",
                    evidence="insufficient",
                    rank=4,
                ),
                generated("secret-family-a", "a-g5", duplicate="a-g3", rank=5),
                generated("secret-family-b", "b-g1", cluster="b-major-1", rank=1),
                generated("secret-family-c", "c-g1", cluster="c-major-1", rank=1),
            ],
        }
        metadata = {
            "secret-family-a": {
                "public_family_id": "F01",
                "benchmark_tier": "primary",
                "is_journal_case": True,
            },
            "secret-family-b": {
                "public_family_id": "F02",
                "benchmark_tier": "primary",
                "is_journal_case": False,
            },
            "secret-family-c": {
                "public_family_id": "F05",
                "benchmark_tier": "secondary",
                "is_journal_case": False,
            },
        }

        result = ra.compute_privacy_safe_metrics(
            gold_validation,
            generated_validation,
            family_metadata=metadata,
            cost_by_family={"secret-family-a": 2.0, "secret-family-b": 3.0, "secret-family-c": 1.0},
        )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["primary_family_macro_major_cluster_recall_at_5"], 0.75)
        self.assertEqual(result["journal_family_macro_major_cluster_recall_at_5"], 0.5)
        self.assertEqual(result["primary_family_macro_sampled_minor_cluster_recall_at_5"], 1.0)
        self.assertEqual(result["primary_supported_significant_precision_at_5"], 0.6667)
        self.assertEqual(result["primary_valid_novelty_yield_at_5"], 0.1667)
        self.assertEqual(result["primary_duplicate_rate_at_5"], 0.1667)
        self.assertEqual(result["total_cost_usd"], 6.0)
        serialized = json.dumps(result)
        self.assertNotIn("secret-family", serialized)
        self.assertNotIn("reviewer", serialized)
        self.assertNotIn("/Users/", serialized)
        self.assertEqual([row["family_id"] for row in result["families"]], ["F01", "F02", "F05"])

    def test_metrics_remain_pending_until_both_packets_are_complete(self):
        result = ra.compute_privacy_safe_metrics(
            {"status": "ready", "rows": []},
            {"status": "pending_human_adjudication", "pending_fields": ["g1: correctness"]},
        )
        self.assertEqual(result["status"], "pending_human_adjudication")
        self.assertEqual(result["generated_pending_count"], 1)

    def test_manual_duplicate_clusters_are_one_recall_target(self):
        gold_validation = {
            "status": "ready",
            "rows": [
                {
                    "family_id": "family-a",
                    "cluster_id": "major-1",
                    "tier_screen": "major",
                    "include": "yes",
                    "sampled_minor": "no",
                    "duplicate_cluster_ids": json.dumps(["major-2"]),
                },
                {
                    "family_id": "family-a",
                    "cluster_id": "major-2",
                    "tier_screen": "major",
                    "include": "yes",
                    "sampled_minor": "no",
                    "duplicate_cluster_ids": "[]",
                },
            ],
        }
        generated_validation = {
            "status": "ready",
            "rows": [
                {
                    "family_id": "family-a",
                    "case_id": "case",
                    "rank": 1,
                    "correctness": "correct",
                    "significance": "significant",
                    "evidence_sufficiency": "sufficient",
                    "human_match_status": "matched",
                    "confirmed_human_cluster_ids": json.dumps(["major-2"]),
                    "duplicate_status": "unique",
                    "valid_novelty": "no",
                }
            ],
        }
        result = ra.compute_privacy_safe_metrics(gold_validation, generated_validation)
        self.assertEqual(result["families"][0]["major_cluster_count"], 1)
        self.assertEqual(result["families"][0]["major_cluster_recall_at_5"], 1.0)


if __name__ == "__main__":
    unittest.main()
