"""Offline adjudication artifacts and metrics for review-corpus evaluation.

This module deliberately has no dependency on OpenAI or the feedback pipeline.
It turns normalized human-review issues into hash-bound adjudication packets,
validates completed packets, prepares an analogous packet for generated issues,
and computes privacy-safe aggregate metrics only after both gates are complete.

The packets are private working artifacts: they contain issue text and source
locators.  Writers therefore create their directory with mode ``0700`` and
files with mode ``0600``.  The aggregate metrics contain neither text nor
source/reviewer provenance and are suitable for a portable evaluation report.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import tempfile
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Iterator, List, Mapping, Sequence, Tuple


PACKET_VERSION = "feedback-llm-adjudication-v1"
DEFAULT_MINOR_SAMPLE_SIZE = 5
DEFAULT_SAMPLE_SEED = 20260802
DEFAULT_PRIVATE_ROOT = Path.home() / ".feedback_llm"

MAJOR_DECISION_TIERS = {
    "major",
    "major_revision",
    "major_revision_issue",
    "potential_rejection",
    "potential_rejection_reason",
    "rejection",
}
TIER_SCREEN_VALUES = {"major", "minor", "exclude"}
INCLUDE_VALUES = {"yes", "no"}
SEVERITY_VALUES = {
    "potential_rejection_reason",
    "major_revision_issue",
    "minor_revision_issue",
    "nice_to_have",
}
SUPPORT_VALUES = {"supported", "partially_supported", "unsupported", "unclear"}
CORRECTNESS_VALUES = {"correct", "incorrect", "unclear"}
SIGNIFICANCE_VALUES = {"significant", "minor", "not_significant", "unclear"}
EVIDENCE_VALUES = {"sufficient", "partial", "insufficient", "unclear"}
MATCH_STATUS_VALUES = {"matched", "unmatched", "unclear"}
DUPLICATE_STATUS_VALUES = {"unique", "duplicate", "unclear"}
YES_NO_VALUES = {"yes", "no"}

_STOPWORDS = {
    "about", "after", "again", "also", "among", "because", "been", "before",
    "being", "between", "both", "could", "does", "from", "have", "into",
    "more", "most", "other", "paper", "should", "some", "such", "than", "that",
    "their", "there", "these", "they", "this", "through", "under", "very",
    "what", "when", "where", "which", "while", "with", "would", "your",
}

GOLD_COLUMNS = [
    "packet_version",
    "binding_hash",
    "family_id",
    "case_ids",
    "cluster_id",
    "provisional_tier",
    "sampled_minor",
    "full_adjudication_required",
    "issue_count",
    "representative_issue_id",
    "representative_text",
    "member_issue_ids",
    "source_ids",
    "reviewer_ids",
    "source_locators",
    "tier_screen",
    "include",
    "canonical_issue",
    "severity",
    "evidentiary_support",
    "duplicate_cluster_ids",
    "exclusion_reason",
    "adjudicator_notes",
]

GOLD_EDITABLE_COLUMNS = (
    "tier_screen",
    "include",
    "canonical_issue",
    "severity",
    "evidentiary_support",
    "duplicate_cluster_ids",
    "exclusion_reason",
    "adjudicator_notes",
)
GOLD_IMMUTABLE_COLUMNS = tuple(
    column for column in GOLD_COLUMNS if column not in GOLD_EDITABLE_COLUMNS
)

GENERATED_COLUMNS = [
    "packet_version",
    "binding_hash",
    "gold_binding_hash",
    "family_id",
    "case_id",
    "rank",
    "generated_issue_id",
    "pipeline_issue_id",
    "generated_text",
    "evidence_ids",
    "proposed_human_cluster_id",
    "proposed_match_score",
    "proposed_shared_terms",
    "correctness",
    "significance",
    "evidence_sufficiency",
    "human_match_status",
    "confirmed_human_cluster_ids",
    "duplicate_status",
    "duplicate_of_generated_id",
    "valid_novelty",
    "adjudicator_notes",
]

GENERATED_EDITABLE_COLUMNS = (
    "correctness",
    "significance",
    "evidence_sufficiency",
    "human_match_status",
    "confirmed_human_cluster_ids",
    "duplicate_status",
    "duplicate_of_generated_id",
    "valid_novelty",
    "adjudicator_notes",
)
GENERATED_IMMUTABLE_COLUMNS = tuple(
    column for column in GENERATED_COLUMNS if column not in GENERATED_EDITABLE_COLUMNS
)

PARTIAL_GOLD_BINDING_VERSION = "feedback-llm-partial-gold-v1"


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_hash(value: Any) -> str:
    """Return a stable SHA-256 digest for JSON-serializable evaluation state."""
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def _family_id(issue: Mapping[str, Any]) -> str:
    value = _clean(issue.get("family_id") or issue.get("manuscript_family_id"))
    if not value:
        raise ValueError("Every normalized issue must have family_id or manuscript_family_id.")
    return value


def _case_id(issue: Mapping[str, Any]) -> str:
    return _clean(issue.get("case_id") or issue.get("manuscript_version_id") or "case")


def _issue_text(issue: Mapping[str, Any]) -> str:
    return _clean(issue.get("issue_text") or issue.get("text"))


def _issue_id(issue: Mapping[str, Any]) -> str:
    supplied = _clean(issue.get("atomic_issue_id") or issue.get("issue_id") or issue.get("id"))
    if supplied:
        return supplied
    payload = {
        "family_id": _family_id(issue),
        "case_id": _case_id(issue),
        "source_id": _clean(issue.get("source_id") or issue.get("review_file")),
        "reviewer_id": _clean(issue.get("reviewer_id")),
        "locator": issue.get("source_locator") or issue.get("locator") or "",
        "text": _issue_text(issue),
    }
    return "HI_" + stable_hash(payload)[:16]


def _tier_priority(value: Any) -> int:
    tier = _clean(value).lower()
    if tier in {"potential_rejection_reason", "potential_rejection", "rejection"}:
        return 4
    if tier in {"major_revision_issue", "major_revision", "major"}:
        return 3
    if tier in {"minor_revision_issue", "minor_revision", "minor"}:
        return 2
    if tier in {"nice_to_have", "nice to have"}:
        return 1
    return 0


def _provisional_tier(issues: Sequence[Mapping[str, Any]]) -> str:
    return "major" if any(_clean(item.get("decision_tier")).lower() in MAJOR_DECISION_TIERS for item in issues) else "minor"


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in _STOPWORDS
    }


def _text_similarity(left: str, right: str) -> Tuple[float, List[str]]:
    left_tokens = _tokens(left)
    right_tokens = _tokens(right)
    if not left_tokens or not right_tokens:
        return 0.0, []
    shared = sorted(left_tokens & right_tokens)
    jaccard = len(shared) / len(left_tokens | right_tokens)
    containment = len(shared) / min(len(left_tokens), len(right_tokens))
    return max(jaccard, containment * 0.85), shared


def _binding_issue(issue: Mapping[str, Any]) -> Dict[str, Any]:
    """Select fields whose changes invalidate human adjudication."""
    return {
        "family_id": _family_id(issue),
        "case_id": _case_id(issue),
        "issue_id": _issue_id(issue),
        "issue_text": _issue_text(issue),
        "decision_tier": _clean(issue.get("decision_tier")),
        "issue_type": _clean(issue.get("issue_type")),
        "source_id": _clean(issue.get("source_id") or issue.get("review_file")),
        "source_hash": _clean(issue.get("source_hash")),
        "source_locator": issue.get("source_locator") or issue.get("locator") or "",
        "extraction_rule": issue.get("extraction_rule") or issue.get("extraction_rule_hash") or "",
    }


def _should_merge(left: Mapping[str, Any], right: Mapping[str, Any]) -> bool:
    left_group = _clean(left.get("duplicate_group") or left.get("cluster_key"))
    right_group = _clean(right.get("duplicate_group") or right.get("cluster_key"))
    if left_group and left_group == right_group:
        return True
    left_text = _issue_text(left).lower()
    right_text = _issue_text(right).lower()
    if left_text and left_text == right_text:
        return True
    score, shared = _text_similarity(left_text, right_text)
    type_compatible = (
        not _clean(left.get("issue_type"))
        or not _clean(right.get("issue_type"))
        or _clean(left.get("issue_type")) == _clean(right.get("issue_type"))
    )
    return type_compatible and len(shared) >= 4 and score >= 0.64


def _default_family_clusters(issues: Sequence[Mapping[str, Any]]) -> List[List[Mapping[str, Any]]]:
    """Conservatively cluster duplicates/strong lexical overlaps in one family."""
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    ungrouped: List[Mapping[str, Any]] = []
    for issue in issues:
        declared = _clean(issue.get("cluster_id"))
        if declared:
            grouped["declared:" + declared].append(issue)
        else:
            ungrouped.append(issue)

    clusters = [grouped[key] for key in sorted(grouped)]
    ordered = sorted(
        ungrouped,
        key=lambda item: (
            -_tier_priority(item.get("decision_tier")),
            _clean(item.get("issue_type")),
            _issue_id(item),
        ),
    )
    for issue in ordered:
        destination = None
        for cluster in clusters:
            if any(_should_merge(issue, member) for member in cluster):
                destination = cluster
                break
        if destination is None:
            clusters.append([issue])
        else:
            destination.append(issue)
    return clusters


def _external_family_clusters(
    issues: Sequence[Mapping[str, Any]],
    clusterer: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]],
) -> List[List[Mapping[str, Any]]]:
    """Adapt an existing clusterer while replacing order-based cluster IDs."""
    issue_map = {_issue_id(issue): issue for issue in issues}
    adapted = []
    for issue in sorted(issues, key=_issue_id):
        item = dict(issue)
        item["atomic_issue_id"] = _issue_id(issue)
        adapted.append(item)
    external = clusterer(adapted)
    clusters: List[List[Mapping[str, Any]]] = []
    assigned: set[str] = set()
    for cluster in external:
        ids = [str(value) for value in cluster.get("issue_ids", [])]
        members = [issue_map[issue_id] for issue_id in ids if issue_id in issue_map]
        if not members and cluster.get("members"):
            members = [issue_map[_issue_id(item)] for item in cluster["members"] if _issue_id(item) in issue_map]
        if members:
            clusters.append(members)
            assigned.update(_issue_id(item) for item in members)
    clusters.extend([[issue] for issue in issues if _issue_id(issue) not in assigned])
    return clusters


def cluster_normalized_issues(
    issues: Sequence[Mapping[str, Any]],
    clusterer: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]] | None = None,
) -> List[Dict[str, Any]]:
    """Create deterministic, family-scoped clusters from normalized issues.

    Passing ``feedback_pipeline.cluster_human_review_issues`` as ``clusterer``
    reuses the pipeline's semantic matcher.  Regardless of the supplied
    clusterer's numbering, this adapter assigns stable content-derived IDs and
    never permits a cluster to cross a manuscript-family boundary.
    """
    families: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    seen_ids: set[Tuple[str, str]] = set()
    for issue in issues:
        family = _family_id(issue)
        issue_id = _issue_id(issue)
        key = (family, issue_id)
        if key in seen_ids:
            raise ValueError(f"Duplicate normalized issue ID in family {family}: {issue_id}")
        seen_ids.add(key)
        families[family].append(issue)

    result: List[Dict[str, Any]] = []
    for family in sorted(families):
        family_issues = families[family]
        raw_clusters = (
            _external_family_clusters(family_issues, clusterer)
            if clusterer is not None
            else _default_family_clusters(family_issues)
        )
        for members in raw_clusters:
            ordered_members = sorted(members, key=_issue_id)
            member_ids = [_issue_id(item) for item in ordered_members]
            representative = max(
                ordered_members,
                key=lambda item: (
                    _tier_priority(item.get("decision_tier")),
                    len(_issue_text(item)),
                    _issue_id(item),
                ),
            )
            cluster_id = "HC_" + stable_hash({"family_id": family, "issue_ids": member_ids})[:14]
            result.append(
                {
                    "cluster_id": cluster_id,
                    "family_id": family,
                    "case_ids": sorted({_case_id(item) for item in ordered_members}),
                    "provisional_tier": _provisional_tier(ordered_members),
                    "representative_issue_id": _issue_id(representative),
                    "representative_text": _issue_text(representative),
                    "issue_count": len(ordered_members),
                    "issue_ids": member_ids,
                    "source_ids": sorted(
                        {
                            _clean(item.get("source_id") or item.get("review_file"))
                            for item in ordered_members
                            if _clean(item.get("source_id") or item.get("review_file"))
                        }
                    ),
                    "reviewer_ids": sorted(
                        {
                            _clean(item.get("reviewer_id"))
                            for item in ordered_members
                            if _clean(item.get("reviewer_id"))
                        }
                    ),
                    "source_locators": [
                        item.get("source_locator") or item.get("locator")
                        for item in ordered_members
                        if item.get("source_locator") or item.get("locator")
                    ],
                    "member_bindings": [_binding_issue(item) for item in ordered_members],
                }
            )
    return sorted(result, key=lambda item: (item["family_id"], item["cluster_id"]))


def select_full_adjudication_clusters(
    clusters: Sequence[Mapping[str, Any]],
    minor_sample_size: int = DEFAULT_MINOR_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> set[str]:
    """Select all provisional majors and a stable hash-ranked minor sample."""
    if minor_sample_size < 0:
        raise ValueError("minor_sample_size must be non-negative")
    selected = {
        str(cluster["cluster_id"])
        for cluster in clusters
        if _clean(cluster.get("provisional_tier")).lower() == "major"
    }
    by_family: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for cluster in clusters:
        if _clean(cluster.get("provisional_tier")).lower() != "major":
            by_family[_clean(cluster.get("family_id"))].append(cluster)
    for family, candidates in by_family.items():
        ranked = sorted(
            candidates,
            key=lambda item: stable_hash(
                {"seed": seed, "family_id": family, "cluster_id": item["cluster_id"]}
            ),
        )
        selected.update(str(item["cluster_id"]) for item in ranked[:minor_sample_size])
    return selected


def gold_binding_hash(
    clusters: Sequence[Mapping[str, Any]],
    binding_context: Mapping[str, Any] | None = None,
    minor_sample_size: int = DEFAULT_MINOR_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
) -> str:
    payload_clusters = []
    for cluster in sorted(clusters, key=lambda item: (item["family_id"], item["cluster_id"])):
        payload_clusters.append(
            {
                "cluster_id": cluster["cluster_id"],
                "family_id": cluster["family_id"],
                "case_ids": cluster.get("case_ids", []),
                "provisional_tier": cluster.get("provisional_tier"),
                "issue_ids": cluster.get("issue_ids", []),
                "member_bindings": cluster.get("member_bindings", []),
            }
        )
    return stable_hash(
        {
            "packet_version": PACKET_VERSION,
            "kind": "gold",
            "seed": seed,
            "minor_sample_size": minor_sample_size,
            "clusters": payload_clusters,
            "binding_context": dict(binding_context or {}),
        }
    )


def _json_cell(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _parse_json_cell(value: Any, default: Any) -> Any:
    text = str(value or "").strip()
    if not text:
        return default
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return default


def _parse_json_list_cell(value: Any) -> tuple[List[Any], bool]:
    """Return a JSON list and whether the original cell was valid JSON-list data."""
    text = str(value or "").strip()
    if not text:
        return [], True
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return [], False
    return (parsed, True) if isinstance(parsed, list) else ([], False)


def _private_root_path(private_root: str | Path | None = None) -> Path:
    return Path(private_root or DEFAULT_PRIVATE_ROOT).expanduser().resolve()


def _private_path(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
    label: str,
    allow_root: bool = False,
) -> tuple[Path, Path]:
    root = _private_root_path(private_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if (candidate == root and not allow_root) or (
        candidate != root and root not in candidate.parents
    ):
        raise ValueError(f"{label} must stay below {root}")
    return candidate, root


def _ensure_private_directory(path: Path, root: Path) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"private output must stay below {root}")
    if root.exists():
        if not root.is_dir():
            raise ValueError(f"private root is not a directory: {root}")
        os.chmod(root, 0o700)
    else:
        root.mkdir(parents=True, mode=0o700)
        os.chmod(root, 0o700)
    current = root
    for part in path.relative_to(root).parts:
        current = current / part
        if current.exists():
            if not current.is_dir():
                raise ValueError(f"private output parent is not a directory: {current}")
            os.chmod(current, 0o700)
        else:
            os.mkdir(current, mode=0o700)
        os.chmod(current, 0o700)


def _prepare_private_directory(
    output_dir: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Path:
    path, root = _private_path(
        output_dir,
        private_root=private_root,
        label="adjudication output",
        allow_root=True,
    )
    _ensure_private_directory(path, root)
    return path


def _assert_private_input(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Path:
    candidate, root = _private_path(
        path,
        private_root=private_root,
        label="adjudication input",
    )
    file_mode = stat.S_IMODE(candidate.stat().st_mode)
    if file_mode != 0o600:
        raise ValueError(
            f"adjudication input permissions must be 0600, got {oct(file_mode)}"
        )
    current = candidate.parent
    while True:
        mode = stat.S_IMODE(current.stat().st_mode)
        if mode != 0o700:
            raise ValueError(
                f"private directory permissions must be 0700 for {current}, got {oct(mode)}"
            )
        if current == root:
            break
        current = current.parent
    return candidate


def _atomic_private_write(
    path: Path,
    writer: Callable[[Any], None],
    *,
    newline: str | None = None,
) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(
            descriptor,
            "w",
            encoding="utf-8",
            newline=newline,
        ) as handle:
            writer(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass
        raise


def _write_private_text(path: Path, text: str) -> None:
    _atomic_private_write(path, lambda handle: handle.write(text))


def _write_csv(path: Path, columns: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    def write(handle: Any) -> None:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    _atomic_private_write(path, write, newline="")


def _markdown_text(text: Any) -> str:
    return _clean(text).replace("|", "\\|")


def write_gold_adjudication_packet(
    issues: Sequence[Mapping[str, Any]],
    output_dir: str | Path,
    *,
    binding_context: Mapping[str, Any] | None = None,
    clusterer: Callable[[List[Dict[str, Any]]], List[Dict[str, Any]]] | None = None,
    minor_sample_size: int = DEFAULT_MINOR_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
    prefix: str = "gold_adjudication",
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Write CSV and Markdown packets for human-cluster adjudication."""
    clusters = cluster_normalized_issues(issues, clusterer=clusterer)
    if not clusters:
        raise ValueError("Cannot prepare a gold adjudication packet with no issues.")
    selected = select_full_adjudication_clusters(clusters, minor_sample_size, seed)
    binding = gold_binding_hash(clusters, binding_context, minor_sample_size, seed)
    rows: List[Dict[str, Any]] = []
    for cluster in clusters:
        is_minor_sample = (
            cluster["provisional_tier"] == "minor" and cluster["cluster_id"] in selected
        )
        rows.append(
            {
                "packet_version": PACKET_VERSION,
                "binding_hash": binding,
                "family_id": cluster["family_id"],
                "case_ids": _json_cell(cluster.get("case_ids", [])),
                "cluster_id": cluster["cluster_id"],
                "provisional_tier": cluster["provisional_tier"],
                "sampled_minor": "yes" if is_minor_sample else "no",
                "full_adjudication_required": "yes" if cluster["cluster_id"] in selected else "no",
                "issue_count": cluster["issue_count"],
                "representative_issue_id": cluster["representative_issue_id"],
                "representative_text": cluster["representative_text"],
                "member_issue_ids": _json_cell(cluster["issue_ids"]),
                "source_ids": _json_cell(cluster.get("source_ids", [])),
                "reviewer_ids": _json_cell(cluster.get("reviewer_ids", [])),
                "source_locators": _json_cell(cluster.get("source_locators", [])),
                "tier_screen": "",
                "include": "",
                "canonical_issue": "",
                "severity": "",
                "evidentiary_support": "",
                "duplicate_cluster_ids": "[]",
                "exclusion_reason": "",
                "adjudicator_notes": "",
            }
        )

    output = _prepare_private_directory(output_dir, private_root=private_root)
    csv_path = output / f"{prefix}.csv"
    markdown_path = output / f"{prefix}.md"
    _write_csv(csv_path, GOLD_COLUMNS, rows)

    lines = [
        "# Gold feedback adjudication",
        "",
        f"- Packet version: `{PACKET_VERSION}`",
        f"- Binding hash: `{binding}`",
        f"- Cluster count: {len(rows)}",
        f"- Full adjudication rows: {len(selected)}",
        f"- Minor sample rule: up to {minor_sample_size} per family, seed {seed}",
        "",
        "Every cluster requires `tier_screen` (`major`, `minor`, or `exclude`). "
        "Rows marked FULL also require the substantive adjudication fields. "
        "Edit the CSV; this Markdown file is a reading packet.",
    ]
    current_family = None
    for row in rows:
        if row["family_id"] != current_family:
            current_family = row["family_id"]
            lines.extend(["", f"## Family `{current_family}`", ""])
        mode = "FULL" if row["full_adjudication_required"] == "yes" else "TIER ONLY"
        lines.extend(
            [
                f"### [{mode}] `{row['cluster_id']}` ({row['provisional_tier']})",
                "",
                _markdown_text(row["representative_text"]),
                "",
                f"Members: {row['member_issue_ids']}",
                f"Sources: {row['source_ids']}",
                f"Reviewers: {row['reviewer_ids']}",
                f"Locators: {row['source_locators']}",
            ]
        )
    _write_private_text(markdown_path, "\n".join(lines) + "\n")
    return {
        "status": "pending_human_adjudication",
        "binding_hash": binding,
        "cluster_count": len(rows),
        "full_adjudication_count": len(selected),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "clusters": clusters,
        "rows": rows,
    }


def _read_csv(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> List[Dict[str, str]]:
    private_path = _assert_private_input(path, private_root=private_root)
    with private_path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def gold_row_requirements(
    row: Mapping[str, Any],
    *,
    family_by_cluster: Mapping[str, str] | None = None,
) -> Dict[str, Any]:
    """Return the conditional requirements and completion state for one gold row."""
    cluster_id = _clean(row.get("cluster_id")) or "<missing>"
    family_id = _clean(row.get("family_id"))
    known_families = dict(family_by_cluster or {cluster_id: family_id})
    pending: List[str] = []
    errors: List[str] = []

    duplicate_ids, duplicates_valid = _parse_json_list_cell(
        row.get("duplicate_cluster_ids")
    )
    if not duplicates_valid:
        errors.append("duplicate_cluster_ids must be a JSON list")
    else:
        for duplicate_id in map(str, duplicate_ids):
            if duplicate_id == cluster_id:
                errors.append("cannot list itself as a duplicate")
            elif duplicate_id not in known_families:
                errors.append(f"duplicate cluster does not exist: {duplicate_id}")
            elif known_families[duplicate_id] != family_id:
                errors.append("duplicate cluster must be in the same family")

    tier = _clean(row.get("tier_screen")).lower()
    tier_screened = tier in TIER_SCREEN_VALUES
    full_required = (
        _clean(row.get("full_adjudication_required")).lower() == "yes"
        or tier == "major"
    )
    if not tier_screened:
        pending.append("tier_screen")
    else:
        include = _clean(row.get("include")).lower()
        if tier == "exclude":
            if include != "no":
                pending.append("include=no")
            if not _clean(row.get("exclusion_reason")):
                pending.append("exclusion_reason")
        elif full_required:
            if include not in INCLUDE_VALUES:
                pending.append("include")
            elif include == "no":
                if not _clean(row.get("exclusion_reason")):
                    pending.append("exclusion_reason")
            else:
                if not _clean(row.get("canonical_issue")):
                    pending.append("canonical_issue")
                severity = _clean(row.get("severity"))
                if severity not in SEVERITY_VALUES:
                    pending.append("severity")
                elif tier == "major" and severity not in {
                    "potential_rejection_reason",
                    "major_revision_issue",
                }:
                    errors.append("major tier requires major severity")
                elif tier == "minor" and severity not in {
                    "minor_revision_issue",
                    "nice_to_have",
                }:
                    errors.append("minor tier requires minor severity")
                if _clean(row.get("evidentiary_support")) not in SUPPORT_VALUES:
                    pending.append("evidentiary_support")

    return {
        "cluster_id": cluster_id,
        "tier": tier,
        "tier_screened": tier_screened,
        "full_required": full_required,
        "pending_fields": pending,
        "errors": errors,
        "complete": not pending and not errors,
    }


def gold_adjudication_progress(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Summarize tier-screen and full-label progress for the adjudication UI."""
    family_by_cluster = {
        _clean(row.get("cluster_id")): _clean(row.get("family_id")) for row in rows
    }
    states = [
        gold_row_requirements(row, family_by_cluster=family_by_cluster) for row in rows
    ]
    families: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "tier_done": 0, "full_total": 0, "full_done": 0, "remaining": 0}
    )
    for row, state in zip(rows, states):
        family = _clean(row.get("family_id"))
        family_progress = families[family]
        family_progress["total"] += 1
        family_progress["tier_done"] += int(state["tier_screened"])
        family_progress["full_total"] += int(state["full_required"])
        family_progress["full_done"] += int(
            state["full_required"] and state["complete"]
        )
        family_progress["remaining"] += int(not state["complete"])
    return {
        "total": len(rows),
        "tier_done": sum(int(state["tier_screened"]) for state in states),
        "full_total": sum(int(state["full_required"]) for state in states),
        "full_done": sum(
            int(state["full_required"] and state["complete"]) for state in states
        ),
        "complete": sum(int(state["complete"]) for state in states),
        "remaining": sum(int(not state["complete"]) for state in states),
        "families": {family: dict(values) for family, values in sorted(families.items())},
    }


def eligible_duplicate_clusters(
    rows: Sequence[Mapping[str, Any]],
    cluster_id: str,
) -> List[Dict[str, Any]]:
    """Return same-family duplicate candidates, excluding the current cluster."""
    current = next(
        (row for row in rows if _clean(row.get("cluster_id")) == cluster_id),
        None,
    )
    if current is None:
        raise ValueError(f"Unknown cluster_id: {cluster_id}")
    family_id = _clean(current.get("family_id"))
    return sorted(
        [
            dict(row)
            for row in rows
            if _clean(row.get("family_id")) == family_id
            and _clean(row.get("cluster_id")) != cluster_id
        ],
        key=lambda row: _clean(row.get("cluster_id")),
    )


def validate_gold_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_binding_hash: str | None = None,
    expected_cluster_ids: Iterable[str] | None = None,
) -> Dict[str, Any]:
    """Validate gold rows and report incomplete, invalid, or stale state."""
    errors: List[str] = []
    pending: List[str] = []
    bindings = {_clean(row.get("binding_hash")) for row in rows if _clean(row.get("binding_hash"))}
    binding = next(iter(bindings)) if len(bindings) == 1 else ""
    stale = False
    if not rows:
        errors.append("packet has no cluster rows")
    if any(_clean(row.get("packet_version")) != PACKET_VERSION for row in rows):
        errors.append(f"packet_version must be {PACKET_VERSION}")
    if len(bindings) != 1:
        errors.append("packet must contain exactly one nonempty binding_hash")
    if expected_binding_hash and binding != expected_binding_hash:
        stale = True
        errors.append("binding hash does not match current corpus/extraction state")

    ids = [_clean(row.get("cluster_id")) for row in rows]
    if any(not cluster_id for cluster_id in ids):
        errors.append("every row must have cluster_id")
    if len(ids) != len(set(ids)):
        errors.append("cluster_id values must be unique")
    if expected_cluster_ids is not None and set(ids) != set(expected_cluster_ids):
        stale = True
        errors.append("packet cluster set does not match current corpus clusters")

    family_by_cluster = {
        _clean(row.get("cluster_id")): _clean(row.get("family_id")) for row in rows
    }

    for row in rows:
        state = gold_row_requirements(row, family_by_cluster=family_by_cluster)
        cluster_id = state["cluster_id"]
        pending.extend(f"{cluster_id}: {field}" for field in state["pending_fields"])
        errors.extend(f"{cluster_id}: {error}" for error in state["errors"])

    if stale:
        status = "stale"
    elif errors:
        status = "invalid"
    elif pending:
        status = "pending_human_adjudication"
    else:
        status = "ready"
    return {
        "status": status,
        "binding_hash": binding,
        "rows": [dict(row) for row in rows],
        "errors": errors,
        "pending_fields": pending,
        "cluster_count": len(rows),
        "completed": status == "ready",
    }


def load_gold_adjudication(
    csv_path: str | Path,
    *,
    expected_binding_hash: str | None = None,
    clusters: Sequence[Mapping[str, Any]] | None = None,
    binding_context: Mapping[str, Any] | None = None,
    minor_sample_size: int = DEFAULT_MINOR_SAMPLE_SIZE,
    seed: int = DEFAULT_SAMPLE_SEED,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a gold packet and detect changes to bound source/extraction state."""
    expected_ids = None
    if clusters is not None:
        expected_ids = [str(item["cluster_id"]) for item in clusters]
        computed = gold_binding_hash(clusters, binding_context, minor_sample_size, seed)
        if expected_binding_hash and expected_binding_hash != computed:
            raise ValueError("Provided expected_binding_hash disagrees with current clusters/context.")
        expected_binding_hash = computed
    rows = _read_csv(csv_path, private_root=private_root)
    result = validate_gold_rows(
        rows,
        expected_binding_hash=expected_binding_hash,
        expected_cluster_ids=expected_ids,
    )
    if clusters is not None:
        selected = select_full_adjudication_clusters(clusters, minor_sample_size, seed)
        expected = {}
        for cluster in clusters:
            cluster_id = str(cluster["cluster_id"])
            expected[cluster_id] = {
                "family_id": _clean(cluster.get("family_id")),
                "case_ids": list(cluster.get("case_ids", [])),
                "provisional_tier": _clean(cluster.get("provisional_tier")),
                "sampled_minor": (
                    "yes"
                    if cluster.get("provisional_tier") == "minor" and cluster_id in selected
                    else "no"
                ),
                "full_adjudication_required": "yes" if cluster_id in selected else "no",
                "issue_count": str(cluster.get("issue_count", "")),
                "representative_issue_id": _clean(cluster.get("representative_issue_id")),
                "representative_text": _clean(cluster.get("representative_text")),
                "member_issue_ids": list(cluster.get("issue_ids", [])),
                "source_ids": list(cluster.get("source_ids", [])),
                "reviewer_ids": list(cluster.get("reviewer_ids", [])),
                "source_locators": list(cluster.get("source_locators", [])),
            }
        immutable_errors = []
        for row in rows:
            cluster_id = _clean(row.get("cluster_id"))
            target = expected.get(cluster_id)
            if target is None:
                continue
            for field in (
                "family_id",
                "provisional_tier",
                "sampled_minor",
                "full_adjudication_required",
                "issue_count",
                "representative_issue_id",
                "representative_text",
            ):
                if _clean(row.get(field)) != target[field]:
                    immutable_errors.append(f"{cluster_id}: immutable field changed: {field}")
            for field in (
                "case_ids",
                "member_issue_ids",
                "source_ids",
                "reviewer_ids",
                "source_locators",
            ):
                if _parse_json_cell(row.get(field), None) != target[field]:
                    immutable_errors.append(f"{cluster_id}: immutable field changed: {field}")
        if immutable_errors:
            result["errors"].extend(immutable_errors)
            result["status"] = "stale"
            result["completed"] = False
    result["csv_path"] = str(Path(csv_path).expanduser())
    return result


def project_partial_gold_adjudication(
    gold_validation: Mapping[str, Any],
) -> Dict[str, Any]:
    """Freeze the currently completed gold rows for an exploratory pilot.

    This projection never treats an incomplete tier screen as negative evidence.
    It is allowed only when every row selected for full adjudication, including
    rows promoted to ``major`` during screening, is complete.  The derived
    binding covers both the source-bound packet and every scoring-relevant label
    in the selected rows, so later labeling changes invalidate generated work.
    """
    source_status = _clean(gold_validation.get("status"))
    if source_status in {"stale", "invalid"}:
        raise ValueError(
            "Partial-gold evaluation requires a current, valid adjudication "
            f"packet (status={source_status})."
        )
    if source_status not in {"ready", "pending_human_adjudication"}:
        raise ValueError(
            "Partial-gold evaluation requires a loaded adjudication packet "
            f"(status={source_status or 'missing'})."
        )

    rows = [dict(row) for row in gold_validation.get("rows", [])]
    if not rows:
        raise ValueError("Partial-gold evaluation requires at least one gold row.")
    family_by_cluster = {
        _clean(row.get("cluster_id")): _clean(row.get("family_id")) for row in rows
    }
    states = [
        gold_row_requirements(row, family_by_cluster=family_by_cluster) for row in rows
    ]
    invalid_rows = [
        state["cluster_id"] for state in states if state.get("errors")
    ]
    if invalid_rows:
        raise ValueError(
            "Partial-gold evaluation cannot use invalid rows: "
            + ", ".join(sorted(invalid_rows))
        )
    incomplete_full = [
        state["cluster_id"]
        for state in states
        if state["full_required"] and not state["complete"]
    ]
    if incomplete_full:
        raise ValueError(
            "Partial-gold evaluation requires every full-adjudication row and "
            "promoted major to be complete "
            f"({len(incomplete_full)} incomplete)."
        )

    completed_rows = sorted(
        [row for row, state in zip(rows, states) if state["complete"]],
        key=lambda row: (_clean(row.get("family_id")), _clean(row.get("cluster_id"))),
    )
    if not completed_rows:
        raise ValueError("Partial-gold evaluation has no completed rows to score.")
    scoring_rows = [
        row
        for row in completed_rows
        if (
            _clean(row.get("full_adjudication_required")).lower() == "yes"
            or _clean(row.get("tier_screen")).lower() == "major"
        )
        and _clean(row.get("include")).lower() == "yes"
    ]
    if not scoring_rows:
        raise ValueError(
            "Partial-gold evaluation has no fully adjudicated included rows to score."
        )
    scoring_ids = {_clean(row.get("cluster_id")) for row in scoring_rows}
    for row in scoring_rows:
        duplicate_ids = _parse_json_cell(row.get("duplicate_cluster_ids"), [])
        unavailable = sorted(set(map(str, duplicate_ids or [])) - scoring_ids)
        if unavailable:
            raise ValueError(
                "Partial-gold duplicate links must point to scoring rows "
                f"({_clean(row.get('cluster_id'))}: {', '.join(unavailable)})."
            )

    def bound_row(row: Mapping[str, Any]) -> Dict[str, Any]:
        duplicates = _parse_json_cell(row.get("duplicate_cluster_ids"), [])
        return {
            "family_id": _clean(row.get("family_id")),
            "cluster_id": _clean(row.get("cluster_id")),
            "tier_screen": _clean(row.get("tier_screen")).lower(),
            "include": _clean(row.get("include")).lower(),
            "canonical_issue": _clean(row.get("canonical_issue")),
            "severity": _clean(row.get("severity")).lower(),
            "evidentiary_support": _clean(row.get("evidentiary_support")).lower(),
            "duplicate_cluster_ids": sorted(map(str, duplicates or [])),
            "exclusion_reason": _clean(row.get("exclusion_reason")),
        }

    source_binding = _clean(gold_validation.get("binding_hash"))
    if not source_binding:
        raise ValueError("Partial-gold evaluation requires a nonempty source binding hash.")
    binding = stable_hash(
        {
            "projection_version": PARTIAL_GOLD_BINDING_VERSION,
            "source_binding_hash": source_binding,
            "completed_rows": [bound_row(row) for row in completed_rows],
        }
    )
    progress = gold_adjudication_progress(rows)
    coverage = {
        "total_cluster_count": progress["total"],
        "adjudicated_cluster_count": len(completed_rows),
        "unadjudicated_cluster_count": progress["total"] - len(completed_rows),
        "adjudicated_cluster_rate": round(len(completed_rows) / progress["total"], 4),
        "tier_screened_count": progress["tier_done"],
        "full_required_count": progress["full_total"],
        "full_required_completed_count": progress["full_done"],
        "eligible_scoring_cluster_count": len(scoring_rows),
        "excluded_cluster_count": sum(
            _clean(row.get("include")).lower() == "no" for row in completed_rows
        ),
        "tier_only_non_scoring_cluster_count": sum(
            _clean(row.get("include")).lower() not in INCLUDE_VALUES
            for row in completed_rows
        ),
    }
    return {
        "status": "ready",
        "completed": True,
        "gold_mode": "partial",
        "evaluation_scope": "partial_gold_pilot",
        "denominator_scope": "adjudicated_clusters_only",
        "exhaustive_major_recall": False,
        "binding_hash": binding,
        "source_binding_hash": source_binding,
        "rows": scoring_rows,
        "cluster_count": len(scoring_rows),
        "coverage": coverage,
        "errors": [],
        "pending_fields": [],
    }


def _file_revision(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def _gold_editor_lock(path: Path) -> Iterator[None]:
    """Serialize editor saves across local browser sessions on POSIX systems."""
    try:
        import fcntl
    except ImportError as exc:  # pragma: no cover - the launcher targets macOS/POSIX
        raise RuntimeError(
            "Guarded adjudication saves require a POSIX file-lock implementation."
        ) from exc
    lock_path = path.with_name(f".{path.name}.lock")
    descriptor = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        os.fchmod(descriptor, 0o600)
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def load_gold_editor_state(
    csv_path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a private gold packet with a revision token for guarded UI edits."""
    private_path = _assert_private_input(csv_path, private_root=private_root)
    for _ in range(3):
        before = _file_revision(private_path)
        rows = _read_csv(private_path, private_root=private_root)
        after = _file_revision(private_path)
        if before == after:
            break
    else:
        raise RuntimeError("The adjudication file changed repeatedly while it was being read.")
    validation = validate_gold_rows(rows)
    return {
        "csv_path": str(private_path),
        "revision": after,
        "rows": rows,
        "validation": validation,
        "progress": gold_adjudication_progress(rows),
    }


def save_gold_editor_rows(
    csv_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_revision: str,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Atomically save UI edits while preserving bound packet fields.

    Incomplete human labels are valid drafts. Invalid rows, immutable-field
    changes, and revisions from an older browser view are rejected before the
    file is replaced.
    """
    private_path = _assert_private_input(csv_path, private_root=private_root)
    with _gold_editor_lock(private_path):
        current = load_gold_editor_state(private_path, private_root=private_root)
        if not expected_revision or current["revision"] != expected_revision:
            raise RuntimeError(
                "The adjudication file changed since this page loaded. Reload before saving."
            )

        proposed = [dict(row) for row in rows]
        current_rows = current["rows"]
        if len(proposed) != len(current_rows):
            raise ValueError("The adjudication row count cannot change in the editor.")
        missing = [
            f"row {index}: {column}"
            for index, row in enumerate(proposed, start=1)
            for column in GOLD_COLUMNS
            if column not in row
        ]
        if missing:
            raise ValueError(
                "Adjudication rows are missing required columns: " + ", ".join(missing)
            )

        immutable_changes: List[str] = []
        for current_row, proposed_row in zip(current_rows, proposed):
            cluster_id = _clean(current_row.get("cluster_id")) or "<missing>"
            for column in GOLD_IMMUTABLE_COLUMNS:
                if str(proposed_row.get(column, "")) != str(current_row.get(column, "")):
                    immutable_changes.append(f"{cluster_id}: {column}")
        if immutable_changes:
            raise ValueError(
                "The editor cannot change bound packet fields: "
                + ", ".join(immutable_changes)
            )

        expected_binding = current["validation"].get("binding_hash")
        expected_ids = [row["cluster_id"] for row in current_rows]
        validation = validate_gold_rows(
            proposed,
            expected_binding_hash=expected_binding,
            expected_cluster_ids=expected_ids,
        )
        if validation["status"] in {"invalid", "stale"}:
            details = validation["errors"] or [validation["status"]]
            raise ValueError("Adjudication edits are invalid: " + "; ".join(details))

        cleaned = [
            {column: str(row.get(column, "")) for column in GOLD_COLUMNS}
            for row in proposed
        ]
        _write_csv(private_path, GOLD_COLUMNS, cleaned)
        return load_gold_editor_state(private_path, private_root=private_root)


def _generated_id(issue: Mapping[str, Any]) -> str:
    existing = _clean(issue.get("generated_issue_id"))
    if existing:
        return existing
    return "GI_" + stable_hash(
        {
            "family_id": _family_id(issue),
            "case_id": _case_id(issue),
            "pipeline_issue_id": _clean(issue.get("id")),
            "text": _issue_text(issue),
            "evidence_ids": issue.get("evidence_ids", []),
        }
    )[:16]


def select_generated_top_k(
    generated_issues: Sequence[Mapping[str, Any]], top_k: int = 5
) -> List[Dict[str, Any]]:
    """Select final generated issues by explicit rank, preserving list order as fallback."""
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    groups: Dict[Tuple[str, str], List[Tuple[int, Mapping[str, Any]]]] = defaultdict(list)
    for position, issue in enumerate(generated_issues, start=1):
        groups[(_family_id(issue), _case_id(issue))].append((position, issue))
    selected: List[Dict[str, Any]] = []
    for (family, case), entries in sorted(groups.items()):
        ranked = sorted(
            entries,
            key=lambda pair: (
                int(pair[1].get("rank")) if str(pair[1].get("rank", "")).isdigit() else pair[0],
                pair[0],
            ),
        )[:top_k]
        for fallback_rank, (_position, issue) in enumerate(ranked, start=1):
            item = dict(issue)
            item["family_id"] = family
            item["case_id"] = case
            item["rank"] = int(issue.get("rank")) if str(issue.get("rank", "")).isdigit() else fallback_rank
            item["generated_issue_id"] = _generated_id(issue)
            selected.append(item)
    return selected


def _best_gold_match(issue: Mapping[str, Any], gold_rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    family = _family_id(issue)
    best_id = ""
    best_score = 0.0
    best_shared: List[str] = []
    for row in gold_rows:
        if _clean(row.get("family_id")) != family:
            continue
        if _clean(row.get("include")).lower() != "yes":
            continue
        score, shared = _text_similarity(_issue_text(issue), _clean(row.get("representative_text")))
        cluster_id = _clean(row.get("cluster_id"))
        if (score, cluster_id) > (best_score, best_id):
            best_id = cluster_id
            best_score = score
            best_shared = shared
    return {
        "cluster_id": best_id,
        "score": round(best_score, 4),
        "shared_terms": best_shared[:12],
    }


def generated_binding_hash(
    selected_issues: Sequence[Mapping[str, Any]],
    gold_binding: str,
    run_binding_context: Mapping[str, Any] | None = None,
    top_k: int = 5,
) -> str:
    payload = [
        {
            "family_id": _family_id(issue),
            "case_id": _case_id(issue),
            "rank": int(issue["rank"]),
            "generated_issue_id": _generated_id(issue),
            "pipeline_issue_id": _clean(issue.get("id")),
            "text": _issue_text(issue),
            "evidence_ids": issue.get("evidence_ids", []),
        }
        for issue in selected_issues
    ]
    return stable_hash(
        {
            "packet_version": PACKET_VERSION,
            "kind": "generated",
            "gold_binding_hash": gold_binding,
            "top_k": top_k,
            "issues": payload,
            "run_binding_context": dict(run_binding_context or {}),
        }
    )


def write_generated_adjudication_packet(
    generated_issues: Sequence[Mapping[str, Any]],
    gold_validation: Mapping[str, Any],
    output_dir: str | Path,
    *,
    run_binding_context: Mapping[str, Any] | None = None,
    top_k: int = 5,
    prefix: str = "generated_adjudication",
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Write the manual top-K generated-issue adjudication packet."""
    if gold_validation.get("status") != "ready":
        raise ValueError("Gold adjudication must be complete and current before generated adjudication.")
    selected = select_generated_top_k(generated_issues, top_k=top_k)
    if not selected:
        raise ValueError("Cannot prepare generated adjudication with no generated issues.")
    gold_binding = _clean(gold_validation.get("binding_hash"))
    binding = generated_binding_hash(selected, gold_binding, run_binding_context, top_k)
    gold_rows = list(gold_validation.get("rows", []))
    rows: List[Dict[str, Any]] = []
    for issue in selected:
        proposed = _best_gold_match(issue, gold_rows)
        rows.append(
            {
                "packet_version": PACKET_VERSION,
                "binding_hash": binding,
                "gold_binding_hash": gold_binding,
                "family_id": _family_id(issue),
                "case_id": _case_id(issue),
                "rank": issue["rank"],
                "generated_issue_id": _generated_id(issue),
                "pipeline_issue_id": _clean(issue.get("id")),
                "generated_text": _issue_text(issue),
                "evidence_ids": _json_cell(issue.get("evidence_ids", [])),
                "proposed_human_cluster_id": proposed["cluster_id"],
                "proposed_match_score": proposed["score"],
                "proposed_shared_terms": _json_cell(proposed["shared_terms"]),
                "correctness": "",
                "significance": "",
                "evidence_sufficiency": "",
                "human_match_status": "",
                "confirmed_human_cluster_ids": "[]",
                "duplicate_status": "",
                "duplicate_of_generated_id": "",
                "valid_novelty": "",
                "adjudicator_notes": "",
            }
        )

    output = _prepare_private_directory(output_dir, private_root=private_root)
    csv_path = output / f"{prefix}.csv"
    markdown_path = output / f"{prefix}.md"
    _write_csv(csv_path, GENERATED_COLUMNS, rows)
    lines = [
        "# Generated up-to-five adjudication",
        "",
        f"- Packet version: `{PACKET_VERSION}`",
        f"- Binding hash: `{binding}`",
        f"- Gold binding hash: `{gold_binding}`",
        f"- Generated rows: {len(rows)}",
        "",
        "Every row requires correctness, significance, evidence sufficiency, "
        "human-match status, duplicate status, and valid-novelty labels. "
        "Proposed matches are lexical suggestions only and require confirmation.",
    ]
    for row in rows:
        lines.extend(
            [
                "",
                f"## `{row['family_id']}` / `{row['case_id']}` / rank {row['rank']}",
                "",
                _markdown_text(row["generated_text"]),
                "",
                f"Generated ID: `{row['generated_issue_id']}`",
                f"Proposed cluster: `{row['proposed_human_cluster_id']}` "
                f"(score {row['proposed_match_score']})",
                f"Evidence IDs: {row['evidence_ids']}",
            ]
        )
    _write_private_text(markdown_path, "\n".join(lines) + "\n")
    return {
        "status": "pending_human_adjudication",
        "binding_hash": binding,
        "gold_binding_hash": gold_binding,
        "row_count": len(rows),
        "csv_path": str(csv_path),
        "markdown_path": str(markdown_path),
        "selected_issues": selected,
        "rows": rows,
    }


def generated_row_requirements(
    row: Mapping[str, Any],
    *,
    generated_rows: Sequence[Mapping[str, Any]] | None = None,
    gold_cluster_families: Mapping[str, str] | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """Return conditional label requirements for one generated issue row."""
    issue_id = _clean(row.get("generated_issue_id")) or "<missing>"
    family_id = _clean(row.get("family_id"))
    case_id = _clean(row.get("case_id"))
    known_generated = {
        _clean(item.get("generated_issue_id")): item
        for item in (generated_rows or [row])
        if _clean(item.get("generated_issue_id"))
    }
    known_gold = {
        str(cluster_id): _clean(cluster_family)
        for cluster_id, cluster_family in (gold_cluster_families or {}).items()
    }
    pending: List[str] = []
    errors: List[str] = []

    rank_text = _clean(row.get("rank"))
    if not rank_text.isdigit() or int(rank_text) <= 0:
        errors.append("rank must be a positive integer")
    elif top_k is not None and int(rank_text) > top_k:
        errors.append(f"rank must not exceed top_k={top_k}")

    checks = {
        "correctness": CORRECTNESS_VALUES,
        "significance": SIGNIFICANCE_VALUES,
        "evidence_sufficiency": EVIDENCE_VALUES,
        "human_match_status": MATCH_STATUS_VALUES,
        "duplicate_status": DUPLICATE_STATUS_VALUES,
        "valid_novelty": YES_NO_VALUES,
    }
    for field, allowed in checks.items():
        value = _clean(row.get(field)).lower()
        if not value:
            pending.append(field)
        elif value not in allowed:
            errors.append(f"{field} must be one of: {', '.join(sorted(allowed))}")

    match_status = _clean(row.get("human_match_status")).lower()
    confirmed, confirmed_valid = _parse_json_list_cell(
        row.get("confirmed_human_cluster_ids")
    )
    if not confirmed_valid:
        errors.append("confirmed_human_cluster_ids must be a JSON list")
    else:
        confirmed_ids = [str(cluster_id) for cluster_id in confirmed]
        if match_status == "matched" and not confirmed_ids:
            pending.append("confirmed_human_cluster_ids")
        if match_status == "unmatched" and confirmed_ids:
            errors.append("unmatched row cannot confirm human clusters")
        if gold_cluster_families is not None:
            for cluster_id in confirmed_ids:
                if cluster_id not in known_gold:
                    errors.append(f"confirms an unknown human cluster: {cluster_id}")
                elif known_gold[cluster_id] != family_id:
                    errors.append("confirmed human cluster must be in the same family")

    duplicate_status = _clean(row.get("duplicate_status")).lower()
    duplicate_of = _clean(row.get("duplicate_of_generated_id"))
    if duplicate_status == "duplicate":
        if not duplicate_of:
            pending.append("duplicate_of_generated_id")
        elif duplicate_of not in known_generated or duplicate_of == issue_id:
            errors.append("duplicate target must be another generated issue in the packet")
        else:
            target = known_generated[duplicate_of]
            if (
                _clean(target.get("family_id")) != family_id
                or _clean(target.get("case_id")) != case_id
            ):
                errors.append("duplicate target must be in the same family and case")
    if duplicate_status == "unique" and duplicate_of:
        errors.append("unique row cannot have duplicate_of_generated_id")

    novelty = _clean(row.get("valid_novelty")).lower()
    if novelty == "yes":
        if match_status != "unmatched":
            errors.append("valid novelty must be unmatched")
        if _clean(row.get("correctness")).lower() != "correct":
            errors.append("valid novelty must be correct")
        if _clean(row.get("evidence_sufficiency")).lower() != "sufficient":
            errors.append("valid novelty must have sufficient evidence")
        if duplicate_status != "unique":
            errors.append("valid novelty must be unique")

    return {
        "generated_issue_id": issue_id,
        "pending_fields": pending,
        "errors": errors,
        "complete": not pending and not errors,
    }


def generated_adjudication_progress(
    rows: Sequence[Mapping[str, Any]],
    *,
    gold_cluster_families: Mapping[str, str] | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """Summarize generated-issue label progress for the local editor."""
    states = [
        generated_row_requirements(
            row,
            generated_rows=rows,
            gold_cluster_families=gold_cluster_families,
            top_k=top_k,
        )
        for row in rows
    ]
    families: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "complete": 0, "remaining": 0}
    )
    cases: Dict[str, Dict[str, int]] = defaultdict(
        lambda: {"total": 0, "complete": 0, "remaining": 0}
    )
    for row, state in zip(rows, states):
        family_id = _clean(row.get("family_id"))
        case_key = f"{family_id}/{_clean(row.get('case_id'))}"
        for bucket in (families[family_id], cases[case_key]):
            bucket["total"] += 1
            bucket["complete"] += int(state["complete"])
            bucket["remaining"] += int(not state["complete"])
    return {
        "total": len(rows),
        "complete": sum(int(state["complete"]) for state in states),
        "remaining": sum(int(not state["complete"]) for state in states),
        "families": {key: dict(value) for key, value in sorted(families.items())},
        "cases": {key: dict(value) for key, value in sorted(cases.items())},
    }


def validate_generated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_binding_hash: str | None = None,
    expected_gold_binding_hash: str | None = None,
    expected_generated_issue_ids: Iterable[str] | None = None,
    valid_gold_cluster_ids: Iterable[str] | None = None,
    gold_cluster_families: Mapping[str, str] | None = None,
    top_k: int | None = None,
) -> Dict[str, Any]:
    """Validate all manual labels in a generated top-K packet."""
    errors: List[str] = []
    pending: List[str] = []
    bindings = {_clean(row.get("binding_hash")) for row in rows if _clean(row.get("binding_hash"))}
    gold_bindings = {
        _clean(row.get("gold_binding_hash")) for row in rows if _clean(row.get("gold_binding_hash"))
    }
    binding = next(iter(bindings)) if len(bindings) == 1 else ""
    gold_binding = next(iter(gold_bindings)) if len(gold_bindings) == 1 else ""
    stale = False
    if not rows:
        errors.append("packet has no generated issue rows")
    if any(_clean(row.get("packet_version")) != PACKET_VERSION for row in rows):
        errors.append(f"packet_version must be {PACKET_VERSION}")
    if len(bindings) != 1:
        errors.append("packet must contain exactly one nonempty binding_hash")
    if len(gold_bindings) != 1:
        errors.append("packet must contain exactly one nonempty gold_binding_hash")
    if expected_binding_hash and binding != expected_binding_hash:
        stale = True
        errors.append("binding hash does not match current baseline output")
    if expected_gold_binding_hash and gold_binding != expected_gold_binding_hash:
        stale = True
        errors.append("gold binding hash does not match current adjudication")

    generated_ids = {_clean(row.get("generated_issue_id")) for row in rows}
    if "" in generated_ids:
        errors.append("every row must have generated_issue_id")
    if len(generated_ids) != len(rows):
        errors.append("generated_issue_id values must be unique")
    if expected_generated_issue_ids is not None and generated_ids != set(expected_generated_issue_ids):
        stale = True
        errors.append("packet generated-issue set does not match current baseline output")
    rank_keys: set[Tuple[str, str, int]] = set()
    ranks_by_case: Dict[Tuple[str, str], set[int]] = defaultdict(set)
    known_gold_families = {
        str(cluster_id): _clean(family_id)
        for cluster_id, family_id in (gold_cluster_families or {}).items()
    }
    gold_ids_known = valid_gold_cluster_ids is not None
    gold_ids = set(str(value) for value in (valid_gold_cluster_ids or []))
    for row in rows:
        issue_id = _clean(row.get("generated_issue_id")) or "<missing>"
        rank_text = _clean(row.get("rank"))
        if rank_text.isdigit() and int(rank_text) > 0:
            rank_value = int(rank_text)
            rank_key = (
                _clean(row.get("family_id")),
                _clean(row.get("case_id")),
                rank_value,
            )
            if rank_key in rank_keys:
                errors.append(f"{issue_id}: rank must be unique within family/case")
            rank_keys.add(rank_key)
            ranks_by_case[rank_key[:2]].add(rank_value)
        state = generated_row_requirements(
            row,
            generated_rows=rows,
            gold_cluster_families=(
                known_gold_families
                if gold_cluster_families is not None
                else None
            ),
            top_k=top_k,
        )
        pending.extend(f"{issue_id}: {field}" for field in state["pending_fields"])
        errors.extend(f"{issue_id}: {error}" for error in state["errors"])

        confirmed, confirmed_valid = _parse_json_list_cell(
            row.get("confirmed_human_cluster_ids")
        )
        if (
            confirmed_valid
            and gold_ids_known
            and any(str(cluster_id) not in gold_ids for cluster_id in confirmed)
        ):
            errors.append(f"{issue_id}: confirms an unknown human cluster")

    if top_k is not None:
        for case_key, ranks in sorted(ranks_by_case.items()):
            expected_ranks = set(range(1, len(ranks) + 1))
            if ranks != expected_ranks:
                errors.append(
                    f"case {case_key!r}: ranks must be contiguous from 1"
                )

    if stale:
        status = "stale"
    elif errors:
        status = "invalid"
    elif pending:
        status = "pending_human_adjudication"
    else:
        status = "ready"
    return {
        "status": status,
        "binding_hash": binding,
        "gold_binding_hash": gold_binding,
        "rows": [dict(row) for row in rows],
        "errors": errors,
        "pending_fields": pending,
        "row_count": len(rows),
        "completed": status == "ready",
    }


def load_generated_adjudication(
    csv_path: str | Path,
    *,
    expected_binding_hash: str | None = None,
    expected_gold_binding_hash: str | None = None,
    valid_gold_cluster_ids: Iterable[str] | None = None,
    gold_cluster_families: Mapping[str, str] | None = None,
    generated_issues: Sequence[Mapping[str, Any]] | None = None,
    run_binding_context: Mapping[str, Any] | None = None,
    top_k: int = 5,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    rows = _read_csv(csv_path, private_root=private_root)
    selected = None
    expected_ids = None
    if generated_issues is not None:
        selected = select_generated_top_k(generated_issues, top_k=top_k)
        expected_ids = [_generated_id(item) for item in selected]
        gold_binding = expected_gold_binding_hash
        if not gold_binding:
            row_gold_bindings = {
                _clean(row.get("gold_binding_hash"))
                for row in rows
                if _clean(row.get("gold_binding_hash"))
            }
            if len(row_gold_bindings) == 1:
                gold_binding = next(iter(row_gold_bindings))
        computed = generated_binding_hash(
            selected,
            gold_binding or "",
            run_binding_context,
            top_k,
        )
        if expected_binding_hash and expected_binding_hash != computed:
            raise ValueError("Provided expected_binding_hash disagrees with current generated issues/context.")
        expected_binding_hash = computed
    elif run_binding_context is not None and rows:
        row_issues = [
            {
                "family_id": row.get("family_id", ""),
                "case_id": row.get("case_id", ""),
                "rank": row.get("rank", ""),
                "generated_issue_id": row.get("generated_issue_id", ""),
                "id": row.get("pipeline_issue_id", ""),
                "text": row.get("generated_text", ""),
                "evidence_ids": _parse_json_cell(row.get("evidence_ids"), []),
            }
            for row in rows
        ]
        selected = select_generated_top_k(row_issues, top_k=top_k)
        expected_ids = [_generated_id(item) for item in selected]
        gold_binding = expected_gold_binding_hash or _clean(
            rows[0].get("gold_binding_hash")
        )
        computed = generated_binding_hash(
            selected,
            gold_binding,
            run_binding_context,
            top_k,
        )
        if expected_binding_hash and expected_binding_hash != computed:
            raise ValueError(
                "Generated adjudication immutable fields do not match the baseline binding."
            )
        expected_binding_hash = computed
    result = validate_generated_rows(
        rows,
        expected_binding_hash=expected_binding_hash,
        expected_gold_binding_hash=expected_gold_binding_hash,
        expected_generated_issue_ids=expected_ids,
        valid_gold_cluster_ids=valid_gold_cluster_ids,
        gold_cluster_families=gold_cluster_families,
        top_k=top_k,
    )
    if selected is not None:
        expected = {
            _generated_id(item): {
                "family_id": _family_id(item),
                "case_id": _case_id(item),
                "rank": str(item["rank"]),
                "pipeline_issue_id": _clean(item.get("id")),
                "generated_text": _issue_text(item),
                "evidence_ids": item.get("evidence_ids", []),
            }
            for item in selected
        }
        immutable_errors = []
        for row in rows:
            issue_id = _clean(row.get("generated_issue_id"))
            target = expected.get(issue_id)
            if target is None:
                continue
            for field in ("family_id", "case_id", "rank", "pipeline_issue_id", "generated_text"):
                if _clean(row.get(field)) != target[field]:
                    immutable_errors.append(f"{issue_id}: immutable field changed: {field}")
            if _parse_json_cell(row.get("evidence_ids"), None) != target["evidence_ids"]:
                immutable_errors.append(f"{issue_id}: immutable field changed: evidence_ids")
        if immutable_errors:
            result["errors"].extend(immutable_errors)
            result["status"] = "stale"
            result["completed"] = False
    result["csv_path"] = str(Path(csv_path).expanduser())
    return result


def load_generated_editor_state(
    csv_path: str | Path,
    *,
    expected_binding_hash: str | None = None,
    expected_gold_binding_hash: str | None = None,
    gold_cluster_families: Mapping[str, str] | None = None,
    run_binding_context: Mapping[str, Any] | None = None,
    top_k: int = 5,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a generated packet with a revision token for guarded UI edits."""
    private_path = _assert_private_input(csv_path, private_root=private_root)
    for _ in range(3):
        before = _file_revision(private_path)
        validation = load_generated_adjudication(
            private_path,
            expected_binding_hash=expected_binding_hash,
            expected_gold_binding_hash=expected_gold_binding_hash,
            gold_cluster_families=gold_cluster_families,
            run_binding_context=run_binding_context,
            top_k=top_k,
            private_root=private_root,
        )
        after = _file_revision(private_path)
        if before == after:
            break
    else:
        raise RuntimeError(
            "The generated adjudication file changed repeatedly while it was being read."
        )
    rows = list(validation.get("rows", []))
    return {
        "csv_path": str(private_path),
        "revision": after,
        "rows": rows,
        "validation": validation,
        "progress": generated_adjudication_progress(
            rows,
            gold_cluster_families=gold_cluster_families,
            top_k=top_k,
        ),
    }


def save_generated_editor_rows(
    csv_path: str | Path,
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_revision: str,
    expected_binding_hash: str | None = None,
    expected_gold_binding_hash: str | None = None,
    gold_cluster_families: Mapping[str, str] | None = None,
    run_binding_context: Mapping[str, Any] | None = None,
    top_k: int = 5,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Atomically save generated-label drafts without changing bound output fields."""
    private_path = _assert_private_input(csv_path, private_root=private_root)
    with _gold_editor_lock(private_path):
        current = load_generated_editor_state(
            private_path,
            expected_binding_hash=expected_binding_hash,
            expected_gold_binding_hash=expected_gold_binding_hash,
            gold_cluster_families=gold_cluster_families,
            run_binding_context=run_binding_context,
            top_k=top_k,
            private_root=private_root,
        )
        if not expected_revision or current["revision"] != expected_revision:
            raise RuntimeError(
                "The generated adjudication file changed since this page loaded. "
                "Reload before saving."
            )
        if current["validation"].get("status") == "stale":
            details = current["validation"].get("errors") or [
                current["validation"].get("status", "invalid")
            ]
            raise ValueError(
                "The current generated adjudication packet is not editable: "
                + "; ".join(details)
            )

        proposed = [dict(row) for row in rows]
        current_rows = current["rows"]
        if len(proposed) != len(current_rows):
            raise ValueError(
                "The generated adjudication row count cannot change in the editor."
            )
        missing = [
            f"row {index}: {column}"
            for index, row in enumerate(proposed, start=1)
            for column in GENERATED_COLUMNS
            if column not in row
        ]
        if missing:
            raise ValueError(
                "Generated adjudication rows are missing required columns: "
                + ", ".join(missing)
            )

        current_order = [
            _clean(row.get("generated_issue_id")) for row in current_rows
        ]
        proposed_order = [_clean(row.get("generated_issue_id")) for row in proposed]
        if proposed_order != current_order:
            raise ValueError(
                "The generated adjudication row order cannot change in the editor."
            )

        immutable_changes: List[str] = []
        for current_row, proposed_row in zip(current_rows, proposed):
            issue_id = _clean(current_row.get("generated_issue_id")) or "<missing>"
            for column in GENERATED_IMMUTABLE_COLUMNS:
                if str(proposed_row.get(column, "")) != str(
                    current_row.get(column, "")
                ):
                    immutable_changes.append(f"{issue_id}: {column}")
        if immutable_changes:
            raise ValueError(
                "The editor cannot change bound generated packet fields: "
                + ", ".join(immutable_changes)
            )

        current_validation = current["validation"]
        validation = validate_generated_rows(
            proposed,
            expected_binding_hash=(
                expected_binding_hash
                or _clean(current_validation.get("binding_hash"))
            ),
            expected_gold_binding_hash=(
                expected_gold_binding_hash
                or _clean(current_validation.get("gold_binding_hash"))
            ),
            expected_generated_issue_ids=current_order,
            gold_cluster_families=gold_cluster_families,
            top_k=top_k,
        )
        if validation["status"] in {"invalid", "stale"}:
            details = validation["errors"] or [validation["status"]]
            raise ValueError(
                "Generated adjudication edits are invalid: " + "; ".join(details)
            )

        cleaned = [
            {column: str(row.get(column, "")) for column in GENERATED_COLUMNS}
            for row in proposed
        ]
        _write_csv(private_path, GENERATED_COLUMNS, cleaned)
        return load_generated_editor_state(
            private_path,
            expected_binding_hash=expected_binding_hash,
            expected_gold_binding_hash=expected_gold_binding_hash,
            gold_cluster_families=gold_cluster_families,
            run_binding_context=run_binding_context,
            top_k=top_k,
            private_root=private_root,
        )


def _safe_ratio(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return round(numerator / denominator, 4)


def _macro(values: Iterable[float | None]) -> float | None:
    usable = [value for value in values if value is not None]
    if not usable:
        return None
    return round(sum(usable) / len(usable), 4)


def _deduplicated_gold_sets(
    family_rows: Sequence[Mapping[str, Any]],
) -> Tuple[set[str], set[str], Dict[str, str]]:
    """Return major/minor cluster groups after manual duplicate corrections."""
    included = {
        _clean(row.get("cluster_id")): row
        for row in family_rows
        if _clean(row.get("include")).lower() == "yes"
    }
    parent = {cluster_id: cluster_id for cluster_id in included}

    def find(cluster_id: str) -> str:
        while parent[cluster_id] != cluster_id:
            parent[cluster_id] = parent[parent[cluster_id]]
            cluster_id = parent[cluster_id]
        return cluster_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root == right_root:
            return
        keep, merge = sorted([left_root, right_root])
        parent[merge] = keep

    for cluster_id, row in included.items():
        duplicates = _parse_json_cell(row.get("duplicate_cluster_ids"), [])
        if isinstance(duplicates, list):
            for duplicate_id in map(str, duplicates):
                if duplicate_id in included:
                    union(cluster_id, duplicate_id)

    cluster_to_group = {cluster_id: find(cluster_id) for cluster_id in included}
    group_rows: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for cluster_id, row in included.items():
        group_rows[cluster_to_group[cluster_id]].append(row)
    major_groups = {
        group
        for group, rows in group_rows.items()
        if any(_clean(row.get("tier_screen")).lower() == "major" for row in rows)
    }
    sampled_minor_groups = {
        group
        for group, rows in group_rows.items()
        if group not in major_groups
        and any(
            _clean(row.get("tier_screen")).lower() == "minor"
            and _clean(row.get("sampled_minor")).lower() == "yes"
            for row in rows
        )
    }
    return major_groups, sampled_minor_groups, cluster_to_group


def _family_metadata(
    family_id: str, metadata: Mapping[str, Mapping[str, Any]]
) -> Dict[str, Any]:
    item = dict(metadata.get(family_id, {}))
    journal_value = item.get("is_journal_case", item.get("journal_case", False))
    is_journal = (
        journal_value
        if isinstance(journal_value, bool)
        else _clean(journal_value).lower() in {"1", "true", "yes"}
    )
    return {
        "public_family_id": _clean(item.get("public_family_id")) or "F_" + stable_hash(family_id)[:8],
        "benchmark_tier": _clean(item.get("benchmark_tier")).lower() or "primary",
        "is_journal_case": is_journal,
        "case_count": max(1, int(item.get("case_count", 1) or 1)),
    }


def compute_privacy_safe_metrics(
    gold_validation: Mapping[str, Any],
    generated_validation: Mapping[str, Any],
    *,
    family_metadata: Mapping[str, Mapping[str, Any]] | None = None,
    cost_by_family: Mapping[str, float] | None = None,
    top_k: int = 5,
    output_cardinality_mode: str = "exact_k",
) -> Dict[str, Any]:
    """Compute aggregate metrics after both human-adjudication gates pass.

    The returned object intentionally contains only pseudonymous family IDs,
    counts, rates, cost numbers, and status.  It never copies issue text,
    reviewer IDs, locators, evidence text, or filesystem paths.
    """
    if output_cardinality_mode not in {"exact_k", "up_to_k"}:
        raise ValueError("output_cardinality_mode must be one of: exact_k, up_to_k")
    if gold_validation.get("status") != "ready" or generated_validation.get("status") != "ready":
        pending_result = {
            "status": "pending_human_adjudication",
            "gold_status": gold_validation.get("status", "missing"),
            "generated_status": generated_validation.get("status", "missing"),
            "gold_pending_count": len(gold_validation.get("pending_fields", [])),
            "generated_pending_count": len(generated_validation.get("pending_fields", [])),
        }
        if gold_validation.get("gold_mode") == "partial":
            pending_result.update(
                {
                    "gold_mode": "partial",
                    "evaluation_scope": "partial_gold_pilot",
                    "denominator_scope": "adjudicated_clusters_only",
                    "exhaustive_major_recall": False,
                    "gold_coverage": dict(gold_validation.get("coverage", {})),
                }
            )
        return pending_result

    metadata = family_metadata or {}
    costs = cost_by_family or {}
    gold_rows = list(gold_validation.get("rows", []))
    generated_rows = []
    for row in generated_validation.get("rows", []):
        rank = _clean(row.get("rank"))
        if rank.isdigit() and int(rank) <= top_k:
            generated_rows.append(row)
    families = sorted(
        (set(metadata) if output_cardinality_mode == "up_to_k" else set())
        | {
            _clean(row.get("family_id"))
            for row in gold_rows + generated_rows
            if _clean(row.get("family_id"))
        }
    )

    confirmed_by_family: Dict[str, set[str]] = defaultdict(set)
    for row in generated_rows:
        if (
            _clean(row.get("correctness")).lower() == "correct"
            and _clean(row.get("evidence_sufficiency")).lower() == "sufficient"
            and _clean(row.get("human_match_status")).lower() == "matched"
        ):
            confirmed_by_family[_clean(row.get("family_id"))].update(
                str(item) for item in _parse_json_cell(row.get("confirmed_human_cluster_ids"), [])
            )

    per_family: List[Dict[str, Any]] = []
    for family in families:
        meta = _family_metadata(family, metadata)
        family_gold = [row for row in gold_rows if _clean(row.get("family_id")) == family]
        family_generated = [row for row in generated_rows if _clean(row.get("family_id")) == family]
        major_ids, sampled_minor_ids, cluster_to_group = _deduplicated_gold_sets(family_gold)
        matched = {
            cluster_to_group[cluster_id]
            for cluster_id in confirmed_by_family.get(family, set())
            if cluster_id in cluster_to_group
        }
        supported_significant = sum(
            1
            for row in family_generated
            if _clean(row.get("correctness")).lower() == "correct"
            and _clean(row.get("significance")).lower() == "significant"
            and _clean(row.get("evidence_sufficiency")).lower() == "sufficient"
        )
        novel = sum(1 for row in family_generated if _clean(row.get("valid_novelty")).lower() == "yes")
        duplicates = sum(
            1 for row in family_generated if _clean(row.get("duplicate_status")).lower() == "duplicate"
        )
        count = len(family_generated)
        evaluation_slots = (
            top_k * meta["case_count"]
            if output_cardinality_mode == "up_to_k"
            else count
        )
        per_family.append(
            {
                "family_id": meta["public_family_id"],
                "benchmark_tier": meta["benchmark_tier"],
                "is_journal_case": meta["is_journal_case"],
                "major_cluster_count": len(major_ids),
                "major_clusters_recalled_at_5": len(major_ids & matched),
                "major_cluster_recall_at_5": _safe_ratio(len(major_ids & matched), len(major_ids)),
                "sampled_minor_cluster_count": len(sampled_minor_ids),
                "sampled_minor_clusters_recalled_at_5": len(sampled_minor_ids & matched),
                "sampled_minor_cluster_recall_at_5": _safe_ratio(
                    len(sampled_minor_ids & matched), len(sampled_minor_ids)
                ),
                "generated_issue_count": count,
                "evaluation_slot_count": evaluation_slots,
                "unfilled_issue_slot_count": max(0, evaluation_slots - count),
                "supported_significant_issue_count": supported_significant,
                "supported_significant_precision_at_5": _safe_ratio(
                    supported_significant, evaluation_slots
                ),
                "valid_novel_issue_count": novel,
                "valid_novelty_yield_at_5": _safe_ratio(novel, evaluation_slots),
                "duplicate_issue_count": duplicates,
                "duplicate_rate_at_5": _safe_ratio(duplicates, count),
                "cost_usd": round(float(costs.get(family, 0.0)), 6),
            }
        )

    primary = [row for row in per_family if row["benchmark_tier"] == "primary"]
    journal = [row for row in primary if row["is_journal_case"]]
    secondary = [row for row in per_family if row["benchmark_tier"] != "primary"]
    primary_generated = sum(row["generated_issue_count"] for row in primary)
    primary_slots = sum(row["evaluation_slot_count"] for row in primary)
    primary_supported = sum(row["supported_significant_issue_count"] for row in primary)
    primary_novel = sum(row["valid_novel_issue_count"] for row in primary)
    primary_duplicates = sum(row["duplicate_issue_count"] for row in primary)
    all_generated = sum(row["generated_issue_count"] for row in per_family)
    all_slots = sum(row["evaluation_slot_count"] for row in per_family)
    all_supported = sum(row["supported_significant_issue_count"] for row in per_family)
    all_novel = sum(row["valid_novel_issue_count"] for row in per_family)
    all_duplicates = sum(row["duplicate_issue_count"] for row in per_family)
    result = {
        "status": "complete",
        "top_k": top_k,
        "output_cardinality_mode": output_cardinality_mode,
        "missing_slot_policy": (
            "count_as_miss_for_supported_precision_and_novelty_yield"
            if output_cardinality_mode == "up_to_k"
            else "not_applicable"
        ),
        "primary_family_count": len(primary),
        "secondary_family_count": len(secondary),
        "primary_family_macro_major_cluster_recall_at_5": _macro(
            row["major_cluster_recall_at_5"] for row in primary
        ),
        "journal_family_macro_major_cluster_recall_at_5": _macro(
            row["major_cluster_recall_at_5"] for row in journal
        ),
        "primary_family_macro_sampled_minor_cluster_recall_at_5": _macro(
            row["sampled_minor_cluster_recall_at_5"] for row in primary
        ),
        "primary_supported_significant_precision_at_5": _safe_ratio(
            primary_supported, primary_slots
        ),
        "primary_valid_novelty_yield_at_5": _safe_ratio(primary_novel, primary_slots),
        "primary_duplicate_rate_at_5": _safe_ratio(primary_duplicates, primary_generated),
        "all_family_supported_significant_precision_at_5": _safe_ratio(all_supported, all_slots),
        "all_family_valid_novelty_yield_at_5": _safe_ratio(all_novel, all_slots),
        "all_family_duplicate_rate_at_5": _safe_ratio(all_duplicates, all_generated),
        "total_generated_issue_count": all_generated,
        "primary_evaluation_slot_count": primary_slots,
        "all_family_evaluation_slot_count": all_slots,
        "total_unfilled_issue_slot_count": sum(
            row["unfilled_issue_slot_count"] for row in per_family
        ),
        "total_cost_usd": round(sum(float(costs.get(family, 0.0)) for family in families), 6),
        "families": per_family,
    }
    if gold_validation.get("gold_mode") == "partial":
        result["status"] = "partial_gold_pilot"
        result["evaluation_scope"] = "partial_gold_pilot"
        result["denominator_scope"] = "adjudicated_clusters_only"
        result["exhaustive_major_recall"] = False
        result["novelty_reference_scope"] = "partial_gold_plus_manual_judgment"
        result["top_k_policy"] = output_cardinality_mode
        result["gold_coverage"] = dict(gold_validation.get("coverage", {}))
        result[
            "adjudicated_subset_primary_family_macro_major_cluster_recall_at_5"
        ] = result.pop("primary_family_macro_major_cluster_recall_at_5")
        result[
            "adjudicated_subset_journal_family_macro_major_cluster_recall_at_5"
        ] = result.pop("journal_family_macro_major_cluster_recall_at_5")
        for family in result["families"]:
            family["adjudicated_major_cluster_count"] = family.pop(
                "major_cluster_count"
            )
            family["adjudicated_major_clusters_recalled_at_5"] = family.pop(
                "major_clusters_recalled_at_5"
            )
            family["adjudicated_major_cluster_recall_at_5"] = family.pop(
                "major_cluster_recall_at_5"
            )
            if output_cardinality_mode == "up_to_k":
                family["duplicate_rate_among_returned_at_5"] = family.pop(
                    "duplicate_rate_at_5"
                )
        result["primary_families_with_adjudicated_major_denominator"] = sum(
            row["benchmark_tier"] == "primary"
            and row["adjudicated_major_cluster_count"] > 0
            for row in result["families"]
        )
        result["journal_families_with_adjudicated_major_denominator"] = sum(
            row["benchmark_tier"] == "primary"
            and row["is_journal_case"]
            and row["adjudicated_major_cluster_count"] > 0
            for row in result["families"]
        )
        if output_cardinality_mode == "up_to_k":
            result["primary_duplicate_rate_among_returned_at_5"] = result.pop(
                "primary_duplicate_rate_at_5"
            )
            result["all_family_duplicate_rate_among_returned_at_5"] = result.pop(
                "all_family_duplicate_rate_at_5"
            )
    return result


__all__ = [
    "DEFAULT_MINOR_SAMPLE_SIZE",
    "DEFAULT_PRIVATE_ROOT",
    "DEFAULT_SAMPLE_SEED",
    "GENERATED_COLUMNS",
    "GENERATED_EDITABLE_COLUMNS",
    "GENERATED_IMMUTABLE_COLUMNS",
    "GOLD_COLUMNS",
    "GOLD_EDITABLE_COLUMNS",
    "GOLD_IMMUTABLE_COLUMNS",
    "PACKET_VERSION",
    "PARTIAL_GOLD_BINDING_VERSION",
    "cluster_normalized_issues",
    "compute_privacy_safe_metrics",
    "eligible_duplicate_clusters",
    "generated_binding_hash",
    "generated_adjudication_progress",
    "generated_row_requirements",
    "gold_adjudication_progress",
    "gold_binding_hash",
    "gold_row_requirements",
    "load_generated_adjudication",
    "load_generated_editor_state",
    "load_gold_editor_state",
    "load_gold_adjudication",
    "project_partial_gold_adjudication",
    "save_gold_editor_rows",
    "save_generated_editor_rows",
    "select_full_adjudication_clusters",
    "select_generated_top_k",
    "stable_hash",
    "validate_generated_rows",
    "validate_gold_rows",
    "write_generated_adjudication_packet",
    "write_gold_adjudication_packet",
]
