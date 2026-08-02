"""Private, manifest-driven human-feedback corpus import.

The module deliberately separates private source material from repository data.
Manifests point to local files, bind every file to a SHA-256 digest, and produce
records compatible with the historical-review corpus shape used by
``feedback_pipeline``.  Raw paths and reviewer names are never included in the
portable audit.

The importer is offline.  It does not download cloud placeholders or call an
API, and it adds no dependency beyond PyMuPDF, which the application already
uses for PDF input.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import tempfile
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple
from xml.etree import ElementTree as ET


MANIFEST_VERSION = "v1"
IMPORTER_VERSION = "review-corpus-manifest-v1"
MANIFEST_SOURCE_KIND = "manifest_human_feedback"
DEFAULT_PRIVATE_ROOT = Path.home() / ".feedback_llm"

ALLOWED_EXTRACTORS = {
    "pdf_annotations",
    "pdf_text",
    "docx_comments",
    "docx_body",
    "markdown",
    "text",
}
ALLOWED_BENCHMARK_TIERS = {"primary", "secondary", "deferred"}
ALLOWED_DISPOSITIONS = {
    "evaluation",
    "deferred",
    "memory_only",
    "action_only",
    "exclude",
}
EXACT_VERSION_MATCHES = {"exact_embedded", "exact_submission", "version_matched"}
ALLOWED_VERSION_MATCHES = EXACT_VERSION_MATCHES | {"near_exact", "unknown"}
IMPORTABLE_DISPOSITIONS = {"evaluation", "memory_only"}
SUPPORTED_MANUSCRIPT_SUFFIXES = {".pdf", ".docx", ".md", ".markdown", ".txt"}
FORBIDDEN_EVALUATION_SOURCE_TOKENS = {
    "ai",
    "automated",
    "derivative",
    "digest",
    "generated",
    "llm",
    "machine",
    "rebuttal",
    "response",
    "summary",
    "synthetic",
}

_W_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_W = f"{{{_W_NS}}}"
_NUMBERED_ITEM_RE = re.compile(
    r"^\s*(?:#{1,6}\s*)?(?:\*\*)?(\d{1,3})[.)](?:\*\*)?\s+",
    re.MULTILINE,
)
_BULLET_RE = re.compile(r"^\s*(?:[-*+]|\u2022)\s+")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+")


class ManifestValidationError(ValueError):
    """Raised when a private corpus manifest fails closed validation."""

    def __init__(self, errors: str | Sequence[str]):
        self.errors = [errors] if isinstance(errors, str) else list(errors)
        super().__init__("Invalid review corpus manifest:\n- " + "\n- ".join(self.errors))


def sha256_file(path: str | Path) -> str:
    """Return the lowercase SHA-256 digest for *path* without loading it at once."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _case_paper_id(family_id: str, case_id: str) -> str:
    """Return a case pseudonym; family-level holdout uses ``family_id``."""
    basis = f"{family_id}\0{case_id}".encode("utf-8")
    return f"paper_{hashlib.sha256(basis).hexdigest()[:16]}"


def _normalize_manifest_version(value: Any) -> str:
    if value in {1, "1", "v1"}:
        return MANIFEST_VERSION
    return str(value or "")


def _is_identifier(value: Any) -> bool:
    return bool(
        isinstance(value, str)
        and re.fullmatch(r"[a-z0-9][a-z0-9_.-]{0,99}", value)
    )


def _is_digest(value: Any) -> bool:
    return bool(isinstance(value, str) and re.fullmatch(r"[a-f0-9]{64}", value))


def _expand_path(raw_path: str, base_dir: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(raw_path)))
    if not expanded.is_absolute():
        expanded = base_dir / expanded
    return expanded.resolve()


def _private_root_path(private_root: str | Path | None = None) -> Path:
    return Path(private_root or DEFAULT_PRIVATE_ROOT).expanduser().resolve()


def _path_below_private_root(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
    label: str,
) -> Tuple[Path, Path]:
    root = _private_root_path(private_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    candidate = candidate.resolve()
    if candidate == root or root not in candidate.parents:
        raise ManifestValidationError(f"{label} must stay below {root}")
    return candidate, root


def _assert_private_file(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
    label: str,
) -> Path:
    candidate, root = _path_below_private_root(
        path,
        private_root=private_root,
        label=label,
    )
    try:
        file_mode = stat.S_IMODE(candidate.stat().st_mode)
    except FileNotFoundError:
        raise
    if file_mode != 0o600:
        raise ManifestValidationError(
            f"{label} permissions must be 0600, got {oct(file_mode)}"
        )
    current = candidate.parent
    while True:
        directory_mode = stat.S_IMODE(current.stat().st_mode)
        if directory_mode != 0o700:
            raise ManifestValidationError(
                f"private directory permissions must be 0700 for {current}, "
                f"got {oct(directory_mode)}"
            )
        if current == root:
            break
        current = current.parent
    return candidate


def _ensure_private_directory(path: Path, root: Path) -> None:
    """Create/chmod *root* and descendants before any private bytes are written."""
    if path != root and root not in path.parents:
        raise ManifestValidationError(f"private output must stay below {root}")
    if root.exists():
        if not root.is_dir():
            raise ManifestValidationError(f"private root is not a directory: {root}")
        os.chmod(root, 0o700)
    else:
        # The default root is directly below an existing home directory.  Tests
        # may inject a deeper root, so create missing ancestors first but never
        # write private content until the root itself is mode 0700.
        root.mkdir(parents=True, mode=0o700)
        os.chmod(root, 0o700)
    relative_parts = path.relative_to(root).parts
    current = root
    for part in relative_parts:
        current = current / part
        if current.exists():
            if not current.is_dir():
                raise ManifestValidationError(
                    f"private output parent is not a directory: {current}"
                )
            os.chmod(current, 0o700)
        else:
            os.mkdir(current, mode=0o700)
        os.chmod(current, 0o700)


def _file_is_offline(path: Path) -> bool:
    """Detect common non-materialized cloud placeholders without fetching them."""
    try:
        info = path.stat()
    except OSError:
        return False
    offline_flags = getattr(stat, "UF_OFFLINE", 0) | getattr(stat, "SF_DATALESS", 0)
    return bool(offline_flags and getattr(info, "st_flags", 0) & offline_flags)


def _validate_file_entry(
    entry: Any,
    *,
    label: str,
    base_dir: Path,
    allowed_suffixes: Iterable[str] | None = None,
) -> Tuple[List[str], Path | None]:
    errors: List[str] = []
    if not isinstance(entry, Mapping):
        return [f"{label} must be an object"], None
    raw_path = entry.get("path")
    if not isinstance(raw_path, str) or not raw_path.strip():
        return [f"{label}.path must be a nonempty string"], None
    if not _is_digest(entry.get("sha256")):
        errors.append(f"{label}.sha256 must be a lowercase 64-character digest")
    path = _expand_path(raw_path, base_dir)
    if allowed_suffixes and path.suffix.lower() not in set(allowed_suffixes):
        errors.append(f"{label} has unsupported file type: {path.suffix or '<none>'}")
    if not path.exists():
        errors.append(f"{label} is missing: {raw_path}")
        return errors, path
    if not path.is_file():
        errors.append(f"{label} is not a regular file: {raw_path}")
        return errors, path
    if _file_is_offline(path):
        errors.append(f"{label} is an unreadable cloud placeholder: {raw_path}")
        return errors, path
    try:
        size = path.stat().st_size
    except OSError as exc:
        errors.append(f"{label} cannot be inspected: {exc}")
        return errors, path
    if size <= 0:
        errors.append(f"{label} is empty or an unreadable cloud placeholder: {raw_path}")
        return errors, path
    if _is_digest(entry.get("sha256")):
        try:
            actual = sha256_file(path)
        except OSError as exc:
            errors.append(f"{label} cannot be read: {exc}")
        else:
            if actual != entry["sha256"]:
                errors.append(
                    f"{label} SHA-256 is stale: expected {entry['sha256']}, got {actual}"
                )
    return errors, path


def _selector_errors(selectors: Any, label: str) -> List[str]:
    if selectors is None:
        return []
    if not isinstance(selectors, Mapping):
        return [f"{label} must be an object"]
    errors: List[str] = []
    allowed = {
        "pages",
        "page_start",
        "page_end",
        "start_heading",
        "end_heading",
        "start_marker",
        "end_marker",
        "include_item_numbers",
        "comment_ids",
        "annotation_types",
        "min_words",
    }
    unknown = sorted(set(selectors) - allowed)
    if unknown:
        errors.append(f"{label} contains unsupported keys: {', '.join(unknown)}")
    for key in ("pages", "include_item_numbers"):
        value = selectors.get(key)
        if value is not None and (
            not isinstance(value, list)
            or not value
            or any(not isinstance(item, int) or item < 1 for item in value)
        ):
            errors.append(f"{label}.{key} must be a nonempty list of positive integers")
    if "comment_ids" in selectors:
        value = selectors["comment_ids"]
        if not isinstance(value, list) or not value or any(
            not isinstance(item, (str, int)) for item in value
        ):
            errors.append(f"{label}.comment_ids must be a nonempty string/integer list")
    if "annotation_types" in selectors:
        value = selectors["annotation_types"]
        if not isinstance(value, list) or not value or any(
            not isinstance(item, str) or not item.strip() for item in value
        ):
            errors.append(f"{label}.annotation_types must be a nonempty string list")
    for key in ("page_start", "page_end", "min_words"):
        value = selectors.get(key)
        if value is not None and (not isinstance(value, int) or value < 1):
            errors.append(f"{label}.{key} must be a positive integer")
    if (
        isinstance(selectors.get("page_start"), int)
        and isinstance(selectors.get("page_end"), int)
        and selectors["page_start"] > selectors["page_end"]
    ):
        errors.append(f"{label}.page_start cannot exceed page_end")
    for key in ("start_heading", "end_heading", "start_marker", "end_marker"):
        value = selectors.get(key)
        if value is not None and (not isinstance(value, str) or not value.strip()):
            errors.append(f"{label}.{key} must be a nonempty string")
    return errors


def load_review_manifest(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a private manifest and validate schema, file hashes, and permissions."""
    manifest_path = _assert_private_file(
        path,
        private_root=private_root,
        label="review manifest",
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read {manifest_path.name}: {exc}") from exc
    validate_review_manifest(manifest, base_dir=manifest_path.parent)
    return manifest


def is_review_manifest(path: str | Path) -> bool:
    """Return whether a JSON file declares the supported review-manifest version."""
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.suffix.lower() != ".json":
        return False
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return (
        isinstance(value, Mapping)
        and _normalize_manifest_version(value.get("manifest_version")) == MANIFEST_VERSION
        and isinstance(value.get("cases"), list)
    )


def _snapshot_integrity_payload(value: Mapping[str, Any]) -> Dict[str, Any]:
    """Return canonical snapshot content, excluding only echoed integrity fields."""
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"binding_hash", "corpus_fingerprint"}
    }
    audit = payload.get("audit")
    if isinstance(audit, Mapping):
        payload["audit"] = {
            key: item
            for key, item in audit.items()
            if key not in {"binding_hash", "corpus_fingerprint"}
        }
    return payload


def _snapshot_integrity_hash(value: Mapping[str, Any]) -> str:
    return _sha256_json(_snapshot_integrity_payload(value))


def _normalized_corpus_structure_errors(value: Any) -> List[str]:
    if not isinstance(value, Mapping):
        return ["normalized corpus root must be an object"]
    errors: List[str] = []
    if _normalize_manifest_version(value.get("manifest_version")) != MANIFEST_VERSION:
        errors.append("normalized corpus manifest_version must be 'v1'")
    if not isinstance(value.get("records"), list):
        errors.append("normalized corpus records must be a list")
    if not isinstance(value.get("issues"), list):
        errors.append("normalized corpus issues must be a list")
    if not isinstance(value.get("stats"), Mapping):
        errors.append("normalized corpus stats must be an object")
    if not isinstance(value.get("audit"), Mapping):
        errors.append("normalized corpus audit must be an object")
    return errors


def _normalized_corpus_errors(value: Any) -> List[str]:
    errors = _normalized_corpus_structure_errors(value)
    if not isinstance(value, Mapping):
        return errors
    binding_hash = value.get("binding_hash")
    corpus_fingerprint = value.get("corpus_fingerprint")
    if not _is_digest(binding_hash):
        errors.append("normalized corpus binding_hash must be a SHA-256 digest")
    if not _is_digest(corpus_fingerprint):
        errors.append("normalized corpus corpus_fingerprint must be a SHA-256 digest")
    if binding_hash != corpus_fingerprint:
        errors.append("normalized corpus binding_hash does not match corpus_fingerprint")
    audit = value.get("audit")
    if isinstance(audit, Mapping):
        if audit.get("corpus_fingerprint") != corpus_fingerprint:
            errors.append("normalized corpus audit fingerprint does not match")
        if audit.get("binding_hash") != binding_hash:
            errors.append("normalized corpus audit binding hash does not match")
    if isinstance(value, Mapping):
        try:
            computed = _snapshot_integrity_hash(value)
        except (TypeError, ValueError):
            errors.append("normalized corpus contains non-canonical JSON content")
        else:
            if corpus_fingerprint != computed:
                errors.append(
                    "normalized corpus integrity hash does not match canonical content"
                )
    return errors


def reseal_normalized_corpus(corpus: Mapping[str, Any]) -> Dict[str, Any]:
    """Recompute integrity echoes after trusted deterministic in-memory enrichment.

    This operation is intentionally separate from :func:`load_private_corpus`.
    Saved snapshots are always verified before they are returned and are never
    auto-resealed. Callers should use this only on a corpus built from a
    hash-validated manifest during the same process.
    """
    errors = _normalized_corpus_structure_errors(corpus)
    if errors:
        raise ManifestValidationError(errors)
    sealed = dict(corpus)
    sealed["audit"] = dict(corpus["audit"])
    sealed.pop("binding_hash", None)
    sealed.pop("corpus_fingerprint", None)
    sealed["audit"].pop("binding_hash", None)
    sealed["audit"].pop("corpus_fingerprint", None)
    fingerprint = _snapshot_integrity_hash(sealed)
    sealed["corpus_fingerprint"] = fingerprint
    sealed["binding_hash"] = fingerprint
    sealed["audit"]["corpus_fingerprint"] = fingerprint
    sealed["audit"]["binding_hash"] = fingerprint
    return sealed


def is_normalized_review_corpus(path: str | Path) -> bool:
    """Return whether *path* is a minimally valid normalized private snapshot."""
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.suffix.lower() != ".json":
        return False
    try:
        value = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return not _normalized_corpus_errors(value)


def load_private_corpus(
    path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Load a normalized snapshot, rejecting unsafe paths, modes, or tampering."""
    corpus_path = _assert_private_file(
        path,
        private_root=private_root,
        label="private corpus",
    )
    try:
        value = json.loads(corpus_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestValidationError(f"cannot read private corpus: {exc}") from exc
    errors = _normalized_corpus_errors(value)
    if errors:
        raise ManifestValidationError(errors)
    return dict(value)


def validate_review_manifest(
    manifest: Mapping[str, Any],
    *,
    base_dir: str | Path,
) -> None:
    """Fail closed on schema, uniqueness, local-materialization, and hash errors."""
    root = Path(base_dir).expanduser().resolve()
    errors: List[str] = []
    if not isinstance(manifest, Mapping):
        raise ManifestValidationError("manifest root must be an object")
    if _normalize_manifest_version(manifest.get("manifest_version")) != MANIFEST_VERSION:
        errors.append("manifest_version must be 'v1'")
    if not _is_identifier(manifest.get("corpus_id")):
        errors.append("corpus_id must be a lowercase, non-identifying stable identifier")
    cases = manifest.get("cases")
    if not isinstance(cases, list) or not cases:
        errors.append("cases must be a nonempty list")
        cases = []

    case_ids: set[str] = set()
    source_ids: set[str] = set()
    source_references: List[Tuple[str, List[str]]] = []

    for case_index, case in enumerate(cases):
        case_label = f"cases[{case_index}]"
        if not isinstance(case, Mapping):
            errors.append(f"{case_label} must be an object")
            continue
        family_id = case.get("family_id")
        case_id = case.get("case_id")
        tier = case.get("benchmark_tier")
        disposition = case.get("disposition", "evaluation")
        if not _is_identifier(family_id):
            errors.append(f"{case_label}.family_id is invalid")
        if not _is_identifier(case_id):
            errors.append(f"{case_label}.case_id is invalid")
        elif case_id in case_ids:
            errors.append(f"duplicate case_id: {case_id}")
        else:
            case_ids.add(case_id)
        if tier not in ALLOWED_BENCHMARK_TIERS:
            errors.append(
                f"{case_label}.benchmark_tier must be one of {sorted(ALLOWED_BENCHMARK_TIERS)}"
            )
        if disposition not in ALLOWED_DISPOSITIONS:
            errors.append(f"{case_label}.disposition is invalid: {disposition}")
        if "classification" in case and (
            not isinstance(case["classification"], str) or not case["classification"].strip()
        ):
            errors.append(f"{case_label}.classification must be a nonempty string")

        manuscript_files = case.get("manuscript_files")
        if not isinstance(manuscript_files, list) or not manuscript_files:
            errors.append(f"{case_label}.manuscript_files must be a nonempty ordered list")
            manuscript_files = []
        roles: set[str] = set()
        for file_index, file_entry in enumerate(manuscript_files):
            file_label = f"{case_label}.manuscript_files[{file_index}]"
            file_errors, _ = _validate_file_entry(
                file_entry,
                label=file_label,
                base_dir=root,
                allowed_suffixes=SUPPORTED_MANUSCRIPT_SUFFIXES,
            )
            errors.extend(file_errors)
            if isinstance(file_entry, Mapping):
                role = file_entry.get("role", "main" if file_index == 0 else "supplement")
                if not _is_identifier(role):
                    errors.append(f"{file_label}.role is invalid")
                elif role in roles:
                    errors.append(f"{case_label} has duplicate manuscript role: {role}")
                else:
                    roles.add(role)

        sources = case.get("sources")
        if not isinstance(sources, list) or not sources:
            errors.append(f"{case_label}.sources must be a nonempty list")
            sources = []
        eligible_primary = 0
        for source_index, source in enumerate(sources):
            source_label = f"{case_label}.sources[{source_index}]"
            if not isinstance(source, Mapping):
                errors.append(f"{source_label} must be an object")
                continue
            source_id = source.get("source_id")
            extractor = source.get("extractor")
            reviewer_id = source.get("reviewer_id")
            source_disposition = source.get("disposition", disposition)
            version_match = source.get("version_match")
            if not _is_identifier(source_id):
                errors.append(f"{source_label}.source_id is invalid")
            elif source_id in source_ids:
                errors.append(f"duplicate source_id: {source_id}")
            else:
                source_ids.add(source_id)
            if extractor not in ALLOWED_EXTRACTORS:
                errors.append(
                    f"{source_label}.extractor must be one of {sorted(ALLOWED_EXTRACTORS)}"
                )
            if not _is_identifier(reviewer_id):
                errors.append(
                    f"{source_label}.reviewer_id must be a pseudonymous stable identifier"
                )
            if "provenance" not in source:
                errors.append(f"{source_label}.provenance is required")
            elif not isinstance(source["provenance"], str) or not source["provenance"].strip():
                errors.append(f"{source_label}.provenance must be a nonempty string")
            if "source_type" not in source:
                errors.append(f"{source_label}.source_type is required")
            elif not isinstance(source["source_type"], str) or not source["source_type"].strip():
                errors.append(f"{source_label}.source_type must be a nonempty string")
            if source_disposition == "evaluation":
                source_labels = " ".join(
                    str(source.get(field, ""))
                    for field in ("provenance", "source_type")
                )
                source_tokens = set(re.findall(r"[a-z0-9]+", source_labels.casefold()))
                forbidden_tokens = sorted(
                    source_tokens & FORBIDDEN_EVALUATION_SOURCE_TOKENS
                )
                if forbidden_tokens:
                    errors.append(
                        f"{source_label} evaluation source is AI-generated, derivative, "
                        "or response material: " + ", ".join(forbidden_tokens)
                    )
            if "feedback_date" in source and (
                not isinstance(source["feedback_date"], str)
                or not re.fullmatch(r"\d{4}-\d{2}(?:-\d{2})?", source["feedback_date"])
            ):
                errors.append(f"{source_label}.feedback_date must be YYYY-MM or YYYY-MM-DD")
            if "duplicate_group" in source and not _is_identifier(source["duplicate_group"]):
                errors.append(f"{source_label}.duplicate_group is invalid")
            if "disposition" not in source:
                errors.append(f"{source_label}.disposition is required")
            if source_disposition not in ALLOWED_DISPOSITIONS:
                errors.append(f"{source_label}.disposition is invalid: {source_disposition}")
            if version_match not in ALLOWED_VERSION_MATCHES:
                errors.append(f"{source_label}.version_match is invalid: {version_match}")
            if tier == "primary" and source_disposition == "evaluation":
                eligible_primary += 1
                if version_match not in EXACT_VERSION_MATCHES:
                    errors.append(
                        f"{source_label} is primary evaluation feedback but is not version matched"
                    )
            errors.extend(_selector_errors(source.get("selectors"), f"{source_label}.selectors"))
            file_errors, source_path = _validate_file_entry(
                source,
                label=source_label,
                base_dir=root,
                allowed_suffixes=None,
            )
            errors.extend(file_errors)
            if source_path and extractor in ALLOWED_EXTRACTORS:
                expected_suffixes = {
                    "pdf_annotations": {".pdf"},
                    "pdf_text": {".pdf"},
                    "docx_comments": {".docx"},
                    "docx_body": {".docx"},
                    "markdown": {".md", ".markdown"},
                    "text": {".txt"},
                }[extractor]
                if source_path.suffix.lower() not in expected_suffixes:
                    errors.append(
                        f"{source_label} extractor {extractor} does not match {source_path.suffix}"
                    )
            supersedes = source.get("supersedes", [])
            if supersedes is None:
                supersedes = []
            if not isinstance(supersedes, list) or any(
                not _is_identifier(item) for item in supersedes
            ):
                errors.append(f"{source_label}.supersedes must be a list of source IDs")
            elif _is_identifier(source_id):
                source_references.append((source_id, supersedes))
        if tier == "primary" and disposition == "evaluation" and eligible_primary == 0:
            errors.append(f"{case_label} is primary but has no eligible evaluation feedback")

    for source_id, supersedes in source_references:
        for superseded in supersedes:
            if superseded == source_id:
                errors.append(f"source {source_id} cannot supersede itself")
            elif superseded not in source_ids:
                errors.append(f"source {source_id} supersedes unknown source {superseded}")

    if errors:
        raise ManifestValidationError(errors)


def _selected_page_numbers(page_count: int, selectors: Mapping[str, Any]) -> List[int]:
    if "pages" in selectors:
        selected = selectors["pages"]
    else:
        start = selectors.get("page_start", 1)
        end = selectors.get("page_end", page_count)
        selected = list(range(start, end + 1))
    invalid = [page for page in selected if page > page_count]
    if invalid:
        raise ManifestValidationError(
            f"page selector exceeds document length ({page_count}): {invalid}"
        )
    return selected


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def _apply_text_selectors(text: str, selectors: Mapping[str, Any]) -> str:
    selected = text
    lower = selected.casefold()
    start_value = selectors.get("start_heading") or selectors.get("start_marker")
    end_value = selectors.get("end_heading") or selectors.get("end_marker")
    if start_value:
        offset = lower.find(str(start_value).casefold())
        if offset < 0:
            raise ManifestValidationError(f"start selector not found: {start_value}")
        selected = selected[offset:]
        lower = selected.casefold()
    if end_value:
        offset = lower.find(str(end_value).casefold())
        if offset < 0:
            raise ManifestValidationError(f"end selector not found: {end_value}")
        selected = selected[:offset]
    return selected.strip()


def _numbered_item_blocks(text: str) -> List[Tuple[int, str]]:
    matches = list(_NUMBERED_ITEM_RE.finditer(text))
    blocks: List[Tuple[int, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        block = text[match.start():end].strip()
        if block:
            blocks.append((int(match.group(1)), block))
    return blocks


def _atomize_text(text: str, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    text = _apply_text_selectors(text, selectors)
    numbered = _numbered_item_blocks(text)
    requested_items = selectors.get("include_item_numbers")
    if requested_items is not None:
        available = {number for number, _ in numbered}
        missing = sorted(set(requested_items) - available)
        if missing:
            raise ManifestValidationError(f"numbered item selectors not found: {missing}")
        requested = set(requested_items)
        numbered = [(number, block) for number, block in numbered if number in requested]
    if numbered:
        return [
            {
                "text": _normalize_space(block),
                "locator": f"item:{number}",
                "anchor_text": "",
            }
            for number, block in numbered
            if _normalize_space(block)
        ]

    min_words = int(selectors.get("min_words", 4))
    paragraphs = re.split(r"\n\s*\n+", text)
    issues: List[Dict[str, str]] = []
    pending_heading = ""
    for paragraph in paragraphs:
        cleaned = _normalize_space(paragraph)
        if not cleaned:
            continue
        if _HEADING_RE.match(paragraph) and len(paragraph.splitlines()) == 1:
            pending_heading = re.sub(r"^\s*#{1,6}\s+", "", paragraph).strip()
            continue
        lines = paragraph.splitlines()
        bullet_starts = [index for index, line in enumerate(lines) if _BULLET_RE.match(line)]
        if bullet_starts:
            for bullet_index, start in enumerate(bullet_starts):
                end = bullet_starts[bullet_index + 1] if bullet_index + 1 < len(bullet_starts) else len(lines)
                bullet = _normalize_space("\n".join(lines[start:end]))
                if len(bullet.split()) >= min_words:
                    issues.append(
                        {
                            "text": bullet,
                            "locator": f"bullet:{len(issues) + 1}",
                            "anchor_text": pending_heading,
                        }
                    )
            pending_heading = ""
            continue
        if len(cleaned.split()) >= min_words:
            issues.append(
                {
                    "text": cleaned,
                    "locator": f"block:{len(issues) + 1}",
                    "anchor_text": pending_heading,
                }
            )
            pending_heading = ""
    return issues


def _pdf_text_by_page(path: Path, selectors: Mapping[str, Any]) -> List[Tuple[int, str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover - declared application dependency
        raise ManifestValidationError("PyMuPDF is required for PDF extraction") from exc
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ManifestValidationError(f"cannot open PDF source: {exc}") from exc
    try:
        pages = _selected_page_numbers(document.page_count, selectors)
        output: List[Tuple[int, str]] = []
        for number in pages:
            page = document[number - 1]
            blocks = page.get_text("blocks", sort=True)
            text = "\n\n".join(
                str(block[4]).strip()
                for block in blocks
                if len(block) >= 5 and str(block[4]).strip()
            )
            output.append((number, text))
        return output
    finally:
        document.close()


def _extract_pdf_text(path: Path, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    page_text = _pdf_text_by_page(path, selectors)
    joined = "\n\n".join(text for _, text in page_text)
    issues = _atomize_text(joined, selectors)
    page_label = ",".join(str(number) for number, _ in page_text)
    for issue in issues:
        issue["locator"] = f"pages:{page_label};{issue['locator']}"
    return issues


def _extract_pdf_annotations(path: Path, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    try:
        import fitz
    except ImportError as exc:  # pragma: no cover
        raise ManifestValidationError("PyMuPDF is required for PDF extraction") from exc
    try:
        document = fitz.open(path)
    except Exception as exc:
        raise ManifestValidationError(f"cannot open annotated PDF: {exc}") from exc
    output: List[Dict[str, str]] = []
    try:
        pages = _selected_page_numbers(document.page_count, selectors)
        type_filter = {item.casefold() for item in selectors.get("annotation_types", [])}
        for page_number in pages:
            page = document[page_number - 1]
            annotations = page.annots()
            if annotations is None:
                continue
            for annotation_index, annotation in enumerate(annotations, start=1):
                annotation_type = str(annotation.type[1] or annotation.type[0])
                if type_filter and annotation_type.casefold() not in type_filter:
                    continue
                content = _normalize_space((annotation.info or {}).get("content", ""))
                if not content:
                    continue
                anchor = _normalize_space(page.get_text("text", clip=annotation.rect))
                output.append(
                    {
                        "text": content,
                        "locator": f"page:{page_number};annotation:{annotation_index}",
                        "anchor_text": anchor,
                        "annotation_type": annotation_type,
                    }
                )
    finally:
        document.close()
    return output


def _docx_xml(path: Path, member: str) -> ET.Element:
    try:
        with zipfile.ZipFile(path) as archive:
            payload = archive.read(member)
    except KeyError as exc:
        raise ManifestValidationError(f"DOCX is missing {member}") from exc
    except (OSError, zipfile.BadZipFile) as exc:
        raise ManifestValidationError(f"cannot read DOCX source: {exc}") from exc
    try:
        return ET.fromstring(payload)
    except ET.ParseError as exc:
        raise ManifestValidationError(f"malformed DOCX XML in {member}: {exc}") from exc


def _element_text(element: ET.Element) -> str:
    parts: List[str] = []
    for node in element.iter():
        if node.tag == f"{_W}t" and node.text:
            parts.append(node.text)
        elif node.tag in {f"{_W}tab", f"{_W}br", f"{_W}cr"}:
            parts.append("\n")
    return _normalize_space(" ".join(parts))


def _docx_paragraphs(path: Path) -> List[str]:
    document = _docx_xml(path, "word/document.xml")
    return [
        text
        for paragraph in document.iter(f"{_W}p")
        if (text := _element_text(paragraph))
    ]


def _extract_docx_body(path: Path, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    return _atomize_text("\n\n".join(_docx_paragraphs(path)), selectors)


def _docx_comment_anchors(path: Path) -> Dict[str, str]:
    document = _docx_xml(path, "word/document.xml")
    anchors: Dict[str, List[str]] = {}
    active: List[str] = []
    for node in document.iter():
        if node.tag == f"{_W}commentRangeStart":
            comment_id = node.attrib.get(f"{_W}id", "")
            if comment_id:
                active.append(comment_id)
                anchors.setdefault(comment_id, [])
        elif node.tag == f"{_W}commentRangeEnd":
            comment_id = node.attrib.get(f"{_W}id", "")
            if comment_id in active:
                active.remove(comment_id)
        elif node.tag == f"{_W}t" and node.text:
            for comment_id in active:
                anchors.setdefault(comment_id, []).append(node.text)
    return {
        comment_id: _normalize_space(" ".join(parts))
        for comment_id, parts in anchors.items()
    }


def _extract_docx_comments(path: Path, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    comments = _docx_xml(path, "word/comments.xml")
    anchors = _docx_comment_anchors(path)
    requested = {str(value) for value in selectors.get("comment_ids", [])}
    output: List[Dict[str, str]] = []
    found: set[str] = set()
    for comment in comments.iter(f"{_W}comment"):
        comment_id = comment.attrib.get(f"{_W}id", "")
        if requested and comment_id not in requested:
            continue
        found.add(comment_id)
        content = _element_text(comment)
        if not content:
            continue
        output.append(
            {
                "text": content,
                "locator": f"comment:{comment_id}",
                "anchor_text": anchors.get(comment_id, ""),
                "annotation_type": "DOCX comment",
            }
        )
    missing = sorted(requested - found)
    if missing:
        raise ManifestValidationError(f"DOCX comment selectors not found: {missing}")
    return output


def _extract_text_file(path: Path, selectors: Mapping[str, Any]) -> List[Dict[str, str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise ManifestValidationError(f"text source is not UTF-8: {path.name}") from exc
    return _atomize_text(text, selectors)


def extract_feedback_source(
    path: str | Path,
    extractor: str,
    selectors: Mapping[str, Any] | None = None,
) -> List[Dict[str, str]]:
    """Extract atomic feedback units from one already-validated local source."""
    source_path = Path(path)
    selection = dict(selectors or {})
    functions = {
        "pdf_annotations": _extract_pdf_annotations,
        "pdf_text": _extract_pdf_text,
        "docx_comments": _extract_docx_comments,
        "docx_body": _extract_docx_body,
        "markdown": _extract_text_file,
        "text": _extract_text_file,
    }
    if extractor not in functions:
        raise ManifestValidationError(f"unsupported extractor: {extractor}")
    issues = functions[extractor](source_path, selection)
    min_words = int(selection.get("min_words", 1 if extractor in {"pdf_annotations", "docx_comments"} else 4))
    return [issue for issue in issues if len(issue.get("text", "").split()) >= min_words]


def extract_manuscript_text(path: str | Path) -> str:
    """Extract manuscript text without importing PDF annotation comments."""
    manuscript_path = Path(path)
    suffix = manuscript_path.suffix.lower()
    if suffix == ".pdf":
        return "\n\n".join(text for _, text in _pdf_text_by_page(manuscript_path, {}))
    if suffix == ".docx":
        return "\n\n".join(_docx_paragraphs(manuscript_path))
    if suffix in {".md", ".markdown", ".txt"}:
        try:
            return manuscript_path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise ManifestValidationError(
                f"manuscript text is not UTF-8: {manuscript_path.name}"
            ) from exc
    raise ManifestValidationError(f"unsupported manuscript file type: {suffix}")


def extract_ordered_manuscript_bundle(
    manuscript_files: Sequence[Mapping[str, Any]],
    *,
    base_dir: str | Path,
) -> str:
    """Extract an ordered main-manuscript/supplement bundle for evaluation."""
    root = Path(base_dir).expanduser().resolve()
    chunks: List[str] = []
    for index, file_entry in enumerate(manuscript_files):
        path = _expand_path(str(file_entry["path"]), root)
        role = str(file_entry.get("role", "main" if index == 0 else "supplement"))
        chunks.append(f"\n\n[[MANUSCRIPT_FILE:{index + 1};ROLE:{role}]]\n\n")
        chunks.append(extract_manuscript_text(path))
    return "".join(chunks).strip()


def _source_fingerprint(source: Mapping[str, Any]) -> str:
    return _sha256_json(
        {
            "importer_version": IMPORTER_VERSION,
            "source_sha256": source["sha256"],
            "extractor": source["extractor"],
            "selectors": source.get("selectors", {}),
        }
    )


def _record_for_source(
    *,
    case: Mapping[str, Any],
    source: Mapping[str, Any],
    manuscript_paths: Sequence[Path],
    extracted: Sequence[Mapping[str, str]],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    family_id = str(case["family_id"])
    case_id = str(case["case_id"])
    paper_id = _case_paper_id(family_id, case_id)
    source_id = str(source["source_id"])
    reviewer_id = str(source["reviewer_id"])
    fingerprint = _source_fingerprint(source)
    manuscript_files = [str(path) for path in manuscript_paths]
    manuscript_hashes = [str(entry["sha256"]) for entry in case["manuscript_files"]]
    source_type = str(source.get("source_type", source.get("provenance", "human_feedback")))
    sections = [{"reviewer_id": reviewer_id, "text": item["text"]} for item in extracted]
    record: Dict[str, Any] = {
        "paper_id": paper_id,
        "family_id": family_id,
        "manuscript_family_id": family_id,
        "manuscript_version_id": case_id,
        "case_id": case_id,
        "manifest_case": True,
        "classification": str(case.get("classification", "")),
        "source_id": source_id,
        "review_file": source_id,
        "title": "",
        "journal": str(case.get("journal", "")),
        "manuscript": "",
        "date": str(source.get("feedback_date", case.get("feedback_date", ""))),
        "decision": str(case.get("decision", "")),
        "source": str(source.get("provenance", "human_feedback")),
        "source_kind": MANIFEST_SOURCE_KIND,
        "sections": sections,
        "raw_text": "\n\n".join(item["text"] for item in extracted),
        "matched_paper_files": manuscript_files,
        "manuscript_files": manuscript_files,
        "manuscript_hashes": manuscript_hashes,
        "match_status": str(source["version_match"]),
        "quality_flag": "use",
        "benchmark_tier": str(case["benchmark_tier"]),
        "disposition": str(source.get("disposition", case.get("disposition", "evaluation"))),
        "source_sha256": str(source["sha256"]),
        "source_hash": str(source["sha256"]),
        "source_type": source_type,
        "extraction_fingerprint": fingerprint,
    }
    issues: List[Dict[str, Any]] = []
    for index, extracted_issue in enumerate(extracted, start=1):
        text = extracted_issue["text"]
        issue_hash = hashlib.sha256(_normalize_space(text).casefold().encode("utf-8")).hexdigest()
        issue: Dict[str, Any] = {
            "paper_id": paper_id,
            "family_id": family_id,
            "manuscript_family_id": family_id,
            "manuscript_version_id": case_id,
            "case_id": case_id,
            "manifest_case": True,
            "source_id": source_id,
            "review_file": source_id,
            "journal": str(case.get("journal", "")),
            "decision": str(case.get("decision", "")),
            "review_round": str(case.get("review_round", "")),
            "reviewer_id": reviewer_id,
            "atomic_issue_id": f"{case_id}_{source_id}_{index:03d}",
            "issue_text": text,
            "issue_type": str(source.get("issue_type", "unclassified")),
            "decision_tier": str(source.get("decision_tier", "unadjudicated")),
            "action_requested": "",
            "tone": "",
            "paper_section": "",
            "reviewer_confidence": "",
            "design_type": str(case.get("design_type", "unclear")),
            "source_kind": MANIFEST_SOURCE_KIND,
            "matched_paper_files": manuscript_files,
            "manuscript_files": manuscript_files,
            "manuscript_hashes": manuscript_hashes,
            "match_status": str(source["version_match"]),
            "quality_flag": "use",
            "benchmark_tier": str(case["benchmark_tier"]),
            "disposition": record["disposition"],
            "provenance": str(source.get("provenance", "human_feedback")),
            "source_locator": extracted_issue.get("locator", ""),
            "anchor_text": extracted_issue.get("anchor_text", ""),
            "annotation_type": extracted_issue.get("annotation_type", ""),
            "version_match": str(source["version_match"]),
            "source_sha256": str(source["sha256"]),
            "source_hash": str(source["sha256"]),
            "source_type": source_type,
            "extraction_fingerprint": fingerprint,
            "issue_hash": issue_hash,
            "cluster_id": "",
            "match_confidence": "unreviewed",
            "adjudication_status": "unreviewed",
        }
        issues.append(issue)
    return record, issues


def build_review_corpus_from_manifest(
    manifest_or_path: Mapping[str, Any] | str | Path,
    *,
    base_dir: str | Path | None = None,
    include_deferred: bool = False,
    private_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Validate and import a private manifest into a normalized corpus.

    The returned corpus contains private issue text and resolved manuscript paths.
    Store it only with :func:`write_private_corpus`.  ``audit`` is path-free and
    suitable for portable aggregate reporting.
    """
    if isinstance(manifest_or_path, Mapping):
        if "records" in manifest_or_path or "issues" in manifest_or_path:
            errors = _normalized_corpus_errors(manifest_or_path)
            if errors:
                raise ManifestValidationError(errors)
            return dict(manifest_or_path)
        if base_dir is None:
            raise ManifestValidationError("base_dir is required for an in-memory manifest")
        manifest = dict(manifest_or_path)
        root = Path(base_dir).expanduser().resolve()
        validate_review_manifest(manifest, base_dir=root)
    else:
        manifest_path = _assert_private_file(
            manifest_or_path,
            private_root=private_root,
            label="review corpus input",
        )
        try:
            candidate = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            candidate = None
        if isinstance(candidate, Mapping) and (
            "records" in candidate or "issues" in candidate
        ):
            return load_private_corpus(
                manifest_path,
                private_root=private_root,
            )
        manifest = load_review_manifest(manifest_path, private_root=private_root)
        root = manifest_path.parent

    corpus_id = str(manifest["corpus_id"])
    records: List[Dict[str, Any]] = []
    issues: List[Dict[str, Any]] = []
    source_audit: List[Dict[str, Any]] = []
    duplicate_sources: List[Dict[str, str]] = []
    superseded = {
        superseded_id
        for case in manifest["cases"]
        for source in case["sources"]
        for superseded_id in source.get("supersedes", [])
    }
    seen_source_signatures: Dict[str, str] = {}
    seen_issue_keys: Dict[Tuple[str, str, str], str] = {}
    eligible_issues_by_primary_case: Counter[str] = Counter()

    for case in manifest["cases"]:
        case_disposition = case.get("disposition", "evaluation")
        manuscript_paths = [
            _expand_path(str(entry["path"]), root) for entry in case["manuscript_files"]
        ]
        for source in case["sources"]:
            source_id = str(source["source_id"])
            disposition = str(source.get("disposition", case_disposition))
            if source_id in superseded:
                source_audit.append(
                    {
                        "source_id": source_id,
                        "case_id": case["case_id"],
                        "status": "superseded",
                        "source_sha256": source["sha256"],
                        "issue_count": 0,
                    }
                )
                continue
            if disposition not in IMPORTABLE_DISPOSITIONS and not (
                include_deferred and disposition == "deferred"
            ):
                source_audit.append(
                    {
                        "source_id": source_id,
                        "case_id": case["case_id"],
                        "status": f"not_imported:{disposition}",
                        "source_sha256": source["sha256"],
                        "issue_count": 0,
                    }
                )
                continue
            source_path = _expand_path(str(source["path"]), root)
            signature = _sha256_json(
                {
                    "source_sha256": source["sha256"],
                    "extractor": source["extractor"],
                    "selectors": source.get("selectors", {}),
                    # Preserve genuinely independent reviewers even in the
                    # unlikely event that they submit identical bytes.  A
                    # declared duplicate_group intentionally overrides this.
                    "reviewer_key": source.get("duplicate_group", source["reviewer_id"]),
                }
            )
            if signature in seen_source_signatures:
                kept_id = seen_source_signatures[signature]
                duplicate_sources.append({"source_id": source_id, "kept_source_id": kept_id})
                source_audit.append(
                    {
                        "source_id": source_id,
                        "case_id": case["case_id"],
                        "status": "duplicate_source",
                        "source_sha256": source["sha256"],
                        "duplicate_of": kept_id,
                        "issue_count": 0,
                    }
                )
                continue
            seen_source_signatures[signature] = source_id
            extracted = extract_feedback_source(
                source_path,
                str(source["extractor"]),
                source.get("selectors", {}),
            )
            record, source_issues = _record_for_source(
                case=case,
                source=source,
                manuscript_paths=manuscript_paths,
                extracted=extracted,
            )
            duplicate_group = str(source.get("duplicate_group", ""))
            kept_issues: List[Dict[str, Any]] = []
            for issue in source_issues:
                reviewer_key = duplicate_group or str(issue["reviewer_id"])
                key = (
                    str(case["case_id"]),
                    reviewer_key,
                    str(issue["issue_hash"]),
                )
                if key in seen_issue_keys:
                    continue
                seen_issue_keys[key] = str(issue["atomic_issue_id"])
                kept_issues.append(issue)
            if kept_issues:
                record["sections"] = [
                    {"reviewer_id": issue["reviewer_id"], "text": issue["issue_text"]}
                    for issue in kept_issues
                ]
                record["raw_text"] = "\n\n".join(issue["issue_text"] for issue in kept_issues)
                records.append(record)
                issues.extend(kept_issues)
                if case["benchmark_tier"] == "primary" and disposition == "evaluation":
                    eligible_issues_by_primary_case[str(case["case_id"])] += len(kept_issues)
            source_audit.append(
                {
                    "source_id": source_id,
                    "case_id": case["case_id"],
                    "status": "imported" if kept_issues else "empty_after_extraction",
                    "source_sha256": source["sha256"],
                    "extraction_fingerprint": _source_fingerprint(source),
                    "extracted_issue_count": len(source_issues),
                    "issue_count": len(kept_issues),
                }
            )

    empty_primary = [
        str(case["case_id"])
        for case in manifest["cases"]
        if case["benchmark_tier"] == "primary"
        and case.get("disposition", "evaluation") == "evaluation"
        and eligible_issues_by_primary_case[str(case["case_id"])] == 0
    ]
    if empty_primary:
        raise ManifestValidationError(
            f"primary cases produced no eligible human feedback: {', '.join(empty_primary)}"
        )

    manifest_fingerprint_payload = {
        "manifest_version": MANIFEST_VERSION,
        "importer_version": IMPORTER_VERSION,
        "corpus_id": corpus_id,
        "cases": [
            {
                "family_id": case["family_id"],
                "case_id": case["case_id"],
                "benchmark_tier": case["benchmark_tier"],
                "manuscripts": [entry["sha256"] for entry in case["manuscript_files"]],
                "sources": [
                    {
                        "source_id": source["source_id"],
                        "sha256": source["sha256"],
                        "fingerprint": _source_fingerprint(source),
                        "disposition": source.get(
                            "disposition", case.get("disposition", "evaluation")
                        ),
                    }
                    for source in case["sources"]
                ],
            }
            for case in manifest["cases"]
        ],
    }
    manifest_binding_hash = _sha256_json(manifest_fingerprint_payload)
    tier_rank = {"primary": 0, "secondary": 1, "deferred": 2}
    family_tiers: Dict[str, str] = {}
    for case in manifest["cases"]:
        family_id = str(case["family_id"])
        tier = str(case["benchmark_tier"])
        previous = family_tiers.get(family_id)
        if previous is None or tier_rank[tier] < tier_rank[previous]:
            family_tiers[family_id] = tier
    stats = {
        "records": len(records),
        "issues": len(issues),
        "cases": len(manifest["cases"]),
        "families": len({case["family_id"] for case in manifest["cases"]}),
        "primary_cases": sum(
            case["benchmark_tier"] == "primary" for case in manifest["cases"]
        ),
        "secondary_cases": sum(
            case["benchmark_tier"] == "secondary" for case in manifest["cases"]
        ),
        "duplicate_sources": len(duplicate_sources),
        "records_with_papers": sum(bool(record["matched_paper_files"]) for record in records),
        "matched_pdf_files": len(
            {
                path
                for record in records
                for path in record["matched_paper_files"]
                if Path(path).suffix.lower() == ".pdf"
            }
        ),
        "source_kind": MANIFEST_SOURCE_KIND,
        "excluded_low_confidence_records": 0,
        "raw_review_records": 0,
    }
    audit = {
        "corpus_id": corpus_id,
        "manifest_version": MANIFEST_VERSION,
        "importer_version": IMPORTER_VERSION,
        "manifest_binding_hash": manifest_binding_hash,
        "case_ids": [case["case_id"] for case in manifest["cases"]],
        "family_tiers": family_tiers,
        "source_audit": source_audit,
        "duplicate_sources": duplicate_sources,
        "stats": stats,
    }
    corpus = {
        "corpus_id": corpus_id,
        "manifest_version": MANIFEST_VERSION,
        "importer_version": IMPORTER_VERSION,
        "records": records,
        "issues": issues,
        "paper_matches": {
            record["review_file"]: {
                "matched_paper_files": list(record["matched_paper_files"]),
                "match_status": record["match_status"],
            }
            for record in records
        },
        "stats": stats,
        "audit": audit,
        "excluded_records": [
            item["source_id"]
            for item in source_audit
            if item["status"] != "imported"
        ],
    }
    return reseal_normalized_corpus(corpus)


def write_private_corpus(
    corpus: Mapping[str, Any],
    output_path: str | Path,
    *,
    private_root: str | Path | None = None,
) -> Path:
    """Atomically write corpus JSON below a mode-0700 root as a mode-0600 file."""
    target, root = _path_below_private_root(
        output_path,
        private_root=private_root,
        label="private corpus output",
    )
    if "records" in corpus or "issues" in corpus:
        errors = _normalized_corpus_errors(corpus)
        if errors:
            raise ManifestValidationError(errors)
    _ensure_private_directory(target.parent, root)
    payload = json.dumps(corpus, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{target.name}.",
        suffix=".tmp",
        dir=target.parent,
        text=True,
    )
    temporary_path = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, target)
        directory_descriptor = os.open(target.parent, os.O_RDONLY)
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
    return target


__all__ = [
    "ALLOWED_BENCHMARK_TIERS",
    "ALLOWED_DISPOSITIONS",
    "ALLOWED_EXTRACTORS",
    "DEFAULT_PRIVATE_ROOT",
    "IMPORTER_VERSION",
    "MANIFEST_SOURCE_KIND",
    "MANIFEST_VERSION",
    "ManifestValidationError",
    "build_review_corpus_from_manifest",
    "extract_feedback_source",
    "extract_manuscript_text",
    "extract_ordered_manuscript_bundle",
    "is_normalized_review_corpus",
    "is_review_manifest",
    "load_private_corpus",
    "load_review_manifest",
    "reseal_normalized_corpus",
    "sha256_file",
    "validate_review_manifest",
    "write_private_corpus",
]
