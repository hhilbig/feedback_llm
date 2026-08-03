"""Local Streamlit editor for private feedback adjudication.

Run with:

    streamlit run adjudication_app.py --server.address 127.0.0.1 \
        --browser.gatherUsageStats false

The app has no OpenAI dependency and accepts only adjudication files below the
configured private root (``~/.feedback_llm`` by default).
"""

from __future__ import annotations

import html
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import streamlit as st

from review_adjudication import (
    eligible_duplicate_clusters,
    generated_adjudication_progress,
    generated_row_requirements,
    gold_adjudication_progress,
    gold_row_requirements,
    load_generated_editor_state,
    load_gold_editor_state,
    save_generated_editor_rows,
    save_gold_editor_rows,
)


TIER_LABELS = {
    "major": "Major concern",
    "minor": "Minor concern",
    "exclude": "Exclude",
}
INCLUDE_LABELS = {
    "yes": "Yes, keep it",
    "no": "No, exclude it",
}
SEVERITY_LABELS = {
    "potential_rejection_reason": "Potential rejection reason",
    "major_revision_issue": "Major revision issue",
    "minor_revision_issue": "Minor revision issue",
    "nice_to_have": "Nice to have",
}
SUPPORT_LABELS = {
    "supported": "Supported",
    "partially_supported": "Partially supported",
    "unsupported": "Unsupported",
    "unclear": "Unclear",
}
CORRECTNESS_LABELS = {
    "correct": "Correct",
    "incorrect": "Incorrect",
    "unclear": "Unclear",
}
SIGNIFICANCE_LABELS = {
    "significant": "Significant",
    "minor": "Minor",
    "not_significant": "Not significant",
    "unclear": "Unclear",
}
EVIDENCE_LABELS = {
    "sufficient": "Sufficient",
    "partial": "Partial",
    "insufficient": "Insufficient",
    "unclear": "Unclear",
}
MATCH_LABELS = {
    "matched": "Matched",
    "unmatched": "No adjudicated match",
    "unclear": "Unclear",
}
DUPLICATE_LABELS = {
    "unique": "Unique",
    "duplicate": "Duplicate",
    "unclear": "Unclear",
}
NOVELTY_LABELS = {
    "yes": "Yes",
    "no": "No",
}
QUEUE_LABELS = (
    "Full labels",
    "Tier screen",
    "All clusters",
)
GROUP_LABELS = (
    "All selected rows",
    "Provisional majors",
    "Sampled minors",
    "Other minors",
)


def _private_root() -> Path:
    configured = os.getenv("FEEDBACK_LLM_PRIVATE_ROOT")
    return Path(configured or Path.home() / ".feedback_llm").expanduser().resolve()


def _default_csv_path(private_root: Path) -> Path:
    configured = os.getenv("FEEDBACK_LLM_ADJUDICATION_PATH")
    if configured:
        return Path(configured).expanduser()
    preferred = (
        private_root
        / "private_feedback_pilot_v1"
        / "adjudication"
        / "final"
        / "gold_adjudication.csv"
    )
    if preferred.exists():
        return preferred
    candidates = sorted(
        private_root.glob("**/gold_adjudication.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else preferred


def _default_generated_csv_path(private_root: Path) -> Path:
    configured = os.getenv("FEEDBACK_LLM_GENERATED_ADJUDICATION_PATH")
    if configured:
        return Path(configured).expanduser()
    preferred = (
        private_root
        / "private_feedback_pilot_v1"
        / "adjudication"
        / "final"
        / "generated_adjudication.csv"
    )
    if preferred.exists():
        return preferred
    candidates = sorted(
        private_root.glob("**/generated_adjudication.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else preferred


def _binding_source(csv_path: str | Path, private_root: Path) -> Path | None:
    """Find the nearest private manifest, falling back to a normalized snapshot."""
    private_root = private_root.expanduser().resolve()
    path = Path(csv_path).expanduser().resolve()
    fallback = None
    for parent in (path.parent, *path.parents):
        if parent != private_root and private_root not in parent.parents:
            break
        manifest = parent / "review_manifest.json"
        if manifest.is_file():
            return manifest
        corpus = parent / "corpus.json"
        if fallback is None and corpus.is_file():
            fallback = corpus
        if parent == private_root:
            break
    return fallback


def _check_current_binding(
    csv_path: str | Path,
    private_root: Path,
) -> dict[str, Any]:
    """Rebuild current clusters from the nearest corpus source and check the packet."""
    source = _binding_source(csv_path, private_root)
    if source is None:
        return {
            "available": False,
            "verified": False,
            "save_allowed": True,
            "status": "unverified",
            "errors": [],
        }
    try:
        from feedback_pipeline import _load_current_gold_adjudication, load_review_corpus

        corpus = load_review_corpus(source, private_root=private_root)
        current = _load_current_gold_adjudication(
            corpus,
            csv_path,
            private_root=private_root,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        return {
            "available": True,
            "verified": False,
            "save_allowed": False,
            "status": "stale",
            "errors": [str(exc)],
        }
    binding_ok = current["status"] != "stale"
    return {
        "available": True,
        "verified": binding_ok,
        "save_allowed": binding_ok,
        "status": current["status"],
        "errors": list(current.get("errors", [])),
    }


def _private_json(path: str | Path, private_root: Path) -> dict[str, Any]:
    candidate = Path(path).expanduser().resolve()
    root = private_root.expanduser().resolve()
    if candidate == root or root not in candidate.parents:
        raise ValueError(f"Private metadata must stay below {root}")
    mode = candidate.stat().st_mode & 0o777
    if mode != 0o600:
        raise ValueError(
            f"Private metadata permissions must be 0600, got {oct(mode)}"
        )
    value = json.loads(candidate.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("Private metadata must contain a JSON object")
    return value


def _file_signature(path: str | Path | None) -> tuple[str, int, int] | None:
    if path is None:
        return None
    candidate = Path(path).expanduser().resolve()
    if not candidate.is_file():
        return None
    file_stat = candidate.stat()
    return (str(candidate), file_stat.st_mtime_ns, file_stat.st_size)


def _check_current_generated_binding(
    csv_path: str | Path,
    private_root: Path,
) -> dict[str, Any]:
    """Verify generated rows against the run audit and current projected gold."""
    path = Path(csv_path).expanduser().resolve()
    audit_path = path.parent / "baseline_run.local_audit.json"
    gold_path = path.parent / "gold_adjudication.csv"
    source = _binding_source(gold_path, private_root)
    if source is None:
        return {
            "available": False,
            "verified": False,
            "save_allowed": False,
            "status": "unverified",
            "errors": ["Cannot find the private manifest or corpus snapshot."],
        }
    try:
        from feedback_pipeline import (
            _gold_adjudication_for_evaluation,
            _load_current_gold_adjudication,
            _manifest_baseline_audit_errors,
            _review_eval_generated_binding_context,
            load_review_corpus,
        )

        audit = _private_json(audit_path, private_root)
        corpus = load_review_corpus(source, private_root=private_root)
        audit_errors = _manifest_baseline_audit_errors(corpus, audit)
        if audit_errors:
            raise ValueError("; ".join(audit_errors))
        run_metadata = audit.get("run_metadata", {})
        gold_mode = _clean(run_metadata.get("gold_mode") or "complete").lower()
        raw_gold = _load_current_gold_adjudication(
            corpus,
            gold_path,
            private_root=private_root,
        )
        gold = _gold_adjudication_for_evaluation(raw_gold, gold_mode)
        if gold.get("status") != "ready":
            raise ValueError(
                "The current gold packet is not ready under the recorded gold mode."
            )
        recorded_gold_binding = _clean(run_metadata.get("gold_binding_hash"))
        if recorded_gold_binding != _clean(gold.get("binding_hash")):
            raise ValueError(
                "The gold labels changed after generation; regenerate the baseline packet."
            )
        expected_binding = _clean(
            audit.get("generated_adjudication", {}).get("binding_hash")
        )
        if not expected_binding:
            raise ValueError("The run audit has no canonical generated-packet binding.")
        gold_cluster_families = {
            _clean(row.get("cluster_id")): _clean(row.get("family_id"))
            for row in gold.get("rows", [])
        }
        top_k = int(run_metadata.get("top_k", 5) or 5)
        run_context = _review_eval_generated_binding_context(
            corpus,
            run_metadata,
            gold_mode=gold_mode,
        )
        editor = load_generated_editor_state(
            path,
            expected_binding_hash=expected_binding,
            expected_gold_binding_hash=gold.get("binding_hash"),
            gold_cluster_families=gold_cluster_families,
            run_binding_context=run_context,
            top_k=top_k,
            private_root=private_root,
        )
        validation = editor["validation"]
        if validation.get("status") in {"invalid", "stale"}:
            raise ValueError(
                "; ".join(validation.get("errors", []))
                or "The generated packet is invalid or stale."
            )
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        return {
            "available": True,
            "verified": False,
            "save_allowed": False,
            "status": "stale",
            "errors": [str(exc)],
        }
    return {
        "available": True,
        "verified": True,
        "save_allowed": True,
        "status": validation.get("status", "pending_human_adjudication"),
        "errors": [],
        "editor": editor,
        "expected_binding_hash": expected_binding,
        "expected_gold_binding_hash": gold.get("binding_hash"),
        "gold_cluster_families": gold_cluster_families,
        "gold_rows": [dict(row) for row in gold.get("rows", [])],
        "run_binding_context": run_context,
        "top_k": top_k,
    }


def _clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def _family_label(value: Any) -> str:
    return _clean(value).replace("_", " ").replace("-", " ").title()


def _json_list(value: Any) -> list[Any]:
    try:
        parsed = json.loads(str(value or "[]"))
    except json.JSONDecodeError:
        return []
    return parsed if isinstance(parsed, list) else []


def _row_states(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    family_by_cluster = {
        _clean(row.get("cluster_id")): _clean(row.get("family_id")) for row in rows
    }
    return {
        _clean(row.get("cluster_id")): gold_row_requirements(
            row,
            family_by_cluster=family_by_cluster,
        )
        for row in rows
    }


def _filter_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    queue: str,
    family: str,
    group: str,
    unfinished_only: bool,
) -> list[dict[str, Any]]:
    states = _row_states(rows)
    selected: list[dict[str, Any]] = []
    for original in rows:
        row = dict(original)
        cluster_id = _clean(row.get("cluster_id"))
        state = states[cluster_id]
        if queue == "Full labels" and not state["full_required"]:
            continue
        if queue == "Tier screen" and state["full_required"]:
            promoted_major = (
                _clean(row.get("full_adjudication_required")) != "yes"
                and state["tier"] == "major"
            )
            if not promoted_major:
                continue
        if family != "All families" and _clean(row.get("family_id")) != family:
            continue
        if group == "Provisional majors" and _clean(row.get("provisional_tier")) != "major":
            continue
        if group == "Sampled minors" and _clean(row.get("sampled_minor")) != "yes":
            continue
        if group == "Other minors" and (
            _clean(row.get("provisional_tier")) != "minor"
            or _clean(row.get("sampled_minor")) == "yes"
        ):
            continue
        if unfinished_only and state["complete"]:
            continue
        selected.append(row)
    return selected


def _build_row_edit(
    row: Mapping[str, Any],
    *,
    tier: str,
    include: str,
    canonical_issue: str,
    severity: str,
    evidentiary_support: str,
    duplicate_cluster_ids: Sequence[str],
    exclusion_reason: str,
    adjudicator_notes: str,
) -> dict[str, Any]:
    """Apply one form submission and clear fields that no longer apply."""
    updated = dict(row)
    updated.update(
        {
            "tier_screen": tier,
            "include": include,
            "canonical_issue": canonical_issue.strip(),
            "severity": severity,
            "evidentiary_support": evidentiary_support,
            "duplicate_cluster_ids": json.dumps(
                sorted(set(map(str, duplicate_cluster_ids))), ensure_ascii=False
            ),
            "exclusion_reason": exclusion_reason.strip(),
            "adjudicator_notes": adjudicator_notes.strip(),
        }
    )
    full_required = (
        _clean(updated.get("full_adjudication_required")).lower() == "yes"
        or tier == "major"
    )
    if tier == "exclude":
        updated.update(
            {
                "include": "no",
                "canonical_issue": "",
                "severity": "",
                "evidentiary_support": "",
                "duplicate_cluster_ids": "[]",
            }
        )
    elif not full_required:
        updated.update(
            {
                "include": "",
                "canonical_issue": "",
                "severity": "",
                "evidentiary_support": "",
                "duplicate_cluster_ids": "[]",
                "exclusion_reason": "",
            }
        )
    elif include == "no":
        updated.update(
            {
                "canonical_issue": "",
                "severity": "",
                "evidentiary_support": "",
                "duplicate_cluster_ids": "[]",
            }
        )
    elif include == "yes":
        updated["exclusion_reason"] = ""
    return updated


def _generated_row_states(
    rows: Sequence[Mapping[str, Any]],
    *,
    gold_cluster_families: Mapping[str, str],
    top_k: int,
) -> dict[str, dict[str, Any]]:
    return {
        _clean(row.get("generated_issue_id")): generated_row_requirements(
            row,
            generated_rows=rows,
            gold_cluster_families=gold_cluster_families,
            top_k=top_k,
        )
        for row in rows
    }


def _filter_generated_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    states: Mapping[str, Mapping[str, Any]],
    family: str,
    unfinished_only: bool,
) -> list[dict[str, Any]]:
    selected = []
    for original in rows:
        row = dict(original)
        issue_id = _clean(row.get("generated_issue_id"))
        if family != "All families" and _clean(row.get("family_id")) != family:
            continue
        if unfinished_only and states[issue_id]["complete"]:
            continue
        selected.append(row)
    return selected


def _generated_match_candidates(
    gold_rows: Sequence[Mapping[str, Any]],
    family_id: str,
) -> list[dict[str, Any]]:
    return sorted(
        [
            dict(row)
            for row in gold_rows
            if _clean(row.get("family_id")) == _clean(family_id)
            and _clean(row.get("include")).lower() == "yes"
        ],
        key=lambda row: _clean(row.get("cluster_id")),
    )


def _generated_duplicate_candidates(
    rows: Sequence[Mapping[str, Any]],
    current_id: str,
) -> list[dict[str, Any]]:
    current = next(
        row
        for row in rows
        if _clean(row.get("generated_issue_id")) == _clean(current_id)
    )
    return sorted(
        [
            dict(row)
            for row in rows
            if _clean(row.get("generated_issue_id")) != _clean(current_id)
            and _clean(row.get("family_id")) == _clean(current.get("family_id"))
            and _clean(row.get("case_id")) == _clean(current.get("case_id"))
        ],
        key=lambda row: int(_clean(row.get("rank")) or 0),
    )


def _build_generated_row_edit(
    row: Mapping[str, Any],
    *,
    correctness: str,
    significance: str,
    evidence_sufficiency: str,
    human_match_status: str,
    confirmed_human_cluster_ids: Sequence[str],
    duplicate_status: str,
    duplicate_of_generated_id: str,
    valid_novelty: str,
    adjudicator_notes: str,
) -> dict[str, Any]:
    """Apply one generated-issue decision and clear conditional fields."""
    confirmed = (
        sorted(set(map(str, confirmed_human_cluster_ids)))
        if human_match_status == "matched"
        else []
    )
    duplicate_of = (
        str(duplicate_of_generated_id or "")
        if duplicate_status == "duplicate"
        else ""
    )
    novelty_eligible = (
        correctness == "correct"
        and evidence_sufficiency == "sufficient"
        and human_match_status == "unmatched"
        and duplicate_status == "unique"
    )
    updated = dict(row)
    updated.update(
        {
            "correctness": correctness,
            "significance": significance,
            "evidence_sufficiency": evidence_sufficiency,
            "human_match_status": human_match_status,
            "confirmed_human_cluster_ids": json.dumps(
                confirmed, ensure_ascii=False
            ),
            "duplicate_status": duplicate_status,
            "duplicate_of_generated_id": duplicate_of,
            "valid_novelty": valid_novelty if novelty_eligible else "no",
            "adjudicator_notes": adjudicator_notes.strip(),
        }
    )
    return updated


def _clear_form_state(cluster_id: str | None = None) -> None:
    prefix = f"edit::{cluster_id}::" if cluster_id else "edit::"
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            del st.session_state[key]


def _navigate(cluster_id: str, *, clear_cluster: str | None = None) -> None:
    if clear_cluster:
        _clear_form_state(clear_cluster)
    st.session_state["selected_cluster_id"] = cluster_id
    st.session_state.pop("cluster_picker", None)
    st.session_state.pop("active_cluster_id", None)
    st.session_state.pop("displayed_revision", None)


def _clear_generated_form_state(issue_id: str | None = None) -> None:
    prefix = f"genedit::{issue_id}::" if issue_id else "genedit::"
    for key in list(st.session_state):
        if str(key).startswith(prefix):
            del st.session_state[key]


def _navigate_generated(issue_id: str, *, clear_issue: str | None = None) -> None:
    if clear_issue:
        _clear_generated_form_state(clear_issue)
    st.session_state["generated_selected_id"] = issue_id
    st.session_state.pop("generated_issue_picker", None)
    st.session_state.pop("generated_active_id", None)
    st.session_state.pop("generated_displayed_revision", None)


def _option_index(options: Sequence[str], value: Any) -> int | None:
    cleaned = _clean(value).lower()
    return options.index(cleaned) if cleaned in options else None


def _next_id(ids: Sequence[str], current: str, *, reverse: bool = False) -> str:
    if not ids:
        return current
    try:
        index = ids.index(current)
    except ValueError:
        return ids[-1] if reverse else ids[0]
    offset = -1 if reverse else 1
    return ids[(index + offset) % len(ids)]


def _next_incomplete_id(
    ordered_ids: Sequence[str],
    current: str,
    states: Mapping[str, Mapping[str, Any]],
) -> str:
    if current not in ordered_ids:
        candidates = [cluster_id for cluster_id in ordered_ids if not states[cluster_id]["complete"]]
        return candidates[0] if candidates else current
    start = ordered_ids.index(current)
    rotated = list(ordered_ids[start + 1 :]) + list(ordered_ids[: start + 1])
    candidates = [cluster_id for cluster_id in rotated if not states[cluster_id]["complete"]]
    return candidates[0] if candidates else current


def _render_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --desk-ink: #172033;
          --desk-navy: #263f70;
          --desk-blue: #3567d4;
          --desk-paper: #f6f8fc;
          --desk-panel: #ffffff;
          --desk-rule: #cfd9e8;
          --desk-muted: #637087;
          --desk-amber: #bd6511;
          --desk-green: #167568;
        }
        .stApp { background: var(--desk-paper); color: var(--desk-ink); }
        .block-container { max-width: 1120px; padding-top: 3.8rem; padding-bottom: 4rem; }
        h1, h2, h3 {
          font-family: Charter, "Iowan Old Style", "Palatino Linotype", serif;
          color: var(--desk-ink);
          letter-spacing: -0.018em;
        }
        p, label, [data-testid="stWidgetLabel"] {
          font-family: "Avenir Next", Avenir, "Helvetica Neue", sans-serif;
        }
        [data-testid="stSidebar"] {
          background: #eaf0f8;
          border-right: 1px solid var(--desk-rule);
        }
        [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
          font-family: "Avenir Next", Avenir, sans-serif;
          letter-spacing: 0.01em;
        }
        [data-testid="stToolbar"] { display: none; }
        .desk-eyebrow {
          color: var(--desk-blue);
          font: 700 0.72rem/1.3 "SFMono-Regular", Consolas, monospace;
          letter-spacing: 0.14em;
          text-transform: uppercase;
          margin-bottom: 0.4rem;
        }
        .desk-subtitle { color: var(--desk-muted); font-size: 0.98rem; margin: -0.5rem 0 1.35rem; }
        .ledger {
          display: grid;
          grid-template-columns: repeat(3, minmax(0, 1fr));
          border: 1px solid var(--desk-rule);
          background: var(--desk-panel);
          margin: 1rem 0 1.4rem;
          box-shadow: 0 8px 24px rgba(38, 63, 112, 0.06);
        }
        .ledger-cell { padding: 0.85rem 1rem 0.75rem; border-right: 1px solid var(--desk-rule); }
        .ledger-cell:last-child { border-right: 0; }
        .ledger-label {
          color: var(--desk-muted);
          font: 700 0.67rem/1.2 "SFMono-Regular", Consolas, monospace;
          letter-spacing: 0.1em;
          text-transform: uppercase;
        }
        .ledger-value { color: var(--desk-ink); font: 650 1.2rem/1.5 "Avenir Next", sans-serif; }
        .ledger-track { height: 3px; background: #e7ecf4; margin-top: 0.35rem; overflow: hidden; }
        .ledger-fill { height: 100%; background: var(--desk-blue); }
        .issue-card {
          position: relative;
          background: var(--desk-panel);
          border: 1px solid var(--desk-rule);
          border-left: 5px solid var(--desk-navy);
          padding: 1.45rem 1.6rem 1.5rem;
          margin: 0.35rem 0 0.75rem;
          box-shadow: 0 10px 28px rgba(38, 63, 112, 0.07);
        }
        .issue-kicker {
          color: var(--desk-blue);
          font: 700 0.68rem/1.3 "SFMono-Regular", Consolas, monospace;
          letter-spacing: 0.1em;
          text-transform: uppercase;
          margin-bottom: 0.72rem;
        }
        .issue-text {
          color: var(--desk-ink);
          font: 500 1.17rem/1.62 Charter, "Iowan Old Style", serif;
          white-space: pre-wrap;
        }
        .issue-meta { color: var(--desk-muted); font-size: 0.82rem; margin-top: 0.9rem; }
        .generated-card { border-left-color: var(--desk-amber); }
        .slot-strip {
          display: flex;
          align-items: stretch;
          gap: 0.42rem;
          margin: 0.4rem 0 1rem;
        }
        .slot {
          min-width: 2.55rem;
          padding: 0.42rem 0.55rem;
          border: 1px solid var(--desk-rule);
          background: #edf1f7;
          color: var(--desk-muted);
          text-align: center;
          font: 700 0.72rem/1.25 "SFMono-Regular", Consolas, monospace;
        }
        .slot.done { background: #e2f2ee; border-color: #9bcfc3; color: var(--desk-green); }
        .slot.active {
          background: #fff2df;
          border-color: #d99a51;
          color: #914a08;
          box-shadow: inset 0 -3px 0 var(--desk-amber);
        }
        .suggestion-card {
          border: 1px solid #d7deea;
          background: #f1f4f9;
          padding: 0.85rem 1rem;
          margin: 0.55rem 0 0.9rem;
        }
        .suggestion-label {
          color: var(--desk-muted);
          font: 700 0.67rem/1.2 "SFMono-Regular", Consolas, monospace;
          letter-spacing: 0.08em;
          text-transform: uppercase;
          margin-bottom: 0.35rem;
        }
        .suggestion-text { color: var(--desk-ink); font-size: 0.94rem; line-height: 1.5; }
        .evidence-chip {
          display: inline-block;
          color: #6e430f;
          background: #fff2df;
          border: 1px solid #edc793;
          padding: 0.14rem 0.42rem;
          margin: 0.2rem 0.28rem 0 0;
          font: 700 0.68rem/1.3 "SFMono-Regular", Consolas, monospace;
        }
        .mode-chip {
          display: inline-block;
          color: var(--desk-navy);
          background: #e8eef9;
          border: 1px solid #c7d4e9;
          padding: 0.18rem 0.48rem;
          margin-right: 0.35rem;
          font: 700 0.68rem/1.3 "SFMono-Regular", Consolas, monospace;
          letter-spacing: 0.04em;
          text-transform: uppercase;
        }
        div[data-testid="stForm"] {
          background: rgba(255, 255, 255, 0.68);
          border: 1px solid var(--desk-rule);
          padding: 1rem 1.1rem 1.2rem;
        }
        div[data-testid="stForm"] h3 { margin-top: 0.15rem; }
        .local-note {
          border-left: 3px solid var(--desk-green);
          padding: 0.35rem 0 0.35rem 0.75rem;
          color: var(--desk-muted);
          font-size: 0.84rem;
        }
        .stButton > button, .stFormSubmitButton > button { border-radius: 3px; }
        *:focus-visible { outline: 3px solid #8eb0f5 !important; outline-offset: 2px; }
        @media (max-width: 720px) {
          .block-container { padding-top: 3.4rem; }
          .ledger { grid-template-columns: 1fr; }
          .ledger-cell { border-right: 0; border-bottom: 1px solid var(--desk-rule); }
          .ledger-cell:last-child { border-bottom: 0; }
          .issue-card { padding: 1.1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _render_ledger(
    progress: Mapping[str, Any],
    status: str,
    *,
    binding_verified: bool,
) -> None:
    total = max(int(progress["total"]), 1)
    full_total = max(int(progress["full_total"]), 1)
    tier_pct = 100 * int(progress["tier_done"]) / total
    full_pct = 100 * int(progress["full_done"]) / full_total
    complete_pct = 100 * int(progress["complete"]) / total
    if status == "ready" and binding_verified:
        status_label = "Ready"
    elif status == "ready":
        status_label = "Labels done; binding unchecked"
    else:
        status_label = f"{progress['remaining']} remaining"
    st.markdown(
        f"""
        <div class="ledger" aria-label="Adjudication progress">
          <div class="ledger-cell">
            <div class="ledger-label">Tier screen</div>
            <div class="ledger-value">{progress['tier_done']} / {progress['total']}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{tier_pct:.2f}%"></div></div>
          </div>
          <div class="ledger-cell">
            <div class="ledger-label">Full labels</div>
            <div class="ledger-value">{progress['full_done']} / {progress['full_total']}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{full_pct:.2f}%"></div></div>
          </div>
          <div class="ledger-cell">
            <div class="ledger-label">Packet status</div>
            <div class="ledger-value">{html.escape(status_label)}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{complete_pct:.2f}%"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_generated_ledger(
    progress: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    states: Mapping[str, Mapping[str, Any]],
    status: str,
    *,
    binding_verified: bool,
) -> None:
    cases: dict[tuple[str, str], list[str]] = {}
    for row in rows:
        key = (_clean(row.get("family_id")), _clean(row.get("case_id")))
        cases.setdefault(key, []).append(_clean(row.get("generated_issue_id")))
    complete_cases = sum(
        all(states[issue_id]["complete"] for issue_id in issue_ids)
        for issue_ids in cases.values()
    )
    total = max(int(progress["total"]), 1)
    complete = int(progress["complete"])
    complete_pct = 100 * complete / total
    case_total = max(len(cases), 1)
    case_pct = 100 * complete_cases / case_total
    if status == "ready" and binding_verified:
        status_label = "Ready to finalize"
    elif status == "ready":
        status_label = "Labels done; binding unchecked"
    else:
        status_label = f"{progress['remaining']} remaining"
    st.markdown(
        f"""
        <div class="ledger" aria-label="Generated issue labeling progress">
          <div class="ledger-cell">
            <div class="ledger-label">Issues labeled</div>
            <div class="ledger-value">{complete} / {progress['total']}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{complete_pct:.2f}%"></div></div>
          </div>
          <div class="ledger-cell">
            <div class="ledger-label">Manuscripts complete</div>
            <div class="ledger-value">{complete_cases} / {len(cases)}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{case_pct:.2f}%"></div></div>
          </div>
          <div class="ledger-cell">
            <div class="ledger-label">Packet status</div>
            <div class="ledger-value">{html.escape(status_label)}</div>
            <div class="ledger-track"><div class="ledger-fill" style="width:{complete_pct:.2f}%"></div></div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_gold_editor(private_root: Path) -> None:
    default_path = str(_default_csv_path(private_root))
    csv_path = st.sidebar.text_input(
        "Gold adjudication CSV",
        value=default_path,
        help=f"The file must remain below {private_root}.",
        key="gold_csv_path",
    )
    if st.sidebar.button(
        "Reload from disk", use_container_width=True, key="gold_reload"
    ):
        _clear_form_state()
        for key in (
            "selected_cluster_id",
            "cluster_picker",
            "active_cluster_id",
            "displayed_revision",
            "binding_check_token",
            "binding_check_result",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    try:
        editor = load_gold_editor_state(csv_path, private_root=private_root)
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Cannot open the adjudication packet: {exc}")
        st.info(
            "Choose an existing gold_adjudication.csv below the private root. "
            "The file must use mode 0600 and its parent directories mode 0700."
        )
        st.stop()

    rows = editor["rows"]
    validation = editor["validation"]
    progress = editor["progress"]
    states = _row_states(rows)
    families = sorted({_clean(row.get("family_id")) for row in rows})

    source = _binding_source(editor["csv_path"], private_root)
    source_signature = None
    if source is not None:
        source_stat = source.stat()
        source_signature = (str(source), source_stat.st_mtime_ns, source_stat.st_size)
    binding_token = (
        editor["csv_path"],
        validation.get("binding_hash", ""),
        source_signature,
    )
    if st.session_state.get("binding_check_token") != binding_token:
        with st.spinner("Checking source and extraction binding..."):
            st.session_state["binding_check_result"] = _check_current_binding(
                editor["csv_path"], private_root
            )
        st.session_state["binding_check_token"] = binding_token
    binding_check = st.session_state["binding_check_result"]

    queue = st.sidebar.radio(
        "Queue",
        QUEUE_LABELS,
        index=0,
        help="Start with Full labels, then use Tier screen for the remaining quick decisions.",
        key="gold_queue",
    )
    family = st.sidebar.selectbox(
        "Family",
        ["All families", *families],
        format_func=lambda value: value if value == "All families" else _family_label(value),
        key="gold_family",
    )
    group = st.sidebar.selectbox("Sample", GROUP_LABELS, key="gold_group")
    unfinished_only = st.sidebar.checkbox(
        "Needs work only", value=True, key="gold_unfinished_only"
    )

    scoped_rows = _filter_rows(
        rows,
        queue=queue,
        family=family,
        group=group,
        unfinished_only=unfinished_only,
    )
    scoped_ids = [_clean(row.get("cluster_id")) for row in scoped_rows]

    st.sidebar.markdown("### Progress")
    st.sidebar.progress(
        progress["tier_done"] / max(progress["total"], 1),
        text=f"Tier screen: {progress['tier_done']} of {progress['total']}",
    )
    st.sidebar.progress(
        progress["full_done"] / max(progress["full_total"], 1),
        text=f"Full labels: {progress['full_done']} of {progress['full_total']}",
    )
    st.sidebar.markdown(
        '<div class="local-note">Save writes directly to the private CSV. '
        "No feedback is sent to an API.</div>",
        unsafe_allow_html=True,
    )
    if binding_check["verified"]:
        st.sidebar.success("Source and extraction binding verified")
    elif binding_check["available"]:
        st.sidebar.error("Binding check failed; saving is disabled")
    else:
        st.sidebar.warning("Corpus binding not found; the paid CLI must recheck it")

    st.markdown('<div class="desk-eyebrow">Private review desk</div>', unsafe_allow_html=True)
    st.title("Adjudicate historical feedback")
    st.markdown(
        '<div class="desk-subtitle">One feedback cluster at a time. '
        "Start with the full-label sample, then finish the quick tier screen.</div>",
        unsafe_allow_html=True,
    )
    _render_ledger(
        progress,
        validation["status"],
        binding_verified=binding_check["verified"],
    )

    flash = st.session_state.pop("adjudication_flash", None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)

    if validation["errors"]:
        st.error("The packet has validation errors. Fix them before treating it as complete.")
        with st.expander("Show validation errors"):
            for error in validation["errors"]:
                st.write(f"- {error}")

    if binding_check["available"] and not binding_check["verified"]:
        st.error(
            "This packet no longer matches the current source or extraction state. "
            "Saving is disabled. Rebuild the adjudication packet before labeling it."
        )
        with st.expander("Show binding-check errors"):
            for error in binding_check["errors"]:
                st.write(f"- {error}")

    if not scoped_rows:
        if progress["remaining"] == 0:
            st.success("All gold-feedback labels are complete and field-valid.")
        else:
            st.info("No rows match these filters. Change the queue, family, sample, or Needs work filter.")
        st.stop()

    selected = st.session_state.get("selected_cluster_id")
    if selected not in scoped_ids:
        selected = scoped_ids[0]
        st.session_state["selected_cluster_id"] = selected
        st.session_state.pop("cluster_picker", None)
    selected_index = scoped_ids.index(selected)
    picked = st.sidebar.selectbox(
        "Jump to cluster",
        scoped_ids,
        index=selected_index,
        key="cluster_picker",
        format_func=lambda cluster_id: (
            f"{scoped_ids.index(cluster_id) + 1}/{len(scoped_ids)} · "
            f"{_family_label(next(row['family_id'] for row in scoped_rows if row['cluster_id'] == cluster_id))}"
        ),
    )
    if picked != selected:
        _clear_form_state(selected)
        selected = picked
        st.session_state["selected_cluster_id"] = selected
        st.session_state.pop("active_cluster_id", None)
        st.session_state.pop("displayed_revision", None)

    row = next(dict(item) for item in rows if _clean(item.get("cluster_id")) == selected)
    state = states[selected]
    if st.session_state.get("active_cluster_id") != selected:
        st.session_state["active_cluster_id"] = selected
        st.session_state["displayed_revision"] = editor["revision"]

    position = scoped_ids.index(selected) + 1
    mode = "Full labels" if state["full_required"] else "Tier only"
    chips = [mode, f"Provisional {_clean(row.get('provisional_tier'))}"]
    if _clean(row.get("sampled_minor")) == "yes":
        chips.append("Sampled minor")
    chip_html = "".join(
        f'<span class="mode-chip">{html.escape(chip)}</span>' for chip in chips
    )
    issue_text = html.escape(str(row.get("representative_text") or ""))
    st.markdown(
        f"""
        <div class="issue-card">
          <div class="issue-kicker">{html.escape(_family_label(row.get('family_id')))} · row {position} of {len(scoped_ids)}</div>
          <div class="issue-text">{issue_text}</div>
          <div class="issue-meta">{chip_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.expander("Source and cluster details"):
        details = {
            "cluster_id": row.get("cluster_id"),
            "case_ids": _json_list(row.get("case_ids")),
            "issue_count": row.get("issue_count"),
            "reviewer_ids": _json_list(row.get("reviewer_ids")),
            "source_ids": _json_list(row.get("source_ids")),
            "source_locators": _json_list(row.get("source_locators")),
            "member_issue_ids": _json_list(row.get("member_issue_ids")),
        }
        st.json(details)

    show_details = state["full_required"] or state["tier"] == "exclude"
    canonical_key = f"edit::{selected}::canonical"
    if show_details and st.button(
        "Use representative wording",
        key=f"edit::{selected}::copy_wording",
        help="Copies the displayed feedback into the editable canonical-issue field.",
    ):
        st.session_state[canonical_key] = str(row.get("representative_text") or "")

    duplicate_rows = eligible_duplicate_clusters(rows, selected)
    duplicate_lookup = {
        _clean(candidate.get("cluster_id")): candidate for candidate in duplicate_rows
    }
    selected_duplicates = [
        str(value)
        for value in _json_list(row.get("duplicate_cluster_ids"))
        if str(value) in duplicate_lookup
    ]

    with st.form(f"adjudication_form::{selected}"):
        st.subheader("Tier decision")
        tier_options = list(TIER_LABELS)
        tier = st.radio(
            "How consequential is this concern?",
            tier_options,
            index=_option_index(tier_options, row.get("tier_screen")),
            format_func=TIER_LABELS.get,
            horizontal=True,
            key=f"edit::{selected}::tier",
        )
        st.caption(
            "Major means a possible rejection reason or major-revision issue. "
            "Exclude means boilerplate, generated material, a response, or otherwise ineligible feedback."
        )

        include = _clean(row.get("include")).lower()
        canonical_issue = str(row.get("canonical_issue") or "")
        severity = _clean(row.get("severity")).lower()
        support = _clean(row.get("evidentiary_support")).lower()
        duplicates = selected_duplicates
        exclusion_reason = str(row.get("exclusion_reason") or "")

        if show_details:
            st.subheader("Full label")
            include_options = list(INCLUDE_LABELS)
            include = st.radio(
                "Keep this as a valid human concern?",
                include_options,
                index=_option_index(include_options, row.get("include")),
                format_func=INCLUDE_LABELS.get,
                horizontal=True,
                key=f"edit::{selected}::include",
            )
            canonical_issue = st.text_area(
                "Canonical issue wording",
                value=canonical_issue,
                height=120,
                key=canonical_key,
                help="State the concern once in your preferred wording. Edit the source wording if needed.",
            )
            severity_options = list(SEVERITY_LABELS)
            severity = st.selectbox(
                "Severity",
                severity_options,
                index=_option_index(severity_options, row.get("severity")),
                format_func=SEVERITY_LABELS.get,
                placeholder="Choose a severity",
                key=f"edit::{selected}::severity",
            )
            support_options = list(SUPPORT_LABELS)
            support = st.selectbox(
                "Evidentiary support",
                support_options,
                index=_option_index(support_options, row.get("evidentiary_support")),
                format_func=SUPPORT_LABELS.get,
                placeholder="Choose support",
                help="Judge whether the feedback itself identifies enough manuscript evidence for the concern.",
                key=f"edit::{selected}::support",
            )
            duplicates = st.multiselect(
                "Duplicate of another cluster (optional)",
                options=list(duplicate_lookup),
                default=selected_duplicates,
                format_func=lambda cluster_id: (
                    f"{cluster_id} · "
                    f"{_clean(duplicate_lookup[cluster_id].get('representative_text'))[:95]}"
                ),
                key=f"edit::{selected}::duplicates",
                help="Only clusters from this manuscript family are available.",
            )
            exclusion_reason = st.text_area(
                "Exclusion reason",
                value=exclusion_reason,
                height=82,
                key=f"edit::{selected}::exclusion",
                help="Required when you choose Exclude or No, exclude it.",
            )

        notes = st.text_area(
            "Notes (optional)",
            value=str(row.get("adjudicator_notes") or ""),
            height=82,
            key=f"edit::{selected}::notes",
        )
        save_col, next_col = st.columns([1, 1])
        save = save_col.form_submit_button(
            "Save",
            use_container_width=True,
            disabled=not binding_check["save_allowed"],
        )
        save_next = next_col.form_submit_button(
            "Save and next",
            type="primary",
            use_container_width=True,
            disabled=not binding_check["save_allowed"],
        )

    if save or save_next:
        updated_row = _build_row_edit(
            row,
            tier=tier or "",
            include=include or "",
            canonical_issue=canonical_issue,
            severity=severity or "",
            evidentiary_support=support or "",
            duplicate_cluster_ids=duplicates,
            exclusion_reason=exclusion_reason,
            adjudicator_notes=notes,
        )
        proposed = [updated_row if item["cluster_id"] == selected else dict(item) for item in rows]
        try:
            saved = save_gold_editor_rows(
                editor["csv_path"],
                proposed,
                expected_revision=st.session_state["displayed_revision"],
                private_root=private_root,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
        else:
            saved_states = _row_states(saved["rows"])
            saved_state = saved_states[selected]
            if saved_state["complete"]:
                message = "Saved. This cluster is complete."
            else:
                missing = ", ".join(saved_state["pending_fields"])
                message = f"Draft saved. Still needed: {missing}."
            st.session_state["adjudication_flash"] = ("success", message)
            target = selected
            if save_next and saved_state["complete"]:
                target = _next_incomplete_id(scoped_ids, selected, saved_states)
            _navigate(target, clear_cluster=selected)
            st.rerun()

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    if nav_left.button("← Previous", use_container_width=True):
        _navigate(_next_id(scoped_ids, selected, reverse=True), clear_cluster=selected)
        st.rerun()
    nav_mid.caption("Previous and Skip discard unsaved form changes.")
    if nav_right.button("Skip for now →", use_container_width=True):
        _navigate(_next_id(scoped_ids, selected), clear_cluster=selected)
        st.rerun()


def _render_generated_editor(private_root: Path) -> None:
    default_path = str(_default_generated_csv_path(private_root))
    csv_path = st.sidebar.text_input(
        "Generated adjudication CSV",
        value=default_path,
        help=(
            "Choose the canonical generated_adjudication.csv. Checkpoint and "
            "in-progress files are recovery artifacts, not labeling targets."
        ),
        key="generated_csv_path",
    )
    if st.sidebar.button(
        "Reload from disk", use_container_width=True, key="generated_reload"
    ):
        _clear_generated_form_state()
        for key in (
            "generated_selected_id",
            "generated_issue_picker",
            "generated_active_id",
            "generated_displayed_revision",
            "generated_binding_check_token",
            "generated_binding_check_result",
        ):
            st.session_state.pop(key, None)
        st.rerun()

    try:
        basic_editor = load_generated_editor_state(
            csv_path,
            private_root=private_root,
        )
    except (OSError, ValueError, RuntimeError) as exc:
        st.error(f"Cannot open the generated-issue packet: {exc}")
        st.info(
            "Choose the canonical generated_adjudication.csv below the private "
            "root. It must have mode 0600 inside 0700 directories."
        )
        st.stop()

    packet_path = Path(basic_editor["csv_path"])
    gold_path = packet_path.parent / "gold_adjudication.csv"
    audit_path = packet_path.parent / "baseline_run.local_audit.json"
    source = _binding_source(gold_path, private_root)
    binding_token = (
        basic_editor["csv_path"],
        basic_editor["revision"],
        _file_signature(gold_path),
        _file_signature(audit_path),
        _file_signature(source),
    )
    if st.session_state.get("generated_binding_check_token") != binding_token:
        with st.spinner("Checking baseline, gold, and source bindings..."):
            st.session_state[
                "generated_binding_check_result"
            ] = _check_current_generated_binding(csv_path, private_root)
        st.session_state["generated_binding_check_token"] = binding_token
    binding_check = st.session_state["generated_binding_check_result"]
    editor = binding_check.get("editor", basic_editor)
    rows = editor["rows"]
    validation = editor["validation"]
    gold_cluster_families = binding_check.get("gold_cluster_families", {})
    top_k = int(binding_check.get("top_k", 5) or 5)
    states = _generated_row_states(
        rows,
        gold_cluster_families=gold_cluster_families,
        top_k=top_k,
    )
    progress = generated_adjudication_progress(
        rows,
        gold_cluster_families=gold_cluster_families,
        top_k=top_k,
    )
    families = sorted({_clean(row.get("family_id")) for row in rows})

    family = st.sidebar.selectbox(
        "Family",
        ["All families", *families],
        format_func=lambda value: (
            value if value == "All families" else _family_label(value)
        ),
        key="generated_family",
    )
    unfinished_only = st.sidebar.checkbox(
        "Needs work only",
        value=True,
        key="generated_unfinished_only",
    )
    scoped_rows = _filter_generated_rows(
        rows,
        states=states,
        family=family,
        unfinished_only=unfinished_only,
    )
    scoped_ids = [
        _clean(row.get("generated_issue_id")) for row in scoped_rows
    ]

    st.sidebar.markdown("### Progress")
    st.sidebar.progress(
        progress["complete"] / max(progress["total"], 1),
        text=f"Issues labeled: {progress['complete']} of {progress['total']}",
    )
    st.sidebar.markdown(
        '<div class="local-note">Save writes only to the private CSV. '
        "No manuscript or feedback text is sent to an API.</div>",
        unsafe_allow_html=True,
    )
    if binding_check["verified"]:
        st.sidebar.success("Run, gold, and source bindings verified")
    else:
        st.sidebar.error("Binding check failed; saving is disabled")

    st.markdown(
        '<div class="desk-eyebrow">Cold-baseline review desk</div>',
        unsafe_allow_html=True,
    )
    st.title("Label generated feedback")
    st.markdown(
        '<div class="desk-subtitle">Judge one final critique at a time. '
        "These labels determine whether the cold baseline found the human concerns "
        "and whether unmatched critiques are valid additions.</div>",
        unsafe_allow_html=True,
    )
    _render_generated_ledger(
        progress,
        rows,
        states,
        validation["status"],
        binding_verified=binding_check["verified"],
    )

    flash = st.session_state.pop("generated_adjudication_flash", None)
    if flash:
        kind, message = flash
        getattr(st, kind)(message)

    if validation["errors"]:
        st.error("The generated packet has validation errors. Saving is disabled.")
        with st.expander("Show validation errors"):
            for error in validation["errors"]:
                st.write(f"- {error}")
    if not binding_check["verified"]:
        st.error(
            "This packet does not match the current baseline run and partial-gold "
            "binding. Saving is disabled so stale labels cannot overwrite it."
        )
        with st.expander("Show binding-check errors"):
            for error in binding_check.get("errors", []):
                st.write(f"- {error}")

    if not scoped_rows:
        if progress["remaining"] == 0:
            st.success(
                "All generated issues are field-valid. The benchmark can now be "
                "finalized offline."
            )
        else:
            st.info(
                "No issues match these filters. Change the family or turn off "
                "Needs work only."
            )
        st.stop()

    selected = st.session_state.get("generated_selected_id")
    if selected not in scoped_ids:
        selected = scoped_ids[0]
        st.session_state["generated_selected_id"] = selected
        st.session_state.pop("generated_issue_picker", None)
    selected_index = scoped_ids.index(selected)
    picked = st.sidebar.selectbox(
        "Jump to issue",
        scoped_ids,
        index=selected_index,
        key="generated_issue_picker",
        format_func=lambda issue_id: (
            f"{scoped_ids.index(issue_id) + 1}/{len(scoped_ids)} · "
            f"{_family_label(next(row['family_id'] for row in scoped_rows if row['generated_issue_id'] == issue_id))} · "
            f"rank {next(row['rank'] for row in scoped_rows if row['generated_issue_id'] == issue_id)}"
        ),
    )
    if picked != selected:
        _clear_generated_form_state(selected)
        selected = picked
        st.session_state["generated_selected_id"] = selected
        st.session_state.pop("generated_active_id", None)
        st.session_state.pop("generated_displayed_revision", None)

    row = next(
        dict(item)
        for item in rows
        if _clean(item.get("generated_issue_id")) == selected
    )
    state = states[selected]
    if st.session_state.get("generated_active_id") != selected:
        st.session_state["generated_active_id"] = selected
        st.session_state["generated_displayed_revision"] = editor["revision"]

    case_rows = sorted(
        [
            item
            for item in rows
            if _clean(item.get("family_id")) == _clean(row.get("family_id"))
            and _clean(item.get("case_id")) == _clean(row.get("case_id"))
        ],
        key=lambda item: int(_clean(item.get("rank")) or 0),
    )
    slots = []
    for candidate in case_rows:
        candidate_id = _clean(candidate.get("generated_issue_id"))
        classes = ["slot"]
        if states[candidate_id]["complete"]:
            classes.append("done")
        if candidate_id == selected:
            classes.append("active")
        slots.append(
            f'<span class="{" ".join(classes)}">{html.escape(_clean(candidate.get("rank")))}</span>'
        )
    st.markdown(
        '<div class="slot-strip" aria-label="Issue ranks for this manuscript">'
        + "".join(slots)
        + "</div>",
        unsafe_allow_html=True,
    )

    evidence_ids = [str(value) for value in _json_list(row.get("evidence_ids"))]
    evidence_html = "".join(
        f'<span class="evidence-chip">{html.escape(value)}</span>'
        for value in evidence_ids
    ) or '<span class="issue-meta">No evidence IDs recorded</span>'
    position = scoped_ids.index(selected) + 1
    st.markdown(
        f"""
        <div class="issue-card generated-card">
          <div class="issue-kicker">{html.escape(_family_label(row.get('family_id')))} · rank {html.escape(_clean(row.get('rank')))} · issue {position} of {len(scoped_ids)}</div>
          <div class="issue-text">{html.escape(str(row.get('generated_text') or ''))}</div>
          <div class="issue-meta">Manuscript evidence cited</div>
          <div>{evidence_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    gold_rows = binding_check.get("gold_rows", [])
    gold_candidates = _generated_match_candidates(
        gold_rows,
        _clean(row.get("family_id")),
    )
    gold_lookup = {
        _clean(candidate.get("cluster_id")): candidate
        for candidate in gold_candidates
    }
    proposed_id = _clean(row.get("proposed_human_cluster_id"))
    proposed = gold_lookup.get(proposed_id)
    if proposed:
        proposed_text = _clean(
            proposed.get("canonical_issue") or proposed.get("representative_text")
        )
        st.markdown(
            f"""
            <div class="suggestion-card">
              <div class="suggestion-label">Lexical suggestion · requires confirmation</div>
              <div class="suggestion-text">{html.escape(proposed_text)}</div>
              <div class="issue-meta">Match score {html.escape(_clean(row.get('proposed_match_score')))} · {html.escape(proposed_id)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if st.button(
            "Use suggested match",
            key=f"genedit::{selected}::use_suggestion",
            help="Marks this issue as matched and selects the suggested adjudicated concern.",
        ):
            st.session_state[f"genedit::{selected}::match"] = "matched"
            st.session_state[f"genedit::{selected}::confirmed"] = [proposed_id]
            st.rerun()
    else:
        st.caption(
            "No same-family lexical suggestion is available. You can still select "
            "a different adjudicated concern below."
        )

    with st.expander("Run details and sibling critiques"):
        st.json(
            {
                "generated_issue_id": row.get("generated_issue_id"),
                "pipeline_issue_id": row.get("pipeline_issue_id"),
                "case_id": row.get("case_id"),
                "proposed_human_cluster_id": proposed_id,
                "proposed_match_score": row.get("proposed_match_score"),
                "proposed_shared_terms": _json_list(
                    row.get("proposed_shared_terms")
                ),
            }
        )
        st.markdown("**Other final critiques for this manuscript**")
        for sibling in case_rows:
            if _clean(sibling.get("generated_issue_id")) == selected:
                continue
            st.caption(f"Rank {_clean(sibling.get('rank'))}")
            st.write(str(sibling.get("generated_text") or ""))

    st.subheader("Your assessment")
    correctness_options = list(CORRECTNESS_LABELS)
    correctness = st.radio(
        "Is the critique correct?",
        correctness_options,
        index=_option_index(correctness_options, row.get("correctness")),
        format_func=CORRECTNESS_LABELS.get,
        horizontal=True,
        key=f"genedit::{selected}::correctness",
    )

    significance_col, evidence_col = st.columns(2)
    significance_options = list(SIGNIFICANCE_LABELS)
    significance = significance_col.radio(
        "How significant is it?",
        significance_options,
        index=_option_index(significance_options, row.get("significance")),
        format_func=SIGNIFICANCE_LABELS.get,
        key=f"genedit::{selected}::significance",
    )
    evidence_options = list(EVIDENCE_LABELS)
    evidence_sufficiency = evidence_col.radio(
        "Is the cited evidence sufficient?",
        evidence_options,
        index=_option_index(evidence_options, row.get("evidence_sufficiency")),
        format_func=EVIDENCE_LABELS.get,
        key=f"genedit::{selected}::evidence",
    )

    match_options = list(MATCH_LABELS)
    human_match_status = st.radio(
        "Does it match an adjudicated human concern?",
        match_options,
        index=_option_index(match_options, row.get("human_match_status")),
        format_func=MATCH_LABELS.get,
        horizontal=True,
        key=f"genedit::{selected}::match",
    )
    confirmed = []
    if human_match_status == "matched":
        current_confirmed = [
            str(value)
            for value in _json_list(row.get("confirmed_human_cluster_ids"))
            if str(value) in gold_lookup
        ]
        confirmed = st.multiselect(
            "Confirmed human concern(s)",
            options=list(gold_lookup),
            default=current_confirmed,
            format_func=lambda cluster_id: (
                f"{_clean(gold_lookup[cluster_id].get('canonical_issue') or gold_lookup[cluster_id].get('representative_text'))[:125]} · {cluster_id}"
            ),
            key=f"genedit::{selected}::confirmed",
            help="Only included adjudicated concerns from this manuscript family are shown.",
        )
    elif not gold_candidates:
        st.caption(
            "This family has no included concern in the current partial-gold scoring set."
        )
    else:
        st.caption(
            f"‘No adjudicated match’ means no match among the {len(gold_rows)} "
            "currently adjudicated scoring concerns, not that no reviewer ever raised it."
        )

    duplicate_candidates = _generated_duplicate_candidates(rows, selected)
    duplicate_lookup = {
        _clean(candidate.get("generated_issue_id")): candidate
        for candidate in duplicate_candidates
    }
    duplicate_options = list(DUPLICATE_LABELS)
    duplicate_status = st.radio(
        "Is this a duplicate of another final critique for this manuscript?",
        duplicate_options,
        index=_option_index(duplicate_options, row.get("duplicate_status")),
        format_func=DUPLICATE_LABELS.get,
        horizontal=True,
        key=f"genedit::{selected}::duplicate_status",
    )
    duplicate_of = ""
    if duplicate_status == "duplicate":
        duplicate_ids = list(duplicate_lookup)
        current_duplicate = _clean(row.get("duplicate_of_generated_id"))
        duplicate_of = st.selectbox(
            "Duplicate of",
            duplicate_ids,
            index=(
                duplicate_ids.index(current_duplicate)
                if current_duplicate in duplicate_ids
                else None
            ),
            placeholder="Choose the matching critique",
            format_func=lambda issue_id: (
                f"Rank {_clean(duplicate_lookup[issue_id].get('rank'))} · "
                f"{_clean(duplicate_lookup[issue_id].get('generated_text'))[:115]}"
            ),
            key=f"genedit::{selected}::duplicate_of",
        ) or ""

    novelty_eligible = (
        correctness == "correct"
        and evidence_sufficiency == "sufficient"
        and human_match_status == "unmatched"
        and duplicate_status == "unique"
    )
    novelty_key = f"genedit::{selected}::novelty"
    if not novelty_eligible:
        st.session_state[novelty_key] = "no"
    novelty_options = list(NOVELTY_LABELS)
    valid_novelty = st.radio(
        "Is this a valid novel contribution from the pipeline?",
        novelty_options,
        index=(
            _option_index(novelty_options, row.get("valid_novelty"))
            if novelty_eligible
            else novelty_options.index("no")
        ),
        format_func=NOVELTY_LABELS.get,
        horizontal=True,
        disabled=not novelty_eligible,
        key=novelty_key,
        help=(
            "Yes is available only for a correct, sufficiently supported, unique "
            "issue with no match in the adjudicated human set."
        ),
    )
    if not novelty_eligible:
        st.caption(
            "Novelty is set to No unless the issue is correct, sufficiently "
            "supported, unmatched, and unique."
        )

    notes = st.text_area(
        "Notes (optional)",
        value=str(row.get("adjudicator_notes") or ""),
        height=82,
        key=f"genedit::{selected}::notes",
    )
    save_col, next_col = st.columns([1, 1])
    save = save_col.button(
        "Save",
        use_container_width=True,
        disabled=not binding_check["save_allowed"],
        key=f"generated_save::{selected}",
    )
    save_next = next_col.button(
        "Save and next",
        type="primary",
        use_container_width=True,
        disabled=not binding_check["save_allowed"],
        key=f"generated_save_next::{selected}",
    )

    if save or save_next:
        updated_row = _build_generated_row_edit(
            row,
            correctness=correctness or "",
            significance=significance or "",
            evidence_sufficiency=evidence_sufficiency or "",
            human_match_status=human_match_status or "",
            confirmed_human_cluster_ids=confirmed,
            duplicate_status=duplicate_status or "",
            duplicate_of_generated_id=duplicate_of,
            valid_novelty=valid_novelty or "no",
            adjudicator_notes=notes,
        )
        proposed_rows = [
            updated_row
            if _clean(item.get("generated_issue_id")) == selected
            else dict(item)
            for item in rows
        ]
        try:
            saved = save_generated_editor_rows(
                editor["csv_path"],
                proposed_rows,
                expected_revision=st.session_state[
                    "generated_displayed_revision"
                ],
                expected_binding_hash=binding_check["expected_binding_hash"],
                expected_gold_binding_hash=binding_check[
                    "expected_gold_binding_hash"
                ],
                gold_cluster_families=binding_check["gold_cluster_families"],
                run_binding_context=binding_check["run_binding_context"],
                top_k=top_k,
                private_root=private_root,
            )
        except (ValueError, RuntimeError, OSError) as exc:
            st.error(str(exc))
        else:
            saved_states = _generated_row_states(
                saved["rows"],
                gold_cluster_families=binding_check["gold_cluster_families"],
                top_k=top_k,
            )
            saved_state = saved_states[selected]
            if saved_state["complete"]:
                message = "Saved. This generated issue is complete."
            else:
                missing = ", ".join(saved_state["pending_fields"])
                message = f"Draft saved. Still needed: {missing}."
            st.session_state["generated_adjudication_flash"] = (
                "success",
                message,
            )
            target = selected
            if save_next and saved_state["complete"]:
                target = _next_incomplete_id(
                    scoped_ids,
                    selected,
                    saved_states,
                )
            _navigate_generated(target, clear_issue=selected)
            st.rerun()

    nav_left, nav_mid, nav_right = st.columns([1, 2, 1])
    if nav_left.button(
        "← Previous", use_container_width=True, key=f"generated_previous::{selected}"
    ):
        _navigate_generated(
            _next_id(scoped_ids, selected, reverse=True),
            clear_issue=selected,
        )
        st.rerun()
    nav_mid.caption("Previous and Skip discard unsaved label changes.")
    if nav_right.button(
        "Skip for now →",
        use_container_width=True,
        key=f"generated_skip::{selected}",
    ):
        _navigate_generated(
            _next_id(scoped_ids, selected),
            clear_issue=selected,
        )
        st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="Private Feedback Adjudication",
        page_icon="✓",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _render_styles()
    private_root = _private_root()

    st.sidebar.markdown("## Review desk")
    st.sidebar.caption("Private, local, and offline")
    generated_exists = _default_generated_csv_path(private_root).is_file()
    tasks = ("Generated issues", "Historical feedback")
    task = st.sidebar.radio(
        "Task",
        tasks,
        index=0 if generated_exists else 1,
        key="adjudication_task",
        help="Generated issues are the current cold-baseline outputs. Historical feedback is the gold-label packet.",
    )
    st.sidebar.divider()
    if task == "Generated issues":
        _render_generated_editor(private_root)
    else:
        _render_gold_editor(private_root)


if __name__ == "__main__":
    main()
