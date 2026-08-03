"""Local Streamlit editor for private gold-feedback adjudication.

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
    gold_adjudication_progress,
    gold_row_requirements,
    load_gold_editor_state,
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
    default_path = str(_default_csv_path(private_root))
    csv_path = st.sidebar.text_input(
        "Gold adjudication CSV",
        value=default_path,
        help=f"The file must remain below {private_root}.",
    )
    if st.sidebar.button("Reload from disk", use_container_width=True):
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
    )
    family = st.sidebar.selectbox(
        "Family",
        ["All families", *families],
        format_func=lambda value: value if value == "All families" else _family_label(value),
    )
    group = st.sidebar.selectbox("Sample", GROUP_LABELS)
    unfinished_only = st.sidebar.checkbox("Needs work only", value=True)

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


if __name__ == "__main__":
    main()
