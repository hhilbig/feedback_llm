"""
Streamlit web interface for the feedback pipeline.

Run with: streamlit run streamlit_app.py
Or double-click: run_app.command (macOS) / run_app.bat (Windows)
"""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import streamlit.components.v1 as components

# --- History Management ---
HISTORY_DIR = Path.home() / ".feedback_llm"
HISTORY_FILE = HISTORY_DIR / "history.json"
MAX_HISTORY_ENTRIES = 50


def load_history() -> list[dict]:
    """Load feedback history from disk."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def save_history(entries: list[dict]) -> None:
    """Save feedback history to disk."""
    HISTORY_DIR.mkdir(parents=True, exist_ok=True)
    # Keep only the most recent entries
    entries = entries[-MAX_HISTORY_ENTRIES:]
    with open(HISTORY_FILE, "w") as f:
        json.dump(entries, f, indent=2)


def add_history_entry(paper_text: str, result: dict, model: str, num_agents: int) -> str:
    """Add a new entry to history and return its ID."""
    entry_id = str(uuid.uuid4())[:8]
    entry = {
        "id": entry_id,
        "timestamp": datetime.now().isoformat(),
        "title": _extract_paper_title(paper_text),
        "paper_preview": paper_text[:100].replace("\n", " ").strip(),
        "model": model,
        "num_agents": num_agents,
        "meta_review": result["meta_review"],
        "report_markdown": result.get("report_markdown") or result["meta_review"],
        "actual_usage": result.get("actual_usage"),
    }
    history = load_history()
    history.append(entry)
    save_history(history)
    return entry_id


def _style_meta_review(md: str) -> str:
    """Replace [REQUIRED] and [SUGGESTED] tags with colored HTML badges."""
    md = md.replace(
        "[REQUIRED]",
        '<span style="background-color:#ff4b4b22;color:#ff4b4b;padding:2px 6px;'
        'border-radius:4px;font-weight:600">[REQUIRED]</span>',
    )
    md = md.replace(
        "[SUGGESTED]",
        '<span style="background-color:#1f77b422;color:#1f77b4;padding:2px 6px;'
        'border-radius:4px;font-weight:600">[SUGGESTED]</span>',
    )
    return md


def copy_button_js(text: str, button_id: str = "copy_btn") -> None:
    """Render a JavaScript-based copy button that works in browsers."""
    # Escape for JS string literal
    escaped = text.replace("\\", "\\\\").replace("`", "\\`").replace("$", "\\$")
    html = f"""
    <button id="{button_id}" onclick="copyText()" style="
        background-color: #f0f2f6;
        border: 1px solid #d0d0d0;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        cursor: pointer;
        font-size: 14px;
    ">📋 Copy to Clipboard</button>
    <script>
    function copyText() {{
        const text = `{escaped}`;
        navigator.clipboard.writeText(text).then(() => {{
            document.getElementById("{button_id}").innerText = "✓ Copied!";
            setTimeout(() => {{
                document.getElementById("{button_id}").innerText = "📋 Copy to Clipboard";
            }}, 2000);
        }}).catch(err => {{
            alert("Failed to copy: " + err);
        }});
    }}
    </script>
    """
    components.html(html, height=50)


def _extract_paper_title(paper_text: str, max_len: int = 60) -> str:
    """Extract a meaningful title from paper text, skipping LaTeX preamble."""
    import re

    skip_prefixes = (
        "\\documentclass", "\\usepackage", "\\begin{document}",
        "\\newcommand", "\\renewcommand", "\\setlength", "\\input",
        "\\maketitle", "\\pagestyle", "\\bibliographystyle",
    )

    lines = paper_text.split("\n")
    for line in lines:
        stripped = line.strip()
        # Skip blank lines and comments
        if not stripped or stripped.startswith("%"):
            continue
        # Skip common preamble commands
        if any(stripped.startswith(p) for p in skip_prefixes):
            continue
        # Extract content from \title{...}
        title_match = re.match(r"\\title\{(.+)\}", stripped)
        if title_match:
            stripped = title_match.group(1)
        # Strip wrapping commands like \textbf{...}, \section*{...}, \section{...}
        stripped = re.sub(r"\\(?:textbf|textit|emph|section\*?|subsection\*?|chapter\*?)\{([^}]*)\}", r"\1", stripped)
        # Remove remaining backslash commands (e.g. \centering, \large)
        stripped = re.sub(r"\\[a-zA-Z]+\*?", "", stripped).strip()
        # Skip if nothing meaningful remains
        if not stripped or len(stripped) < 5:
            continue
        # Truncate
        if len(stripped) > max_len:
            stripped = stripped[:max_len - 3] + "..."
        return stripped

    # Fallback: first 60 chars of raw text
    fallback = paper_text[:max_len].replace("\n", " ").strip()
    if len(fallback) > max_len:
        fallback = fallback[:max_len - 3] + "..."
    return fallback


def _extract_topic_from_meta_review(meta_review: str) -> str:
    """Extract a short topic phrase from the meta-review's opening paragraphs.

    Looks for quoted phrases (which typically name the paper's key concepts)
    and picks the longest one as the most descriptive.
    """
    import re

    # Search first 800 chars for quoted phrases (straight and curly quotes)
    quotes = re.findall(
        r'["\u201c]([^"\u201c\u201d]{4,50})["\u201d]',
        meta_review[:800],
    )
    # Filter out generic phrases that aren't topic-specific
    generic = {"what is new", "what is the contribution", "why does this matter"}
    quotes = [q for q in quotes if q.lower().strip() not in generic]
    if quotes:
        # Pick the longest phrase as the most descriptive
        best = max(quotes, key=len)
        return best.strip().capitalize()
    return ""


st.set_page_config(
    page_title="Paper Feedback Pipeline",
    page_icon="📝",
    layout="wide",
)

# Check for API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error("OPENAI_API_KEY not found in environment. Add it to your .env file.")
    st.stop()

# --- Session State Initialization ---
if "current_result" not in st.session_state:
    st.session_state.current_result = None
if "current_paper_text" not in st.session_state:
    st.session_state.current_paper_text = None
if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None

st.title("📝 Paper Feedback Pipeline")
st.markdown("""
A single AI response is often hit-or-miss. This tool uses multiple specialized reviewers
(Theorists, Methodologists, Rivals, and a design-specific reviewer) to surface different
failure modes. Feedback is tied to evidence IDs, verified against the manuscript, routed
through cheaper models where possible, and synthesized into a final report.
""")

# --- How it Works ---
with st.expander("How it works"):
    st.markdown("""
This pipeline is evidence-first: it builds a manuscript map, generates diverse critiques,
verifies each critique against cited evidence, and only then synthesizes the final report.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         YOUR PAPER TEXT                             │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. EVIDENCE MAP                                                    │
│     Builds stable evidence IDs for sections, paragraphs, tables,    │
│     figures, equations, and appendices. Suspicious instruction-like │
│     manuscript text is quarantined before review.                   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. GENERATION (with Diversity Seeds)                               │
│     8 specialized agents review evidence-indexed context:           │
│     • 2 Theorists (assumptions / causal mechanisms / frameworks)    │
│     • 2 Rivals (confounders / competing mechanisms)                 │
│     • 2 Methodologists (identification / measurement)               │
│     • 1 Design specialist (DiD / IV / RD / survey / etc.)           │
│     • 1 Editor (clarity, structure, organization)                   │
│     Each proposal must cite evidence IDs or mark itself inferential.│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. GROUNDING CHECK                                                 │
│     Flags proposals that reference tables, figures, or sections     │
│     not found in the paper (hallucination guardrail).               │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. DOMAIN SCORING + ROUTING                                        │
│     Cheap models score identification, measurement/sample,          │
│     interpretation, theory/contribution, evidence support,          │
│     actionability, and severity. Severe or ambiguous cases escalate.│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. VERIFICATION-FIRST ADJUDICATION                                 │
│     A verifier decides whether each critique is supported,          │
│     inferential, demoted, or removed. Rewrite is clarity-only and   │
│     cannot add new factual claims or evidence IDs.                  │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. DEDUP + PRESENTATION CLUSTERS                                   │
│     Evidence-aware deduplication protects severe minority issues.   │
│     Clustering only labels related comments for presentation.       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  7. SYNTHESIS                                                       │
│     A Meta-Reviewer writes evidence-linked guidance on design,      │
│     measurement/sample, interpretation, theory/contribution, and    │
│     writing when writing is a binding issue.                        │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       FINAL META-REVIEW                             │
└─────────────────────────────────────────────────────────────────────┘
```

**Why this approach?**
- **Evidence map**: Creates auditable manuscript IDs before critique generation
- **Design-aware ensemble**: Adds a design-specific reviewer for DiD, IV, RD, surveys, and related designs
- **Domain scoring**: Prioritizes validity, measurement/sample risk, interpretation, evidence support, and actionability
- **Model routing**: Uses cheaper current models for routine stages and escalates hard cases
- **Verification first**: Checks support and severity before any rewrite
- **Protected minority critiques**: Keeps severe evidence-distinct issues from being collapsed away
""")

# --- Sidebar: Settings ---
with st.sidebar:
    st.header("Settings")

    from feedback_pipeline import GENERATION_MODEL, current_model_options

    model_options = current_model_options()
    model = st.selectbox(
        "Model",
        options=model_options,
        index=model_options.index(GENERATION_MODEL),
        help=(
            "Generation model. Other stages are routed automatically: "
            "cheap models for routine scoring/formatting and frontier models for synthesis."
        ),
    )

    agents = st.select_slider(
        "Number of Agents",
        options=[8, 16, 24, 32],
        value=8,
        help="More agents = more diverse feedback, but higher cost.",
    )

    top_k = st.slider(
        "Top-K Proposals",
        min_value=3,
        max_value=15,
        value=5,
        help="Number of top proposals to include in the meta-review.",
    )

    st.divider()
    st.markdown("**Cost Warning**: Using many agents with premium models can be expensive. Start small to test.")

    # --- Sidebar: History ---
    st.divider()
    st.header("History")
    history = load_history()
    if not history:
        st.caption("No previous feedback runs yet.")
    else:
        # Show most recent 10 entries
        for entry in reversed(history[-10:]):
            title = entry.get("title", "")
            if not title or title.lstrip().startswith("\\"):
                # Old entries without title — try meta-review, then paper_preview
                title = _extract_topic_from_meta_review(entry.get("meta_review", ""))
            if not title or title.lstrip().startswith("\\"):
                title = _extract_paper_title(entry.get("paper_preview", ""))
            if not title or title.lstrip().startswith("\\"):
                title = "Paper feedback"
            if len(title) > 40:
                title = title[:37] + "..."
            model_info = entry.get("model", "")
            date = entry["timestamp"][:10]
            label = f"{title}\n{model_info} · {entry.get('num_agents', '?')} agents · {date}"
            if st.button(label, key=f"history_{entry['id']}", use_container_width=True):
                st.session_state.selected_history_id = entry["id"]
                st.session_state.current_result = None  # Clear current to show history
                st.rerun()

    # Button to clear history selection and show current result
    if st.session_state.selected_history_id:
        if st.button("← Back to Current", use_container_width=True):
            st.session_state.selected_history_id = None
            st.rerun()

# --- Main: Input ---
st.header("1. Provide Your Paper")

input_method = st.radio(
    "Input method:",
    options=["Paste text", "Upload PDF"],
    horizontal=True,
)

paper_text = ""

if input_method == "Paste text":
    paper_text = st.text_area(
        "Paste your paper text here:",
        height=300,
        placeholder="Copy and paste your paper text from Overleaf, Word, or any editor...",
    )
else:
    uploaded_file = st.file_uploader(
        "Upload a PDF file:",
        type=["pdf"],
    )
    if uploaded_file is not None:
        try:
            import fitz  # pymupdf

            # Read PDF from uploaded bytes
            pdf_bytes = uploaded_file.read()
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text_parts = []
            for page in doc:
                text_parts.append(page.get_text())
            doc.close()
            paper_text = "\n".join(text_parts)

            if paper_text.strip():
                st.success(f"Extracted {len(paper_text):,} characters from PDF.")
                with st.expander("Preview extracted text"):
                    st.text(paper_text[:2000] + ("..." if len(paper_text) > 2000 else ""))
            else:
                st.error("Could not extract text from PDF. It may be scanned/image-based.")
        except ImportError:
            st.error("pymupdf is not installed. Run: pip install pymupdf")
        except Exception as e:
            st.error(f"Error reading PDF: {e}")

# --- Main: Run ---
st.header("2. Generate Feedback")

can_run = bool(paper_text.strip())

if not paper_text.strip():
    st.info("Provide paper text above to continue.")
else:
    # Show cost estimate before running
    from feedback_pipeline import estimate_cost_before_run

    estimate = estimate_cost_before_run(
        paper_text,
        num_agents=agents,
        gen_model=model,
        top_k=top_k,
    )
    estimated_cost = estimate["estimated_total_cost_usd"]

    st.info(f"**Estimated cost: ${estimated_cost:.2f}** (actual cost may vary)")

if st.button("Generate Feedback", type="primary", disabled=not can_run):
    from feedback_pipeline import full_feedback_pipeline

    # Clear any history selection when generating new feedback
    st.session_state.selected_history_id = None

    # Progress display
    progress_bar = st.progress(0)
    status_text = st.empty()

    def update_progress(step: int, total: int, message: str):
        progress_bar.progress(step / total)
        status_text.markdown(f"**Step {step} of {total}:** {message}")

    try:
        result = asyncio.run(
            full_feedback_pipeline(
                paper_text,
                num_agents=agents,
                gen_model=model,
                top_k=top_k,
                progress_callback=update_progress,
            )
        )

        progress_bar.progress(1.0)
        status_text.empty()

        # Store in session state
        st.session_state.current_result = result
        st.session_state.current_paper_text = paper_text

        # Save to history
        add_history_entry(paper_text, result, model, agents)

        st.rerun()  # Rerun to display results from session state

    except ValueError as e:
        st.error(f"Configuration Error: {e}")
    except Exception as e:
        st.error(f"Error: {e}")

# --- Results Display (from session state or history) ---
from feedback_pipeline import _format_cost_estimate

# Determine what to display
display_result = None
display_source = None

if st.session_state.selected_history_id:
    # Show historical feedback
    history = load_history()
    for entry in history:
        if entry["id"] == st.session_state.selected_history_id:
            display_result = {
                "meta_review": entry["meta_review"],
                "report_markdown": entry.get("report_markdown") or entry["meta_review"],
                "actual_usage": entry.get("actual_usage") or entry.get("cost_estimate"),
            }
            hist_title = entry.get("title", "")
            if not hist_title or hist_title.lstrip().startswith("\\"):
                hist_title = _extract_topic_from_meta_review(entry.get("meta_review", ""))
            if not hist_title or hist_title.lstrip().startswith("\\"):
                hist_title = _extract_paper_title(entry.get("paper_preview", ""))
            if not hist_title or hist_title.lstrip().startswith("\\"):
                hist_title = "Paper feedback"
            display_source = f"History: {hist_title} ({entry['model']}, {entry['num_agents']} agents, {entry['timestamp'][:10]})"
            break
elif st.session_state.current_result:
    # Show current result
    display_result = st.session_state.current_result
    display_source = "Current run"

if display_result:
    st.success("Feedback generated!" if display_source == "Current run" else f"Viewing: {display_source}")

    st.header("3. Results")
    meta_review_md = display_result["meta_review"]
    parts = meta_review_md.split("## Proposed Revisions", 1)
    st.markdown(_style_meta_review(parts[0]), unsafe_allow_html=True)
    if len(parts) > 1:
        st.divider()
        st.markdown(
            _style_meta_review("## Proposed Revisions" + parts[1]),
            unsafe_allow_html=True,
        )

    # Export options
    meta_review_text = display_result.get("report_markdown") or display_result["meta_review"]
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            label="Download Full Markdown",
            data=meta_review_text,
            file_name="feedback.md",
            mime="text/markdown",
        )
    with col2:
        copy_button_js(meta_review_text)

    # Display actual usage / cost
    usage = display_result.get("actual_usage") or display_result.get("cost_estimate")
    if usage:
        label = "Actual Token Usage" if usage.get("source") == "actual" else "Cost Estimate"
        with st.expander(label):
            st.text(_format_cost_estimate(usage))
