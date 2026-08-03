# AGENTS.md

This file provides guidance to Codex when working with this repository.

## Project Overview

This is an async feedback pipeline for quantitative social science papers using OpenAI's API. It builds an evidence map before critique generation, routes routine stages to cheaper current models, verifies feedback against manuscript evidence, and synthesizes an evidence-linked final report.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run web app
streamlit run streamlit_app.py

# Run private, offline adjudication app
streamlit run adjudication_app.py --server.address 127.0.0.1 --browser.gatherUsageStats false

# CLI alternatives
python3 -m feedback_pipeline --clipboard
python3 -m feedback_pipeline --pdf paper.pdf
python3 -m feedback_pipeline --file paper.txt

# Offline tests
python3 -m unittest discover -s tests
python3 -m py_compile feedback_pipeline.py streamlit_app.py adjudication_app.py tests/*.py
```

## Architecture

The pipeline is mostly in `feedback_pipeline.py`:

1. **Evidence map**: Sanitizes manuscript text, quarantines instruction-like lines, assigns stable evidence IDs to sections, paragraphs, tables, figures, equations, and appendices, and extracts a structured evidence map.
2. **Design-aware generation**: Eight reviewers generate one evidence-linked critique each: 2 theorists, 2 rivals, 2 methodologists, 1 design specialist, and 1 editor.
3. **Grounding check**: Regex guardrail flags missing table, figure, section, appendix, column, panel, and equation references.
4. **Domain scoring and escalation**: Proposals are scored on identification risk, measurement/sample risk, interpretation risk, theory/contribution risk, evidence support, actionability, severity, and confidence. Severe or ambiguous items can escalate to the frontier model.
5. **Selection and evidence-aware deduplication**: High-quality proposals are deduplicated while preserving severe evidence-distinct minority critiques.
6. **Verification-first adjudication**: Verifier keeps, demotes, or removes proposals based on manuscript support, severity, counter-evidence, and actionability.
7. **Constrained rewrite and presentation clustering**: Rewrite is clarity-only and cannot add factual claims or evidence IDs. Clustering only annotates related proposals for presentation.
8. **Meta-review**: Synthesizes verified proposals into evidence-linked sections on identification/design, measurement/sample, empirical interpretation, theory/contribution, and writing only if material.

## Key Design Decisions

- **Evidence first**: Later stages cite stable evidence IDs rather than raw text spans.
- **Prompt injection defense**: Hidden/control characters are stripped and instruction-like manuscript lines are quarantined.
- **Structured outputs**: Evidence extraction, generation, scoring, verification, and rewrite use JSON Schema outputs.
- **Model routing**: Defaults are `gpt-5.6-terra` for generation/scoring/verification, `gpt-5.6-luna` for rewrite/simple labeling, and `gpt-5.6-sol` for editorial triage/meta-review/escalation. Preserve reasoning effort at `none` for Terra/Luna and `medium` for Sol unless representative evaluations justify a change.
- **Verification before rewrite**: The pipeline does not use a substantive critique/revision loop. Rewrite follows verification and is clarity-only.
- **Protected minority critiques**: Severe identification, measurement/sample, and interpretation issues survive deduplication when they cite distinct evidence.
- **Presentation-only clustering**: Clusters never merge or remove proposals.
- **Async throughout**: API calls use `AsyncOpenAI` with `asyncio.gather` where appropriate.

## Configuration

- API key: Set `OPENAI_API_KEY` in `.env` or the environment.
- Model registry: `MODEL_REGISTRY` in `feedback_pipeline.py`.
- Routing defaults: `DEFAULT_MODEL_ROUTING`.
- Thresholds: `IMPORTANCE_THRESHOLD = 3`, `COMPOSITE_THRESHOLD = 3.0`.
- Embeddings: `text-embedding-3-small`.
