## feedback_llm (very lightweight prototype)

This repository contains a **minimal, async feedback pipeline** for quantitative social science papers.
It is intentionally small: one core module, a few functions, and no framework or heavy dependencies.

### What it does

- **Generation**: 8 specialized workers (theorists, rival researchers, methodologists, editors) each propose one high-impact piece of feedback.
- **Scoring**: Each proposal is scored on importance, specificity, actionability, and uniqueness.
- **Critique**: High-quality proposals receive a short Delphi-style critique before synthesis.
- **Selection (Python only)**: Proposals are ranked and classified using simple, deterministic logic
  (no hidden logic inside prompts).
- **Meta-review**: All high-quality proposals feed into a short meta-review plus global priorities.

### Design principles

- **Lightweight by default**: keep the core small and inspectable.
- **Async, but simple**: a single async pipeline, plus a small synchronous `feedback(paper_text)` wrapper.
- **Deterministic control**: ranking, thresholds, and filtering all live in Python.
- **Easily extensible**: prompts and thresholds can be refined without changing the overall structure.

If you add new features, **avoid bloat**:

- Prefer small functions over new modules or frameworks.
- Avoid adding heavy dependencies unless absolutely necessary.
- Keep configuration minimal and close to the code.

### Input & output

- **Input**: plain-text paper content, which can be the full manuscript or a partial excerpt (e.g., just the introduction). The generation workers are instructed to do the best possible job with whatever text they receive.
- **Primary output**: a concise meta-review synthesizing high-quality proposals.
- **Detailed output**: `full_feedback_pipeline` also returns raw proposals, scoring metadata, selection decisions, and a `cost_estimate` (tiktoken-based).
- **Default models**: generation and scoring use `gpt-5`; the meta-review uses `gpt-5.1`. Adjust these in `feedback_pipeline.py` if your account prefers different models.

### Quick start

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key (for example in your shell):

```bash
export OPENAI_API_KEY="sk-..."
```

3. Use the pipeline from Python:

```python
from feedback_pipeline import feedback

paper_text = "... your paper text ..."
meta_review = feedback(paper_text)
print(meta_review)
```

For more detailed outputs (proposals, scores, and selection metadata), import and call
`full_feedback_pipeline` from `feedback_pipeline.py`.

### CLI usage

You can also run the pipeline from the command line:

- **From a file**:

```bash
python -m feedback_pipeline --file path/to/paper.txt
```

- **Piped stdin**:

```bash
cat path/to/paper.txt | python -m feedback_pipeline
```

- **Interactive paste**:

```bash
python -m feedback_pipeline --paste
# Paste text, then type ::END:: on a new line and press Enter
```

If you omit `--file` and `--paste`, the CLI will prompt automatically when run in a TTY.

All commands print the meta-review to stdout while progress updates (e.g., “Generating proposals…”) stream to stderr. Add `--estimate-cost` to display an approximate token/cost breakdown based on `tiktoken`.
