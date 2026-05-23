# Paper Feedback Pipeline

AI-assisted feedback for quantitative social science papers. The pipeline is evidence-first: it maps the manuscript, generates role-diverse critiques, verifies each critique against the manuscript, ranks issues with a domain-specific rubric, and synthesizes an evidence-linked final report.

## Getting Started

### Step 1: Get an OpenAI API Key

1. Go to [platform.openai.com](https://platform.openai.com)
2. Sign up or log in
3. Go to API Keys and create a new key
4. Copy the key

### Step 2: Save Your API Key

Create a file called `.env` in this folder:

```text
OPENAI_API_KEY=sk-your-key-here
```

### Step 3: Run the App

Double-click `run_app.command`.

The first run may take 1-2 minutes while dependencies are checked. Later runs should start faster.

### Step 4: Use the App

1. Paste paper text or upload a PDF.
2. Adjust model, agent count, and top-k settings if needed.
3. Click "Generate Feedback".
4. Review the evidence-linked report and its evidence lookup appendix.

## How It Works

The current pipeline has eight stages:

1. **Evidence map**: Strips hidden/control characters, quarantines instruction-like manuscript text, indexes sections, paragraphs, tables, figures, equations, and appendices, and extracts a structured manuscript map.
2. **Design-aware generation**: Eight agents generate one critique each. The panel includes 2 theorists, 2 rivals, 2 methodologists, 1 design specialist, and 1 editor. The design specialist adapts to DiD, IV, RD, experiments, surveys, descriptive work, panel observational designs, qualitative work, and mixed methods.
3. **Grounding check**: Regex guardrails flag missing table, figure, section, appendix, column, panel, and equation references.
4. **Domain scoring with routing**: Cheap models score identification risk, measurement/sample risk, interpretation risk, theory/contribution risk, evidence support, actionability, severity, and confidence. Severe, ambiguous, low-confidence, or low-agreement items escalate to the frontier model.
5. **Selection and evidence-aware deduplication**: High-quality proposals are deduplicated with embeddings, but severe evidence-distinct identification, measurement/sample, and interpretation critiques are protected.
6. **Verification-first adjudication**: A verifier decides whether each critique is supported, partially supported, inferential, unsupported, or contradicted, then keeps, demotes, or removes it.
7. **Constrained rewrite and presentation clustering**: Rewrite is clarity-only and cannot add factual claims or evidence IDs. Clustering only labels related proposals for presentation; it does not merge or remove critiques.
8. **Meta-review plus evidence lookup**: The final report prioritizes identification/design, measurement/sample construction, empirical interpretation, theory/contribution, and writing only when writing is material. A deterministic appendix lists the manuscript excerpts for cited evidence IDs.

## Settings

| Setting | What it does |
|---------|--------------|
| **Model** | Generation model. Other stages route automatically to cheaper or stronger models. |
| **Number of Agents** | More agents increase breadth and cost. Must be a multiple of 8. |
| **Top-K Proposals** | Number of top insights emphasized in the final report. |

Default routing uses `gpt-5.4-mini` for generation, scoring, and verification, `gpt-5.4-nano` for constrained rewrites and simple labeling, and `gpt-5.5` for meta-review and escalations.

Cost note: each run uses the paid OpenAI API. The app shows an estimate before a run and actual usage afterward.

## Troubleshooting

**"OPENAI_API_KEY not found"**
- Make sure `.env` exists in this folder.
- Make sure it contains `OPENAI_API_KEY=sk-...` with no spaces around `=`.

**App will not start**
- Check that Python 3 is installed: `python3 --version`.
- Install dependencies if needed: `pip install -r requirements.txt`.

**PDF upload does not work**
- Some PDFs are scanned images rather than embedded text. Paste text instead.

## Advanced: Command Line

```bash
source .venv/bin/activate

python -m feedback_pipeline --clipboard
python -m feedback_pipeline --pdf paper.pdf
python -m feedback_pipeline --file paper.txt

python -m feedback_pipeline --agents 16 --model gpt-5.4-mini --top-k 10 --file paper.txt
python -m feedback_pipeline --file paper.txt --no-evidence-appendix
```

See [DESIGN.md](DESIGN.md) for implementation details.
