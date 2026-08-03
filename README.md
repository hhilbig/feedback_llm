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
4. Review the editorial diagnosis and evidence-linked report. Use comprehensive audit mode when you want the full deterministic checklist appendix.

## How It Works

The current pipeline has nine stages:

1. **Evidence map and substantive design profile**: Strips hidden/control characters, quarantines instruction-like manuscript text, indexes sections, paragraphs, tables, figures, equations, and appendices, extracts a structured manuscript map, and deterministically flags applicable design risks such as DD/DDD inference, repeated-cross-section composition, and text-as-data validation.
2. **Design-aware generation**: Eight agents generate one critique each. The panel includes 2 theorists, 2 rivals, 2 methodologists, 1 design specialist, and 1 editor. The design specialist adapts to DiD, IV, RD, experiments, surveys, descriptive work, panel observational designs, qualitative work, and mixed methods; the prompts also receive the substantive checklist findings as omission cues.
3. **Grounding check**: Regex guardrails flag missing table, figure, section, appendix, column, panel, and equation references.
4. **Domain scoring with routing**: Cheap models score identification risk, measurement/sample risk, interpretation risk, theory/contribution risk, evidence support, actionability, severity, and confidence. Severe, ambiguous, low-confidence, or low-agreement items escalate to the frontier model.
5. **Selection and evidence-aware deduplication**: High-quality proposals are deduplicated with embeddings, but severe evidence-distinct identification, measurement/sample, and interpretation critiques are protected.
6. **Verification-first adjudication**: A verifier decides whether each critique is supported, partially supported, inferential, unsupported, or contradicted, then keeps, demotes, or removes it.
7. **Constrained rewrite and presentation clustering**: Rewrite is clarity-only and cannot add factual claims or evidence IDs. Clustering only labels related proposals for presentation; it does not merge or remove critiques.
8. **Editorial triage**: A decision-impact stage builds a clear 5-8 item problem list when enough non-marginal problems are available. Each problem receives both a rejection-risk label (high, conditional, low, none) and a decision-tier label (potential rejection reason, major revision issue, minor revision issue, nice-to-have, drop).
9. **Editorial report plus appendices**: The final report starts with an editorial summary, then a numbered problem list with rejection-risk labels and notes on non-rejection issues. Detailed evidence lookup and substantive coverage appendices are opt-in.

## Settings

| Setting | What it does |
|---------|--------------|
| **Model** | Generation model. Other stages route automatically to cheaper or stronger models. |
| **Number of Agents** | More agents increase breadth and cost. Must be a multiple of 8. |
| **Top-K Proposals** | Number of top insights emphasized in the final report. |
| **Review mode** | Editorial triage shows clear problems with rejection-risk labels. Comprehensive audit also appends deterministic coverage and evidence appendices. |

Default routing uses `gpt-5.6-terra` for generation, scoring, and verification, `gpt-5.6-luna` for constrained rewrites and simple labeling, and `gpt-5.6-sol` for editorial triage, meta-review, and escalations. Reasoning effort is explicitly `none` for Terra/Luna and `medium` for Sol to preserve the effective behavior of the previous routing.

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

python -m feedback_pipeline --agents 16 --model gpt-5.6-terra --top-k 10 --file paper.txt
python -m feedback_pipeline --file paper.txt --include-evidence-appendix
python -m feedback_pipeline --file paper.txt --include-audit-appendix

# Inspect a legacy review archive or a hash-bound private manifest without calling the API.
python -m feedback_pipeline --inspect-review-corpus --review-corpus /path/to/journal_reviews_inbox_2026-06-04

# Validate a private manifest and write a normalized snapshot below ~/.feedback_llm.
python -m feedback_pipeline \
  --inspect-review-corpus \
  --review-corpus ~/.feedback_llm/review_manifest.json \
  --review-corpus-output corpora/private_feedback_pilot_v1.json

# Prepare the private gold-cluster CSV and reading packet. This makes no API calls.
python -m feedback_pipeline \
  --review-corpus ~/.feedback_llm/review_manifest.json \
  --prepare-review-adjudication ~/.feedback_llm/adjudication/private_feedback_pilot_v1

# Optional raw Gmail sidecars can be stored under:
# /path/to/journal_reviews_inbox_2026-06-04/raw_gmail_exports/*.md
# When a sidecar contains "- Review file: `...`", it replaces the digest text for that record.

# Include forwarded/low-confidence archive entries only for local inspection.
python -m feedback_pipeline --inspect-review-corpus --review-corpus /path/to/journal_reviews_inbox_2026-06-04 --include-low-confidence-reviews

# Use anonymized, digest-derived historical reviewer patterns for tone/salience calibration.
# The final report does not expose old review examples.
python -m feedback_pipeline --file paper.txt --review-corpus /path/to/journal_reviews_inbox_2026-06-04

# Use an API-safe structured reviewer prior. The pipeline runs a cold paper-only
# pass first, then uses the prior only for targeted gap checks and triage calibration.
python -m feedback_pipeline --file paper.txt --review-prior outputs/private_review_prior.json

# Distill a local private review archive into an API-safe structured reviewer-prior artifact.
# The local audit file keeps exact support metadata and should stay out of git.
python -m feedback_pipeline \
  --distill-review-prior /path/to/journal_reviews_inbox_2026-06-04 \
  --review-prior-output outputs/private_review_prior.json \
  --review-prior-audit-output outputs/private_review_prior.local_audit.json

# Estimate a cold, manuscript-only baseline without calling the API.
python -m feedback_pipeline \
  --eval-review-corpus ~/.feedback_llm/review_manifest.json \
  --eval-memory-mode none \
  --eval-output outputs/review_eval_plan.json

# Estimate the explicit partial-gold pilot from completed labels only.
# This also makes no API calls and reports completed and scoring denominators.
python -m feedback_pipeline \
  --eval-review-corpus ~/.feedback_llm/review_manifest.json \
  --eval-memory-mode none \
  --eval-adjudication ~/.feedback_llm/adjudication/private_feedback_pilot_v1/gold_adjudication.csv \
  --eval-gold-mode partial \
  --eval-output outputs/review_eval_partial_plan.json

# Paid mode requires completed, current gold labels and an explicit dollar ceiling.
python -m feedback_pipeline \
  --eval-review-corpus ~/.feedback_llm/review_manifest.json \
  --eval-memory-mode none \
  --eval-adjudication ~/.feedback_llm/adjudication/private_feedback_pilot_v1/gold_adjudication.csv \
  --eval-max-cost-usd 15 \
  --eval-output outputs/review_eval_baseline.json \
  --eval-run-api

# After labeling generated_adjudication.csv, finalize privacy-safe metrics offline.
# Pass --eval-gold-mode partial here if the paid run used partial gold.
python -m feedback_pipeline \
  --review-corpus ~/.feedback_llm/review_manifest.json \
  --eval-adjudication ~/.feedback_llm/adjudication/private_feedback_pilot_v1/gold_adjudication.csv \
  --eval-generated-adjudication ~/.feedback_llm/adjudication/private_feedback_pilot_v1/generated_adjudication.csv \
  --finalize-review-eval ~/.feedback_llm/adjudication/private_feedback_pilot_v1/baseline_run.local_audit.json \
  --eval-output outputs/review_eval_metrics.json

# Compare baseline, API-safe prior, and local raw-memory upper bound.
# Dry-run mode estimates the three-way gate cost without calling the API.
python -m feedback_pipeline \
  --eval-review-prior-gate /path/to/journal_reviews_inbox_2026-06-04 \
  --eval-limit 3 \
  --eval-output outputs/review_prior_eval_gate_plan.json

# Paid API mode for the gate, after reviewing the dry-run cost.
python -m feedback_pipeline \
  --eval-review-prior-gate /path/to/journal_reviews_inbox_2026-06-04 \
  --eval-limit 1 \
  --eval-output outputs/review_prior_eval_gate_api.json \
  --eval-max-cost-usd 15 \
  --eval-run-api

# OpenAI Batch API mode for the gate. This writes local JSONL batch files and
# a manifest, then polls each dependent batch stage until completion.
python -m feedback_pipeline \
  --eval-review-prior-gate /path/to/journal_reviews_inbox_2026-06-04 \
  --eval-limit 3 \
  --eval-output outputs/review_prior_eval_gate_batch.json \
  --eval-run-api \
  --eval-max-cost-usd 15 \
  --eval-batch-api
```

The paid examples use illustrative ceilings. Always run the dry plan first,
commit the exact `feedback_llm` implementation being evaluated, and approve its
current total before making API calls.

### Private Feedback Benchmark

`review_manifest.schema.json` defines the private manifest. Each evaluated case
binds one ordered manuscript bundle to one or more human-feedback sources using
SHA-256 hashes. `family_id` is the leakage boundary; `case_id` identifies one
reviewed manuscript version. The importer supports PDF annotations, Word
comments, review PDFs, Markdown, text files, and Word body text. It rejects
unsupported formats, stale hashes, empty files, dataless cloud placeholders,
and evaluation sources labeled as AI-generated, derivative, or response material.

The manifest, normalized corpus, gold labels, generated-issue labels, and local
audit stay below `~/.feedback_llm/` with private permissions. The repository
contains only a pseudonymous example manifest. Saved evaluation JSON is a
path-free aggregate projection; human feedback and generated issue text remain
in the ignored local adjudication artifacts.

To label either private packet without editing CSV cells, double-click
`run_adjudication_app.command`, or run:

```bash
streamlit run adjudication_app.py \
  --server.address 127.0.0.1 \
  --browser.gatherUsageStats false
```

When a canonical `generated_adjudication.csv` exists, the app opens the generated
issue workflow by default. It shows the full critique, cited evidence IDs, the
five ranked critiques for that manuscript, and any same-family lexical match
suggestion. For each issue, label correctness, significance, evidence
sufficiency, human-cluster match, duplication, and valid novelty. **Save and
next** moves to the next unfinished issue.

Use the **Task** switch for the historical-feedback workflow. That screen shows
one gold cluster at a time; start with **Full labels**, then use **Tier screen**
for the remaining quick decisions. Both workflows write only their designated
label fields to the private CSV. The app rechecks the source, gold, and baseline
bindings before each save, makes no API calls, and refuses files outside
`~/.feedback_llm/`.

Gold validation has two passes. Every deduplicated cluster receives a quick
major/minor/exclude screen. Full fields are required for every major cluster and
for five deterministically sampled minor clusters per family (seed `20260802`).
Strict evaluation remains the default and requires the complete tier screen.
`--eval-gold-mode partial` is an explicit exploratory alternative. It runs only
after every sampled row and every row screened as major has complete labels. It
uses fully adjudicated included rows as scoring targets, reports all completed
screens as coverage, and never describes major-cluster recall as exhaustive.
Changing a completed label or completing another row changes the partial binding
and invalidates any generated-issue packet from the earlier state.

After a paid run, every real final post-verification issue, up to five per case,
receives manual labels for correctness, significance, evidence support,
duplication, human-cluster match, and valid novelty. A successful benchmark case
must return at least one issue; the runner never creates placeholder rows. Missing
slots count as misses for supported-significant precision@5 and valid-novelty
yield@5. Recall keeps the human-cluster denominator, and duplicate rate is
calculated among returned issues only.

The runner atomically rewrites a private in-progress packet after each completed
case and creates a content-addressed private checkpoint before the final audit.
The canonical `generated_adjudication.csv` is written only after all run gates
pass. Generated text and evidence IDs therefore remain recoverable after an
interruption or failed audit but never enter portable output. A completed paid
result remains `pending_human_adjudication` until every returned issue is labeled.
Manifest runs do not calculate provisional recall against raw feedback; scoring
begins only after these manual labels are complete. Changing a source, extraction
rule, manuscript, cluster, output-cardinality policy, or generated issue
invalidates the corresponding labels.

See [DESIGN.md](DESIGN.md) for implementation details.
