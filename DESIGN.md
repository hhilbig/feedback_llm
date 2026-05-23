## Design Overview

This document describes the current implementation in `feedback_pipeline.py` and
`streamlit_app.py`. The system is an evidence-first, cost-routed feedback pipeline
for quantitative social science manuscripts.

### Pipeline Flow

```text
Paper text
  |
  |-- Evidence map
  |     Strip hidden/control characters
  |     Quarantine instruction-like manuscript text
  |     Index sections, paragraphs, tables, figures, equations, appendices
  |     Extract research question, design, sample, measures, claims, results
  |
  |-- Design-aware generation
  |     8 specialized agents produce evidence-linked proposals
  |     Agent block: 2 theorists, 2 rivals, 2 methodologists,
  |     1 design specialist, 1 editor
  |
  |-- Grounding check
  |     Regex guardrail flags missing table, figure, section, appendix,
  |     column, panel, and equation references
  |
  |-- Domain scoring and escalation
  |     Cheap judge scores validity-relevant dimensions twice
  |     Severe, ambiguous, low-confidence, or low-agreement items escalate
  |
  |-- Selection and evidence-aware deduplication
  |     Keep high-quality proposals by domain composite
  |     Protect severe evidence-distinct minority critiques
  |
  |-- Verification-first adjudication
  |     Verifier checks support, severity, evidence IDs, counter-evidence,
  |     and actionability, then keeps, demotes, or removes proposals
  |
  |-- Constrained rewrite and presentation clustering
  |     Rewrite is clarity-only and cannot add factual claims or evidence IDs
  |     Clustering labels related proposals only; it does not merge/drop them
  |
  |-- Meta-review
        Final evidence-linked markdown report with prioritized revisions
```

### Entry Points

**Streamlit app:** `streamlit_app.py`

- Requires `OPENAI_API_KEY` from `.env` or the environment.
- Accepts pasted text or uploaded PDFs.
- Lets the user choose generation model, number of agents, and `top_k`.
- Shows a pre-run cost estimate.
- Displays progress across the pipeline stages.
- Saves recent runs to `~/.feedback_llm/history.json`, keeping the most recent 50 entries.
- Lets the user download the final report as markdown or copy it from the browser.

**CLI:** `python -m feedback_pipeline`

- Input sources are mutually exclusive: `--file`, `--paste`, `--clipboard`, or `--pdf`.
- If no input flag is passed, the CLI reads piped stdin, then `paper.txt` if present,
  then falls back to interactive paste.
- `--agents` must be a multiple of 8.
- `--model` must be one of the allowed model keys in `MODEL_REGISTRY`.
- `--top-k` controls how many top proposals are emphasized in the meta-review.
- Actual token usage and cost are printed unless `--no-cost-estimate` is passed.

### Models and Routing

The model registry is defined in `MODEL_REGISTRY`. Defaults are:

- Generation: `gpt-5.4-mini`
- Scoring: `gpt-5.4-mini`
- Verification: `gpt-5.4-mini`
- Rewrite and simple labels: `gpt-5.4-nano`
- Meta-review: `gpt-5.5`
- Escalation: `gpt-5.5`

Previous GPT-5 family models remain allowed for reproducibility. The registry includes
pricing so estimates and actual usage can be reconciled by stage.

### Evidence Map

The ingestion layer builds deterministic manuscript evidence IDs before any critique
generation:

- `SEC###` for sections
- `P###` for paragraphs
- `TBL###` for tables
- `FIG###` for figures
- `EQ###` for equations
- `APP###` for appendices
- `Q###` for quarantined instruction-like lines

`sanitize_manuscript_text()` removes zero-width/control characters and quarantines
prompt-injection-like text. `build_deterministic_evidence_index()` creates auditable
IDs. `extract_manuscript_evidence_map()` then uses strict structured output to extract:

- research question
- research design
- estimand
- sample
- measures
- main claims
- identification assumptions
- main results
- robustness checks
- tables, figures, appendices
- limitations

The extracted map is used by generation, verification, and meta-review.

### Generation

Agents are created in blocks of 8:

- 2 Theorists: contribution, logic, assumptions, mechanisms, and theoretical framing.
- 2 Rival researchers: alternative explanations, rival mechanisms, omitted variables,
  contextual factors, and selection effects.
- 2 Methodologists: identification clarity, measurement, sample construction, data limits,
  and statistical interpretation.
- 1 Design specialist: method-specific threats for DiD, IV, RD, experiments, surveys,
  descriptive work, panel observational designs, qualitative work, and mixed methods.
- 1 Editor: clarity, organization, and structure.

Generation uses `FEEDBACK_PROPOSAL_SCHEMA`. Each proposal includes:

- `id`
- `dimension`
- `issue_family`
- `affected_claim_ids`
- `evidence_ids`
- `support_status`
- `severity`
- `confidence`
- `text`
- `diagnostic_next_steps`

Generation runs in parallel. If some workers fail, the pipeline keeps successful
proposals; if all fail, it aborts.

### Grounding Check

Generated proposals are annotated before scoring. The guardrail extracts references
such as `Table 1`, `Figure 2`, `Section 4.1`, `Appendix A`, `Column 2`, `Panel B`,
and `Equation 1`, then checks whether those strings appear in normalized manuscript
text.

Grounding failures add:

- `grounding_flag = True`
- `missing_refs = [...]`

Flagged proposals remain available, but verifier and meta-review prompts treat
unsupported specifics skeptically.

### Scoring and Escalation

The scorer uses `SCORING_SCHEMA`, not a generic importance/specificity rubric. It
scores:

- `identification_risk`
- `measurement_sample_risk`
- `interpretation_risk`
- `theory_contribution_risk`
- `evidence_support`
- `actionability`
- `severity`
- `confidence`

Each proposal is scored twice with swapped context/rubric order. For compatibility
with older downstream code, `importance` aliases `severity`, `specificity` aliases
`evidence_support`, and `uniqueness` is a diversity priority rather than a quality
component.

The composite is:

```text
0.35 * severity
+ 0.25 * evidence_support
+ 0.20 * actionability
+ 0.20 * max(domain risk scores)
```

It is adjusted by reviewer agreement and scorer confidence. Items escalate to the
frontier model when they are severe and ambiguous, low-confidence, low-agreement,
or high-impact with weak evidence support.

### Selection and Deduplication

Selection keeps:

- top proposals by domain composite
- low-value IDs for diagnostics
- high-quality proposals by threshold
- diversity-priority proposals
- proposals grouped by dimension

High-quality proposals are semantically deduplicated with `text-embedding-3-small`
when embeddings are available. If embeddings fail, the code falls back to lexical
Jaccard similarity.

Deduplication is evidence-aware. Severe critiques in protected issue families
(`identification_design`, `measurement_sample`, `results_interpretation`) are
preserved when they cite different evidence or claims, even if their wording is
semantically similar to another proposal.

### Verification and Rewrite

The current pipeline does not use a discussant critique/revision loop. Instead,
`run_verification_round()` applies `VERIFICATION_SCHEMA` to adjudicate each
high-quality proposal:

- `decision`: keep, demote, or remove
- `support_assessment`: supported, partially supported, inferential, unsupported,
  or contradicted
- `verified_severity`
- `supported_evidence_ids`
- `missing_or_invalid_evidence_ids`
- `counter_evidence_ids`
- actionability and confidence

`apply_verification_decisions()` removes contradicted/unsupported proposals, demotes
weakly supported ones, and preserves verifier metadata. `run_constrained_rewrite_round()`
then rewrites only for clarity. The rewrite prompt forbids new factual claims, new
tables, new variables, new results, or new evidence IDs.

### Presentation Clustering

Clustering is presentation-only. It annotates proposals with:

- `cluster_id`
- `cluster_size`
- `source_ids`

It does not synthesize, replace, merge, or remove proposals. This prevents rare
high-severity critiques from being hidden by a broader cluster label.

### Meta-review Output

The meta-review receives:

- high-quality proposals by dimension
- globally strongest proposals by domain composite
- diversity-priority proposals
- verifier adjudications
- all high-quality proposals with evidence IDs, support status, verification status,
  risk scores, and cluster metadata

The final report starts with `## Narrative Summary` and covers:

1. Identification and design
2. Measurement and sample construction
3. Empirical interpretation
4. Theory and contribution
5. Writing and structure, only if writing is a binding issue

It then writes `## Proposed Revisions`, a prioritized numbered list of 3-5 revisions.
Each revision is marked `[REQUIRED]` or `[SUGGESTED]`, includes evidence IDs and support
status, and gives a one-sentence justification.

The CLI and app download output append `## Evidence Lookup` after the narrative report.
This deterministic appendix extracts cited evidence IDs from the final report and prints
their manuscript type, section, source lines, and excerpt. It does not make another API
call. Use `--no-evidence-appendix` for narrative-only CLI output.

### Cost Tracking

`estimate_cost_before_run()` estimates costs from prompt templates, paper length,
agent count, `top_k`, assumed output lengths, routing defaults, and known model
prices. During a run, `UsageTracker` records token usage by stage and
`compute_actual_cost()` computes actual costs, including cached-input pricing when
reported by the API.

Tracked stages include evidence map, generation, scoring, score escalation,
verification, rewrite, clustering, embeddings, and meta-review.

### Reliability and Safety Features

- The OpenAI client is lazy, so imports and offline tests do not require an API key.
- System prompts treat manuscript text as untrusted data.
- Instruction-like manuscript lines are quarantined before review agents see text.
- Structured stages use strict JSON Schema outputs.
- Chat calls retry transient rate-limit, connection, and timeout errors.
- Generation supports partial failure recovery.
- Embedding failure during deduplication falls back to lexical similarity.
- Presentation clustering failure does not abort the run.

### Tests

The `tests/` directory uses `unittest` and mocked API calls for deterministic coverage.
Current test modules cover:

- model routing and structured output helpers
- evidence-map indexing and quarantine
- evidence-ID-aware generation
- verification-first adjudication
- domain scoring, escalation, protected deduplication, and presentation clustering
- mocked end-to-end pipeline behavior
