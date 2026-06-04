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
  |     Build deterministic substantive profile and checklist findings
  |     for DD/DDD, inference, repeated cross-sections, and text-as-data
  |
  |-- Design-aware generation
  |     8 specialized agents produce evidence-linked proposals
  |     Agent block: 2 theorists, 2 rivals, 2 methodologists,
  |     1 design specialist, 1 editor
  |     Optional anonymized review-digest memory calibrates tone and reviewer salience
  |
  |-- Grounding check
  |     Regex guardrail flags missing table, figure, section, appendix,
  |     column, panel, and equation references
  |
  |-- Domain scoring and escalation
  |     Cheap judge scores validity-relevant dimensions twice
  |     Severe, ambiguous, low-confidence, or low-agreement items escalate
  |     Optional historical-review calibration adds separate reviewer-likelihood
  |     and decision-risk scores without replacing scientific-validity scores
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
  |-- Editorial triage
  |     Classify verified issues by publication-decision relevance
  |     Enforce caps on rejection reasons, major blockers, and non-blocking items
  |
  |-- Editorial report
        Decision-relevant report plus optional audit and evidence lookup appendices
```

### Entry Points

**Streamlit app:** `streamlit_app.py`

- Requires `OPENAI_API_KEY` from `.env` or the environment.
- Accepts pasted text or uploaded PDFs.
- Lets the user choose generation model, number of agents, and `top_k`.
- Lets the user choose editorial triage or comprehensive audit mode.
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
- `--top-k` controls how many top proposals are emphasized before editorial triage.
- `--include-evidence-appendix` appends deterministic evidence excerpts for cited IDs.
- `--include-audit-appendix` appends the full deterministic substantive checklist and evidence lookup.
- Actual token usage and cost are printed unless `--no-cost-estimate` is passed.
- `--inspect-review-corpus --review-corpus PATH` loads an archived review folder,
  prints record/issue/paper-match counts, and exits without calling the API.
- `--include-low-confidence-reviews` includes forwarded/low-confidence records in
  local corpus inspection only.
- `--review-corpus PATH` on a normal run enables private historical-review memory
  for generation and editorial triage.
- `--eval-review-corpus PATH` builds whole-paper held-out splits from the archive,
  estimates per-split API cost, and optionally saves a JSON plan without calling
  the API.
- `--eval-run-api` actually runs the held-out evaluation. It should be used only
  after reviewing the dry-run estimate because it calls the paid API.

### Historical Review Corpus

The optional review-corpus layer turns a private archive of past journal decisions
and matched papers into local calibration data. The current archive files are
extracted review digests, not certified raw referee-report exports. The layer does
not train a model and does not expose old reviews in the final report. The current
implementation:

- Parses archived Markdown review digests and `papers/PAPER_MATCHES.md`.
- Links each review record to matched paper PDFs when available.
- Excludes forwarded/low-confidence records by default.
- Reads optional raw Gmail sidecars from `raw_gmail_exports/*.md`. When a sidecar
  names a `Review file`, its raw reviewer/editor sections replace the digest
  sections for that record. Records without sidecars still use digests.
- Atomizes reviewer/editor sections into raw-or-digest-derived issue candidates with issue
  type, decision tier, action-request, tone, paper section, reviewer-confidence
  heuristic, design type, and paper-match metadata.
- Uses pseudonymous paper IDs and redacts titles, submission IDs, message IDs,
  emails, URLs, and obvious author identifiers before prompt memory is built.
- Retrieves similar historical issues with a transparent lexical baseline plus
  design/type metadata. If no issue clears the similarity threshold, no memory
  block is sent; there is no generic fallback to unrelated major comments.
- Builds review-memory prompt context for tone, specificity, and reviewer salience
  only. Prompts explicitly forbid importing facts from historical reviews.
- Adds `reviewer_likelihood_score`, `decision_risk_score`, and
  `similar_issue_ids` to proposals when `--review-corpus` is enabled.
- Provides local held-out evaluation scaffolding through
  `compare_generated_to_human_issues()`. Matching now uses a local semantic
  matcher: weighted issue-concept/token features retrieve the nearest held-out
  review issue, and a rule-based verifier labels matched, partially matched, or
  novel/unmatched issues. It is still local and non-API, not a paid LLM verifier.
- Provides a whole-paper held-out eval harness through
  `run_historical_review_eval()`. Each split removes every review round for one
  pseudonymous paper ID from review memory, runs or plans the pipeline on the
  matched paper, and compares generated issues against the held-out issue
  candidates. Dry-run mode estimates costs only and makes no API calls. API-mode
  JSON keeps compact generated-issue summaries so matching logic can be audited
  later without rerunning the paid pipeline.

The reviewer-likelihood score is intentionally separate from scientific validity.
A comment can be likely to appear in a referee report without being correct, and a
valid concern can be absent from the historical corpus.

### Models and Routing

The model registry is defined in `MODEL_REGISTRY`. Defaults are:

- Generation: `gpt-5.4-mini`
- Scoring: `gpt-5.4-mini`
- Verification: `gpt-5.4-mini`
- Rewrite and simple labels: `gpt-5.4-nano`
- Editorial triage: `gpt-5.5`
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

The extracted map is used by generation, verification, editorial triage, and final reporting.

### Substantive Design Profile

After evidence-map construction, deterministic checkers classify substantive designs,
data types, and key risks. This stage deliberately ignores spelling, prose polish, and
LaTeX formatting. It focuses on omission risks that matter for quantitative review:

- DD/DDD and event-study checks for parallel trends, treatment timing, anticipation,
  inference level, group-time cell sizes, and placebo groups.
- Survey and repeated-cross-section checks for composition, sampling, weights, and
  small treated groups.
- Text-as-data and LLM-coded outcome checks for validation samples, prompt/held-out
  separation, model-version reproducibility, confusion matrices, prevalence,
  measurement error, conditional outcomes, and article-level temporal dependence.

The resulting `substantive_profile` and `substantive_checks` are passed to generation,
verification, and editorial triage as omission cues. They are not treated as proof of a flaw:
LLM stages must still ground any substantive critique in evidence IDs or clearly mark it
as inferential.

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

Flagged proposals remain available, but verifier and final-report prompts treat
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

### Editorial Triage

Before the final report is written, `editorial_triage()` classifies verified proposals
and unresolved substantive checklist findings into a clear problem list with separate
decision-risk labels.

Rejection-risk labels:

- `high`
- `conditional`
- `low`
- `none`

Decision-tier labels:

- `potential_rejection_reason`
- `major_revision_issue`
- `minor_revision_issue`
- `nice_to_have`
- `drop`

This stage asks two separate questions: whether an issue is important enough to appear
in a serious problem list, and how much rejection risk it carries. The triage schema
records why the problem matters, what would make it rejection-level, why it is not
currently rejection-level, the minimum fix, fixability, affected core claim, evidence
strength, existing mitigations, output location, and recommended action.

Hard post-processing rules enforce editorial discipline:

- at most 2 potential rejection reasons
- at most 8 clear problems in the main problem list
- demote verifier-demoted issues by one level
- demote rejection-level checklist diagnostics unless triage gives a conditional/high
  rejection-risk rationale

### Editorial Report Output

The final report receives:

- editorial triage classifications
- issue inputs used for triage
- verified high-quality proposals as context

The final report starts with `## Editorial Summary` and then writes:

1. `## Clear Problems and Rejection Risk`
2. `## Notes on Non-Rejection Issues`

The main problem list is capped at 5-8 clear problems when enough non-marginal issues
are available. It explicitly says whether the extracted evidence shows no clear
rejection-level flaw, conditional rejection risks, potential rejection reasons, mostly
major-revision issues, or mostly minor issues. It does not print the full coverage
audit or evidence lookup by default.

Use `--include-evidence-appendix` to append `## Evidence Lookup` after the narrative
report. This deterministic appendix extracts cited evidence IDs from the final report
and prints their manuscript type, section, source lines, and excerpt. It does not make
another API call.

When `--include-audit-appendix` or comprehensive audit mode is used, the output appends
both `## Evidence Lookup` and `## Substantive Coverage Audit`. The coverage checklist
reports the detected design/data profile, unresolved deterministic substantive
findings, and whether the final report visibly addressed applicable review categories
such as inference, measurement, sample construction, robustness, interpretation, and
text-as-data validation.

### Cost Tracking

`estimate_cost_before_run()` estimates costs from prompt templates, paper length,
agent count, `top_k`, assumed output lengths, routing defaults, and known model
prices. During a run, `UsageTracker` records token usage by stage and
`compute_actual_cost()` computes actual costs, including cached-input pricing when
reported by the API.

Tracked stages include evidence map, generation, scoring, score escalation,
verification, rewrite, clustering, embeddings, editorial triage, and meta-review.

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
- editorial triage caps and decision-impact report prompting
- mocked end-to-end pipeline behavior
