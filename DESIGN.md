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
  |-- Cold design-aware generation
  |     8 specialized agents produce evidence-linked proposals
  |     Agent block: 2 theorists, 2 rivals, 2 methodologists,
  |     1 design specialist, 1 editor
  |     With `--review-prior`, this first pass remains paper-only
  |
  |-- Optional structured reviewer-prior gap pass
  |     Select applicable, unsatisfied priors after the cold pass
  |     Generate targeted proposals only for missing high-salience checks
  |     Priors may raise checks but cannot supply facts or evidence support
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
  |     Structured priors can calibrate reviewer likelihood and decision tier
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
- `--inspect-review-corpus --review-corpus PATH` loads either a legacy archive,
  a private manifest, or a normalized private snapshot and exits without an API call.
- `--review-corpus-output PATH` writes the normalized private snapshot below
  `~/.feedback_llm/` with mode `0600`.
- `--prepare-review-adjudication DIR` writes the hash-bound gold-cluster CSV and
  Markdown reading packet without an API call.
- `--include-low-confidence-reviews` includes forwarded/low-confidence records in
  local corpus inspection only.
- `--review-corpus PATH` on a normal run enables private historical-review memory
  for generation and editorial triage.
- `--review-prior PATH` on a normal run enables the API-safe structured reviewer
  prior. The pipeline first runs cold generation without the prior, then uses it
  for targeted post-cold-pass gap checks and editorial triage calibration.
- `--distill-review-prior PATH` builds the structured reviewer-prior artifact
  locally from a private review archive without calling the API.
- `--eval-review-corpus PATH` builds whole-paper held-out splits from the archive,
  estimates per-split API cost, and optionally saves a JSON plan without calling
  the API.
- `--eval-memory-mode none` runs the cold baseline. Historical feedback remains
  local for scoring and neither a review corpus nor a review prior reaches the pipeline.
- `--eval-adjudication PATH` supplies completed gold labels for a manifest run.
- `--eval-gold-mode complete` is the default and requires every cluster screen.
  `--eval-gold-mode partial` freezes only the currently completed labels after
  confirming that every sampled row and every row screened as major is fully
  adjudicated. Partial results use an adjudicated-only denominator and are never
  reported as exhaustive major recall.
- `--finalize-review-eval LOCAL_AUDIT` combines the completed gold and generated
  packets into privacy-safe aggregate metrics without an API call.
- `--eval-max-cost-usd N` is mandatory with `--eval-run-api`. Evaluation performs
  a complete zero-call preflight and rejects a total above the ceiling.
- `--eval-review-prior-gate PATH` compares three held-out modes: no memory
  baseline, API-safe structured reviewer prior, and local raw-memory upper bound.
  Dry-run mode estimates the full three-way cost without calling the API.
- `--eval-batch-api` routes the reviewer-prior gate's chat-completion calls
  through OpenAI Batch API. Because the pipeline has dependent stages, batching
  is stage-by-stage: each concurrent wave is submitted, polled to completion, and
  then the next local stage proceeds. Local JSONL inputs and a batch manifest are
  written under the requested batch output directory.
- `--eval-run-api` actually runs the held-out evaluation. Manifest evaluation
  requires current gold under the selected gold mode. The runner checks the
  remaining cost allowance before each manuscript family.

### Historical Review Corpus

The optional review-corpus layer accepts two inputs. The legacy path reads archived
review digests. The manifest path binds exact manuscript versions to raw human
feedback for evaluation. Neither path trains a model. Historical feedback enters a
normal paper run only when review memory is explicitly enabled.

#### Manifest corpus

The manifest is private JSON governed by `review_manifest.schema.json`. Each case
contains a family ID, case ID, ordered manuscript files, benchmark tier, version-match
status, source disposition, extraction rules, and SHA-256 hashes. The family ID is
the holdout boundary, so renamed manuscripts and later rounds from the same research
project cannot enter calibration memory for that family.

Supported extractors are PDF annotations, Word comments, review-PDF text, Markdown,
text, and Word body text. PDF manuscript extraction ignores annotation comments.
Word comment extraction reads Office XML directly and keeps the selected manuscript
anchor. Source selectors can restrict pages, headings, markers, numbered items,
comment IDs, and annotation types. RTF and XLSX are unsupported. Missing, empty,
stale, or dataless cloud files fail before cost estimation. Evaluation sources
must declare human provenance and source type; AI-generated, derivative-summary,
and response material fail closed.

The importer emits normalized records and atomic issues with pseudonymous reviewer
IDs, source locators, anchor text, source/manuscript hashes, family/case provenance,
benchmark tier, and disposition. It removes byte-identical sources and declared
duplicate representations while retaining independent reviewers. Private snapshots
stay below `~/.feedback_llm/` with directory mode `0700` and file mode `0600`.

#### Legacy archive and shared evaluation behavior

The legacy archive files are extracted review digests unless a raw Gmail sidecar is
available. The implementation:

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
- Provides local held-out evaluation scaffolding for legacy archives through
  `compare_generated_to_human_issues()`. Matching uses a local semantic
  matcher: weighted issue-concept/token features retrieve the nearest held-out
  review issue, and a rule-based verifier labels matched, partially matched, or
  novel/unmatched issues. Manifest benchmarks do not use this provisional
  comparison because their generated issues require manual labels.
- Provides a whole-paper held-out eval harness through
  `run_historical_review_eval()`. Manifest splits exclude the complete manuscript
  family; legacy splits retain paper-ID fallback behavior. The cold baseline passes
  `review_corpus=None` and `review_prior=None` explicitly. It scores the final
  post-verification `top_proposals[:5]`, not the larger candidate pool. A
  successful manifest case may return one to five real issues; the evaluator
  never pads a short result with placeholder rows.
- Plans every selected case and totals its estimated cost before any API request.
  Paid mode requires a positive cost ceiling and checks the remaining allowance
  before each case. A paid pilot rejects partial selections and requires the fixed
  four-primary/one-secondary composition, including three primary journal cases,
  before its first request. A manifest paid run also requires a clean committed
  `feedback_llm` worktree. The run records the commit, dirty flag, routing, reasoning
  effort, reviewer-role counts, thresholds, hashes, token usage, and costs.
- Writes portable evaluation JSON containing pseudonymous IDs, counts, scalar
  metrics, hashes, and costs only. Paths, reviewer identities, human text,
  generated issue text, locators, and adjudicator notes remain in local private
  artifacts.
- Provides a reviewer-prior deployment gate through
  `run_review_prior_eval_gate()`. Each split distills the safe prior from training
  reviews only, with the held-out paper-review pair excluded, then compares
  baseline, safe-prior, and local raw-memory modes on major human issue-cluster
  recall, deduplicated reviewer-likelihood precision, unsupported-claim rate, and
  duplicate laundry-list rate.
- Provides local reviewer-prior distillation through
  `distill_review_prior_from_corpus()`. This turns private review issue clusters
  into a structured, API-safe prior artifact with controlled reviewer-concern
  templates, executable `raise_if_missing`, `demote_if_present`, and `suppress_if`
  rules, bucketed support levels, decision-tier priors, and a privacy audit. Exact
  support counts, issue IDs, review files, and low-support exclusions are written
  only to the separate local audit metadata file.

The reviewer-likelihood score is intentionally separate from scientific validity.
A comment can be likely to appear in a referee report without being correct, and a
valid concern can be absent from the historical corpus.

The reviewer-prior artifact follows the same rule: priors may raise checks, rank
salience, or shape final wording, but they may not support factual claims or affect
verification. Manuscript evidence remains the only source of support for a critique.

#### Baseline adjudication and metrics

Automatic clustering proposes deduplicated human concerns, but it does not create
the final gold labels. The gold packet lists every cluster for a tier screen. It
requires full adjudication for all major clusters and a hash-ranked sample of five
minor clusters per family using seed `20260802`. The packet records inclusion,
canonical wording, severity, evidentiary support, duplicate corrections, and an
exclusion reason where relevant. Its binding hash changes when source content,
manuscript content, extraction rules, or cluster membership changes.

After a paid run, a second packet contains the final post-verification issues, up
to five per case. Every row requires correctness, significance, evidence
sufficiency, match status, duplicate status, and valid-novelty labels. Lexical
matching supplies suggestions only. An unmatched generated concern is not counted
as false until a person labels it. Each successful case must contain one to five
contiguously ranked issues; no synthetic fifth row is added.

The benchmark treats each case as five available output slots. Missing slots count
as misses for supported-significant precision@5 and valid-novelty yield@5. Human
cluster recall keeps its human-target denominator, while duplicate rate uses only
the issues actually returned. The run metadata and generated-packet binding include
this output-cardinality policy.

After every completed case, the runner atomically rewrites a private in-progress
packet containing full issue text and evidence IDs. After all cases finish, it
writes a content-addressed checkpoint before applying the final run audit. A failed
audit does not create or replace the canonical packet, but its checkpoint remains
available locally. Neither checkpoint metadata nor checkpoint content enters the
portable projection.

A completed paid generation run remains `pending_human_adjudication` until all
returned issue labels are current; incomplete family runs cannot reach that state.
Final aggregate metrics are produced only when the generated packet is current and
complete. In complete-gold mode, the gold packet must also contain every tier
screen; results report family-macro major-cluster recall, the journal-only subset,
sampled-minor recall, supported-significant precision, valid-novelty yield,
duplicate rate, and cost. Secondary cases remain outside the primary aggregate.

Partial-gold mode is an explicit exploratory exception to the complete-screen
gate. It first verifies that every deterministic minor-sample row and every row
screened as major has all required fields. It binds the source packet, the set of
completed rows, and every scoring-relevant label into a new evaluation hash.
Only fully adjudicated rows marked `include=yes` can be proposed as matches,
confirmed as human-cluster matches, or enter metric denominators. Completed
exclusions and tier-only minor screens count toward coverage but not scoring.
Completing another row or changing a substantive label invalidates the earlier
generated packet. Partial output reports completed-row coverage, the eligible
scoring count, and the number of primary families with a validated-major
denominator. It omits unqualified exhaustive major-recall field names.

### Models and Routing

The model registry is defined in `MODEL_REGISTRY`. Defaults are:

- Generation: `gpt-5.6-terra` (`reasoning_effort="none"`)
- Scoring: `gpt-5.6-terra` (`reasoning_effort="none"`)
- Verification: `gpt-5.6-terra` (`reasoning_effort="none"`)
- Rewrite and simple labels: `gpt-5.6-luna` (`reasoning_effort="none"`)
- Editorial triage: `gpt-5.6-sol` (`reasoning_effort="medium"`)
- Meta-review: `gpt-5.6-sol` (`reasoning_effort="medium"`)
- Escalation: `gpt-5.6-sol` (`reasoning_effort="medium"`)

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
- manifest validation, PDF/Word/text extraction, source deduplication, ordered
  manuscript bundles, dataless cloud rejection, and legacy-loader compatibility
- family-level leakage controls, cold-baseline prompt isolation, up-to-five scoring,
  cost ceilings, portable output privacy, and deterministic adjudication artifacts
