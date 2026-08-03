# feedback_llm Status

## Current State

The private-feedback benchmark infrastructure is implemented. It imports
hash-bound manuscript-review pairs, prepares private adjudication packets, runs
a feedback-free cold baseline behind an explicit cost ceiling, and emits only
pseudonymous aggregate results outside the private corpus directory. Reviewer
prompts, model routing, reasoning settings, thresholds, and report synthesis
have not changed in this milestone.

The five-family source freeze is complete. All manuscript and feedback files
are materialized, copied into the private corpus, and hash-verified. The offline
import produces 12 human-review records, 150 issue candidates, and 145 clusters.
The gold-adjudication packet is partly complete. A human has completed 52 of 145
cluster rows. All 36 rows that currently require full adjudication are complete;
30 are included scoring targets, 16 are completed exclusions, and six are
tier-screened minor rows outside the deterministic sample. The remaining 93
rows still need a tier screen before the strict benchmark can run.

The cold-baseline dry run estimated a total API cost of $9.7836. The first paid
attempt cost $9.8542 and kept historical feedback out of every prompt. It returned
5, 5, 4, 4, and 5 final issues across the five cases. The old exact-five audit gate
therefore marked the run incomplete even though all manuscripts were evaluated.
That attempt retained cost and run metadata but not enough untruncated issue text
to create a defensible adjudication packet.

The benchmark now accepts one to five real final issues per case. Missing slots
count as misses for supported-significant precision@5 and valid-novelty yield@5;
duplicate rate remains conditional on returned issues. The runner writes a private
full-text checkpoint after each completed case and before the final audit. A fresh
five-manuscript rerun is authorized with a $10 ceiling. The German AJPS source
still includes only verbatim reviewer passages from the response memo. Author
responses and the merged resubmission remain hash-bound deferred sources and are
not evaluation targets.

## Implemented

1. Manifest v1 with family-level leakage boundaries, ordered manuscript
   bundles, source hashes, pseudonymous reviewers, provenance, selectors,
   benchmark tiers, and source dispositions.
2. Offline extraction for PDF annotations, Word comments, review PDFs, and
   Markdown, text, or Word body feedback. The importer rejects stale hashes,
   unsupported files, missing files, and cloud placeholders.
3. Deterministic issue atomization, source deduplication, provisional
   clustering, major-cluster screening, and a five-minor-cluster sample per
   family using seed `20260802`.
4. Hash-bound private gold and generated-issue adjudication packets. Changes to
   source content, extraction rules, cluster membership, output-cardinality
   policy, or baseline output invalidate the corresponding labels. Incremental
   and content-addressed checkpoints preserve full generated text and evidence
   IDs without entering portable output.
5. Cold evaluation mode that passes neither a review corpus nor a reviewer
   prior to the pipeline and scores only the final post-verification issues, up
   to five per case.
6. Zero-call dry runs, finite positive cost ceilings, complete preflight cost
   checks, fixed pilot-composition checks before the first request, and
   remaining-budget checks before later cases.
7. Portable output projection that removes paths, reviewer identities, source
   text, generated text, locators, and notes.
8. Tests for extractors, mixed human/generated feedback, deduplication,
   family holdout, cold-run isolation, deterministic sampling, tamper detection,
   cost gates, partial-run rejection, and output privacy.
9. An explicit partial-gold pilot mode that freezes completed labels, scores only
   the 30 fully adjudicated included clusters, and labels major recall as
   non-exhaustive. The strict complete-gold gate remains the default.
10. An up-to-five output policy with fixed five-slot denominators for supported
    precision and novelty yield, unchanged human-cluster recall denominators,
    and returned-issue denominators for duplicate rates.

## Source Freeze

| Family | Current status | Pilot treatment |
|---|---|---|
| Great Recession | Ready: exact annotated manuscript, 59 nonempty annotations | Primary |
| CA Insurance | Ready: exact submission, editor report, and three raw reviewer reports | Primary, journal |
| German elections | Ready: original manuscript and appendix plus 30 response-free, verbatim reviewer items | Primary, journal |
| DDR | Ready: exact CPS manuscript and supplement plus nine issues from three reviewers | Primary, journal |
| Insurers | Ready: manuscript plus human items 1–7; generated items 8–20 excluded | Secondary only |

## Next Validation Gate

1. Run the full offline suite and commit the output-policy and checkpoint change
   so the rerun has a clean, frozen code state.
2. Re-run the zero-call partial-gold estimate. Proceed only if the current corpus
   and gold bindings remain valid and the estimate is below the approved $10
   ceiling.
3. Run the authorized five-manuscript cold baseline and verify that all five cases
   finish, actual cost stays below $10, and every split records zero review-memory
   inputs.
4. Verify the canonical generated packet, content-addressed checkpoint, private
   permissions, portable-output redaction, and binding consistency.
5. Label every returned generated issue, then finalize explicitly partial,
   non-exhaustive major-recall results plus sampled-minor, precision, novelty,
   duplicate, and cost metrics.
6. Screen the remaining 93 rows later if an exhaustive benchmark is still worth
   the additional labeling time.

Do not change reviewer prompts, models, routing, reasoning effort, thresholds,
or synthesis until the completed baseline has been human-adjudicated.
