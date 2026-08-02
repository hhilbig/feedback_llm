# feedback_llm Status

## Current State

The private-feedback benchmark infrastructure is implemented. It imports
hash-bound manuscript-review pairs, prepares private adjudication packets, runs
a feedback-free cold baseline behind an explicit cost ceiling, and emits only
pseudonymous aggregate results outside the private corpus directory. Reviewer
prompts, model routing, reasoning settings, thresholds, and report synthesis
have not changed in this milestone.

The five-family source freeze is not yet complete. Two cases are materialized
and hash-verified locally. The exact CA Insurance, German elections, and DDR
submission bundles are still cloud-only Dropbox placeholders, so the importer
correctly rejects them before cost estimation.

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
   source content, extraction rules, cluster membership, or baseline output
   invalidate the corresponding labels.
5. Cold evaluation mode that passes neither a review corpus nor a reviewer
   prior to the pipeline and scores only the final post-verification top five.
6. Zero-call dry runs, finite positive cost ceilings, complete preflight cost
   checks, fixed pilot-composition checks before the first request, and
   remaining-budget checks before later cases.
7. Portable output projection that removes paths, reviewer identities, source
   text, generated text, locators, and notes.
8. Tests for extractors, mixed human/generated feedback, deduplication,
   family holdout, cold-run isolation, deterministic sampling, tamper detection,
   cost gates, partial-run rejection, and output privacy.

## Source Freeze

| Family | Current status | Pilot treatment |
|---|---|---|
| Great Recession | Ready: exact annotated manuscript, 59 nonempty annotations | Primary |
| CA Insurance | Blocked: exact submission and raw reports are cloud-only | Primary, journal |
| German elections | Blocked: exact initial submission, appendix, and review packet are cloud-only | Primary, journal |
| DDR | Blocked: exact CPS manuscript, supplement, and reviews are cloud-only | Primary, journal |
| Insurers | Ready: manuscript plus human items 1–7; generated items 8–20 excluded | Secondary only |

## Next Validation Gate

1. Materialize the three blocked Dropbox bundles and copy them into
   `~/.feedback_llm/` without editing the Dropbox originals.
2. Freeze the complete five-case manifest and verify four primary
   manuscript-feedback pairs by hash.
3. Regenerate the gold packet, screen every cluster, fully adjudicate every
   major cluster, and adjudicate the deterministic minor sample.
4. Run the zero-call cold-baseline cost estimate and obtain explicit approval
   for the reported total.
5. Run the paid baseline only if the completed gold packet is current and the
   estimate is below the approved ceiling.
6. Label every generated top-five issue, then finalize the primary,
   journal-only, sampled-minor, precision, novelty, duplicate, and cost metrics.

Do not change reviewer prompts, models, routing, reasoning effort, thresholds,
or synthesis until the completed baseline has been human-adjudicated.
