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
The gold-adjudication packet is ready but has not been reviewed by a human.

The cold-baseline dry run estimated a total API cost of $9.7836 and made no API
calls. This estimate exceeds the previously discussed $2--$3 range, so no paid
run is authorized. The German AJPS source required one additional safeguard:
only verbatim reviewer passages were transcribed from the response memo. Author
responses and the merged resubmission remain hash-bound deferred sources and
are not evaluation targets.

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
| CA Insurance | Ready: exact submission, editor report, and three raw reviewer reports | Primary, journal |
| German elections | Ready: original manuscript and appendix plus 30 response-free, verbatim reviewer items | Primary, journal |
| DDR | Ready: exact CPS manuscript and supplement plus nine issues from three reviewers | Primary, journal |
| Insurers | Ready: manuscript plus human items 1–7; generated items 8–20 excluded | Secondary only |

## Next Validation Gate

1. Screen all 145 clusters in the gold packet. Complete the required fields for
   all major clusters and the deterministic minor sample (34 full rows).
2. Re-run the zero-call cost estimate after adjudication and obtain explicit
   approval for the reported total. The current estimate is $9.7836.
3. Run the paid baseline only if the completed gold packet is current and the
   estimate is below the approved ceiling.
4. Label every generated top-five issue, then finalize the primary,
   journal-only, sampled-minor, precision, novelty, duplicate, and cost metrics.

Do not change reviewer prompts, models, routing, reasoning effort, thresholds,
or synthesis until the completed baseline has been human-adjudicated.
