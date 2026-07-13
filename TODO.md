# feedback_llm Status

## Current Focus

Implement a privacy-preserving reviewer-prior distillation layer. Raw reviews and
matched paper-review pairings stay local. API-facing runs should use only a
versioned, audited, structured reviewer-prior artifact.

## Implementation Stages

1. Done: structured reviewer-prior schema, privacy audit, and post-cold-pass gap
   selection helpers.
2. Done: local distillation command that turns private review issues into bucketed,
   API-safe priors plus local audit metadata.
3. Done: cold-pass runtime integration: generate paper-only issues first, then use
   reviewer priors only for missing high-salience checks.
4. Done: triage integration where priors can calibrate reviewer likelihood and
   decision tier, but cannot affect manuscript support or verification.
5. Done: held-out evaluation gate comparing baseline, safe prior, local
   raw-memory upper bound, and human issue clusters.

## Guardrails

- Reviewer priors may raise checks, rank salience, or shape final wording.
- Reviewer priors may not verify facts or support manuscript critiques.
- API-safe support evidence is bucketed, not exact.
- Deployment requires privacy audit success and held-out evaluation gains.
