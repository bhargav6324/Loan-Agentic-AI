# Corrected V3 Analysis

## Provenance

- Experiment: `ablation_v2_20260928_023545`
- Git commit: `9243ed50a2655769e4668a72239a88a3f5bc278b`
- Dataset SHA-256: `fae2e0e34f42c524a5c8d107d5acc864ced5e569387cdbbaf964bad263201a06`
- Raw applicant traces: 90,000
- Paper mode: True
- LLM calls: 0

## Corrected fairness reporting

The original seed-level summary read the legacy `audit_bias_flagged` field for A2/A3. The graph stores its result in `fairness_bias_flagged`, so the aggregate value of zero was a reporting error. It did not change any applicant decision.

- A2 fairness audits: 1,376
- A2 fairness flags: 348
- Flag rate among audited cases: 25.29%
- Escalations among all A2 cases: 348 / 15,000 (2.32%)
- Unsafe policy approvals: 0
- Terminal commitment blocks: 0

## A1R versus A2

- Compared cases: 15,000
- Agreement: 14,646 (97.64%)
- Disagreements: 354
- Fairness-driven escalations: 348
- Decision-proposal contract divergences: 6
- Unexplained differences: 0

The six contract divergences are policy-compliant MEDIUM-risk cases where A1R's canonical proposal rejected the case but A2's graph proposal approved it after the fairness check found no bias. They are not policy bypasses, but the project must choose one authoritative proposal contract before making a strict A1R/A2 equivalence claim.

## A2 versus A3

A2 and A3 had 0 mismatches across final decision, commitment status, score, category, policy outcome, and fairness flag. No predictive improvement from A3 should be claimed.

## Conclusion

No ten-seed rerun is required for this reporting correction. Use these corrected fairness files alongside the original immutable experiment outputs. Review the six proposal-contract divergences before freezing the paper's final architecture claims.
