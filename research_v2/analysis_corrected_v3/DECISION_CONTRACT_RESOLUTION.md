# Decision-Contract Resolution

## Problem

Six policy-compliant MEDIUM-risk applicants were rejected by A1R but approved by A2. Their Random Forest confidence values were between 0.51 and 0.59.

A1R applied `canonical_decision_proposal`, which rejects MEDIUM-risk cases when model confidence is below 0.60. A2 ran the fairness analysis and then approved every unflagged MEDIUM-risk case without reapplying that confidence rule.

## Authoritative rule

The canonical deterministic risk/policy proposal contract is authoritative:

- Policy failure or HIGH risk: reject.
- LOW risk: approve.
- MEDIUM risk with model confidence below 0.60: reject.
- MEDIUM risk with model confidence of at least 0.60: approve, subject to required fairness and commitment checks.
- A fairness flag may escalate or block a case, but fairness cannot convert a deterministic rejection into approval.

## Implementation

After the fairness path, the A2 deterministic orchestrator now calls `canonical_decision_proposal` instead of defaulting an unflagged MEDIUM-risk case to approval.

Regression tests cover:

- MEDIUM risk at 0.59 confidence remains rejected.
- MEDIUM risk at 0.60 confidence may be approved.
- A fairness flag escalates instead of approving.

## Experimental consequence

This is a decision-logic correction, not only a reporting correction. The existing V3 outputs remain immutable historical evidence, but they predate this fix. Final paper metrics should come from a new versioned V4 run using the corrected code. The expected direct change is that the six known A1R-reject/A2-approve cases become A2 rejections; the actual V4 result must be measured rather than assumed.
