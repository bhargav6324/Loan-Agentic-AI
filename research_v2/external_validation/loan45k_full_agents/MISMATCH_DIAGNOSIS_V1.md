# Version-1 mismatch diagnosis

The first ten-seed server run is preserved as an audit artifact but must not be
used as the final comparison.

## Observed mismatch

- F1P versus F2 changed 19,098 paired decisions.
- 14,833 changed from reject to approve; 12,875 of those were ground-truth
  rejections.
- 4,265 changed from approve to non-approve.
- F2 and F3 had zero mutual decision mismatches.

## Root cause

The adapted risk agent supplied `max(p, 1-p)` as `risk_ml_confidence`. The
original MEDIUM-risk decision contract uses that scalar as support for approval.
Consequently, high confidence in *rejection* could be interpreted as high
confidence to approve. The static F1P comparator also used a different raw
`p >= 0.5` decision rule, so baseline and graph contracts were not aligned.

## Correction

- Preserve direction by supplying approval probability `p` to the MEDIUM-risk
  approval-support input.
- Make F1P use the same risk-category, policy, reassessment, and directional
  approval contract as F2/F3.
- Save per-row commitment status, fairness flag, risk category, and proposal.
- Report automation coverage and selective accuracy separately from forced
  binary accuracy; escalations are unresolved outcomes, not rejections.

The corrected 200-case pilot produced exact F1P/F2/F3 binary parity, 95%
automation coverage, and 94.21% selective accuracy for F2/F3.
