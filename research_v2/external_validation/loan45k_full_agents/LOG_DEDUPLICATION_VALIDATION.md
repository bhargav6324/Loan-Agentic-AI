# Log and audit deduplication validation

## Cause

The shared state used an additive reducer for `agent_logs`, while reused agents
returned the complete prior state and complete accumulated log list. Sequential
and parallel merges therefore repeatedly re-added old entries. The audit agent
then sealed those duplicate log entries, inflating both counts and runtime.

## Isolated correction

`FullAgentState` now uses an exact-entry, order-preserving unique merge for
`agent_logs` and `errors`. The original synthetic-data graph is unchanged.

## 200-case paired validation (seed 42)

| Configuration | Logs before | Logs after | Audit entries before | Audit entries after | Seconds before | Seconds after |
|---|---:|---:|---:|---:|---:|---:|
| F2 | 502.80 | 11.43 | 249.40 | 6.43 | 1.055 | 0.815 |
| F3 | 1001.60 | 12.43 | 498.80 | 7.43 | 1.401 | 1.040 |

- Compared configuration-case rows: 1,000.
- Decision mismatches: 0.
- Maximum accuracy delta: 0.
- Maximum F1 delta: 0.
- Maximum MCC delta: 0.

Therefore, the authoritative v2 predictive, policy, fairness, reassessment, and
escalation results remain valid. Only duplicated log/audit counts and their
associated runtime overhead are superseded. A ten-seed predictive rerun is not
required for this correction.
