# Phase 3/4 Verification Gate Report

## Status

**Not yet approved for the final ten-seed paper run.** Dataset provenance and
the deterministic A0/A1/A1R path are verified. Native LangGraph A2/A3 must
still pass the full test suite and all-1,500-record parity test in an
environment containing the project dependencies.

## Corrections made in this review

1. Restored the corrected causal dataset generator. The uploaded CSV was the
   corrected dataset, but the uploaded `generate_dataset.py` was the older
   leaky generator.
2. Enforced the binary model contract: `classes_` must be exactly `[0, 1]`,
   where class 1 is default. A legacy three-class model now fails loudly.
3. Added the missing `risk_agent_reassessment` state field so LangGraph cannot
   silently discard reassessment telemetry.
4. Fixed the graph trace parser to read `bias_detected` from the paper-mode
   fairness report instead of the nonexistent `bias_flagged` field.
5. Separated sensitivity-flag generation from downstream action. The flag is
   `abs(signed_delta) > epsilon`; positive and negative deltas receive distinct
   deterministic actions.
6. Ensured the A1 audit does not apply reassessment only to the flipped case;
   A1R/A2/A3 apply reassessment symmetrically to original and flipped scores.
7. Centralized hard-override scores so static and graph paths use the same
   values.
8. Made paper-mode model/schema failures raise instead of silently falling
   back to heuristic-only scoring.
9. Prevented paper mode from starting the learning thread or mutating
   long-term/episodic memory.
10. Corrected policy certificates to include their actual violation list.
11. Removed temporary debug printing from live agent paths.
12. Added aggregate mean/SD/95% CI output, paired seed-level tests, effect
    sizes, Wilcoxon sensitivity tests, and Holm-adjusted p-values.
13. Separated dry-run, validation, and frozen final-result output locations.
14. Defined A2's policy-rejection audit skip as an intentional dynamic-routing
    optimization. Core scores, policy results, commitments, and final decisions
    must match A1R; audit values must match whenever both paths execute the
    audit. A skipped A2 audit is valid only after policy has already made
    approval impossible.
15. Added the always-approve baseline (`C0a`) so accuracy and approval-positive
    F1 are interpreted against the dataset's strong class imbalance.

## Dataset evidence

- Dataset SHA-256:
  `669ff7aeda69a60200732a2b22b7b9cf46bb38681c10892bfb68cc2c38f75f1a`
- Generator SHA-256:
  `bdcb361730e59d193817b0d2f4eef7f7db2e28f998e02de00364f289490e7892`
- Regeneration from the corrected generator reproduced the dataset checksum
  exactly.
- Dataset validation passed: 5,000 rows, 567 defaults (11.34%), valid schema,
  and zero overlap for each frozen seed 42-51.

## Historical evidence preservation

The following historical files retain their expected SHA-256 checksums:

- `research/final_results/ablation_results.csv`:
  `82009a99e59de83e13580d495ff3025285656b1b3b0b5b153eab5503942b03a4`
- `research/final_results/ablation_results_per_seed.csv`:
  `023330353d710cde051669027792cf4ce9bef1a95398a6d1c9b77c65efc1aa92`
- `research/final_results/paired_comparisons.csv`:
  `81e515893edb0358f195ad93af4946adc84220d9fef960605895c34ac858603f`
- `research/final_results/threshold_sweep.csv`:
  `c32e6185f179a1f293e8c488146337a1b4425f5b0a983f1db50bca7b0d653cfc`
- `research/final_results/threshold_sweep_per_seed.csv`:
  `2ca23f74a6c8e9ea085cf8abbe0d93c0ff7bd415e0b7bd4c384e18d61fda2b34`
- `research/final_results/threshold_sweep_trend.csv`:
  `824a55dc41b67ef346a36383ce2a98b69315f24df49305c03a4f290c04335349`

## Seed-42 deterministic validation (not final paper evidence)

The corrected A0/A1/A1R runner evaluated all 1,500 test applicants with zero
failures and zero LLM calls.

| Configuration | Approvals | F1 (approval) | Specificity | Balanced accuracy | Policy-violating approvals |
|---|---:|---:|---:|---:|---:|
| A0 | 1,408 | 0.9434 | 0.3333 | 0.6553 | 92 |
| A1 | 1,268 | 0.9194 | 0.5645 | 0.7339 | 0 |
| A1R | 1,266 | 0.9186 | 0.5645 | 0.7332 | 0 |

These values are a single-seed engineering validation only. They must not be
used as final multi-seed paper claims.

## Important correction to the earlier Antigravity report

The earlier statement “100% logical parity” was not supported by its saved
output. It established identical final binary decisions but still contained
differences in audit execution, audit flags, commitment statuses, composite
scores, and reassessment telemetry. The expanded parity test now compares all
decision-relevant fields over all 1,500 seed-42 test records.

## Required verification commands

Run these from the repository root after installing `requirements.txt`:

```bash
python3 -m pytest tests/test_research_v2/ tests/test_golden_contracts.py -v
python3 -m pytest tests/test_parity.py -v -m slow
```

Only if both commands pass, start the frozen study:

```bash
python3 research_v2/run_agentic_ablation.py \
  --seeds 42,43,44,45,46,47,48,49,50,51 \
  --configurations C0a,A0,A1,A1R,A2,A3 \
  --test-split 0.30 \
  --epsilon 6.0
```

Do not use or publish the legacy files in `research_v2/results/` or the
single-seed validation outputs as final results.
