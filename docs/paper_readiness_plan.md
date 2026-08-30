# Agentic AI Loan Decision — Paper Readiness Plan

## 1. Current Live vs. Research Architecture Differences
- **Model Discrepancy**: The live system uses a 3-class Random Forest (approve, review, reject) trained on 9 features (including location). The research harness uses a binary Random Forest (default, non-default) trained on 8 features.
- **Counterfactual Direction**: The logic computes `cf_score - original_score`. A positive delta means the counterfactual risk is higher, meaning the original profile was scored more leniently.
- **Policy Bypass**: The live `orchestrator_decide` node can currently route LOW-risk cases directly to `auto_approve`, bypassing the `policy_join` node completely.
- **Terminology**: The live system makes multiple references to "human review" and "human escalation," which do not actually exist in the pipeline. The research context uses "override triggers".

## 2. Exact Files That Will Change

**Phase 2: Correctness & Terminology**
- `agents/state.py`: Add `risk_certificate`, `policy_certificate`, `audit_certificate`, `decision_proposal`, `commit_status`, `commit_reason`.
- `orchestrator/graph.py`: Modify edges to enforce a policy commitment barrier.
- `agents/orchestrator_agent.py`: Update routing to propose decisions rather than immediately terminate, forcing a pass through `policy_join`.
- `agents/fairness_agent.py`: Rename mechanism to "gender attribute-flip sensitivity audit", correct delta comments, remove "human review" terms.
- `agents/reporting_agent.py`: Clean up terminology.
- `routes/admin_routes.py`: Clean up terminology.
- `agents/risk_agent.py`: Remove misleading comments claiming the ML model is demographic-blind.

**Phase 3: Unified Contract & Tests**
- `ml/canonical_model_contract.py` (NEW): A unified interface for the binary RF model.
- `tests/test_policy_commitment.py` (NEW)
- `tests/test_counterfactual_direction.py` (NEW)
- `tests/test_feature_schema.py` (NEW)
- `tests/test_parallel_merge.py` (NEW)
- `tests/test_deterministic_replay.py` (NEW)

**Phase 4 & 5: New Agentic Evaluation**
- `research_v2/` (NEW DIRECTORY)
- `research_v2/run_agentic_ablation.py` (NEW)
- `research_v2/publication_assets/` (NEW DIRECTORY)

## 3. Exact Tests To Add
1. **Policy Commitment Barrier**: Test that low-risk cases cannot bypass the policy check and that no terminal state is reached without a `policy_certificate`.
2. **Counterfactual Delta Direction**: Assert that the delta formula correctly identifies harsh vs. lenient scoring.
3. **Feature Schema Alignment**: Assert that gender encoding, feature ordering, and class conventions match strictly between the binary model, the graph, and the research harness.
4. **Parallel Branch Merge**: Assert that parallel Risk and Policy branches do not commit conflicting terminal decisions, and that `policy_join` safely merges them.
5. **Deterministic Replay**: Provide the same input twice and assert the final committed decision is identical.

## 4. Risks That Could Alter Existing Results
- **Enforcing Policy on Low-Risk**: By preventing low-risk cases from bypassing the policy check, some previously "auto-approved" cases might now be rejected due to obscure policy violations. This makes the system safer but may slightly lower the overall approval rate.
- **Note on Historic Results**: Because we are strictly writing new experiments to `research_v2/`, the historical CSV results in `research/final_results/` have zero risk of being altered.

## 5. Rollback Plan
- The legacy `research/` directory is explicitly frozen and will not be touched.
- All live agent modifications (`agents/*.py` and `orchestrator/graph.py`) are tracked in git. If the new `research_v2` pipeline yields logically inconsistent results, we can instantly `git checkout` the agents and orchestrator to their previous state while preserving the `research_v2` output logs for debugging.

## 6. Short Design for the New Agentic AI Evaluation
The evaluation will be housed in `research_v2/` and will use the synthetic binary dataset (N=5000). It will evaluate 5 distinct configurations to isolate the value of the Agentic components:

- **A0 (ML Baseline)**: Random Forest only.
- **A1 (Static Pipeline)**: A rigid, non-agentic workflow (Risk → Policy → Fairness → Decision).
- **A2 (Agentic Orchestration)**: LangGraph agents with deterministic dynamic routing (the current deterministic fallback behavior).
- **A3 (Agentic Self-Correction)**: A2 + deterministic critique/re-evaluation enabled.
- **A4 (Full Live-Agentic)**: A3 + LLM reflection and debate (requires API key, optional).

**Metrics**: Accuracy, Precision, Recall, F1, Specificity, Balanced Accuracy, MCC, ΔDP, ΔEO, Policy Violations, Override Triggers, Decision-Commit Blocks, Routing Frequency, Re-evaluations, Latency.
**Rigor**: 10 paired seeds, Holm-corrected p-values, 95% Confidence Intervals.
