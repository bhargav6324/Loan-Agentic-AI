# Project Grilling — Interview Q&A

---

## 1. Explain your project

I built a multi-agent AI system for automated loan approval. The core idea is that a single ML model can optimize for prediction accuracy, but real-world lending requires more — regulatory compliance, fairness auditing, explainability, and auditability. So instead of one model, I have specialized agents:

- **Intake Agent** — verifies identity (PAN, Aadhaar) using a ReAct loop and checks for fraud
- **Risk Agent** — computes a risk score using a RandomForest ML model blended with heuristic rules
- **Policy Agent** — checks RBI regulatory compliance using RAG over a policy document
- **Fairness Agent** — runs counterfactual bias testing (flips gender/age/location and re-scores)
- **Orchestrator Agent** — dynamically routes applications through the pipeline based on the current state
- **Reporting Agent** — generates human-readable explanations for applicants, admins, and regulators

These agents are orchestrated using a **LangGraph StateGraph** — a directed graph with conditional edges, parallel execution, and cycles. The backend is FastAPI, frontend is React, database is PostgreSQL, and the LLM is llama-3.1-8b via Groq.

The user submits a loan application through the frontend → it triggers the LangGraph pipeline → agents run in a specific order depending on the risk profile → a decision (approved/rejected/escalated) is saved to the database → the frontend displays the full agent timeline with explanations.

---

## 2. Why is it agentic?

Three concrete behaviors make it agentic rather than just a pipeline:

**1. Dynamic tool selection (Data Agent):**
The Data Agent doesn't call the same tools for everyone. It runs a ReAct loop — Reason, Act, Observe — up to 9 iterations. For a salaried employee, it calls PAN, Aadhaar, and CIBIL. For a self-employed person, it also dynamically calls GST verification. For a young applicant, it pulls bank statements. The agent *reasons about what data it needs* based on the profile.

Code: `agents/intake_agent.py`, function `_fallback_reason_next_action()` — it checks employment type, age, and which tools have already been called to decide the next action.

**2. Dynamic routing (Orchestrator):**
The orchestrator doesn't follow a fixed sequence. After each agent completes, it observes the updated state and decides what to do next. If the policy agent finds a critical violation mid-pipeline, the orchestrator skips fairness analysis and goes straight to rejection. If the ML model has low confidence, it requests a risk re-evaluation. The graph has **cycles** — fairness can loop back to risk scoring through the orchestrator.

Code: `agents/orchestrator_agent.py`, function `orchestrator_decide()` — uses `_is_ambiguous()` to gate LLM involvement.

**3. Self-reflection (Risk Agent):**
After computing a risk score, the Risk Agent checks its own output for contradictions. For example, if the credit score is 800 but the computed risk is high, or if DTI is low but the risk came out medium — it flags these inconsistencies and can revise the score. A static pipeline would accept the first score; an agent challenges itself.

Code: `agents/risk_agent.py`, function `_reflect_on_score()`.

**What I would NOT claim:** I won't say everything is LLM-driven. About 85% of orchestrator decisions are handled by deterministic rules. The LLM is invoked only for genuinely ambiguous cases. I consider this a strength — it's how you'd deploy LLM-augmented systems in production: fast deterministic path for routine cases, LLM reasoning for edge cases.

---

## 3. Why LangGraph?

I needed three capabilities that a simple sequential function chain can't provide:

**1. Parallel execution:**
The Risk Agent and Policy Agent are independent — one scores financial risk, the other checks regulatory compliance. They don't need each other's output. LangGraph's fan-out lets me run them simultaneously through two edges from a single `fan_out` node. A `policy_join` node merges their results afterward. This reduces latency.

Code: `orchestrator/graph.py`, lines 193-194:
```python
graph.add_edge("fan_out", "risk_agent")
graph.add_edge("fan_out", "policy_agent")
```

**2. Cycles (loops):**
After the Fairness Agent detects bias, the orchestrator can route back to the Risk Agent for re-evaluation, then back to the orchestrator, then to `final_decision`. This fairness → orchestrator → risk_reeval → orchestrator loop can't be expressed with linear function composition.

Code: Lines 210-211:
```python
graph.add_edge("fairness_agent", "orchestrator_decide")
graph.add_edge("risk_reeval", "orchestrator_decide")
```

**3. State management with reducers:**
When two nodes run in parallel and both write to state, you need merge logic. LangGraph lets me annotate each state field with a reducer. Scalar fields use `_last` (last-write-wins) — fine because parallel branches write to different keys. List fields like `agent_logs` use `operator.add` — both branches' logs get concatenated instead of overwritten.

Code: `agents/state.py`:
```python
risk_score: Annotated[Optional[float], _last]     # scalar — last-write-wins
agent_logs: Annotated[list, operator.add]           # list — concatenate
```

**4. Ablation support:**
For my research paper, I needed to systematically disable agents to measure their individual contribution. LangGraph lets me do:
```python
graph.add_node("fairness_agent", (lambda state: state) if disable_fairness else fairness_agent)
```
One parameter replaces the agent with a no-op. Try doing that cleanly with hardcoded function chains.

---

## 4. How do agents communicate?

Agents do NOT send messages to each other directly. They communicate through a **shared state dictionary** — `AgentState`, defined as a `TypedDict` in `agents/state.py`.

The flow is:
1. A node receives the **full current state** as input
2. The node does its work (calls tools, runs ML model, etc.)
3. The node returns a **partial state update** — only the keys it changed
4. LangGraph **merges** the update into the global state using reducers
5. The next node receives the **updated full state**

Example: The Risk Agent reads `applicant_data` and `credit_report` from state, computes a score, and returns `{"risk_score": 42, "risk_category": "MEDIUM", "risk_factors": [...]}`. The Orchestrator then reads `risk_score` and `risk_category` to decide routing.

There is one case of genuine inter-agent interaction: the Fairness Agent directly calls the Risk Agent's `evaluate_profile()` function with modified features. This simulates what the code calls an "inter-agent debate" — the Fairness Agent challenges the Risk Agent by asking "would you give the same score if the applicant were female?" and the Risk Agent responds by re-scoring. But this is a function call, not message passing.

---

## 5. Explain `AgentState`

`AgentState` is defined in `agents/state.py` as a `TypedDict` with about 40 fields. It's the single data structure that flows through the entire LangGraph pipeline.

**Key design decisions:**

**1. Annotated types with reducers:**
Every field uses `Annotated[Type, reducer]`. This tells LangGraph HOW to merge values when parallel branches update the same state.

```python
risk_score: Annotated[Optional[float], _last]   # Last-write-wins
agent_logs: Annotated[list, operator.add]         # Concatenate lists
```

The `_last` reducer simply returns the newer value. This works because parallel branches (risk and policy) write to **different** keys — `risk_score` vs `policy_violations`. They never conflict.

For `agent_logs`, both branches append entries. `operator.add` concatenates them so no logs are lost.

**2. Grouped by agent:**
The fields are logically grouped:
- Raw input: `applicant_data`, `application_id`, `user_id`
- Data Agent outputs: `pan_verification`, `aadhaar_verification`, `credit_report`
- Fraud Agent outputs: `fraud_detected`, `fraud_reason`
- Risk Agent outputs: `risk_score`, `risk_category`, `risk_factors`
- Policy Agent outputs: `policy_passed`, `policy_violations`
- Fairness Agent outputs: `bias_detected`, `fairness_report`, `fairness_debate_log`
- Decision: `final_decision`, `decision_reasoning`
- Orchestrator: `orchestrator_next_agent`, `orchestrator_history`

**3. `total=False`:**
The TypedDict uses `total=False`, meaning every field is optional. This is necessary because at the start of the pipeline, only `applicant_data` exists. Fields get populated as agents run.

---

## 6. How does the orchestrator decide routing?

The orchestrator (`agents/orchestrator_agent.py`, function `orchestrator_decide()`) uses an **ambiguity-gated** design — two paths:

**Path A — Deterministic (85% of cases):**
Function `_fallback_decide()` applies simple rules:
- Fraud detected or KYC failed → `auto_reject`
- Risk category HIGH → `auto_reject`
- Risk category LOW + policy passed → `auto_approve`
- MEDIUM risk, policy not yet checked → run `policy_agent`
- MEDIUM risk, policy checked, fairness not yet checked → run `fairness_agent`
- Bias detected after fairness → run `risk_reeval`
- All checks done → `final_decision`

**Path B — LLM-driven (15% of ambiguous cases):**
Function `_is_ambiguous()` checks for situations where deterministic rules can't safely decide:
- ML model confidence below 0.55 (the model itself is uncertain)
- Risk is LOW but fraud agent flagged manual KYC (conflicting signals)
- Policy passed but fairness agent detected bias (conflicting signals)
- Pipeline has iterated 3+ times without reaching a terminal decision (stalled)
- Fraud match confidence in the ambiguous band (0.3-0.8)

Only when `_is_ambiguous()` returns True does it call `_call_llm_orchestrator()`. The LLM receives:
- Full state summary (applicant profile, risk score, policy results, fairness results)
- Execution history (what agents already ran and why)
- Available agents with descriptions
- Current iteration count vs MAX_ITERATIONS (12)

The LLM returns JSON: `{"next_agent": "fairness_agent", "reasoning": "Medium risk with low ML confidence, need bias check before deciding."}`

The response is validated — `next_agent` must be in `VALID_AGENTS`. If the LLM returns an invalid agent or fails, it falls back to `_fallback_decide()`.

The log entry tags each decision as `[Deterministic]`, `[LLM-Agentic]`, or `[Fallback]` so you can audit which path was taken.

---

## 7. Where exactly is the LLM used?

The LLM is **llama-3.1-8b-instant** running on Groq. All calls go through a shared client in `agents/llm_client.py` with process-wide rate limiting (2.2s between calls, ~27 RPM, under Groq's 30 RPM free tier limit).

Here are the **7 exact locations** where the LLM is called:

| # | Agent | File | When it fires | What it does |
|---|-------|------|---------------|--------------|
| 1 | Data Agent | `intake_agent.py` ~line 208 | Only when a tool throws an exception | Suggests recovery action (which tool to try next) |
| 2 | Fraud Agent | `intake_agent.py` line 363 | Only for name similarity in 0.60-0.92 band | Classifies as FRAUD/REVIEW/MATCH |
| 3 | Policy Agent | `intake_agent.py` line 670 | Only when RAG retrieval score < 0.25 | Interprets policy text for violations |
| 4 | Planner | `orchestrator_agent.py` line 238 | Every application (first call) | Generates initial execution plan |
| 5 | Orchestrator | `orchestrator_agent.py` line 393 | Only for ambiguous states (~15%) | Decides next agent to route to |
| 6 | Reporting | `reporting_agent.py` line 85 | Every application (final step) | Process trace explanation |
| 7 | Reporting | `reporting_agent.py` line 93+ | Every application (final step) | Applicant message, admin summary, compliance report |

**Agents that do NOT use the LLM at all:**
- Risk Agent — purely ML model + heuristic rules + self-reflection
- Fairness Agent — purely mathematical counterfactual analysis
- auto_approve, auto_reject, final_decision — pure state updates
- policy_join, fan_out — pure routing logic

**Every single LLM call has a deterministic fallback.** If Groq goes down, the system still works — just with less nuanced decisions for edge cases.

---

## 8. Explain Risk Agent

The Risk Agent (`agents/risk_agent.py`) computes a risk score from 0-100. Higher = riskier. It has four layers:

**Layer 1: Hard Override Pre-Screening** (`_check_hard_overrides()`)
Before any scoring, it checks catastrophic conditions:
- `defaults > 3` OR `credit_score < 450` OR `dti_ratio > 0.90` → **HARD_REJECT** (score = 95)
- `credit_score > 820` AND `defaults == 0` AND `dti < 0.15` AND `employment_years > 5` → **HARD_FAST_TRACK** (score = 10)

These bypass all scoring. No ML, no heuristics, no LLM.

**Layer 2: ML Model Prediction** (inside `evaluate_profile()`)
A trained RandomForest (`ml/model/risk_model.joblib`) predicts 3-class probabilities: approve, review, reject. The ML score is calculated as:
```
ml_score = proba[review] × 50 + proba[reject] × 100
```
ML confidence = max probability across the three classes.

The model features: `credit_score, dti_ratio, loan_to_income, employment_score, age, defaults, credit_utilisation, gender_encoded, location_encoded`.

Note: gender and location ARE features in the ML model — this is intentional. It creates **emergent bias** that the Fairness Agent then detects through counterfactual testing. The heuristic scorer is demographic-blind.

**Layer 3: Heuristic Scoring** (`_compute_heuristic_score()`)
Starts at 50 (neutral) and adjusts:
- Credit score 750+ → -20, below 600 → +25
- High DTI (>0.6) → +15
- High LTI (>5x) → +15
- Exponential penalty for defaults: `10 × (2^defaults - 1)`, capped at 40
- Employment stability, loan tenure, loan purpose adjustments
- **Risk interaction penalty**: If 3+ danger flags active simultaneously → +20 (compounding risk)
- **Creditworthiness bonus**: Clean + stable + low debt → up to -15

**Layer 4: Dynamic Confidence-Based Blending**
```python
ml_weight = 0.4 + (0.4 × ml_confidence)   # ranges 0.4 to 0.8
heuristic_weight = 1 - ml_weight
risk_score = ml_weight × ml_score + heuristic_weight × heuristic_score
```
When the ML model is confident (high max probability), it gets up to 80% weight. When uncertain, heuristics dominate at 60%. This is adaptive, not a fixed 60/40 split.

**Layer 5: Self-Reflection** (`_reflect_on_score()`)
After computing the score, the agent checks for internal contradictions:
- Credit score 800+ but risk score > 50? → possible under-weighting of strong credit
- DTI < 0.20 but risk score > 60? → something is inconsistent
- ML and heuristic scores diverge by > 30 points? → low confidence in result

If contradictions are found, the score may be revised. Every revision is logged with its reason.

**Final categorization:** ≤35 = LOW, 35-60 = MEDIUM, >60 = HIGH.

**No LLM is used anywhere in this agent.** It's fully deterministic and reproducible.

---

## 9. Explain RAG

I have two retrieval systems in this project:

### System 1: Policy RAG (TF-IDF based)

**Purpose:** Check loan applications against RBI banking regulations.

**Source document:** `policies/rbi_guidelines.txt` — 20 sections covering age requirements, LTI caps, credit score thresholds, employment rules, AML checks, etc.

**Implementation** (in `agents/intake_agent.py`, class `RAGPolicyEngine`):

1. **Chunking:** The document is split by `##` headings → ~20 chunks, one per regulation section
2. **Vectorization:** `sklearn.feature_extraction.text.TfidfVectorizer` — keyword-based, NOT neural embeddings
3. **Storage:** In-memory (rebuilt each startup). No vector database for this one.
4. **Retrieval:** Given an applicant query string (age, income, credit score), compute `cosine_similarity(query_vec, tfidf_matrix)`, return top-3 chunks + best match score

**The confidence gate:**
- If `retrieval_score ≥ 0.25` → high confidence match → run 5 deterministic rules (age≥21, LTI≤5, defaults≤2, credit≥400, high-value employment). NO LLM needed.
- If `retrieval_score < 0.25` → low confidence → pass the retrieved text to the LLM with `POLICY_RAG_PROMPT`, ask it to identify violations.

In practice, the TF-IDF score is usually above 0.25 for standard applications, so the deterministic path runs most of the time.

**Honest note:** This is keyword-based retrieval, not the dense-embedding RAG that people typically mean when they say "RAG." I'd improve this with sentence transformer embeddings in production.

### System 2: Episodic Memory (ChromaDB — genuine vector search)

**Purpose:** Give the orchestrator historical precedent when routing decisions.

**Implementation** (in `memory/episodic_memory.py`):

1. **Embeddings:** ChromaDB's default `all-MiniLM-L6-v2` sentence transformer
2. **Storage:** ChromaDB persistent client at `data/chroma_db/`, HNSW index with cosine similarity
3. **Storing:** After each decision, `store_episode()` embeds the applicant profile + outcome metadata (decision, risk score, bias detected, debate summary)
4. **Retrieving:** `retrieve_similar_cases()` takes a new applicant profile, finds the top-k most similar past cases

**How it reaches the LLM:** In `orchestrator_agent.py` line 107, `_build_state_summary()` calls `retrieve_similar_cases()` and appends past outcomes to the state summary. When the LLM orchestrator is invoked for ambiguous cases, it sees: "Past similar case #1: approved, no bias. Past similar case #2: rejected, bias detected." This gives it precedent-based reasoning.

This IS genuine semantic vector search with neural embeddings. However, its impact on decisions is indirect — it only adds context to the orchestrator's LLM prompt, and the LLM itself is only called for ~15% of cases.

---

## 10. Explain Fairness Agent

The Fairness Agent (`agents/fairness_agent.py`) performs **counterfactual bias detection**. It runs only for MEDIUM-risk (borderline) cases.

**Core idea:** If changing a protected attribute (gender, age, location) changes the risk score significantly, the scoring system has demographic bias.

**How it works step by step:**

1. Read the original risk score from state (e.g., 45)
2. Deep-copy the entire state
3. Flip ONE protected attribute (e.g., change `applicant_gender` from "male" to "female")
4. Extract features from the modified state using `_extract_features()` (same function the Risk Agent uses)
5. Call `evaluate_profile()` with the modified features — this runs the EXACT same ML + heuristic scoring
6. Get the counterfactual score (e.g., 38)
7. Compute the signed delta: `cf_score - original_score = 38 - 45 = -7`
   - Positive delta = original was scored MORE harshly (possible adverse treatment)
   - Negative delta = original was scored MORE leniently
8. If `|delta| > threshold` → bias detected

**Thresholds:**
- Gender: 5 points (gender should NEVER change the score significantly)
- Age: 10 points (age can legitimately affect score through financial behavior proxies)
- Location: 8 points

**What gets tested:**
- Gender: tests all alternatives (male→female, male→other, etc.)
- Age: tests brackets 25, 35, 45, 55 (skips if difference < 5 years)
- Location: tests urban/semi_urban/rural alternatives

**Adaptive thresholds from long-term memory:**
The agent reads `data/long_term_memory.json`. If historically >60% of women applicants were rejected, it tightens the gender threshold by 2 points (from 5 → 3). This makes bias detection MORE sensitive when systemic patterns exist.

**If bias is detected:**
- `bias_detected = True` in state
- The orchestrator sees this and can route to `risk_reeval` (re-score) or `decision_commit` (which sets status to "escalated" for automated override)

**No LLM is used.** This is pure math — run the same scoring function twice with different inputs, compare the numbers.

**Important research finding:** In our 10-seed ablation study, this agent actually INCREASES demographic parity disparity (ΔDP goes from 0.0188 without fairness agent to 0.0400 with it). This happens because the deterministic override rule disproportionately affects one demographic group more than the other. We report this honestly as an empirical finding.

---

## 11. Why deterministic logic instead of LLM everywhere?

Four reasons:

**1. Reproducibility:**
In financial services, if you reject someone's loan, you need to explain exactly why. An LLM might give different answers for the same input (temperature > 0, different prompt contexts). Deterministic rules give the SAME answer every time. The risk score of 42 is always 42 for the same inputs.

**2. Auditability:**
Regulators need to audit decision-making. A deterministic rule that says "rejected because defaults > 2" is auditable. An LLM that says "I feel like this applicant is risky" is not. Every deterministic step is logged with its exact inputs and outputs.

**3. Cost and latency:**
LLM calls cost money and take 1-3 seconds each. A loan application going through 7 agents with LLM calls each would take 15-20 seconds minimum. With deterministic paths, most applications process in under 2 seconds, with LLM calls only for the ~15% that need nuanced reasoning.

**4. Reliability:**
Groq might go down. The internet might be slow. If the LLM fails, every agent has a deterministic fallback. The system degrades gracefully — it still processes loans, just without LLM-powered nuance for edge cases. If I had LLM everywhere, a Groq outage would make the entire system unusable.

**Where the LLM IS genuinely valuable:**
- Disambiguating names ("Rahul" vs "Rahul Kumar" — is this fraud or a maiden name?)
- Interpreting novel regulatory situations not covered by my 5 hardcoded rules
- Generating human-readable explanations in multiple styles (applicant-friendly vs regulator-friendly)
- Resolving genuinely conflicting signals (low risk + fraud flag)

**The design principle:** Use deterministic code for anything that has a clear rule. Use the LLM for reasoning tasks that require natural language understanding or judgment under ambiguity.

---

## 12. Limitations + Future Improvements

### Current Limitations

**1. Fairness agent increases demographic disparity:**
The counterfactual override rule is a fixed threshold (ε=6.0). It applies the same rule to everyone regardless of population-level distributions. In our ablation study, it improves specificity (+6.19%, p=2.49×10⁻⁷) but increases ΔDP by 0.0211 (p=4.24×10⁻⁵). The override disproportionately affects one demographic group.

**2. Policy RAG is TF-IDF, not semantic:**
Keyword-based retrieval works for structured regulatory text, but misses semantic meaning. A regulation about "over-leveraged borrowers" wouldn't match a query about "high debt ratios" even though they mean the same thing.

**3. Mock external tools:**
PAN, Aadhaar, CIBIL verification tools are all mock functions returning synthetic data. In production, you'd replace these with real API clients (NSDL, UIDAI, CIBIL bureau API). The interface stays identical — same function signatures, same return format.

**4. Synchronous pipeline:**
`loan_pipeline.invoke()` is synchronous and blocks the FastAPI event loop during execution. With many concurrent users, this would be a bottleneck. Production would need a task queue (Celery/RQ) to process loans asynchronously.

**5. Identity index is a JSON file:**
The SHA-256 deduplication index (`data/identity_index.json`) is a flat file. No concurrent write safety, no distributed access. Production would use Redis or a database table.

**6. Admin role is self-assignable:**
Users can register with `role: "admin"` — a security flaw. Production would require admin invitation or a separate admin provisioning flow.

### Future Improvements

**1. Dense embeddings for Policy RAG:**
Replace TF-IDF with sentence-transformer embeddings (like the episodic memory already uses). This would enable semantic matching: "over-leveraged" ≈ "high debt ratio."

**2. Adaptive fairness thresholds:**
Instead of a fixed 5-point threshold for gender, compute the threshold dynamically based on the population distribution of counterfactual score differences. This could prevent the ΔDP increase we observed.

**3. Async task queue:**
Process loans via Celery workers. The API would return immediately with a job ID, and the frontend would poll or use webhooks for results.

**4. Real external API integration:**
Swap mock tools for real NSDL PAN verification, UIDAI Aadhaar authentication, CIBIL/Experian credit bureau APIs. The ReAct loop and tool interface don't change.

**5. LLM observability:**
Add monitoring for LLM latency, token usage, fallback rates, and prompt quality. Track which prompts produce invalid JSON most often.

**6. A/B testing for orchestrator strategies:**
Compare the ambiguity-gated orchestrator against an always-deterministic and always-LLM variant in production. Measure decision quality, latency, and cost.
