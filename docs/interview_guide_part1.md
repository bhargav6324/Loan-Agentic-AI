# Interview Guide — Loan Agentic AI (Part 1 of 2)

## 1. BIG PICTURE

### What problem this solves
Automated loan approval that goes beyond a single ML model by adding **regulatory policy enforcement**, **counterfactual fairness auditing**, and **explainability** — all orchestrated as cooperating agents via LangGraph.

### Complete Input → Output Flow
```
User submits loan app (frontend React form)
  → POST /applications (creates DB record, status=SUBMITTED)
  → POST /applications/{id}/process (triggers pipeline)
      → LangGraph graph.invoke(initial_state)
          → intake_agent (Data Agent ReAct loop + Fraud Agent)
          → planner (LLM generates initial plan)
          → fan_out (parallel split)
              ├── risk_agent (ML + heuristics → risk_score)
              └── policy_agent (RAG + deterministic rules → violations)
          → orchestrator_decide (routes based on risk category)
          → [optional] fairness_agent (counterfactual testing)
          → [optional] risk_reeval
          → policy_join (merges parallel branches)
          → auto_approve / auto_reject / final_decision
          → reporting_agent (LLM explanations + audit + learning)
      → Save Decision + AgentLogs to PostgreSQL
  → Return JSON response to frontend
```

### What is AI/LLM vs Deterministic

| Component | LLM-Driven | Deterministic | Hybrid |
|-----------|:---:|:---:|:---:|
| Data Agent (ReAct loop) | | | ✅ LLM only on tool exception |
| Fraud Agent | | | ✅ LLM only in 0.60-0.92 similarity band |
| Risk Agent | | ✅ | ML model + rule heuristics, NO LLM |
| Policy Agent | | | ✅ LLM only when RAG score < 0.25 |
| Orchestrator | | | ✅ LLM only for ambiguous ~15% cases |
| Fairness Agent | | ✅ | Pure math (counterfactual score diffs) |
| Reporting Agent | ✅ | | LLM generates explanations |
| Planner | ✅ | | LLM generates initial plan |

### Where LangGraph is used
File: `orchestrator/graph.py` — `build_loan_graph()` constructs a `StateGraph(AgentState)` with 13 nodes, conditional edges, and parallel fan-out. LangGraph manages state flow, parallel execution (risk + policy run concurrently), and conditional routing.

### Where LangChain is used
**NOT USED.** The project uses LangGraph directly (from `langgraph.graph`), but does NOT import or use LangChain chains, agents, or tools. All LLM calls go through a custom `agents/llm_client.py` that calls Groq's REST API via `httpx`.

### Where the LLM is used
**Model:** `llama-3.1-8b-instant` via Groq API. Initialized in `agents/llm_client.py` as a raw HTTP client with rate limiting (2.2s between calls).

LLM call sites:
1. `intake_agent.py` line ~208 — Data Agent recovery on tool exception
2. `intake_agent.py` line ~363 — Fraud Agent name mismatch (ambiguous band only)
3. `intake_agent.py` line ~670 — Policy Agent RAG interpretation (low confidence only)
4. `orchestrator_agent.py` line ~238 — Planner initial plan generation
5. `orchestrator_agent.py` line ~393 — Orchestrator routing (ambiguous cases only)
6. `reporting_agent.py` line ~85 — Process explanation generation
7. `reporting_agent.py` — Applicant message, admin summary, compliance report

### Where RAG/vector search is used

**Two separate systems:**

1. **Policy RAG** (`intake_agent.py`, class `RAGPolicyEngine`, line 610):
   - Source: `policies/rbi_guidelines.txt` (20 regulatory sections)
   - Chunking: Split by `##` headings
   - Embeddings: **TF-IDF** (sklearn), NOT neural embeddings
   - Storage: In-memory (no vector DB)
   - Retrieval: Cosine similarity, top-3 chunks
   - Gate: If retrieval_score ≥ 0.25 → deterministic rules; < 0.25 → LLM interprets

2. **Episodic Memory** (`memory/episodic_memory.py`):
   - Uses **ChromaDB** with `all-MiniLM-L6-v2` embeddings
   - Stores past application profiles + outcomes
   - Retrieved by orchestrator for routing precedents (line 107 of orchestrator_agent.py)
   - This IS genuine vector search / semantic retrieval

### Frontend ↔ Backend Communication
- Frontend: React (Vite) at `localhost:5173`
- Backend: FastAPI at `localhost:8000`
- Communication: REST API + Server-Sent Events (SSE) for real-time streaming
- Auth: JWT tokens (access + refresh)

### Architecture Diagram
```
┌─────────────────────────────────────────────────────┐
│                    React Frontend                    │
│  Landing → Register/Login → ApplyLoan → Dashboard   │
│         → ApplicationDetail (agent timeline)        │
└──────────────────────┬──────────────────────────────┘
                       │ REST API + SSE
┌──────────────────────▼──────────────────────────────┐
│              FastAPI Backend (main.py)               │
│  Routes: auth_routes, application_routes,           │
│          process_routes, admin_routes                │
│  Auth: JWT (HS256)  │  Rate Limit: slowapi          │
└──────────────────────┬──────────────────────────────┘
                       │ loan_pipeline.invoke()
┌──────────────────────▼──────────────────────────────┐
│           LangGraph Pipeline (graph.py)             │
│                                                     │
│  intake_agent ──┬── risk_agent ── orchestrator ──┐  │
│                 └── policy_agent ─────────────┐  │  │
│                     fairness_agent ◄──────────┤  │  │
│                     policy_join ◄─────────────┘  │  │
│                     auto_approve/reject/final ◄──┘  │
│                     reporting_agent → END            │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│  PostgreSQL (async)  │  ChromaDB  │  Groq LLM API  │
│  users, applications │  episodic  │  llama-3.1-8b   │
│  decisions, agent_   │  memory    │  via httpx      │
│  logs, documents     │            │                 │
└─────────────────────────────────────────────────────┘
```

---

## 2. TRACING ONE LOAN APPLICATION

**Hypothetical:** Rahul, age 28, male, salaried, income ₹60,000/mo, loan ₹500,000, PAN: ABCDE1234F, Aadhaar: 123456789012

### Step-by-step trace:

**1. API receives it:**
`POST /applications` in `routes/application_routes.py` → validates via `LoanApplicationCreate` schema (`models/schemas.py`) → creates `Application` row in PostgreSQL → returns `application_id`, status=`submitted`

**2. User triggers processing:**
`POST /applications/{id}/process` in `routes/process_routes.py` line 91 → function `process_application()`

**3. Initial state built** (line 126-147):
```python
initial_state = {
    "application_id": "...",
    "user_id": "...",
    "applicant_data": {
        "loan_amount": 500000,
        "applicant_name": "Rahul",
        "applicant_age": 28,
        "monthly_income": 60000,
        "applicant_pan": "ABCDE1234F",
        ...
    },
    "agent_logs": [],
    "errors": [],
}
```

**4. `loan_pipeline.invoke(initial_state)`** — LangGraph starts

**5. Node: `intake_agent`** (`agents/intake_agent.py`, function `intake_agent()` line 751)
- Runs `_run_data()`: ReAct loop (up to 9 iterations)
  - Iteration 1: Calls `verify_pan("ABCDE1234F", "Rahul")` → mock returns `{verified: True, name_on_record: "Rahul Kumar"}`
  - Iteration 2: Calls `verify_aadhaar("123456789012", "Rahul")` → mock returns `{verified: True, ...}`
  - Iteration 3: Calls `get_credit_report("ABCDE1234F")` → mock returns `{credit_score: 720, total_defaults: 0, ...}`
  - Iteration 4: STOP (standard KYC complete, not self-employed)
  - **Source:** Deterministic `_fallback_reason_next_action()`, LLM only on tool exception
- Runs `_run_fraud()`: Checks name similarity
  - "Rahul" vs "Rahul Kumar" → Jaro-Winkler similarity ~0.78 → **ambiguous band** (0.60-0.92) → calls LLM
  - LLM returns `{"status": "REVIEW", "reasoning": "Missing last name"}` → sets `requires_manual_kyc=True`
  - No fraud detected → pipeline continues
- State updates: `pan_verification`, `aadhaar_verification`, `credit_report`, `fraud_detected=False`

**6. Routing: `route_after_intake()`** → no fraud → returns `"planner"`

**7. Node: `planner`** (`orchestrator_agent.py`, `generate_initial_plan()` line 221)
- Calls LLM with applicant summary → LLM returns `{"plan": ["fan_out", "policy_join", "fairness_agent", "final_decision"]}`
- State updates: `plan`, `plan_reasoning`

**8. Node: `fan_out`** → no-op, LangGraph fans out to two edges simultaneously

**9. PARALLEL Node: `risk_agent`** (`agents/risk_agent.py`, function `risk_agent()` line 430)
- `_extract_features()`: credit_score=720, dti_ratio=0.0, loan_to_income=0.694, employment_score=3, age=28, defaults=0
- `_check_hard_overrides()`: No override (score not <450, defaults not >3)
- `evaluate_profile()`:
  - ML model prediction: loads `ml/model/risk_model.joblib`, gets probabilities
  - Heuristic score: starts at 50, credit_score 720 → -10, low DTI → -5, etc. → ~30
  - Dynamic weighting: `ml_weight = 0.4 + 0.4 * confidence` → blended score ~28
  - Self-reflection: checks for contradictions → confirms score
- Result: `risk_score=28`, `risk_category="LOW"`, NO LLM CALL
- State updates: `risk_score`, `risk_category`, `risk_factors`

**10. PARALLEL Node: `policy_agent`** (`intake_agent.py`, `policy_agent_node()` line 764)
- RAG retrieves top-3 policy chunks from `rbi_guidelines.txt` using TF-IDF
- retrieval_score > 0.25 → deterministic path
- `_fallback_policy_check()`: age 28 ≥ 21 ✅, LTI 0.694 ≤ 5 ✅, defaults 0 ≤ 2 ✅, credit 720 ≥ 400 ✅
- Result: `policy_passed=True`, `policy_violations=[]`

**11. Node: `risk_agent` completes → Node: `orchestrator_decide`** (line 439)
- `_is_ambiguous()`: risk_category=LOW, ML confidence high, no conflicts → **NOT ambiguous**
- `_fallback_decide()`: LOW risk, policy not yet visible → returns `"auto_approve"` or routes to `policy_join`
- State updates: `orchestrator_next_agent="policy_join"`

**12. Node: `policy_join`** (graph.py line 101)
- Merges: policy_passed=True + orchestrator wants auto_approve → passes through
- Routes to `auto_approve`

**13. Node: `auto_approve`** (graph.py line 38)
- Sets `final_decision="approved"`, `decision_reasoning="...LOW risk..."`

**14. Node: `reporting_agent`** (`agents/reporting_agent.py`)
- LLM generates applicant message, admin summary, compliance report, process trace
- Stores episode in ChromaDB episodic memory
- Saves to long-term memory JSON

**15. Graph reaches END**

**16. Back in `process_routes.py`:**
- Maps "approved" → `ApplicationStatus.APPROVED`
- Saves `shared_state` (all agent outputs) to `Application.shared_state` JSON column
- Creates `Decision` record with risk_score, risk_factors, explanations
- Saves each `agent_log` entry as `AgentLog` row
- Returns JSON response to frontend

---

## 3. LANGGRAPH IMPLEMENTATION

### State Definition
File: `agents/state.py` — `AgentState(TypedDict)`
- Uses `Annotated` types with custom reducers
- Scalar fields: `_last` reducer (last-write-wins) for parallel safety
- List fields: `operator.add` for `agent_logs` and `errors` (merges from parallel branches)
- ~40 fields covering all agent outputs

### Graph Structure (from `orchestrator/graph.py`)
```
13 Nodes:
  intake_agent, planner, fan_out, risk_agent, policy_agent,
  orchestrator_decide, fairness_agent, risk_reeval, policy_join,
  auto_approve, auto_reject, final_decision, reporting_agent

Edges:
  intake_agent --conditional--> auto_reject | planner
  planner --> fan_out
  fan_out --> risk_agent (parallel)
  fan_out --> policy_agent (parallel)
  risk_agent --> orchestrator_decide
  orchestrator_decide --conditional--> fairness_agent | risk_reeval |
                                       auto_approve | auto_reject |
                                       final_decision | policy_join
  fairness_agent --> orchestrator_decide (loop back!)
  risk_reeval --> orchestrator_decide (loop back!)
  policy_agent --> policy_join
  policy_join --conditional--> auto_approve | auto_reject | final_decision
  auto_approve --> reporting_agent
  auto_reject --> reporting_agent
  final_decision --> reporting_agent
  reporting_agent --> END
```

### Simplified Pseudocode
```
START
  ↓
intake_agent (Data ReAct + Fraud)
  ↓
if fraud/KYC fail → auto_reject → reporting → END
  ↓
planner (LLM generates plan)
  ↓
fan_out ─┬─ risk_agent ──→ orchestrator_decide
         │                      ↓
         │              if LOW → auto_approve
         │              if HIGH → auto_reject
         │              if MEDIUM → fairness_agent ──→ orchestrator_decide (loop)
         │                                              ↓
         └─ policy_agent ──→ policy_join ◄──────── (merges orchestrator decision)
                                  ↓
                          if policy failed → auto_reject
                          else → auto_approve / final_decision
                                  ↓
                          reporting_agent
                                  ↓
                                END
```

### Why designed this way
- **Parallel fan-out**: Risk and Policy run simultaneously (faster)
- **Orchestrator loop**: Can call fairness → re-eval → fairness (up to 12 iterations)
- **Policy join**: Ensures policy violations override any approve decision
- **Ablation support**: `disable_fairness`/`disable_policy` params replace nodes with no-ops

---

## 4. EVERY AGENT

### Agent 1: Data Agent (part of intake_agent)
- **Purpose:** KYC verification and data enrichment via ReAct loop
- **File:** `agents/intake_agent.py`, function `_run_data()` (line 252)
- **Input:** `applicant_data` from state (name, PAN, Aadhaar, employment_type, age)
- **Output:** `pan_verification`, `aadhaar_verification`, `credit_report`, `data_agent_reasoning_trace`
- **Uses LLM?** Only on tool exception (recovery guidance)
- **Uses RAG?** No
- **Uses hardcoded rules?** Yes — `_fallback_reason_next_action()` determines tool order
- **Decision logic:** Iterates up to 9 times. Each iteration: decide next tool → call tool → observe result. Self-employed → also calls GST. Young applicant → also calls bank statements.
- **What if removed?** No KYC data, no credit report → risk agent gets defaults → heuristic-only scoring

### Agent 2: Fraud Agent (part of intake_agent)
- **Purpose:** Identity mismatch detection via name comparison
- **File:** `agents/intake_agent.py`, function `_run_fraud()` (line 410)
- **Input:** `applicant_name`, `pan_verification.name_on_record`, `aadhaar_verification.name`
- **Output:** `fraud_detected`, `requires_manual_kyc`, `fraud_reason`
- **Uses LLM?** Only in ambiguous similarity band (0.60-0.92)
- **Uses hardcoded rules?** Yes — Jaro-Winkler gates; SHA-256 deduplication
- **Decision logic:** similarity ≥ 0.92 → MATCH; < 0.60 → FRAUD; between → LLM decides
- **What if removed?** Identity theft and name mismatches go undetected

### Agent 3: Risk Agent
- **Purpose:** Compute composite risk score (0-100) using ML + heuristics
- **File:** `agents/risk_agent.py`, function `risk_agent()` (line 430)
- **Input:** Features extracted from state (credit_score, dti_ratio, loan_to_income, etc.)
- **Output:** `risk_score`, `risk_category` (LOW/MEDIUM/HIGH), `risk_factors`, `risk_agent_reflection`
- **Uses LLM?** NO — purely computational
- **Uses RAG?** No
- **Uses hardcoded rules?** Yes — heuristic scoring, hard overrides, self-reflection
- **Decision logic:**
  1. Hard override check (defaults>3 → HARD_REJECT, credit>820+clean → HARD_FAST_TRACK)
  2. ML model prediction (RandomForest, 3-class: approve/review/reject)
  3. Heuristic score (rule-based, starts at 50, adjusts per factor)
  4. Dynamic blending: `ml_weight = 0.4 + 0.4 * ml_confidence`
  5. Self-reflection: checks for contradictions (e.g., high credit but high risk)
  6. Categorize: ≤35 LOW, ≤60 MEDIUM, >60 HIGH
- **What if removed?** No risk scoring → no routing → system cannot decide

### Agent 4: Policy Agent
- **Purpose:** Check regulatory compliance (RBI guidelines)
- **File:** `agents/intake_agent.py`, function `_run_policy()` (line 683), node `policy_agent_node()` (line 764)
- **Input:** applicant_data, credit_report
- **Output:** `policy_passed`, `policy_violations`, `policy_rag_reasoning`, `policy_retrieval_score`
- **Uses LLM?** Only when RAG retrieval score < 0.25 (rare)
- **Uses RAG?** Yes — TF-IDF over `policies/rbi_guidelines.txt`
- **Uses hardcoded rules?** Yes — 5 deterministic checks (age≥21, LTI≤5, defaults≤2, credit≥400, high-value employment)
- **Decision logic:** RAG retrieves policy text → if high confidence → deterministic rules; if low → LLM interprets
- **What if removed?** Under-age, over-leveraged, high-default applicants get approved

### Agent 5: Fairness Agent
- **Purpose:** Counterfactual bias detection across gender, age, location
- **File:** `agents/fairness_agent.py`, function `_run_fairness_math()` (line 90)
- **Input:** `risk_score`, `applicant_data`, full state
- **Output:** `fairness_report`, `bias_detected`, `fairness_debate_log`, `debate_transcript`
- **Uses LLM?** NO — pure mathematical counterfactual analysis
- **Uses RAG?** No
- **Uses hardcoded rules?** Yes — thresholds for gender (5pts), age (10pts), location (8pts)
- **Decision logic:** For each protected attribute → flip value → re-run `evaluate_profile()` → compute score delta → if |delta| > threshold → bias_detected=True
- **What if removed?** Demographic bias goes undetected; no counterfactual audit trail

### Agent 6: Orchestrator
- **Purpose:** Dynamic routing — decides which agent runs next
- **File:** `agents/orchestrator_agent.py`, function `orchestrator_decide()` (line 439)
- **Input:** Full state (risk_score, risk_category, policy results, fairness results, history)
- **Output:** `orchestrator_next_agent`, `orchestrator_history`
- **Uses LLM?** Only for ambiguous cases (~15%) — `_is_ambiguous()` gates the LLM call
- **Decision logic:**
  1. Check `_is_ambiguous()` — low ML confidence, conflicting verdicts, stalled pipeline
  2. If NOT ambiguous → `_fallback_decide()` (deterministic rules)
  3. If ambiguous → `_call_llm_orchestrator()` (LLM reasons about next step)
  4. Safety cap: MAX_ITERATIONS=12
- **What if removed?** No routing logic → pipeline cannot branch or loop

### Agent 7: Reporting Agent
- **Purpose:** Generate explanations, audit trail, store to episodic memory
- **File:** `agents/reporting_agent.py`, function `reporting_agent()` (line ~somewhere)
- **Input:** Full final state
- **Output:** `explanation` (applicant_message, admin_summary, compliance_report, process_trace), `audit_trail`, `learning_insights`
- **Uses LLM?** YES — generates all 4 explanation narratives
- **What if removed?** No human-readable explanations, no audit trail, no episodic memory

### auto_approve / auto_reject / final_decision
- **File:** `orchestrator/graph.py` (lines 38, 57, 78)
- **Purpose:** Terminal decision nodes
- **Uses LLM?** NO — pure state updates
- `auto_approve`: Sets `final_decision="approved"`
- `auto_reject`: Sets `final_decision="rejected"` with reason
- `final_decision`: For MEDIUM-risk: checks policy_passed + bias_detected → approved/rejected/escalated

---

## 5. LLM ANALYSIS

### Model: `llama-3.1-8b-instant` via Groq
- Initialized in: `agents/llm_client.py` — raw `httpx.post()` to `https://api.groq.com/openai/v1/chat/completions`
- Rate limited: 2.2s minimum between calls (process-wide lock)

### Prompt locations and purposes:

| # | File | Prompt | Purpose | LLM decides? |
|---|------|--------|---------|:---:|
| 1 | `intake_agent.py` ~line 175 | `DATA_AGENT_REACT_PROMPT` | Recovery on tool exception | Suggests next tool |
| 2 | `intake_agent.py` line 335 | `FRAUD_LLM_PROMPT` | Name mismatch classification | FRAUD/REVIEW/MATCH |
| 3 | `intake_agent.py` line 590 | `POLICY_RAG_PROMPT` | Interpret policy violations | Lists violations |
| 4 | `orchestrator_agent.py` line 144 | `ORCHESTRATOR_PROMPT` | Route to next agent | next_agent choice |
| 5 | `orchestrator_agent.py` line 188 | `ORCHESTRATOR_PLANNER_PROMPT` | Generate initial plan | Plan array |
| 6 | `reporting_agent.py` line 36 | `PROCESS_EXPLANATION_PROMPT` | Audit process trace | Narrative text |
| 7 | `reporting_agent.py` line 93 | `APPLICANT_MESSAGE_PROMPT` | Customer-facing explanation | Narrative text |

### Honest Assessment: Is this genuinely agentic?

**YES, partially.** Here's the honest breakdown:

**Genuinely agentic behaviors:**
1. **Data Agent ReAct loop** — dynamically decides WHICH tools to call based on applicant profile (self-employed → GST, young → bank statements). This is real agentic tool selection.
2. **Orchestrator loop** — can call fairness → re-eval → fairness multiple times. Dynamic routing based on evolving state. The ambiguity gate (`_is_ambiguous()`) is well-designed.
3. **Self-reflection** in Risk Agent — checks its own score for contradictions and revises.
4. **Inter-agent debate** — Fairness Agent calls Risk Agent's `evaluate_profile()` with modified attributes.

**NOT genuinely agentic (deterministic with LLM veneer):**
1. **~85% of orchestrator decisions are deterministic** — the `_fallback_decide()` handles most cases. LLM is only invoked for ambiguous states.
2. **Risk scoring is 100% deterministic** — no LLM involvement at all.
3. **Fairness is 100% mathematical** — counterfactual diffs, no LLM reasoning.
4. **Policy checks are mostly deterministic** — 5 hardcoded rule functions; LLM only when RAG score is low.
5. **The Planner's output is largely ignored** — the orchestrator makes independent decisions after each agent anyway.

**Bottom line:** The system is a **hybrid architecture** where deterministic rules handle routine cases efficiently, and LLM reasoning is invoked only for edge cases. This is actually a GOOD design pattern (cost-effective, reliable), but you should describe it honestly as "ambiguity-gated agentic behavior" rather than claiming everything is LLM-driven.
