# How to Explain This Project in an Interview

## THE OPENING (When they say "Tell me about your project")

> "I built an Agentic AI system for automated loan approval. The idea is — instead
> of using just one ML model to approve or reject a loan, I have multiple specialized
> AI agents that collaborate. One agent verifies identity, one scores risk, one checks
> regulatory compliance, one audits for gender/age bias, and an orchestrator agent
> decides which agents should run and in what order. The whole thing is built with
> LangGraph for orchestration, FastAPI for the backend, React for the frontend, and
> PostgreSQL for storage."

Then STOP. Let the interviewer ask the next question. Don't dump everything at once.

---

## STEP-BY-STEP FLOW (When they ask "Walk me through how it works")

Tell this like a story. Use your hands. Draw on the whiteboard if they have one.

### Step 1: User Submits Application

> "A user goes to the React frontend, fills in a loan form — name, PAN card number,
> Aadhaar number, income, loan amount, employment type, age, gender. They hit submit."

**What actually happens in code:**
- Frontend calls `POST /applications` (in `routes/application_routes.py`)
- Pydantic schema `LoanApplicationCreate` validates the input (PAN format, age range, etc.)
- A row is inserted into the `applications` table in PostgreSQL with `status = SUBMITTED`
- The user gets back an `application_id`

### Step 2: User Triggers Processing

> "Then the user clicks 'Process' on their dashboard. This triggers the AI pipeline."

**What actually happens:**
- Frontend calls `POST /applications/{id}/process` (in `routes/process_routes.py`)
- The backend builds an `initial_state` dictionary with all the applicant data
- Then it calls `loan_pipeline.invoke(initial_state)` — this ONE line runs the entire
  multi-agent pipeline. `loan_pipeline` is a compiled LangGraph StateGraph.

### Step 3: Intake Agent Runs (Data + Fraud)

> "The first agent is the Intake Agent. It has two jobs: verify the applicant's
> identity and check for fraud."

**Data Agent part — the ReAct loop:**
> "For identity verification, I implemented a ReAct pattern — Reason, Act, Observe.
> The agent looks at the applicant's profile and DECIDES which verification tools
> to call. For everyone, it calls PAN verification, Aadhaar verification, and pulls
> the credit report from CIBIL. But if the person is self-employed, it also
> dynamically calls GST verification. If the person is young, it pulls bank
> statements too. This is genuinely agentic — it's not a fixed sequence, the agent
> reasons about what data it needs."

- File: `agents/intake_agent.py`, function `_run_data()` (line 252)
- The tool registry is in `_AVAILABLE_TOOLS` — 5 mock tools (PAN, Aadhaar, CIBIL, GST, Bank)
- The decision logic is in `_fallback_reason_next_action()` — deterministic by default
- LLM is called ONLY if a tool throws an exception (recovery guidance)
- Loop runs up to 9 iterations, stops when all required tools are called

**Fraud Agent part:**
> "After gathering data, the fraud agent checks if the name on the PAN card matches
> the name the applicant provided. I use Jaro-Winkler string similarity.
> If similarity is above 0.92, it's a clear match — no LLM needed.
> If below 0.60, it's clearly fraud — no LLM needed.
> Only in the ambiguous middle band (0.60 to 0.92) — like 'Rahul' vs 'Rahul Kumar' —
> does it call the LLM to decide if it's a typo, a maiden name, or actual fraud."

- File: `agents/intake_agent.py`, function `_check_name_match()` (line 390)
- Also does SHA-256 identity deduplication — hashes PAN/Aadhaar to detect reuse

**If fraud is detected → immediate rejection. Pipeline ends here.**
**If clean → continues to next step.**

### Step 4: Planner Generates Initial Plan

> "Next, a Planner agent looks at the application and generates an initial execution
> plan — basically, which agents should run and in what order. This uses the LLM."

- File: `agents/orchestrator_agent.py`, function `generate_initial_plan()` (line 221)
- LLM receives: applicant summary, available agents, planning rules
- Returns JSON: `{"plan": ["fan_out", "policy_join", "fairness_agent", "final_decision"]}`
- If LLM fails → deterministic fallback plan
- **Important:** This plan is just a starting suggestion. The orchestrator can deviate from it later.

### Step 5: Risk Agent and Policy Agent Run IN PARALLEL

> "Here's where it gets interesting. Two agents run simultaneously — the Risk Agent
> and the Policy Agent. They don't depend on each other, so we run them in parallel
> using LangGraph's fan-out pattern."

**Risk Agent:**
> "The Risk Agent computes a risk score from 0 to 100. It uses two methods:
> 1. A trained RandomForest ML model that predicts approve/review/reject probabilities
> 2. A heuristic rule engine that scores based on credit score, debt-to-income ratio,
>    loan-to-income ratio, defaults, employment stability, etc.
>
> Then it BLENDS them dynamically — when the ML model is confident, it gets more
> weight. When it's uncertain, the heuristics dominate. This is called dynamic
> confidence-based weighting.
>
> After computing the score, the agent does something called self-reflection — it
> checks for contradictions. Like, if the credit score is 800 but the risk score
> came out high, it says 'wait, that doesn't make sense' and adjusts.
>
> Finally it categorizes: below 35 is LOW risk, 35-60 is MEDIUM, above 60 is HIGH."

- File: `agents/risk_agent.py`
- `_extract_features()` → gets features from state
- `_check_hard_overrides()` → defaults > 3 or credit < 450 → instant HARD_REJECT
- `evaluate_profile()` → ML prediction + heuristic scoring + blending + self-reflection
- **NO LLM is used in the risk agent. It's purely computational.**

**Policy Agent (runs in parallel):**
> "At the same time, the Policy Agent checks if the application complies with RBI
> banking regulations. I have a text file with 20 sections of RBI guidelines.
> The agent uses a RAG approach — it retrieves the relevant policy sections using
> TF-IDF similarity, then checks 5 hard rules:
> - Age must be at least 21
> - Loan-to-income ratio must be under 5x
> - Maximum 2 previous defaults
> - Minimum credit score of 400
> - Self-employed people can't get unsecured loans above ₹25 lakhs
>
> If the TF-IDF retrieval confidence is low — meaning the policy text doesn't clearly
> match — only then does it call the LLM to interpret the policy."

- File: `agents/intake_agent.py`, class `RAGPolicyEngine` (line 610)
- RAG: TF-IDF vectorizer over `policies/rbi_guidelines.txt`, cosine similarity retrieval
- Gate: retrieval_score ≥ 0.25 → deterministic rules; < 0.25 → LLM interprets
- Function `_run_policy()` (line 683)

### Step 6: Orchestrator Decides What Happens Next

> "After risk and policy agents finish, the Orchestrator takes over. This is the
> brain of the system.
>
> Here's what makes it smart: it doesn't blindly follow a fixed pipeline. It looks
> at the current state — what's the risk score? Did policy pass? Any fraud flags? —
> and decides what to do next.
>
> For routine cases — like clearly LOW risk with clean policy — it makes the decision
> deterministically. No LLM call needed. But for about 15% of cases that are
> genuinely ambiguous — like low risk but the fraud agent flagged something, or
> policy passed but there's potential bias — it calls the LLM to reason about the
> best next step.
>
> I call this 'ambiguity-gated' orchestration. The LLM only fires when rules
> can't safely decide."

- File: `agents/orchestrator_agent.py`, function `orchestrator_decide()` (line 439)
- `_is_ambiguous()` checks: low ML confidence, conflicting verdicts, stalled pipeline
- If NOT ambiguous → `_fallback_decide()` — deterministic rules
- If ambiguous → `_call_llm_orchestrator()` — LLM gets full state summary + history
- Safety cap: maximum 12 iterations to prevent infinite loops

**Routing outcomes:**
- LOW risk + policy pass → `auto_approve`
- HIGH risk or policy fail → `auto_reject`
- MEDIUM risk → run `fairness_agent` first, then decide

### Step 7: Fairness Agent (for MEDIUM-risk cases)

> "For borderline cases, I run a Fairness Agent that does counterfactual bias
> testing. Here's how it works:
>
> It takes the applicant's risk score — say 45. Then it flips one protected
> attribute — changes gender from male to female — and re-runs the EXACT same
> scoring logic. If the score changes to 38, that's a 7-point delta. If the delta
> exceeds a threshold (5 points for gender), bias is detected.
>
> It does this for three attributes: gender, age bracket, and location (urban vs
> rural). Every delta is logged with its sign — positive means the original was
> scored more harshly.
>
> No LLM is used here. It's pure math — run the scoring function twice with
> different inputs and compare."

- File: `agents/fairness_agent.py`, function `_run_fairness_math()` (line 90)
- `_simulate_counterfactual()` → deep copies state, flips attribute, calls `evaluate_profile()`
- Thresholds: gender = 5 pts, age = 10 pts, location = 8 pts
- Also reads from long-term memory — if historically > 60% of women were rejected, it tightens thresholds

### Step 8: Policy Join (Merging Parallel Results)

> "Remember risk and policy ran in parallel? The policy_join node is where their
> results merge. Its key job: if the policy agent found violations but the
> orchestrator wanted to approve — the policy agent WINS. Violations override
> approvals. This ensures regulatory compliance is never skipped."

- File: `orchestrator/graph.py`, function `policy_join()` (line 101)
- Pure state inspection, no LLM

### Step 9: Terminal Decision

> "Finally, one of three terminal nodes runs:
> - `auto_approve` — for LOW risk, clean applications
> - `auto_reject` — for HIGH risk, fraud, or policy violations
> - `final_decision` — for MEDIUM-risk cases, considering fairness results
>
> If the fairness agent detected bias, the case gets 'escalated' — meaning a human
> admin needs to review it."

- File: `orchestrator/graph.py`, functions at lines 38, 57, 78

### Step 10: Reporting Agent

> "The last agent generates human-readable explanations using the LLM. It creates
> four separate narratives:
> 1. An applicant-facing message (empathetic, simple language)
> 2. An admin summary (technical, with risk factors)
> 3. A compliance report (regulatory audit trail)
> 4. A process trace (how the agents collaborated)
>
> It also stores the case in ChromaDB episodic memory — so future applications
> can be compared to past precedents."

- File: `agents/reporting_agent.py`
- This is where the LLM is used most heavily
- Also stores to `memory/episodic_memory.py` (ChromaDB vector store)

### Step 11: Results Saved and Returned

> "The pipeline returns the final state. The backend saves everything:
> - Updates the application status in PostgreSQL (approved/rejected/escalated)
> - Creates a Decision record with the risk score, violations, fairness report,
>   and all three explanations
> - Saves every agent step as an AgentLog row for auditing
> - Returns a JSON response to the frontend
>
> The frontend shows the user a detailed breakdown — a risk gauge, an agent
> timeline showing which agents ran and what they found, the fairness analysis
> results, and the decision reasoning."

---

## WHEN THEY ASK FOLLOW-UP QUESTIONS

### "What makes this agentic and not just a pipeline?"

> "Three things make it genuinely agentic:
>
> 1. **Dynamic tool selection** — The Data Agent doesn't call the same tools for
>    everyone. A salaried person doesn't need GST verification. The agent reasons
>    about what data it needs based on the profile.
>
> 2. **Dynamic routing** — The orchestrator doesn't follow a fixed sequence. If
>    the policy agent finds a critical violation, it skips fairness analysis and
>    rejects immediately. If the ML model is uncertain, it asks for re-evaluation.
>    The graph has cycles — fairness can loop back to risk for re-scoring.
>
> 3. **Self-reflection** — The risk agent checks its own output for contradictions.
>    It can revise its score if the ML model and heuristic rules disagree.
>
> But I'll be honest — about 85% of decisions go through the deterministic path.
> The LLM is only invoked for the 15% of genuinely ambiguous cases. I think this
> is actually better engineering than calling the LLM for every decision — it's
> faster, cheaper, and more reliable."

### "Where exactly is the LLM used?"

> "The LLM — llama-3.1-8b on Groq — is used in 7 specific places:
> 1. Data Agent: only when a tool throws an exception (recovery)
> 2. Fraud Agent: only for ambiguous name matches (0.60-0.92 similarity)
> 3. Policy Agent: only when RAG retrieval confidence is low
> 4. Planner: generates initial plan
> 5. Orchestrator: only for ambiguous routing decisions
> 6-7. Reporting Agent: generates all explanations
>
> The risk scoring, fairness testing, and policy rule checks are all deterministic.
> No LLM. I did this intentionally — for financial decisions, you want
> reproducibility and auditability, which LLMs can't guarantee."

### "Why LangGraph instead of just calling functions in sequence?"

> "Three reasons:
> 1. **Parallel execution** — Risk and policy agents run simultaneously. LangGraph
>    handles the fan-out and state merging with custom reducers.
> 2. **Cycles** — After fairness detects bias, the orchestrator can loop back to
>    re-evaluate risk. You can't do cycles with a simple function chain.
> 3. **Ablation support** — For my research paper, I needed to disable individual
>    agents. LangGraph lets me pass `disable_fairness=True` and it replaces that
>    node with a no-op lambda. Try doing that cleanly with function calls."

### "How does the RAG work?"

> "I have two retrieval systems:
>
> 1. **Policy RAG** — I have a document with 20 sections of RBI banking guidelines.
>    I chunk it by section headings, vectorize with TF-IDF, and retrieve the top 3
>    relevant chunks using cosine similarity. If the match score is high (above 0.25),
>    I apply deterministic rules. If it's low, the LLM interprets the retrieved text.
>
> 2. **Episodic Memory** — I use ChromaDB with MiniLM embeddings to store past loan
>    applications as vectors. When a new application comes in, the orchestrator
>    retrieves the most similar past cases and their outcomes. This gives the LLM
>    precedent-based reasoning — 'a similar applicant was flagged for bias last time.'"

### "What database do you use and why?"

> "PostgreSQL with async SQLAlchemy. Five tables: users, applications, decisions,
> agent_logs, and documents. The key design choice is the `shared_state` JSON column
> in the applications table — it stores the ENTIRE final LangGraph state, so the
> frontend can read any agent's output without joining multiple tables. Agent logs
> are also stored as separate rows for querying and filtering."

### "What about security?"

> "JWT authentication with access and refresh tokens. Passwords are bcrypt hashed.
> For identity security, I hash PAN and Aadhaar numbers with SHA-256 before storing
> them in the deduplication index — no raw PII is persisted in that index.
> Rate limiting via slowapi. CORS is open in development but restricted in
> production. There's also input validation through Pydantic schemas — PAN format
> must match the regex `[A-Z]{5}[0-9]{4}[A-Z]`, Aadhaar must be exactly 12 digits."

### "What's the biggest limitation?"

> "The counterfactual fairness agent. In our ablation study across 10 random seeds,
> we found that the fairness agent actually INCREASES demographic parity disparity.
> It improves specificity — catching more true defaulters — but the deterministic
> override rule disproportionately affects one demographic group more than the other.
> An adaptive threshold that accounts for population-level distributions would be
> more rigorous. We report this honestly in the paper."

### "What would you improve?"

> "Four things:
> 1. Replace TF-IDF with dense embeddings for policy RAG — actual semantic understanding
> 2. Add an async task queue like Celery — right now processing is synchronous
> 3. Make the fairness threshold adaptive instead of a fixed 5-point cutoff
> 4. Replace mock tools with real APIs — the interface stays the same, just swap
>    the implementations"

---

## IMPORTANT: THINGS TO NEVER SAY

❌ "Every decision is made by the LLM" — it's not, 85% is deterministic
❌ "We use LangChain" — you don't, you use LangGraph directly with httpx
❌ "The RAG uses embeddings" — the Policy RAG uses TF-IDF (but Episodic Memory does use embeddings)
❌ "The system eliminates all bias" — it doesn't, the ablation study shows it can increase ΔDP
❌ "The agents talk to each other" — they communicate through shared state, not messages

## THINGS YOU SHOULD SAY

✅ "The orchestrator is ambiguity-gated — LLM only for the 15% of cases rules can't handle"
✅ "Risk scoring is deterministic — ML model + heuristics, no LLM, for reproducibility"
✅ "The fairness agent uses counterfactual analysis — pure math, no LLM"
✅ "We validated with a 10-seed ablation study with paired t-tests"
✅ "The system degrades gracefully — every LLM call has a deterministic fallback"

---

## QUICK REFERENCE: Key files to mention

| When discussing... | Mention this file |
|--------------------|--------------------|
| "The graph" | `orchestrator/graph.py` — `build_loan_graph()` |
| "Shared state" | `agents/state.py` — `AgentState(TypedDict)` |
| "Risk scoring" | `agents/risk_agent.py` — `evaluate_profile()` |
| "Fairness" | `agents/fairness_agent.py` — `_simulate_counterfactual()` |
| "Policy RAG" | `agents/intake_agent.py` — `RAGPolicyEngine` class |
| "Orchestrator" | `agents/orchestrator_agent.py` — `orchestrator_decide()` |
| "The API endpoint" | `routes/process_routes.py` — `process_application()` |
| "Database" | `models/database.py` — 5 SQLAlchemy ORM models |
| "LLM calls" | `agents/llm_client.py` — rate-limited Groq client |
| "Vector memory" | `memory/episodic_memory.py` — ChromaDB |
