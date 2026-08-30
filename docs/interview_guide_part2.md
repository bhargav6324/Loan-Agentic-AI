# Interview Guide — Loan Agentic AI (Part 2 of 2)

## 6. RAG / KNOWLEDGE RETRIEVAL

### Two retrieval systems exist:

**A. Policy RAG (TF-IDF based — NOT neural)**
- **File:** `agents/intake_agent.py`, class `RAGPolicyEngine` (line 610)
- **Source document:** `policies/rbi_guidelines.txt` (20 sections of RBI banking regulations)
- **Chunking:** Split by `##` headings → ~20 chunks
- **Embeddings:** TF-IDF (`sklearn.feature_extraction.text.TfidfVectorizer`) — keyword-based, NOT semantic
- **Storage:** In-memory (rebuilt on startup, no persistence)
- **Retrieval:** `cosine_similarity(query_vec, tfidf_matrix)`, returns top-3 chunks + best score
- **How it reaches LLM:** Only if `retrieval_score < 0.25` → retrieved text inserted into `POLICY_RAG_PROMPT` → LLM interprets
- **Why RAG:** Enables the policy agent to reference specific regulatory sections rather than hardcoding every possible rule. In practice, the TF-IDF score is usually > 0.25, so deterministic rules run most of the time.
- **Honest note:** This is lightweight keyword retrieval, not the dense-embedding RAG that interviewers typically mean.

**B. Episodic Memory (ChromaDB — genuine vector search)**
- **File:** `memory/episodic_memory.py`
- **Source:** Past loan application profiles stored after each decision
- **Embeddings:** `all-MiniLM-L6-v2` (ChromaDB default sentence transformer)
- **Storage:** ChromaDB persistent client at `data/chroma_db/`
- **Retrieval:** `collection.query(query_texts=[document], n_results=top_k)` — cosine similarity in HNSW index
- **How context reaches agents:** `orchestrator_agent.py` line 107 calls `retrieve_similar_cases()` → past case decisions are appended to the state summary sent to the LLM orchestrator
- **Why:** Allows the orchestrator to learn from precedent — "a similar applicant was rejected last time for bias"
- **Honest note:** This IS genuine semantic vector search. However, its impact on decisions is indirect — it only adds context to the orchestrator's LLM prompt, not to any deterministic rule.

---

## 7. DATABASE

### Engine: PostgreSQL (async via SQLAlchemy + asyncpg)
- **File:** `models/database.py`
- **Connection:** `config.py` → `DATABASE_URL = "postgresql+asyncpg://loanai:loanai123@localhost:5432/loanai_db"`

### Tables:

| Table | Key Columns | Purpose |
|-------|-------------|---------|
| `users` | id, email, password_hash, full_name, role (user/admin), is_active | User accounts |
| `applications` | id, user_id (FK), status (enum), loan_amount, loan_purpose, applicant_name, applicant_pan, monthly_income, employment_type, **shared_state** (JSON) | Loan applications + all agent outputs |
| `decisions` | id, application_id (FK, unique), status, risk_score, risk_category, reasoning, risk_factors (JSON), policy_violations (JSON), fairness_report (JSON), explanation_applicant/reviewer/regulator | Final decisions with explanations |
| `agent_logs` | id, application_id (FK), agent_name, step_number, action, status, duration_ms, output_data (JSON) | Per-step audit trail |
| `documents` | id, application_id (FK), file_name, file_path, file_type, parsed_data (JSON) | Uploaded PDFs |

### Important flow:
1. `POST /applications` → inserts `Application` row (status=SUBMITTED)
2. `POST /applications/{id}/process` → updates status to PROCESSING → runs pipeline → updates status to APPROVED/REJECTED/ESCALATED → creates `Decision` row → creates `AgentLog` rows
3. `GET /applications/{id}` → reads Application + shared_state JSON (contains ALL agent outputs)
4. `GET /admin/escalations` → queries applications WHERE status=ESCALATED

---

## 8. BACKEND API

| Endpoint | Method | Purpose | File |
|----------|--------|---------|------|
| `GET /` | GET | Health check | `main.py` |
| `POST /auth/register` | POST | Create user account | `routes/auth_routes.py` |
| `POST /auth/login` | POST | Get JWT tokens | `routes/auth_routes.py` |
| `POST /auth/refresh` | POST | Refresh access token | `routes/auth_routes.py` |
| `GET /auth/me` | GET | Get current user info | `routes/auth_routes.py` |
| `POST /applications` | POST | Submit loan application | `routes/application_routes.py` |
| `GET /applications` | GET | List user's applications | `routes/application_routes.py` |
| `GET /applications/{id}` | GET | Get application details | `routes/application_routes.py` |
| `POST /applications/{id}/process` | POST | Trigger agentic pipeline | `routes/process_routes.py` |
| `POST /applications/{id}/process_stream` | POST | SSE streaming version | `routes/process_routes.py` |
| `GET /admin/applications` | GET | Admin: list all apps | `routes/admin_routes.py` |
| `GET /admin/escalations` | GET | Admin: escalated cases | `routes/admin_routes.py` |

---

## 9. FRONTEND

### Tech Stack: React + Vite, vanilla CSS

### Pages (`frontend/src/pages/`):
| Page | File | Purpose |
|------|------|---------|
| Landing | `Landing.jsx` | Welcome page |
| Login | `Login.jsx` | JWT login form |
| Register | `Register.jsx` | Account creation |
| Apply Loan | `ApplyLoan.jsx` | Multi-field loan form (name, PAN, income, etc.) |
| Dashboard | `Dashboard.jsx` | Lists user's applications + statuses |
| Application Detail | `ApplicationDetail.jsx` (28KB!) | **Most important** — shows agent timeline, risk gauge, fairness report, explanations |
| Admin Dashboard | `AdminDashboard.jsx` | Admin view of all applications + escalations |

### Key Components (`frontend/src/components/`):
- `AgentPipeline.jsx` — Visual timeline of which agents ran, with status icons
- `RiskGauge.jsx` — Animated gauge showing risk score 0-100
- `Accordion.jsx` — Collapsible sections for agent details
- `Navbar.jsx` — Navigation bar with auth state

### API layer: `frontend/src/api.js`
- Axios instance with JWT interceptor
- Functions: `login()`, `register()`, `submitApplication()`, `processApplication()`, `getApplications()`, `getApplicationDetail()`

### State management: React `useState` + `useEffect` (no Redux/Zustand)

### How results are displayed:
`ApplicationDetail.jsx` reads `application.shared_state` JSON → renders every agent's output in accordion sections → shows risk gauge, fairness debate log, policy violations, LLM explanations

---

## 10. SECURITY AND FAIRNESS

| Feature | Status | Implementation |
|---------|--------|----------------|
| Counterfactual fairness | ✅ Implemented | `fairness_agent.py` — flips gender/age/location, measures score delta |
| Sensitive attribute handling | ✅ Implemented | Gender/age/location used ONLY for counterfactual testing, NOT for ML scoring in heuristics (V3 removed demographic bias from heuristics). But ML model DOES include `gender_encoded` and `location_encoded` as features. |
| Explainability | ✅ Implemented | 4 narratives: applicant, admin, compliance, process trace (LLM-generated) |
| Policy compliance | ✅ Implemented | 5 deterministic RBI rules + RAG over 20-section policy document |
| Audit logs | ✅ Implemented | Every agent step logged to `agent_logs` table with timestamps |
| Identity deduplication | ✅ Implemented | SHA-256 hashing of PAN/Aadhaar in `_IDENTITY_INDEX_PATH` |
| Data validation | ✅ Implemented | Pydantic schemas with regex patterns for PAN, Aadhaar |
| Authentication | ✅ Implemented | JWT (HS256) with access + refresh tokens, `auth/middleware.py` |
| Authorization | ✅ Partial | Role-based (user/admin), but admin role can be set at registration |
| Input sanitization | ⚠️ Basic | Pydantic validation only, no XSS/SQL injection specific defense (SQLAlchemy ORM handles SQL injection) |
| Rate limiting | ✅ Implemented | `slowapi` rate limiter in `gateway/rate_limiter.py` |

---

## 11. FILE PRIORITY

### 🔴 HIGH PRIORITY (Must understand deeply)
| File | Why |
|------|-----|
| `orchestrator/graph.py` | THE LangGraph definition — nodes, edges, routing functions |
| `agents/state.py` | Shared state schema with reducers for parallel safety |
| `agents/orchestrator_agent.py` | Ambiguity-gated LLM orchestration — core "agentic" logic |
| `agents/risk_agent.py` | ML + heuristic scoring, self-reflection, hard overrides |
| `agents/fairness_agent.py` | Counterfactual bias detection — the paper's key contribution |
| `agents/intake_agent.py` | Data ReAct loop + Fraud + Policy agent (3 agents in 1 file) |
| `routes/process_routes.py` | The endpoint that triggers the pipeline + saves results |
| `models/database.py` | Database schema — tables, relationships |

### 🟡 MEDIUM PRIORITY (Should understand)
| File | Why |
|------|-----|
| `agents/llm_client.py` | How LLM calls work — rate limiting, retry logic |
| `agents/reporting_agent.py` | Explainability + audit + episodic memory storage |
| `memory/episodic_memory.py` | ChromaDB vector search — genuine RAG |
| `config.py` | Environment configuration |
| `main.py` | FastAPI app setup, middleware, route registration |
| `frontend/src/pages/ApplicationDetail.jsx` | How agent outputs are visualized |
| `policies/rbi_guidelines.txt` | The policy document used for RAG |

### 🟢 LOW PRIORITY
| File | Why |
|------|-----|
| `tools/mock_*.py` | Mock external APIs (PAN, Aadhaar, CIBIL) |
| `auth/` | Standard JWT auth — nothing novel |
| `models/schemas.py` | Pydantic schemas — standard validation |
| `ml/train_model.py` | Model training script |
| `research/` | Ablation study scripts (academic, not production) |

---

## 12. INTERVIEW Q&A (Top 30)

### A. Basic Project Questions

**Q1: What does this project do?**
A: It's a multi-agent AI system for automated loan approval. Instead of a single ML model making decisions, 6+ specialized agents collaborate — each handling KYC verification, risk scoring, regulatory compliance, fairness auditing, and explainability — orchestrated through a LangGraph state graph.

**Q2: What's the tech stack?**
A: Backend: FastAPI + Python, LangGraph for orchestration, Groq (llama-3.1-8b) for LLM, PostgreSQL (async via SQLAlchemy), ChromaDB for vector memory. Frontend: React + Vite. Auth: JWT.

**Q3: How does a loan application get processed?**
A: User submits form → `POST /applications/{id}/process` → LangGraph `invoke()` runs the state graph → intake (KYC+fraud) → parallel risk+policy → orchestrator routes → optional fairness → terminal decision → reporting → results saved to DB → JSON response.

### B. Architecture Questions

**Q4: Why multi-agent instead of a single model?**
A: A single model optimizes for prediction accuracy. Regulatory compliance, fairness, and explainability are separate concerns that benefit from dedicated agents. The policy agent enforces hard rules that no amount of ML training can guarantee. The fairness agent detects bias through counterfactual testing, which requires re-running the risk model with modified attributes.

**Q5: Why parallel execution for risk and policy?**
A: They're independent — risk scoring doesn't depend on policy, and vice versa. Running them in parallel (via LangGraph fan-out) reduces latency. The `policy_join` node merges their results afterward.

**Q6: What happens if the LLM is unavailable?**
A: Every LLM call has a deterministic fallback. The orchestrator uses `_fallback_decide()`, the fraud agent uses string comparison, the policy agent uses hardcoded rules. The system degrades gracefully — it still works, just without LLM reasoning for edge cases.

### C. LangGraph Questions

**Q7: What is LangGraph and why use it here?**
A: LangGraph is a library for building stateful, cyclic agent workflows as directed graphs. We use it because our pipeline has conditional branching (LOW→approve, HIGH→reject, MEDIUM→fairness loop), parallel execution (risk+policy), and cycles (fairness→orchestrator→risk_reeval→orchestrator). A simple sequential pipeline can't express this.

**Q8: How does state flow between nodes?**
A: All nodes receive the full `AgentState` TypedDict and return a partial update. LangGraph merges updates using reducers — `_last` (last-write-wins) for scalars, `operator.add` for lists. This enables parallel branches to safely write to the same state.

**Q9: What are the conditional edges?**
A: Three: (1) `route_after_intake` — if fraud → auto_reject, else planner. (2) `route_orchestrator` — dispatches to 6 possible nodes based on `orchestrator_next_agent`. (3) `route_after_join` — dispatches to 3 terminal nodes after policy merge.

### D. LLM/Agent Questions

**Q10: Which LLM do you use and why?**
A: `llama-3.1-8b-instant` on Groq. Groq offers free-tier access with fast inference. The 8B model is sufficient because our LLM tasks are structured (JSON output, routing decisions) rather than open-ended generation.

**Q11: How do you prevent LLM hallucinations?**
A: (1) `response_format: {"type": "json_object"}` forces structured output. (2) `temperature: 0.0-0.1` reduces randomness. (3) Every LLM response is validated against known values (e.g., `next_agent` must be in `VALID_AGENTS`). (4) Deterministic fallback on parse failure.

**Q12: What makes the orchestrator "agentic"?**
A: The `_is_ambiguous()` gate. For ~85% of cases, deterministic rules decide the next agent. For the remaining ~15% (low ML confidence, conflicting signals), the LLM observes the full state and reasons about what to do next. The LLM can override the default sequence — e.g., skip fairness for a clearly fraudulent application. This is genuine autonomous decision-making, not just following a fixed pipeline.

### E. RAG Questions

**Q13: How does RAG work in this project?**
A: Two systems. Policy RAG uses TF-IDF over 20 sections of RBI guidelines — it's keyword-based, not semantic. Episodic Memory uses ChromaDB with MiniLM embeddings for semantic similarity search over past loan applications.

**Q14: Why TF-IDF for policy RAG instead of embeddings?**
A: The policy document is structured regulatory text with specific keywords (e.g., "minimum age 21", "LTI cap 5x"). TF-IDF handles keyword matching well for this domain. Also, it's dependency-free (sklearn only) and doesn't need an embedding model.

### F. Backend/API Questions

**Q15: How does SSE streaming work?**
A: `POST /applications/{id}/process_stream` returns a `StreamingResponse` with `text/event-stream` media type. Inside, `loan_pipeline.astream()` yields state updates after each node. Each update is sent as an SSE event with the node name and a human-readable message.

**Q16: How is the database session managed?**
A: FastAPI dependency injection via `get_db()` → yields an `AsyncSession` from `AsyncSessionLocal`. Sessions are scoped to the request and closed in the `finally` block.

### G. Database Questions

**Q17: What is `shared_state` in the applications table?**
A: A JSON column that stores the ENTIRE final LangGraph state — all agent outputs, logs, explanations. This is the single source of truth that the frontend reads to display the complete agent timeline.

**Q18: Why store agent logs separately from shared_state?**
A: `shared_state` is a single JSON blob for fast reads. `AgentLog` rows enable querying/filtering by agent name, step number, or status — useful for admin dashboards and debugging.

### H. Fairness Questions

**Q19: How does counterfactual fairness testing work?**
A: The fairness agent takes the original risk score, flips one protected attribute (e.g., gender male→female), re-runs `evaluate_profile()` with the modified features, and measures the score difference. If |delta| exceeds a threshold (5 pts for gender, 10 for age, 8 for location), bias is flagged.

**Q20: Why does the fairness agent sometimes INCREASE demographic disparity?**
A: Because the override rule is a deterministic threshold. When the distribution of counterfactual score differences differs between demographic groups, the override disproportionately alters one group's approval rate. This is an empirical finding from the ablation study — the fairness agent improves specificity but can increase ΔDP.

### I. Critical/Challenging Questions

**Q21: If 85% of orchestrator decisions are deterministic, why do you need an LLM at all?**
A: The 15% ambiguous cases are where the system adds value over a pure rule engine. These are cases with conflicting signals (e.g., LOW risk but fraud flag, policy passed but bias detected) where hardcoded rules can't safely decide. The LLM weighs trade-offs using the full context.

**Q22: The risk agent doesn't use an LLM — isn't that the most important decision?**
A: Risk scoring is a well-defined numerical problem — ML models and heuristic rules handle it reliably. Using an LLM for score computation would be slower, more expensive, and less reproducible. The LLM is reserved for tasks requiring natural language reasoning (explaining decisions, resolving ambiguity).

**Q23: Your RAG is just TF-IDF — how is that AI?**
A: Fair criticism. The TF-IDF retrieval is a lightweight information retrieval step, not deep semantic understanding. However, the system is designed with a gate: when TF-IDF confidence is high (exact keyword match), deterministic rules apply. When it's low (novel policy situation), the LLM interprets the retrieved text. The real intelligence is in this hybrid design, not in the retrieval method itself.

**Q24: What happens if Groq goes down?**
A: Every LLM-dependent function has a try/except with deterministic fallback. The system continues to work without any LLM — just with less nuanced decisions for edge cases. This is tested and intentional.

**Q25: How do you handle concurrent loan processing?**
A: FastAPI is async. Each `process_application` runs the pipeline synchronously within an async handler. LangGraph `invoke()` is CPU-bound, so concurrent requests queue at the GIL. For production, you'd need a task queue (Celery/RQ). The rate limiter in `gateway/rate_limiter.py` also throttles requests.

**Q26: The mock tools return fake data — how would this work in production?**
A: Replace `tools/mock_pan.py` etc. with real API clients (NSDL PAN verification, UIDAI Aadhaar, CIBIL bureau API). The tool interface stays the same — same input/output signatures. The ReAct loop and fraud agent logic don't change.

**Q27: What are the scalability bottlenecks?**
A: (1) Groq rate limit (30 RPM) — production would need a dedicated LLM endpoint. (2) Synchronous pipeline execution — no parallelism across applications. (3) ChromaDB is single-process — need a distributed vector DB at scale. (4) Identity index is a JSON file — needs a proper distributed cache.

**Q28: Could you replace LangGraph with simple function calls?**
A: For the sequential path, yes. But LangGraph provides: (a) parallel fan-out without manual threading, (b) state reduction for merge conflicts, (c) cycle support for the fairness-risk re-eval loop, (d) streaming via `astream()`, (e) easy ablation via node replacement. It's worth the complexity.

**Q29: What's your biggest technical achievement in this project?**
A: The ambiguity-gated orchestrator design. Most "agentic AI" projects call the LLM for every decision, which is slow and expensive. This system only invokes the LLM when deterministic rules genuinely can't decide (~15% of cases). That's architecturally mature — it's how you'd actually deploy LLM-augmented systems in production.

**Q30: What would you improve?**
A: (1) Replace TF-IDF with dense embeddings for policy RAG. (2) Add async task queue for concurrent processing. (3) Implement proper RBAC instead of allowing admin role at registration. (4) Add integration tests for the full pipeline. (5) Use a real credit bureau API. (6) Add monitoring/observability for LLM calls. (7) The fairness agent's threshold is static — an adaptive threshold based on population statistics would be more rigorous.

---

## 13. CRITICAL WEAKNESSES

| Weakness | Impact | Defense/Improvement |
|----------|--------|---------------------|
| **85% deterministic orchestrator** | Interviewer may question "agentic" claim | "It's ambiguity-gated by design. Production systems should NOT call LLMs for every decision. The agentic behavior emerges for the hard cases that matter." |
| **TF-IDF is not real RAG** | Technically keyword retrieval, not semantic | "TF-IDF is appropriate for structured regulatory text. The system has a second RAG with ChromaDB+MiniLM for semantic search. The design is intentionally hybrid." |
| **Mock tools** | No real external integration | "Same interface. Replace with real APIs by changing one function each. The architecture is integration-ready." |
| **Identity index is a JSON file** | Single-process, no concurrent safety | "Production would use Redis or PostgreSQL for this. The SHA-256 hashing design is sound." |
| **Admin role assignable at registration** | Security flaw | "This is a demo shortcut. Production would require admin invitation or separate admin creation flow." |
| **ML model includes gender_encoded** | Contradiction with fairness claims | "This is intentional — it creates emergent bias that the fairness agent detects via counterfactual testing. The heuristic scorer is demographic-blind (V3). The ML model is the bias vector under test." |
| **No async pipeline execution** | Blocks the event loop | "LangGraph's invoke is synchronous. Production would offload to a Celery worker and poll for completion." |
| **Fairness agent increases ΔDP** | Counter to its purpose | "This is an empirical finding documented in the ablation study. The override rule's deterministic nature can create disparate impact. We report this honestly — it's a research contribution, not a bug." |
| **Rate limit: 30 RPM on Groq** | Can't handle load | "Use a dedicated LLM endpoint (vLLM, TGI) or enterprise Groq plan in production." |
| **LLM explanations may hallucinate** | Inaccurate explanations | "The LLM only EXPLAINS pre-computed results (risk score, violations, etc.) — it never CALCULATES them. If the explanation is wrong, the underlying decision is still correct." |

---

## 14. INTERVIEW CHEAT SHEET

### 30-second explanation
> "I built a multi-agent AI system for loan approval using LangGraph and FastAPI. Instead of one ML model, 6 specialized agents handle KYC verification, risk scoring, regulatory compliance, fairness auditing, and explainability. An LLM-powered orchestrator dynamically routes applications through these agents based on their risk profile. The system uses counterfactual fairness testing — flipping protected attributes and measuring score changes to detect bias."

### 1-minute explanation
> "This is a multi-agent loan approval system with a LangGraph-based state graph. When a user submits a loan application, the Data Agent runs a ReAct loop calling external KYC APIs dynamically. The Fraud Agent detects identity mismatches using fuzzy matching with LLM fallback. Then, in parallel, a Risk Agent scores the application using an ensemble of ML predictions and heuristic rules, while a Policy Agent checks RBI regulatory compliance using RAG over policy documents. An orchestrator — which is ambiguity-gated, meaning it uses deterministic rules for routine cases and LLM reasoning only for genuinely ambiguous ones — decides the next steps. For medium-risk cases, a Fairness Agent runs counterfactual analysis, flipping gender, age, and location to detect bias. Finally, a Reporting Agent generates multi-audience explanations. Everything is logged for audit, stored in PostgreSQL, and displayed in a React frontend with real-time SSE streaming."

### 3-minute deep explanation
> Add to the 1-minute: "The architecture is V7, which introduced parallel fan-out — risk and policy agents run simultaneously via LangGraph edges, then merge at a policy_join node. The orchestrator is the core agentic component: it uses `_is_ambiguous()` to check for conflicting signals (low ML confidence, manual KYC flagged alongside low risk, etc.) and only invokes the LLM for those ~15% of ambiguous cases. The risk agent uses dynamic confidence-based weighting — when the ML model is confident (high max probability), it gets more weight; when uncertain, heuristic rules dominate. It also has a self-reflection loop that checks for contradictions between ML and heuristic scores. The fairness agent doesn't use an LLM — it's pure math: create a deep copy of state, flip a protected attribute, re-run evaluate_profile(), measure the delta. If |delta| > threshold, bias is flagged and the case is escalated. We validated this with a 10-seed controlled ablation study across 6 configurations, with paired t-tests showing statistically significant differences in specificity and balanced accuracy."

### "Why Agentic AI?"
> "Three reasons: (1) The Data Agent dynamically selects which verification tools to call based on the applicant's profile — a salaried employee doesn't need GST verification. (2) The orchestrator can loop — after the fairness agent detects bias, it can re-evaluate risk with adjusted features, creating a multi-turn debate. (3) The system reacts to intermediate results — if the policy agent finds violations, the orchestrator skips fairness and goes straight to rejection. A static pipeline can't do any of this."

### "Why LangGraph?"
> "We need conditional routing, parallel execution, and cycles. LangGraph gives us all three with state reduction for parallel safety. The `StateGraph` with `Annotated` reducers (last-write-wins for scalars, list concatenation for logs) handles the fan-out merge elegantly. We also use `build_loan_graph(disable_fairness=True)` for ablation studies — replacing agents with no-op lambdas."

### "Why LLM?"
> "The LLM handles tasks that require natural language reasoning: disambiguating name mismatches ('Rahul' vs 'Rahul Kumar'), interpreting novel regulatory situations not covered by hardcoded rules, explaining decisions in audience-specific language, and reasoning about conflicting signals in the orchestrator. It does NOT calculate risk scores or check policy rules — those are handled by faster, more reliable deterministic code."

### "Why RAG?"
> "The policy agent needs to check applications against 20 sections of RBI guidelines. Hardcoding every possible regulation is brittle. RAG retrieves the relevant policy sections dynamically. We also have episodic memory via ChromaDB — past loan outcomes are stored as vector embeddings and retrieved when the orchestrator encounters a similar profile, providing precedent-based reasoning."

### Biggest technical challenge
> "Getting parallel state management right. When risk_agent and policy_agent run simultaneously and both write to state, LangGraph needs custom reducers to resolve conflicts. Scalar fields use last-write-wins, but agent_logs need list concatenation. Getting this wrong causes silent data loss."

### Biggest limitation
> "The counterfactual fairness agent increases demographic parity disparity in our ablation study. The deterministic override rule doesn't adapt to the population distribution of score differences across demographic groups. An adaptive threshold would be more rigorous."

### Future improvements
> "(1) Dense embeddings for policy RAG. (2) Async task queue for concurrent processing. (3) Adaptive fairness thresholds. (4) Real external API integration. (5) LLM observability/monitoring. (6) A/B testing framework for different orchestrator strategies."
