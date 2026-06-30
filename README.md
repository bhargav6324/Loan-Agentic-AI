# Agentic AI System for Automated Loan Approval

An autonomous, multi-agent AI system designed for loan approval and ethical bias mitigation in Financial Technology (Fintech). 

The system leverages **6 specialized agents** orchestrated via a LangGraph pipeline to verify identity, analyze financial/credit risk, enforce compliance, audit demographic bias, generate explainable summaries, and immutably log all execution steps.

---

## 🚀 System Quickstart Guide

### 1. Prerequisites
Ensure you have the following installed on your system:
- **Python 3.11** or **3.12**
- **Node.js** (v18 or higher) & **npm**
- **SQLite** (used automatically as default local DB in development)

---

### 2. Backend Setup (FastAPI)

1. **Navigate to the project root:**
   ```bash
   cd /home/bhargav-patil/Loan-Agentic-AI
   ```

2. **Activate the virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Verify/Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify Environment Variables (`.env`):**
   Make sure you have a `.env` file set up in the root directory (based on `.env.example`). The SQLite database connection is pre-configured for local dev:
   ```env
   DATABASE_URL=sqlite+aiosqlite:///./loan_agent.db
   SECRET_KEY=your-jwt-secret-key-here
   APP_ENV=development
   ```

5. **Start the FastAPI Dev Server:**
   ```bash
   uvicorn main:app --reload --host 127.0.0.1 --port 8000
   ```
   *The Swagger API documentation will be available at: http://127.0.0.1:8000/docs*

---

### 3. Frontend Setup (React + Vite)

1. **Open a new terminal window/tab and navigate to the frontend directory:**
   ```bash
   cd /home/bhargav-patil/Loan-Agentic-AI/frontend
   ```

2. **Install frontend dependencies:**
   ```bash
   npm install
   ```

3. **Start the React dev server:**
   ```bash
   npm run dev
   ```
   *The interface will run locally at: http://localhost:5173 (or the URL printed in the terminal)*

---

### 4. Running Pipeline Tests & Evaluations

You can run individual system checks and evaluation pipelines independently of the web servers:

#### A. Run End-to-End Pipeline Simulator (All 6 Agents)
Simulates a full loan application flowing through the Data, Risk (V2), Policy, Fairness, Explainability, and Audit agents:
```bash
# From the root directory (with venv active)
python test_pipeline.py
```

#### B. Run Fairness Evaluation Harness & Baseline Benchmark
Generates biased synthetic profiles, trains baselines (Random Forest, XGBoost, Logistic Regression), runs the Agentic System V2 comparison, and writes the markdown report:
```bash
# 1. Generate biased dataset
python research/data/generate_dataset.py

# 2. Run benchmark harness
python research/evaluation_harness.py
```
*The resulting tables are written to `research/metrics_report.md` and `research/final_comparative_analysis.md`.*
