# Final Experimental Package Readiness Audit & Execution Report

**Date/Time of Audit:** 2026-07-26T16:55:00+05:30  
**Repository Status:** Read-Only (No source files modified)  
**Target Counterfactual Fairness Threshold:** `6.0`

This standalone report presents the verification, execution results, artifact inspection, and readiness assessment for the final experimental package of the Agentic AI Credit Risk & Fairness auditing research paper.

---

## 1. Pre-Execution Verification

Before executing any evaluation suites, an exhaustive code and dependency inspection was conducted across the codebase.

### A. Threshold Parameter Consistency Audit
* **Finding: MAJOR INCONSISTENCY DETECTED.**
* Every evaluation script in the repository currently hardcodes the counterfactual comparison threshold to **`5.0`** (or `5`), NOT the finalized publication threshold of **`6.0`**:
  * **`research/evaluation_harness.py` (Lines 275 & 278):**
    ```python
    if abs(cf_risk_score - risk_score) > 5:
        # ...
        if cf_risk_score < risk_score - 5:
    ```
  * **`research/full_metrics.py` (Lines 194 & 196):**
    ```python
    if abs(cf_risk_score - risk_score) > 5:
        # ...
        if cf_risk_score < risk_score - 5:
    ```
  * **`research/live_spot_check.py` (Lines 102 & 103):**
    ```python
    if abs(cf_risk_score - risk_score) > 5:
        if cf_risk_score < risk_score - 5: return 1
    ```
  * **`agents/fairness_agent.py` (Line 151):**
    ```python
    gender_threshold = max(1.0, 5.0 - systemic_bias_penalty)
    ```

### B. Risk of Overwriting by Old Cached Results
* **Finding:** Executing `research/evaluation_harness.py` and `research/full_metrics.py` directly overwrites existing files in `research/` (`results.csv`, `full_metrics.csv`, `metrics_report.md`, `full_metrics_report.md`, `roc_curves.png`, and `pareto_frontier.png`). Because the scripts still hardcode threshold `5.0`, executing them overwrites the repository with threshold 5.0 results rather than threshold 6.0 results.

### C. Dependency & Environment Health
* **Python Libraries:** All required core packages (`pandas`, `numpy`, `scikit-learn`, `xgboost`, `fairlearn`, `matplotlib`, `langgraph`, `groq`) are installed and functional.
* **Execution Issues Identified:**
  1. **ChromaDB Missing:** `live_spot_check.py` emits repeated runtime warnings (`Failed to retrieve episodic memory: No module named 'chromadb'`). While non-fatal (the agent falls back to stateless ReAct execution), it clutters standard error logs.
  2. **API Rate Limiting Vulnerability:** `live_spot_check.py` executes 30 live LangGraph agent invocations in rapid succession against the Groq API without rate-limit throttling or batching, making it highly susceptible to HTTP 429 Too Many Requests errors.

---

## 2. Final Evaluation Execution Log

In accordance with the required dependency order, execution was initiated without altering any source files:

### Step 1: `python3 research/evaluation_harness.py`
* **Status:** **SUCCESS** (Completed in ~12 seconds).
* **Console Output Summary:**
  ```text
  Training Logistic Regression...
  Training Random Forest...
  Training XGBoost...
  Running Agentic AI Pipeline (v2 — Enhanced Scoring)...
    [chart] ROC curves saved → /home/bhargav-patil/Loan-Agentic-AI/research/figures/roc_curves.png
    [chart] Pareto frontier saved → /home/bhargav-patil/Loan-Agentic-AI/research/figures/pareto_frontier.png
  Evaluation complete. Results saved to results.csv and metrics_report.md.
  Agentic Telemetry: Hard Rejects=7, Fast-Tracks=83, Escalations=72
  ```

### Step 2: `python3 research/full_metrics.py`
* **Status:** **SUCCESS** (Completed in ~55 seconds).
* **Console Output Summary:**
  ```text
  Training Logistic Regression...
  Training Random Forest...
  Training XGBoost...
  Running Agentic AI Pipeline (v2 — Enhanced Scoring)...
  COMPREHENSIVE METRICS REPORT — ML vs AGENTIC SYSTEM
  ✅ Full report saved to full_metrics.csv and full_metrics_report.md
  ```

### Step 3: `python3 research/live_spot_check.py`
* **Status:** **STOPPED / FAILED** (Terminated at Sample 3).
* **Console Output & Explanation:**
  ```text
  Training RF model for simulation benchmark...
  Running spot check on 30 samples...
  Sim   | Live  | Match | Income     | Credit | Gender
  -------------------------------------------------------
  1     | 1     | True  | ₹72864     | 701    | M
  1     | 1     | True  | ₹68609     | 678    | M
  [llm_client] 429 rate limited, retrying in 96.0s (attempt 1/3)
  [llm_client] 429 rate limited, retrying in 755.0s (attempt 1/3)
  ```
* **Why Execution Stopped:** On sample 3, the Groq API returned HTTP 429 (Rate Limit Exceeded), causing the client to enter an exponential backoff sleep of 755 seconds (12.5 minutes). Per your instructions (*"If any script fails, stop, explain why, do not attempt code changes"*), execution was halted.

---

## 3. Generated Artifact Verification

The table below catalogs the exact location, creation/update timestamp, and threshold reflected for every experimental artifact on disk following execution:

| Artifact Name | File Location | Creation / Update Timestamp | Reflects Locked Thresh (6.0)? | Notes & Evidence |
| :--- | :--- | :---: | :---: | :--- |
| **`results.csv`** | `research/results.csv` | 2026-07-26 16:50:39 | ❌ **NO (Reflects 5.0)** | Shows Escalation Rate = `4.8%` (72 cases) and DP Diff = `0.068`. |
| **`full_metrics.csv`** | `research/full_metrics.csv` | 2026-07-26 16:51:41 | ❌ **NO (Reflects 5.0)** | Shows Escalation Rate = `4.8%`, DP Diff = `0.0683`, EO Diff = `0.0405`. |
| **`metrics_report.md`** | `research/metrics_report.md` | 2026-07-26 16:50:39 | ❌ **NO (Reflects 5.0)** | Markdown summary of `results.csv`. |
| **`full_metrics_report.md`** | `research/full_metrics_report.md` | 2026-07-26 16:51:41 | ❌ **NO (Reflects 5.0)** | Markdown summary of `full_metrics.csv`. |
| **`roc_curves.png`** | `research/figures/roc_curves.png` | 2026-07-26 16:50:39 | ❌ **NO (Reflects 5.0)** | Plot reflects predictions generated under threshold 5.0. |
| **`pareto_frontier.png`** | `research/figures/pareto_frontier.png` | 2026-07-26 16:50:39 | ❌ **NO (Reflects 5.0)** | Trade-off curve highlights threshold 5.0 as the operating point. |
| **`live_spot_check`** | Console / None | Failed at 16:58:00 | ❌ **NO (Incomplete)** | Halted at 2/30 samples due to Groq API HTTP 429 rate limiting. |

---

## 4. Experimental Consistency Validation

* **Dataset Size:** Verified at **5,000 rows** (`research/data/credit_risk_dataset.csv`).
* **Test Split Size:** Verified at **1,500 rows** ($30\%$ held-out test split).
* **Random Seed:** Verified at **`random_state=42`** across all train/test partitioning and model initializations.
* **Random Forest Trees:** Verified at **`n_estimators=100`** in `evaluation_harness.py` and `full_metrics.py`.
* **Code Path Integrity:** Verified that no unexpected or modified code paths were executed; the scripts ran strictly in read-only mode.
* **Inconsistencies & Stale Output Report:**
  1. **Stale Benchmark Metrics:** Because the read-only scripts hardcode threshold `5.0`, the generated CSV files report $4.80\%$ escalation rate (72 cases) and Demographic Parity Difference of $0.0683$. These numbers contradict the locked threshold `6.0` calibration results ($3.73\%$ escalation rate, 56 cases, Demographic Parity Diff $0.0626$, Equalized Odds Diff $0.0361$).
  2. **Incomplete Live Validation:** The live spot check could not complete due to external API rate limits, leaving the live-vs-surrogate agreement rate unverified on the latest run.

---

## 5. Final Results Summary Table

Below is the publication-ready summary table formatted exactly as generated by `full_metrics.csv` (retaining exact rounding and string formatting from the read-only script execution):

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | Demographic Parity Diff | Equalized Odds Diff | Escalation Rate |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.868 | 0.8697 | 0.9656 | 0.9152 | 0.7859 | 0.1838 | 0.261 | 0.0% |
| **Random Forest** | 0.8713 | 0.8727 | 0.9665 | 0.9172 | 0.7921 | 0.1777 | 0.2397 | 0.0% |
| **XGBoost** | 0.8627 | 0.8707 | 0.9557 | 0.9112 | 0.7961 | 0.1796 | 0.2552 | 0.0% |
| **Agentic AI System v2** *(Generated @ Thresh 5.0)* | 0.7973 | 0.8024 | 0.962 | 0.875 | 0.7859 | 0.0683 | 0.0405 | 4.8% |
| *(Reference: Calibrated Thresh 6.0 Target)* | *0.7967* | *0.8009* | *0.9638* | *0.8748* | *0.7859* | *0.0626* | *0.0361* | *3.73%* |

---

## 6. Recommended Directory Structure for Archiving

To organize the final conference submission package cleanly without cluttering active source code, we recommend adopting the following hierarchical archive directory structure under `research/`:

```text
research/
├── final_results/
│   ├── csv/
│   │   ├── results.csv
│   │   └── full_metrics.csv
│   ├── figures/
│   │   ├── roc_curves.png
│   │   └── pareto_frontier.png
│   ├── reports/
│   │   ├── metrics_report.md
│   │   ├── full_metrics_report.md
│   │   └── final_comparative_analysis.md
│   ├── traces/
│   │   ├── case_1_react.json
│   │   ├── case_2_reflection.json
│   │   └── case_3_fairness.json
│   └── logs/
│       └── live_spot_check_execution.log
```

---

## 7. Final Readiness Assessment & Checklist

| Artifact / Evaluation Item | Status | Justification / Evidence |
| :--- | :---: | :--- |
| **`results.csv`** | ✅ **Successfully generated** | (⚠ *Warning:* Reflects threshold 5.0 instead of locked 6.0). |
| **`full_metrics.csv`** | ✅ **Successfully generated** | (⚠ *Warning:* Reflects threshold 5.0 instead of locked 6.0). |
| **`metrics_report.md`** | ✅ **Successfully generated** | (⚠ *Warning:* Reflects threshold 5.0 instead of locked 6.0). |
| **`full_metrics_report.md`** | ✅ **Successfully generated** | (⚠ *Warning:* Reflects threshold 5.0 instead of locked 6.0). |
| **`roc_curves.png`** | ✅ **Successfully generated** | High-resolution plot generated and verified. |
| **`pareto_frontier.png`** | ✅ **Successfully generated** | (⚠ *Warning:* Reflects operating point at threshold 5.0). |
| **Live Pipeline Validation** | ❌ **Missing / Failed** | Execution halted at sample 3/30 due to Groq API HTTP 429 rate limiting. |
| **Statistical Significance Tests** | ❌ **Missing** | No confidence interval or McNemar test scripts exist in codebase. |
| **Architectural Ablation Study** | ❌ **Missing** | No ablation script exists to isolate individual agent contributions. |

### Final Question Answer:
> **"Can these experimental artifacts now be considered the official results for the research paper?"**

**NO.** 
These artifacts cannot yet be considered the official, conference-ready results for two critical methodological reasons:

1. **Threshold Parameter Contradiction:** The official research paper implementation is locked at counterfactual threshold **`6.0`**, but the read-only evaluation scripts on disk still hardcode threshold **`5.0`**. Consequently, the generated CSV tables and markdown reports on disk state an escalation rate of $4.80\%$ (72 cases) and Demographic Parity Difference of $0.0683$, directly contradicting the calibrated threshold `6.0` claims ($3.73\%$ escalation rate, 56 cases, Demographic Parity Diff $0.0626$, Equalized Odds Diff $0.0361$).
2. **Incomplete Live vs. Surrogate Validation:** The live LangGraph verification check (`live_spot_check.py`) failed to complete due to external LLM API rate limiting (HTTP 429). Without a completed spot-check log, the paper lacks empirical proof that the deterministic evaluation harness matches the live multi-agent workflow on the final run.

**Requirements to Become Official:** Before final submission, the parameter literal `5` in `evaluation_harness.py`, `full_metrics.py`, and `live_spot_check.py` must be updated to `6.0`, API rate-limit handling (exponential backoff or batching) must be introduced to `live_spot_check.py`, and the entire suite must be re-executed.
