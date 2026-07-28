# Reproducibility Package: LangGraph V7 Multi-Agent Underwriting Pipeline

This document provides complete instructions, parameter specifications, and execution commands required by independent researchers and peer reviewers to reproduce 100% of the experimental results, ablation matrices, case studies, and publication figures reported in the manuscript.

---

## 1. Experimental Specifications & Locked Parameters

| Parameter | Locked Value / Specification | Description |
| :--- | :--- | :--- |
| **Python Version** | `Python 3.10+` (Tested on `3.10.12` / `3.12`) | Core execution runtime. |
| **Random Seed** | `42` | Strictly enforced across all train/test splits, model initialization, and agent simulations to guarantee bit-for-bit deterministic reproducibility. |
| **Counterfactual Fairness Threshold ($\epsilon$)** | `6.0` | Locked decision boundary for counterfactual score disparity ($\lvert \Delta_{\text{CF}} \rvert$). Deltas exceeding $6.0$ trigger mandatory human review escalation. |
| **Dataset** | `credit_risk_dataset.csv` ($N=5,000$) | Balanced credit underwriting dataset containing demographic, financial, and credit history attributes. |
| **Evaluation Split** | Held-out Test Split ($30\%$, $N=1,500$) | Evaluated using `train_test_split(..., test_size=0.3, random_state=42)`. |
| **ML Baseline Architecture** | `RandomForestClassifier(n_estimators=100, random_state=42)` | Unconstrained machine learning baseline model trained on the $70\%$ training split ($N=3,500$). |
| **Multi-Agent Orchestrator** | `LangGraph v7` State Graph | Directed Acyclic Graph (DAG) orchestrating Intake, Planner, Risk, Policy, and Fairness agents with parallel fan-out and join nodes. |

---

## 2. Software Dependencies & Environment Setup

### Required Python Packages
```bash
pip install numpy pandas scikit-learn fairlearn matplotlib langgraph langchain pydantic graphviz
```

### Verified Package Versions
* `numpy >= 1.24.0`
* `pandas >= 2.0.0`
* `scikit-learn >= 1.3.0`
* `fairlearn >= 0.9.0`
* `matplotlib >= 3.7.0`
* `langgraph >= 0.0.30`
* `pydantic >= 2.0`

---

## 3. Repository Structure & Asset Catalog

```text
Loan-Agentic-AI/
├── agents/                       # Read-Only Agent Domain Logic
│   ├── intake_agent.py           # KYC & Fraud verification
│   ├── fairness_agent.py         # Counterfactual bias audit
│   └── orchestrator_agent.py     # Central supervisory router
├── orchestrator/                 # Graph Compilation & Routing
│   └── graph.py                  # LangGraph DAG builder (parameterized for ablation)
├── research/                     # Evaluation & Research Suite
│   ├── data/
│   │   └── credit_risk_dataset.csv # Primary credit underwriting dataset
│   ├── evaluation_harness.py     # Core heuristic & pre-screening domain engine
│   ├── full_metrics.py           # Comprehensive 12-metric evaluation script
│   ├── run_ablation_study.py     # Controlled 4-configuration ablation harness
│   ├── generate_publication_assets.py   # Case study & LaTeX/Markdown table generator
│   ├── generate_publication_figures.py  # 300 DPI PNG & vector SVG figure generator
│   ├── generate_architecture_diagrams.py # Architectural flowchart drawing suite
│   ├── results.csv               # Primary system performance benchmark export
│   └── publication_assets/       # Generated Conference & Journal Assets
│       ├── tables/               # Formatted Table 1 to Table 4 (.md and .tex)
│       └── figures/              # 300 DPI PNG and vector SVG diagrams
└── README.md                     # Project documentation
```

---

## 4. Step-by-Step Execution Order

To regenerate all tables, metrics, case studies, and publication figures from scratch, execute the following terminal commands from the root repository directory (`Loan-Agentic-AI/`) in exact sequential order:

### Step 1: Execute Full Metrics & System Benchmark
Computes overall system performance against the raw Random Forest baseline ($N=1,500$):
```bash
python3 research/full_metrics.py
```
* **Output:** Generates/updates `research/results.csv` and prints comprehensive performance metrics.

### Step 2: Execute Controlled Ablation Study
Runs the 4 architectural configurations (Full System, No-Fairness, No-Policy, RF Baseline) under identical seed and split conditions:
```bash
python3 research/run_ablation_study.py
```
* **Output:** Exports `research/final_results/ablation_results.csv`, `ablation_summary.md`, and `ablation_comparison.png`.

### Step 3: Generate Publication Tables & Extract Case Studies
Automatically identifies representative fairness escalations and generates clean LaTeX (`.tex`) and Markdown (`.md`) publication tables:
```bash
python3 research/generate_publication_assets.py
```
* **Output:** Generates Tables 1 through 4 in `research/publication_assets/tables/`:
  * `table1_overall_performance.md` & `.tex`
  * `table2_ablation_study.md` & `.tex`
  * `table3_rf_vs_agentic.md` & `.tex`
  * `table4_case_studies.md`, `.tex`, & `.csv`

### Step 4: Generate 300 DPI Publication Figures
Produces all statistical performance charts, trade-off curves, and distribution plots in high-resolution raster (`.png`) and vector (`.svg`) formats:
```bash
python3 research/generate_publication_figures.py
```
* **Output:** Generates 5 figures in `research/publication_assets/figures/`:
  * `fig1_roc_curve.png` & `.svg`
  * `fig2_pareto_frontier.png` & `.svg`
  * `fig3_ablation_comparison.png` & `.svg`
  * `fig4_fairness_distribution.png` & `.svg`
  * `fig5_policy_violations.png` & `.svg`

### Step 5: Generate Architecture Flowcharts
Draws Springer/IEEE publication-ready architectural flowcharts:
```bash
python3 research/generate_architecture_diagrams.py
```
* **Output:** Generates 3 flowcharts in `research/publication_assets/figures/`:
  * `arch1_langgraph_topology.png` & `.svg`
  * `arch2_decision_workflow.png` & `.svg`
  * `arch3_fairness_workflow.png` & `.svg`

---

## 5. Verification of Bit-for-Bit Determinism

All scripts utilize `random_state=42` across pandas, scikit-learn, and numpy operations. When running on identical operating system architectures and Python 3.10+ environments, the generated numerical tables and classification metrics are guaranteed to converge to the exact decimal precision reported in the publication package.
