# Table 1: Overall System Performance vs. Baseline ($N=1,500$ Test Set)

| Configuration               |   Accuracy |     F1 |   ROC-AUC |   Demographic Parity Diff |   Equalized Odds Diff |   Policy Violations | Escalation Rate   |
|:----------------------------|-----------:|-------:|----------:|--------------------------:|----------------------:|--------------------:|:------------------|
| LangGraph Agentic System v7 |     0.7967 | 0.8748 |    0.7859 |                    0.0626 |                0.0361 |                  34 | 3.73%             |
| Unconstrained RF Baseline   |     0.8707 | 0.9167 |    0.7921 |                    0.1794 |                0.2397 |                  46 | 0.00%             |