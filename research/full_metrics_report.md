# Comprehensive Metrics Report — ML vs Agentic System

## 1. Performance Metrics

| Model                         |   Accuracy |   Precision |   Recall (Sensitivity) |   F1 Score |   AUC-ROC |   Specificity |    MCC |
|:------------------------------|-----------:|------------:|-----------------------:|-----------:|----------:|--------------:|-------:|
| Logistic Regression           |     0.868  |      0.8697 |                 0.9656 |     0.9152 |    0.7859 |        0.5939 | 0.6391 |
| Random Forest                 |     0.8713 |      0.8727 |                 0.9665 |     0.9172 |    0.7921 |        0.6041 | 0.649  |
| XGBoost                       |     0.8627 |      0.8707 |                 0.9557 |     0.9112 |    0.7961 |        0.6015 | 0.6243 |
| Agentic System v2 (LangGraph) |     0.7973 |      0.8024 |                 0.962  |     0.875  |    0.7859 |        0.335  | 0.4082 |

## 2. Confusion Matrix

| Model                         |   TP |   TN |   FP |   FN |
|:------------------------------|-----:|-----:|-----:|-----:|
| Logistic Regression           | 1068 |  234 |  160 |   38 |
| Random Forest                 | 1069 |  238 |  156 |   37 |
| XGBoost                       | 1057 |  237 |  157 |   49 |
| Agentic System v2 (LangGraph) | 1064 |  132 |  262 |   42 |

## 3. Fairness Metrics

| Model                         |   Male Approval Rate |   Female Approval Rate |   Approval Gap (M-F) |   Demographic Parity Diff |   Equalized Odds Diff |
|:------------------------------|---------------------:|-----------------------:|---------------------:|--------------------------:|----------------------:|
| Logistic Regression           |               0.8911 |                 0.7073 |               0.1838 |                    0.1838 |                0.261  |
| Random Forest                 |               0.8867 |                 0.709  |               0.1777 |                    0.1777 |                0.2397 |
| XGBoost                       |               0.8801 |                 0.7005 |               0.1796 |                    0.1796 |                0.2552 |
| Agentic System v2 (LangGraph) |               0.9109 |                 0.8426 |               0.0683 |                    0.0683 |                0.0405 |

## 4. Agentic Pipeline Telemetry

| Metric | Value |
|---|---|
| Total Test Samples | 1500 |
| Hard Rejects | 7 (0.5%) |
| Hard Fast-Tracks | 83 (5.5%) |
| Fairness Escalations | 72 (4.8%) |


## 5. Trade-off Summary

| Metric | Change |
|---|---|
| Demographic Parity | 61.6% reduction in bias |
| Equalized Odds | 83.1% reduction in bias |
| Approval Gap (M-F) | 61.6% reduction in bias |
| **Accuracy Cost** | 7.4% drop |
| **Specificity Cost** | 0.269 drop (0.60 → 0.34) |
| **MCC Cost** | 0.241 drop (0.65 → 0.41) |
