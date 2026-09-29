# 45K full-agent experiment

This experiment is isolated from all synthetic-data results. It reuses the
original planner, ambiguity-gated orchestrator, commitment node, and reporting
agent. Dataset-incompatible intake, risk, policy, and fairness agents are
adapted to the genuine 45K field contract.

PAN, Aadhaar, salary slips, DTI, and credit utilisation are marked as not
evaluated. They are never fabricated. Groq, Qwen, and all external API calls are
forcibly disabled during research execution.

Configurations:

- F0: approve-all control.
- F1: 45K-native Random Forest.
- F1P: F1 plus dataset-compatible policy.
- F2: full adapted agent graph.
- F3: F2 plus structural critic.

F2/F3 are full-agent *adapters*, not the unchanged original graph, because the
source feature contracts differ.

Version-1 server results are diagnostic only. See `MISMATCH_DIAGNOSIS_V1.md`.
The corrected runner preserves the direction of approval probability and
reports escalations through automation coverage/selective accuracy.
