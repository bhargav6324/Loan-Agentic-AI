# Results preservation and authority

## Version 1 — diagnostic only

- Archive: `loan45k_full_agents_results.tar.gz`
- Archive SHA-256: `768e91414f1d5b80506592072db5421b0c6311b4c5daef4083b49235d8a2dcf6`
- Metrics SHA-256: `64b15691ebf1a2b35911f18cb91720acecf9018a2ef7b9655fd0ae4fd84bda1c`
- Status: preserved for audit and mismatch diagnosis; not valid for final claims.

## Version 2 — authoritative predictive results

- Archive: `loan45k_full_agents_results_v2.tar.gz`
- Archive SHA-256: `c7956ec578873dbd8475fe6c9dc0cf445db98062e6b184949edf85ec3f61846a`
- Metrics SHA-256: `0d74a0a4da7e82e2a42573174861672c3e1d0e60ec73223605cdc6994ef78643`
- Seeds: 42–51.
- Metric rows: 50.
- Prediction traces: 674,900.
- Status: authoritative for the 45K full-agent comparison.

Version 2 predates the log-deduplication correction. Its decisions and
predictive metrics remain valid; only `mean_agent_steps`, `mean_audit_entries`,
and associated runtime overhead contain duplication effects. A 200-case v3
precheck proved the deduplication change decision-neutral. No predictive rerun
is required for the paper's accuracy, policy, fairness, or escalation claims.
