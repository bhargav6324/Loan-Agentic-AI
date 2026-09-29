# Formal cross-dataset comparison

| dataset                          | configuration   | role                        |   accuracy_mean |   f1_mean |   mcc_mean |   policy_violations_mean | automation_coverage_mean   | selective_accuracy_mean   |
|:---------------------------------|:----------------|:----------------------------|----------------:|----------:|-----------:|-------------------------:|:---------------------------|:--------------------------|
| Original 5K synthetic            | C0a             | approve-all control         |          0.7364 |    0.8482 |     0      |                    172.8 | —                          | —                         |
| Original 5K synthetic            | A0              | Random Forest               |          0.8723 |    0.9178 |     0.6528 |                     49   | —                          | —                         |
| Original 5K synthetic            | A1              | Random Forest + policy      |          0.8434 |    0.8969 |     0.5771 |                      0   | —                          | —                         |
| Original 5K synthetic            | A1R             | policy + reassessment       |          0.8419 |    0.8958 |     0.5738 |                      0   | —                          | —                         |
| Original 5K synthetic            | A2              | full graph                  |          0.8316 |    0.888  |     0.5514 |                      0   | —                          | —                         |
| Original 5K synthetic            | A3              | full graph + critic         |          0.8316 |    0.888  |     0.5514 |                      0   | —                          | —                         |
| Published 45K synthetic-enhanced | F0              | approve-all control         |          0.2223 |    0.3637 |     0      |                      4.8 | 1.0000                     | 0.2223                    |
| Published 45K synthetic-enhanced | F1              | Random Forest               |          0.9195 |    0.825  |     0.7736 |                      0.9 | 1.0000                     | 0.9195                    |
| Published 45K synthetic-enhanced | F1P             | policy + reassessment       |          0.9259 |    0.8278 |     0.7814 |                      0   | 1.0000                     | 0.9259                    |
| Published 45K synthetic-enhanced | F2              | full adapted graph          |          0.9257 |    0.8267 |     0.7805 |                      0   | 0.9392                     | 0.9355                    |
| Published 45K synthetic-enhanced | F3              | full adapted graph + critic |          0.9257 |    0.8267 |     0.7805 |                      0   | 0.9392                     | 0.9355                    |

## Interpretation constraints

- Compare governance trends, not raw accuracy: the datasets use different
  populations, labels, feature contracts, and class balances.
- The 45K source is a published, synthetically enhanced benchmark, not
  independent real-bank validation.
- F2/F3 escalations are unresolved outcomes. Their selective accuracy and
  automation coverage must accompany binary metrics.
- F3 and A3 do not improve predictive metrics over their graph counterparts;
  their purpose is structural critique.
- Runtime fields use different measurement units and are not directly compared.
