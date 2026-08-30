# Parameter Provenance for Synthetic Data Generation

| Parameter | Value | Source | One-line Rationale |
| :--- | :--- | :--- | :--- |
| **Target Default Prevalence** | 15.0% | LendingClub Historical Charge-off Data (2007-2018) | Broadly representative of consumer unsecured/subprime retail lending pools. |
| **Marginal Effect of DTI Ratio** | 1.82 | Federal Reserve Board (2018) Report on DTI | A 10 percentage point increase in DTI increases default odds by ~20% (coeff = ln(1.2)/0.10 = 1.82). |
| **Marginal Effect of Loan-to-Income** | 0.15 | CFPB Mortgage & Consumer Risk Studies | Each 1.0 unit increase in LTI multiplier yields approximately 16% higher odds of default (coeff = 0.15). |
| **Marginal Effect of Credit Score** | -5.50 (over 550 pts) | Standard FICO Odds-to-Score Relationship | Traditionally, +20 to +40 points doubles the odds of non-default; this maps to roughly -0.01 per point, or -5.50 over a 550-point range. |
| **Marginal Effect of Prior Defaults** | 1.38 (first), 0.69 (second) | TransUnion/Experian Transition Matrices | A prior default typically multiplies future default odds by 3x-4x (ln(4) = 1.38); subsequent defaults add compounding but diminishing risk. |
| **Label Noise Rate** | 10.0% | Hand & Henley (1997) Consumer Credit Scoring | Unobserved personal/macroeconomic shocks account for ~10-15% of outcome variance independent of underwritten features. |
