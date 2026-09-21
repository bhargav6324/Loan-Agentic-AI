# Generation Provenance

*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*

## Causal Generation Order

The generator follows a one-way simulation order: later variables do not feed back into earlier variables. This prevents target leakage in dataset construction. It does not, by itself, make the implemented gender-flip audit a formal causal-counterfactual analysis.

1. **Demographics:** `applicant_age` (integer 21-64, uniform), `applicant_gender` (0=Male, 1=Female, Bernoulli).
2. **Employment / Income:** `employment_type`, `employment_years`, `monthly_income` (Derived from age).
3. **Loan Request:** `loan_amount`, `loan_tenure_years`, `loan_purpose`.
4. **Debt / Utilisation:** `current_outstanding`, `dti_ratio`, `loan_to_income`, `credit_utilisation`.
5. **Latent Distress:** Hidden true risk variables derived from LTI and income distributions.
6. **Previous Defaults:** `defaults` (Count of prior non-payments).
7. **Biased Credit Score:** `credit_score` (Base score offset by demographic penalties).
8. **Default Probability:** Derived directly from the biased credit score and debt ratios.
9. **Target Generation:** `target_default` (Bernoulli sampling from the default probability).

## Exact Constraints and Specifications
- **Random Seed:** 42
- **Female Credit-Score Penalty:** Exactly **90 points** deducted.
- **Age-Under-30 Penalty:** Exactly **70 points** deducted.
- **Target Definition (`target_default`):** 1 = default, 0 = non-default.
- **Final Defaults Generated:** Exactly 567 defaults out of 5000 records.
- **Final Default Prevalence:** Exactly 11.34%.
- **No Label Flipping:** The target is generated strictly once using a Bernoulli probability derived from financial indicators. No additional label flipping or synthetic noise is added post-generation. 

These constraints are checked by the dataset-governance validation script. CI enforcement depends on that script being included in the repository's actual CI configuration.
