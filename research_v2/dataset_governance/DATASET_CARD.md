# Dataset Card: Synthetic Loan Approval Dataset

*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*

## 1. Dataset Overview
- **Dataset Name:** Synthetic Loan Approval Dataset
- **Version:** 2.0
- **Purpose and Research Use:** This dataset is engineered specifically for evaluating agentic AI pipelines and counterfactual fairness auditing in loan approval systems. It allows for controlled, reproducible experiments to observe how multi-agent architectures resolve structural bias and policy exceptions.
- **Synthetic Nature:** Fully synthetic. No real human data was used or modeled.
- **Number of Records:** 5000

## 2. Dataset Schema & Attribute Definitions
### Eight Random Forest features
1. `credit_score` (Numeric): Synthetic credit score (300-850), intentionally biased downward for females and younger applicants.
2. `dti_ratio` (Numeric): Debt-to-Income ratio.
3. `loan_to_income` (Numeric): Loan amount divided by annual income.
4. `employment_score` (Numeric): Encoded employment stability (1 = self-employed, 2 = business, 3 = salaried).
5. `applicant_age` (Numeric): Age of the applicant (21-64).
6. `defaults` (Numeric): Number of prior loan defaults.
7. `credit_utilisation` (Numeric): Percentage of available credit currently used (0-100%).
8. `applicant_gender` (Categorical): 1 = Female, 0 = Male.

### Additional heuristic inputs
9. `employment_years` (Numeric): Years of active employment.
10. `loan_tenure_years` (Numeric): Requested loan duration in years.
11. `loan_purpose` (String): Categorical purpose (e.g., home, education).

### Additional Policy Agent inputs
12. `monthly_income` (Numeric): Monthly income in INR.
13. `loan_amount` (Numeric): Requested loan amount in INR.
14. `employment_type` (String): Original categorical employment string.
15. `current_outstanding` (Numeric): Current outstanding debt in INR.

## 3. Target Definition and Generation
- **Target Definition (`target_default`):** 1 = the simulated new loan defaults, 0 = the simulated new loan does not default. It is an outcome label, not an approval/rejection decision.
- **Default Prevalence:** 11.34% (567 defaults out of 5000 records).
- **Random Seed:** 42.
- **Generation Process:** The target is generated strictly once using a Bernoulli probability derived from financial indicators (LTI, Credit Score). There is **no additional label flipping**.

## 4. Intentional Demographic Bias
- **Female Bias:** Female applicants receive an intentional **90-point penalty** to their generated credit score.
- **Age Bias:** Applicants under 30 receive an intentional **70-point penalty** to their generated credit score.
- Because the biased credit score enters the default-probability equation, these penalties can also affect the simulated target. They create a controlled bias scenario; they do not guarantee that every fairness metric or every intervention will improve.

## 5. Usage Considerations
- **Intended Uses:** Algorithmic fairness research, agentic architecture validation, and ablation studies.
- **Prohibited Uses:** Training production lending models, generating real-world credit policies, or making financial decisions.
- **Ethical Considerations:** The dataset is deliberately biased and does not reflect real-world creditworthiness. It must not be used to justify, evaluate, or measure actual human behavior.
- **Known Limitations:** The data lacks temporal dynamics, macroeconomic shifts, and real-world noise distributions. 

## 6. Version History and Checksum
- **Version 1.0:** Initial procedural generation.
- **Version 2.0:** Hardened schema, strict 8-feature ML contract, exact 11.34% target prevalence.
- **Dataset Checksum (SHA-256):** Generated and verified by the `dataset_validation.py` harness.
