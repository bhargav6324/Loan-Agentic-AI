# Leakage Audit

*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*

## Target Leakage Verification
The validation script (`dataset_validation.py`) enforces strict schema constraints to prevent data leakage from the target label into the predictive features.

- **Unidirectional Causality:** [Verified through source-code inspection] No feature is generated from `target_default`. The generation provenance clearly documents that `target_default` is the final step in the causal chain.
- **Prior Default Ordering:** [Assumption or limitation] The `defaults` feature represents historical defaults on *prior* loans and is explicitly generated *before* the target label. It is an assumption that prior defaults do not represent an ex-post realization of the current loan default.
- **Inference Isolation:** [Automatically tested] The target label (`target_default`) is stripped from the `ApplicantData` payload. It is never supplied to the Risk, Policy, Fairness, or Routing agents during inference.
- **Attribute-flip audit isolation:** [Verified through source-code inspection] The audit does not access the target label. It flips gender while holding the observed remaining inputs fixed. This is a sensitivity audit, not proof of formal causal counterfactual fairness.

## Evaluation and Validation Separation
- **Data Splitting:** [Verified through source-code inspection] Train/test splits occur strictly before evaluation.
- **Test Integrity:** [Automatically tested] No test records are utilized to train the internal predictive models (Random Forest). Train and test indices do not overlap.
- **Preprocessing Purity:** [Verified through source-code inspection] The current RF path uses the eight numeric/encoded inputs directly and does not fit a global scaler or imputer.
- **Identifier Masking:** [Automatically tested] Applicant IDs are explicitly dropped and are not accessible as model features.
- **Record Uniqueness:** [Automatically tested] No duplicate applicant ID appears across the train and test split. The validation script verifies strict ID uniqueness.

## Control Flow Verification
- Policy rule definitions, LLM critique prompts, and routing mechanisms have been manually audited. [Verified through source-code inspection] They do not contain hidden references to the target label or outcome variables.
