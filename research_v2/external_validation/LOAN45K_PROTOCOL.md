# Loan Approval Classification Dataset (45K) protocol

## Purpose

This dataset is used for comparison with published loan-approval research. It is
not treated as independent real-bank validation.

## Provenance

- Source: `taweilo/loan-approval-classification-data` on Kaggle.
- Publisher description: synthetically enhanced from earlier credit-risk data.
- Expansion method reported by publisher: SMOTENC.
- Target: `loan_status`, where 1 means approved and 0 means rejected.
- Raw data must remain untracked because Kaggle metadata does not expose a clear
  dataset licence through the API.

## Frozen cleaning rules

Rows are excluded if age is outside 20–100, employment experience is negative,
exceeds 70 years, or exceeds age minus 14, income/loan amount is non-positive,
credit score is outside 300–850, or categorical/target values are invalid.

## Evaluation stages

1. E0: dataset-native Random Forest baseline.
2. E1: E0 plus a frozen research policy and gender counterfactual audit.
3. E1R: E1 plus deterministic risk reassessment.
4. E2: E1R plus state-bound evidence certificates and commitment closure.
5. E3: E2 plus a deterministic structural critic.

E-series names distinguish this feature contract from A0-A3. Aadhaar, PAN, DTI,
and credit utilisation are not manufactured; they remain outside this dataset's
evaluation scope.

The dataset-native baseline is not numerically identical to the existing A0
configuration because the feature schemas differ.
