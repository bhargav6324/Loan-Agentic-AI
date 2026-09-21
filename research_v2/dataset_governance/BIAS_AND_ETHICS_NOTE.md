# Bias and Ethics Note

*Aligned with selected good-practice principles from the NIST AI RMF, ISO/IEC 5259, and Datasheets for Datasets.*

## Intentional Bias Insertion
This dataset contains intentional, explicit demographic bias. It was artificially injected to simulate a historically biased legacy credit scoring system. The purpose of this dataset is strictly to evaluate whether downstream agentic AI pipelines can detect, critique, and correct structural unfairness.

- **Gender Penalty:** Female applicants receive a 90-point penalty to their generated credit score.
- **Age Penalty:** Applicants under 30 receive a 70-point penalty to their generated credit score.

## Impact on Ground Truth
The biased credit score is not an isolated feature; it directly impacts the underlying target probability generation. Because the historical labels (`target_default`) were derived heavily from this biased legacy score, the target itself is inherently biased against females and young applicants. 
**The target label is not completely unbiased ground truth.** It represents historical human/system decisions that inherited structural bias.

## Limitations of Audit Mechanisms
The auditing mechanisms tested on this dataset (e.g., the Fairness Agent's counterfactual delta checks) evaluate **direct gender sensitivity only**. 
- They perturb the `applicant_gender` feature and observe the output shift.
- This is an observational counterfactual evaluation, not formal causal counterfactual fairness, because we do not compute full structural equations or account for unobserved confounding variables that might correlate with gender in reality.

## Strict Prohibition on Real-World Generalization
Because this dataset is purely synthetic and intentionally unfair:
- The results and model behaviors observed here **cannot be generalized to real lending environments**.
- This dataset **must not** be used for actual lending decisions, to train production financial models, or to derive real-world policy guidelines.
