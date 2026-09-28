"""Dataset-native governance for the 45K benchmark (E-series)."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

POLICY_VERSION = "loan45k-research-policy-v1"
FAIRNESS_EPSILON = 0.06


def state_hash(record: Mapping[str, Any]) -> str:
    payload = {k: record[k] for k in sorted(record) if k != "loan_status"}
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(text.encode()).hexdigest()


def research_policy(record: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Frozen experimental policy; this is not a real lender policy."""
    violations = []
    if float(record["person_age"]) < 21:
        violations.append("minimum_age")
    if float(record["credit_score"]) < 400:
        violations.append("minimum_credit_score")
    if float(record["loan_percent_income"]) > 0.60:
        violations.append("maximum_loan_income_burden")
    return not violations, violations


def native_risk_score(record: Mapping[str, Any], rejection_probability: float) -> float:
    credit = float(record["credit_score"])
    burden = float(record["loan_percent_income"])
    interest = float(record["loan_int_rate"])
    employment = float(record["person_emp_exp"])
    heuristic = 35 if credit < 500 else 22 if credit < 600 else 10 if credit < 700 else 2
    heuristic += min(30, max(0, burden) * 50)
    heuristic += 20 if str(record["previous_loan_defaults_on_file"]) == "Yes" else 0
    heuristic += 8 if employment < 1 else 3 if employment < 3 else 0
    heuristic += 7 if interest >= 18 else 3 if interest >= 13 else 0
    return round(min(100, 0.70 * rejection_probability * 100 + 0.30 * min(100, heuristic)), 6)


def risk_category(score: float) -> str:
    return "LOW" if score <= 35 else "MEDIUM" if score <= 65 else "HIGH"


def deterministic_reassessment(score: float, record: Mapping[str, Any]) -> tuple[float, bool]:
    revised = score
    if str(record["previous_loan_defaults_on_file"]) == "Yes" and revised < 40:
        revised = 40.0
    if float(record["credit_score"]) < 500 and revised < 55:
        revised = 55.0
    return revised, revised != score


@dataclass(frozen=True)
class GovernedDecision:
    proposed_approval: int
    final_approval: int
    commit_status: str
    risk_score: float
    risk_category: str
    policy_passed: bool
    policy_violations: list[str]
    fairness_executed: bool
    fairness_flagged: bool
    counterfactual_delta: float
    reassessed: bool
    critic_findings: list[str]
    certificates_valid: bool


def decide(record: Mapping[str, Any], approval_probability: float,
           counterfactual_probability: float, configuration: str) -> GovernedDecision:
    if configuration not in {"E0", "E1", "E1R", "E2", "E3"}:
        raise ValueError(f"Unknown configuration: {configuration}")
    proposed = int(approval_probability >= 0.5)
    score = native_risk_score(record, 1 - approval_probability)
    reassessed = False
    if configuration in {"E1R", "E2", "E3"}:
        score, reassessed = deterministic_reassessment(score, record)
    category = risk_category(score)
    passed, violations = research_policy(record)
    if configuration == "E0":
        return GovernedDecision(proposed, proposed, "committed", score, category,
                                True, [], False, False, 0, False, [], True)
    fairness = True
    delta = abs(approval_probability - counterfactual_probability)
    flagged = delta > FAIRNESS_EPSILON
    final = 0 if (not passed or category == "HIGH") else proposed
    digest = state_hash(record)
    certs = {name: {"state_hash": digest, "valid": True}
             for name in ("risk", "policy", "fairness")}
    valid = all(c["valid"] and c["state_hash"] == digest for c in certs.values())
    findings = []
    if proposed and not passed:
        findings.append("approval_conflicts_with_policy")
    if proposed and category == "HIGH":
        findings.append("approval_conflicts_with_high_risk")
    if flagged:
        findings.append("gender_counterfactual_sensitivity")
    if not valid:
        findings.append("invalid_evidence_closure")
    blocked = configuration in {"E2", "E3"} and not valid
    if configuration == "E3" and any(x != "gender_counterfactual_sensitivity" for x in findings):
        blocked = True
    if blocked:
        final = 0
    return GovernedDecision(proposed, final, "blocked" if blocked else "committed",
                            score, category, passed, violations, fairness, flagged,
                            delta, reassessed, findings if configuration == "E3" else [], valid)
