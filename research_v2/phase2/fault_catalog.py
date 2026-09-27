"""Fault types used by the Phase 2 controlled experiments."""

from enum import Enum


class FaultType(str, Enum):
    MISSING_SALARY_SLIP = "missing_salary_slip"
    WRONG_EXTRACTED_INCOME = "wrong_extracted_income"
    CONTRADICTORY_INCOME = "contradictory_income"
    SKIPPED_POLICY_NODE = "skipped_policy_node"
    STALE_RISK_SCORE = "stale_risk_score"
    PREMATURE_COMMITMENT = "premature_commitment"
