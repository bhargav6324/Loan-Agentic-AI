"""
Governance Invariant Tests
══════════════════════════
Tests that enforce the fairness governance invariant across both A1R
(canonical_final_decision) and A2 (graph decision_commit):

  1. Fairness cannot approve a deterministic rejection.
  2. Fairness cannot override a policy rejection.
  3. Policy rejection does not require a fairness audit.
  4. MEDIUM-risk policy-passing cases receive required fairness review.
  5. Missing/stale certificates block commitment.
  6. Valid low-risk cases receive required certificates.
  7. Disagreement classification is mutually exclusive and exhaustive.
"""
import os, sys, pytest
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from core.canonical_contracts import canonical_final_decision


# ─────────────────────────────────────────────────────────────────────────────
# 1. Fairness cannot approve a deterministic rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestFairnessCannotApproveRejection:
    """
    When audit_flagged=True and audit_recommended_action='approve',
    and the decision_proposal is 'reject' (or 'escalate'),
    canonical_final_decision must NOT commit_approve.
    It must escalate for human review.
    """

    def test_fairness_approve_on_reject_proposal_becomes_escalated(self):
        status, reasoning = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=True,
            audit_flagged=True,
            decision_proposal="reject",
            audit_recommended_action="approve",
        )
        assert status != "committed_approve", (
            "Fairness must NOT convert a deterministic rejection into an approval. "
            f"Got status={status}"
        )
        assert status == "automated_escalated_status", (
            f"Expected escalation for human review, got {status}"
        )

    def test_fairness_approve_on_approve_proposal_is_allowed(self):
        """When the risk proposal is also approve, fairness confirmation is fine."""
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=True,
            audit_flagged=True,
            decision_proposal="approve",
            audit_recommended_action="approve",
        )
        assert status == "committed_approve", (
            "When proposal is approve AND fairness confirm approve, should commit_approve"
        )

    def test_fairness_reject_on_approve_proposal_overrides_to_reject(self):
        """Conservative: fairness recommends reject → commit_reject regardless of proposal."""
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=True,
            audit_flagged=True,
            decision_proposal="approve",
            audit_recommended_action="reject",
        )
        assert status == "committed_reject"

    def test_fairness_escalate_proposal_stays_escalated(self):
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=True,
            audit_flagged=True,
            decision_proposal="escalate",
            audit_recommended_action="approve",
        )
        assert status == "automated_escalated_status"


# ─────────────────────────────────────────────────────────────────────────────
# 2. Fairness cannot override a policy rejection
# ─────────────────────────────────────────────────────────────────────────────

class TestFairnessCannotOverridePolicyRejection:
    """
    When policy_passed=False, the final decision must be committed_reject
    (or blocked_policy_violation) regardless of any fairness audit result.
    """

    def test_policy_rejection_with_fairness_approve_is_still_reject(self):
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=False,
            policy_violations=["age < 21"],
            audit_required=True,
            audit_executed=True,
            audit_flagged=True,
            decision_proposal="reject",
            audit_recommended_action="approve",
        )
        assert status == "committed_reject", (
            f"Policy rejection must not be overridden by fairness. Got {status}"
        )

    def test_policy_rejection_with_no_audit_is_still_reject(self):
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=False,
            policy_violations=["age < 21"],
            audit_required=True,
            audit_executed=False,
            audit_flagged=False,
            decision_proposal="reject",
            audit_recommended_action=None,
        )
        # Should NOT be blocked_missing_audit — policy rejection is terminal
        assert status == "committed_reject", (
            f"Policy rejection does not require audit. Got {status}"
        )
        assert status != "blocked_missing_audit"

    def test_policy_rejection_does_not_need_audit(self):
        """Explicit test: policy-failed MEDIUM case with no audit must be committed_reject."""
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=False,
            policy_violations=["defaults > 2"],
            audit_required=True,
            audit_executed=False,
            audit_flagged=False,
            decision_proposal="reject",
            audit_recommended_action=None,
        )
        assert status == "committed_reject"
        assert status != "blocked_missing_audit"


# ─────────────────────────────────────────────────────────────────────────────
# 3. Policy rejection does not require a fairness audit (gate ordering)
# ─────────────────────────────────────────────────────────────────────────────

class TestPolicyRejectionNoAuditRequired:
    """
    Validates the gate ordering in canonical_final_decision:
    policy check must come before the missing-audit gate.
    """

    def test_high_risk_policy_fail_no_audit_is_committed_reject(self):
        status, _ = canonical_final_decision(
            risk_category="HIGH",
            policy_passed=False,
            policy_violations=["LTI > 5x"],
            audit_required=False,
            audit_executed=False,
            audit_flagged=False,
            decision_proposal="reject",
        )
        assert status == "committed_reject"

    def test_medium_risk_policy_fail_no_audit_is_committed_reject(self):
        """Core regression: MEDIUM risk + policy fail + no audit → committed_reject (not blocked_missing_audit)."""
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=False,
            policy_violations=["age < 21"],
            audit_required=True,   # audit would normally be required for MEDIUM
            audit_executed=False,  # but it was not run
            audit_flagged=False,
            decision_proposal="reject",
        )
        assert status == "committed_reject", (
            f"MEDIUM policy-rejected case must be committed_reject (not blocked_missing_audit). Got {status}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# 4. MEDIUM-risk policy-passing cases require fairness review
# ─────────────────────────────────────────────────────────────────────────────

class TestMediumRiskFairnessRequired:
    """
    When policy passed and risk is MEDIUM, the audit is required.
    If it was not executed, the result must be blocked_missing_audit.
    """

    def test_medium_risk_no_audit_is_blocked(self):
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=False,
            audit_flagged=False,
            decision_proposal="approve",
        )
        assert status == "blocked_missing_audit"

    def test_medium_risk_with_audit_is_not_blocked(self):
        status, _ = canonical_final_decision(
            risk_category="MEDIUM",
            policy_passed=True,
            policy_violations=[],
            audit_required=True,
            audit_executed=True,
            audit_flagged=False,
            decision_proposal="approve",
        )
        assert status == "committed_approve"


# ─────────────────────────────────────────────────────────────────────────────
# 5. Missing/stale certificates block commitment (graph decision_commit)
# ─────────────────────────────────────────────────────────────────────────────

class TestCertificateGate:
    """
    Tests the decision_commit node in graph.py.
    Missing risk or policy certificate must block commitment.
    """

    def _make_state(self, risk_cert=True, policy_cert=True, policy_passed=True,
                    risk_category="LOW", proposal="approve", fairness_report=None):
        return {
            "risk_certificate": {"timestamp": 1.0, "risk_score": 20.0} if risk_cert else None,
            "policy_certificate": {"timestamp": 1.0, "passed": policy_passed, "violations_count": 0} if policy_cert else None,
            "policy_passed": policy_passed,
            "policy_violations": [],
            "risk_category": risk_category,
            "decision_proposal": proposal,
            "fairness_report": fairness_report,
            "agent_logs": [],
        }

    def test_missing_risk_certificate_blocks(self):
        from orchestrator.graph import decision_commit
        state = self._make_state(risk_cert=False)
        result = decision_commit(state)
        assert result["commit_status"] == "blocked_missing_certificate"
        assert result["final_decision"] == "escalated"
        assert result["final_decision"] == "escalated"

    def test_missing_policy_certificate_without_state_blocks(self):
        """When policy_passed is also None, must block — genuinely unknown."""
        from orchestrator.graph import decision_commit
        state = self._make_state(policy_cert=False)
        # Remove policy_passed so the recovery path doesn't trigger
        state["policy_passed"] = None
        result = decision_commit(state)
        assert result["commit_status"] == "blocked_missing_certificate"

    def test_missing_policy_cert_cannot_be_reconstructed_from_state(self):
        """Ordinary state fields are not a substitute for a producer certificate."""
        from orchestrator.graph import decision_commit
        state = self._make_state(policy_cert=False, policy_passed=True)
        result = decision_commit(state)
        assert result["commit_status"] == "blocked_missing_certificate"

    def test_both_certs_present_low_risk_approves(self):
        from orchestrator.graph import decision_commit
        state = self._make_state()
        result = decision_commit(state)
        assert result["commit_status"] == "committed_approve"
        assert result["final_decision"] == "approved"


# ─────────────────────────────────────────────────────────────────────────────
# 6. Valid low-risk cases receive required certificates
# ─────────────────────────────────────────────────────────────────────────────

class TestLowRiskCertificates:
    """
    LOW risk cases should not be blocked_missing_certificate when the graph
    has executed correctly and policy_passed is set in state.
    """

    def test_low_risk_without_policy_certificate_blocks(self):
        """The gate must not manufacture a certificate to hide a graph race."""
        from orchestrator.graph import decision_commit
        state = {
            "risk_certificate": {"timestamp": 1.0, "risk_score": 25.0},
            "policy_certificate": None,   # race: cert not yet in state
            "policy_passed": True,        # but policy_agent set this field
            "policy_violations": [],
            "risk_category": "LOW",
            "decision_proposal": "approve",
            "fairness_report": None,
            "agent_logs": [],
        }
        result = decision_commit(state)
        assert result["commit_status"] == "blocked_missing_certificate"


class TestRiskOverrideCertificates:
    """Every successful risk-agent exit must issue a real risk certificate."""

    @staticmethod
    def _state(features):
        annual_income = 600000.0
        return {
            "applicant_data": {
                "monthly_income": annual_income / 12,
                "loan_amount": features.get("loan_to_income", 1.0) * annual_income,
                "applicant_age": 35,
                "applicant_gender": 0,
                "employment_type": "salaried",
                "employment_score": 3,
                "employment_years": features.get("employment_years", 8),
                "loan_tenure_years": 10,
                "loan_purpose": "personal",
                "dti_ratio": features["dti_ratio"],
                "loan_to_income": features.get("loan_to_income", 1.0),
            },
            "credit_report": {
                "credit_score": features["credit_score"],
                "current_outstanding": features["dti_ratio"] * annual_income,
                "total_defaults": features["defaults"],
                "credit_utilisation_pct": 20,
            },
            "agent_logs": [],
            "errors": [],
        }

    def test_hard_reject_issues_certificate(self):
        from agents.risk_agent import risk_agent
        result = risk_agent(self._state({
            "credit_score": 430, "defaults": 0, "dti_ratio": 0.2,
        }))
        cert = result["risk_certificate"]
        assert cert["risk_score"] == 95.0
        assert cert["risk_category"] == "HIGH"
        assert cert["decision_path"] == "hard_reject"

    def test_hard_fast_track_issues_certificate(self):
        from agents.risk_agent import risk_agent
        result = risk_agent(self._state({
            "credit_score": 830, "defaults": 0, "dti_ratio": 0.1,
            "employment_years": 8,
        }))
        cert = result["risk_certificate"]
        assert cert["risk_score"] == 10.0
        assert cert["risk_category"] == "LOW"
        assert cert["decision_path"] == "hard_fast_track"

    def test_low_risk_policy_fail_without_cert_still_blocks(self):
        from orchestrator.graph import decision_commit
        state = {
            "risk_certificate": {"timestamp": 1.0, "risk_score": 25.0},
            "policy_certificate": None,
            "policy_passed": False,
            "policy_violations": ["age < 21"],
            "risk_category": "LOW",
            "decision_proposal": "reject",
            "fairness_report": None,
            "agent_logs": [],
        }
        result = decision_commit(state)
        assert result["commit_status"] == "blocked_missing_certificate"


# ─────────────────────────────────────────────────────────────────────────────
# 7. Disagreement classification is mutually exclusive and exhaustive
# ─────────────────────────────────────────────────────────────────────────────

class TestDisagreementClassification:
    """
    Validates the _classify_disagreement function from the analysis script.
    Each sub-scenario should produce exactly one cause label.
    """

    @pytest.fixture(autouse=True)
    def _import_classify(self):
        import importlib.util
        script = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'scripts',
                         'run_a1r_a2_disagreement_analysis.py')
        )
        spec = importlib.util.spec_from_file_location("da", script)
        mod  = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self._classify = mod._classify_disagreement

    KNOWN_CAUSES = {
        "fairness", "routing", "policy",
        "certificate_commitment_gate", "reassessment", "unexplained",
    }

    def _r(self, **kw):
        """Build a minimal result dict."""
        return {
            "commit_status": kw.get("commit_status", "committed_approve"),
            "audit_bias_flagged": kw.get("audit_bias_flagged", False),
            "fairness_bias_flagged": kw.get("fairness_bias_flagged", False),
            "fairness_recommendation": kw.get("fairness_recommendation", ""),
            "audit_recommended_action": kw.get("audit_recommended_action", ""),
            "deterministic_reassessment_count": kw.get("reassessment_count", 0),
            "composite_score": kw.get("composite_score", 50.0),
            "policy_passed": kw.get("policy_passed", True),
            "risk_category": kw.get("risk_category", "MEDIUM"),
            "final_decision": kw.get("final_decision", "approved"),
        }

    def test_cert_gate_classified_correctly(self):
        r2 = self._r(commit_status="blocked_missing_certificate", final_decision="escalated")
        r1 = self._r(commit_status="committed_approve", final_decision="approved")
        cause = self._classify(r1, r2)
        assert cause == "certificate_commitment_gate"

    def test_blocked_missing_audit_is_cert_gate(self):
        r2 = self._r(commit_status="blocked_missing_audit", final_decision="escalated")
        r1 = self._r(commit_status="committed_reject", final_decision="rejected")
        cause = self._classify(r1, r2)
        assert cause == "certificate_commitment_gate"

    def test_fairness_bias_flag_mismatch(self):
        r1 = self._r(audit_bias_flagged=True, final_decision="approved")
        r2 = self._r(fairness_bias_flagged=False, final_decision="rejected")
        cause = self._classify(r1, r2)
        assert cause == "fairness"

    def test_fairness_override_recommendation_classified_as_fairness(self):
        """A2 fairness_bias_flagged=True + recommendation=override → fairness cause."""
        r1 = self._r(final_decision="approved")
        r2 = self._r(fairness_bias_flagged=True, fairness_recommendation="override",
                     final_decision="escalated")
        cause = self._classify(r1, r2)
        assert cause == "fairness"

    def test_policy_mismatch(self):
        r1 = self._r(policy_passed=True, final_decision="approved")
        r2 = self._r(policy_passed=False, final_decision="rejected")
        cause = self._classify(r1, r2)
        assert cause == "policy"

    def test_routing_risk_category_mismatch(self):
        r1 = self._r(risk_category="MEDIUM", final_decision="approved")
        r2 = self._r(risk_category="HIGH", final_decision="rejected")
        cause = self._classify(r1, r2)
        assert cause == "routing"

    def test_cause_is_always_known(self):
        """Whatever cause is returned, it must be from the known set."""
        scenarios = [
            (self._r(commit_status="blocked_missing_certificate"), self._r()),
            (self._r(audit_bias_flagged=True), self._r(fairness_bias_flagged=False)),
            (self._r(policy_passed=True), self._r(policy_passed=False)),
            (self._r(risk_category="LOW"), self._r(risk_category="HIGH")),
        ]
        for r1, r2 in scenarios:
            cause = self._classify(r1, r2)
            assert cause in self.KNOWN_CAUSES, f"Unknown cause: {cause}"
