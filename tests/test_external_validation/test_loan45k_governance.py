from research_v2.external_validation.loan45k_governance import decide, research_policy, state_hash


def record(**changes):
    base = {"person_age": 35, "person_gender": "female", "person_income": 60000,
            "person_emp_exp": 8, "person_home_ownership": "RENT", "loan_amnt": 10000,
            "loan_intent": "EDUCATION", "loan_int_rate": 10.0,
            "loan_percent_income": 0.17, "cb_person_cred_hist_length": 10,
            "credit_score": 720, "previous_loan_defaults_on_file": "No", "loan_status": 1}
    base.update(changes)
    return base


def test_policy_is_frozen():
    passed, violations = research_policy(record(person_age=19, credit_score=350, loan_percent_income=.7))
    assert not passed
    assert violations == ["minimum_age", "minimum_credit_score", "maximum_loan_income_burden"]


def test_hash_excludes_ground_truth():
    assert state_hash(record(loan_status=0)) == state_hash(record(loan_status=1))


def test_e1_blocks_policy_violation():
    result = decide(record(credit_score=350), .9, .9, "E1")
    assert result.proposed_approval == 1 and result.final_approval == 0


def test_e0_does_not_apply_policy():
    result = decide(record(credit_score=350), .9, .9, "E0")
    assert result.final_approval == 1


def test_e2_closes_current_evidence():
    result = decide(record(), .8, .79, "E2")
    assert result.certificates_valid and result.commit_status == "committed"


def test_e3_critic_records_conflict():
    result = decide(record(credit_score=350), .9, .9, "E3")
    assert "approval_conflicts_with_policy" in result.critic_findings


def test_fairness_flag_is_not_automatic_rejection():
    result = decide(record(), .8, .7, "E3")
    assert result.fairness_flagged and result.final_approval == 1
