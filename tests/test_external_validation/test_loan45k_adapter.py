import pandas as pd

from research_v2.external_validation.loan45k_adapter import REQUIRED_COLUMNS, prepare_loan45k


def test_prepare_loan45k_excludes_impossible_rows(tmp_path):
    valid = {
        "person_age": 30, "person_gender": "female", "person_education": "Bachelor",
        "person_income": 60000, "person_emp_exp": 6, "person_home_ownership": "RENT",
        "loan_amnt": 10000, "loan_intent": "PERSONAL", "loan_int_rate": 10.0,
        "loan_percent_income": 0.17, "cb_person_cred_hist_length": 8,
        "credit_score": 700, "previous_loan_defaults_on_file": "No", "loan_status": 1,
    }
    invalid = dict(valid, person_age=144, person_emp_exp=125)
    path = tmp_path / "loan_data.csv"
    pd.DataFrame([valid, invalid], columns=REQUIRED_COLUMNS).to_csv(path, index=False)
    prepared = prepare_loan45k(path)
    assert len(prepared.data) == 1
    assert prepared.audit["rows_excluded_total"] == 1
    assert prepared.data.loc[0, "gender_encoded"] == 1
    assert prepared.data.loc[0, "loan_to_income"] == 10000 / 60000


def test_prepare_loan45k_rejects_missing_schema(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"person_age": [30]}).to_csv(path, index=False)
    try:
        prepare_loan45k(path)
    except ValueError as exc:
        assert "Missing required columns" in str(exc)
    else:
        raise AssertionError("Expected schema validation failure")
