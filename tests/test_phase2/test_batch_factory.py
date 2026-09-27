import pandas as pd

from research_v2.phase2.batch_factory import build_cases_from_dataframe, build_paired_cases
from research_v2.phase2.fault_catalog import FaultType


def test_batch_factory_excludes_target_label_and_is_deterministic():
    dataframe = pd.DataFrame(
        [
            {"id": 2, "monthly_income": 60000, "loan_amount": 900000, "target_default": 1},
            {"id": 1, "monthly_income": 50000, "loan_amount": 800000, "target_default": 0},
        ]
    )
    first = build_cases_from_dataframe(dataframe, limit=2, seed=42)
    second = build_cases_from_dataframe(dataframe, limit=2, seed=42)
    assert [case.model_dump() for case in first] == [case.model_dump() for case in second]
    assert "target_default" not in first[0].applicant_data


def test_paired_batch_has_clean_and_all_fault_variants():
    dataframe = pd.DataFrame([{"id": 1, "monthly_income": 50000, "loan_amount": 800000}])
    clean = build_cases_from_dataframe(dataframe)
    paired = build_paired_cases(clean)
    assert len(paired) == 1 + len(FaultType)
    assert paired[0].fault_labels == []
    assert len({case.fault_labels[0] for case in paired[1:]}) == len(FaultType)
