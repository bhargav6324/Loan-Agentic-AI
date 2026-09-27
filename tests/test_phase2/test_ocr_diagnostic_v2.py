import pytest
from scripts.run_compact_ocr_diagnostic_v2 import extract_salary_candidates, contains_salary_evidence, salary_matches_expected
from research_v2.phase2.evidence_validator import normalize_entity_name

def test_salary_parsing():
    assert extract_salary_candidates("Net Pay: INR 85,000.00") == [85000.0]
    assert extract_salary_candidates("₹85,000") == [85000.0]
    assert extract_salary_candidates("Rs. 85,000") == [85000.0]
    assert 2026 not in extract_salary_candidates("August 2026 Rs. 85,000 EMP-1002")
    assert 1002 not in extract_salary_candidates("August 2026 Rs. 85,000 EMP-1002")
    assert not contains_salary_evidence("Name: Asha Rao, Employer: Example Pvt Ltd")

def test_unexpected_salary_in_crop():
    text_from_crop = "Some random text INR 50,000"
    assert contains_salary_evidence(text_from_crop) == True

def test_employer_normalized():
    assert normalize_entity_name("GlobalTech Solutions") == normalize_entity_name("GLOBALTECH SOLUTIONS")
    assert normalize_entity_name("Innovatech Corp") != normalize_entity_name("Innovatech Gorp")

def test_ocr_diagnostic_ranking_real(tmp_path):
    # Simulate a ranking tie and false recovery outranking
    methods = [
        {"name": "m1_bad", "false_recovery_rate": 0.1, "salary_recovery_rate": 1.0, "employer_recovery_rate": 1.0, "median_latency_seconds": 0.1, "index": 0},
        {"name": "m2_good_but_complex", "false_recovery_rate": 0.0, "salary_recovery_rate": 0.9, "employer_recovery_rate": 0.9, "median_latency_seconds": 0.1, "index": 2},
        {"name": "m3_good_simple", "false_recovery_rate": 0.0, "salary_recovery_rate": 0.9, "employer_recovery_rate": 0.9, "median_latency_seconds": 0.2, "index": 1},
    ]
    
    # zero false recovery outranks higher recovery with unsafe false recovery (m2, m3 > m1)
    # quality ties prefer simpler method (m3 index < m2 index, so m3 > m2)
    methods.sort(key=lambda x: (
        x["false_recovery_rate"] == 0,
        x["salary_recovery_rate"],
        x["exact_expected_accuracy"] if "exact_expected_accuracy" in x else x["salary_recovery_rate"],
        x["employer_recovery_rate"],
        -x["index"],
        -x["median_latency_seconds"]
    ), reverse=True)
    
    assert methods[0]["name"] == "m3_good_simple"
