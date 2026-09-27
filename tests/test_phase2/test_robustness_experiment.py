import pytest
import json
from pathlib import Path
from research_v2.phase2.robustness_experiment import generate_robustness_variants, run_experiment

def test_generate_robustness_variants(tmp_path):
    output_dir = tmp_path / "images"
    manifest_path = tmp_path / "manifest.json"
    manifest = generate_robustness_variants(output_dir, manifest_path, dry_run=True)
    
    assert len(manifest) > 0
    assert manifest_path.exists()
    assert (output_dir / Path(manifest[0]["image_path"]).name).exists()

def test_run_experiment_correctness(tmp_path):
    # Mock Ollama to be completely deterministic and avoid server restart flakiness
    def mock_generate_json(prompt):
        import re
        emp_match = re.search(r"Employer:\s*(.*)", prompt, re.IGNORECASE)
        inc_match = re.search(r"Net Pay:\s*(.*)", prompt, re.IGNORECASE)
        emp_quote = emp_match.group(1).strip() if emp_match else "EXAMPLE PVT LTD"
        inc_quote = inc_match.group(1).strip() if inc_match else "INR 50,000.00"
        
        res = {
            "monthly_income": 50000,
            "employer_name": "Example Pvt Ltd",
            "employee_name": None,
            "pay_period": None,
            "evidence": {
                "monthly_income": inc_quote,
                "employer_name": emp_quote
            },
            "missing_fields": [],
            "confidence": 0.95
        }
        if "999,999.00" in prompt:
            res["monthly_income"] = 999999
        elif "Fake Company Inc" in prompt:
            res["employer_name"] = "Fake Company Inc"
        elif "Net Pay:" not in prompt and "50,000.00" not in prompt:
            res.pop("monthly_income")
            res["missing_fields"].append("monthly_income")
            res["evidence"].pop("monthly_income", None)
        return res
        
    artifact = run_experiment(
        dry_run=True,
        generator=mock_generate_json,
        output_images_dir=tmp_path / "images",
        manifest_path=tmp_path / "manifest.json",
        results_path=tmp_path / "mocked_logic_results.json",
        evaluation_mode="mocked_logic_test",
    )
    results = artifact["results"]
    assert len(results) == 10
    assert artifact["run_metadata"]["evaluation_mode"] == "mocked_logic_test"
    assert artifact["summary"]["false_block_rate"] == 0.0
    assert artifact["summary"]["fault_detection_rate"] == 1.0
    assert artifact["summary"]["clean_extraction_valid_rate"] == 1.0
    assert artifact["summary"]["clean_income_extraction_accuracy"] == 1.0
    assert artifact["summary"]["clean_employer_extraction_accuracy"] == 1.0
    assert (tmp_path / "mocked_logic_results.json").exists()
    
    for r in results:
        # Pre-reassessment should always be blocked because state version advanced, invalidating risk score cert
        assert r["pre_reassessment_block"] is True
        
        # Enforce all required certificates are checked and passed when gate allows
        if not r["final_commitment_block"]:
            assert set(r["issued_certificate_types"]) == {"income_consistency", "risk_score", "policy_compliance"}
            
    # Clean variants should pass final gate
    clean = [r for r in results if r["variant"] == "clean_baseline"][0]
    assert clean["clean_final_pass"] is True
    assert clean["final_commitment_block"] is False
    assert clean["false_blocking"] is False
    assert clean["detected_fault_code"] == "none"
    
    # Degraded but readable variants should pass
    rotation = [r for r in results if r["variant"] == "rotation_-5"][0]
    assert rotation["clean_final_pass"] is True
    assert rotation["final_commitment_block"] is False
    
    # Altered income should be blocked
    altered_income = [r for r in results if r["variant"] == "altered_income"][0]
    assert altered_income["final_commitment_block"] is True
    assert altered_income["safe_blocking"] is True
    assert altered_income["fault_detected"] is True
    assert altered_income["detected_fault_code"] == "income_contradiction"
    
    # Altered employer should be blocked
    altered_employer = [r for r in results if r["variant"] == "altered_employer"][0]
    assert altered_employer["final_commitment_block"] is True
    assert altered_employer["safe_blocking"] is True
    assert altered_employer["fault_detected"] is True
    assert altered_employer["detected_fault_code"] == "employer_contradiction"
    
    # Critical field crop should be blocked
    critical_crop = [r for r in results if r["variant"] == "critical_field_crop"][0]
    assert critical_crop["final_commitment_block"] is True
    assert critical_crop["safe_blocking"] is True
    assert critical_crop["fault_detected"] is True
    assert critical_crop["detected_fault_code"] == "missing_salary_slip"
    
    # Verify false block rate is 0 in dry run
    assert sum(r["false_blocking"] for r in results) == 0
