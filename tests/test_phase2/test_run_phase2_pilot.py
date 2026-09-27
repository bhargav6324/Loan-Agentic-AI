import pytest
import json
import re
from pathlib import Path
from scripts.run_phase2_pilot import select_pilot_cases, run_pilot

def test_select_pilot_cases_v2(tmp_path):
    manifest_path = Path("research_v2/synthetic_documents/robustness_dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Dataset manifest not found")
        
    selected = select_pilot_cases(manifest_path, seed=101)
    
    # exactly 20 cases
    assert len(selected) == 20
    
    # Layout counts 6-7
    layout_counts = {}
    for c in selected:
        layout_counts[c["layout_id"]] = layout_counts.get(c["layout_id"], 0) + 1
    assert all(6 <= count <= 7 for count in layout_counts.values())
    
    # at least 6 faulty cases
    faults = [c for c in selected if not c["expected_clean"]]
    assert len(faults) >= 6
    
    # each fault category appears at least twice and spans at least two layouts
    for f in ["missing_salary_slip", "income_contradiction", "employer_contradiction"]:
        f_cases = [c for c in selected if c["expected_fault_code"] == f]
        assert len(f_cases) >= 2
        layouts = set(c["layout_id"] for c in f_cases)
        assert len(layouts) >= 2
        
    # semantic contradictions not all from compact
    sem_cases = [c for c in selected if c["expected_fault_code"] in ["income_contradiction", "employer_contradiction"]]
    assert not all(c["layout_id"] == "compact" for c in sem_cases)
    
    # all required degradations
    for d in ["rotation", "blur", "noise", "low_contrast", "low_brightness", "partial_edge_crop"]:
        assert any(c["variant_type"] == d for c in selected)
        
    # all required severities
    for s in ["low", "medium", "high"]:
        assert any(c["severity"] == s for c in selected)
        
    # Test deterministic
    selected2 = select_pilot_cases(manifest_path, seed=101)
    assert [c["variant_id"] for c in selected] == [c["variant_id"] for c in selected2]
    
    # Test missing coverage causes failure
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    # Remove all tabular layouts
    bad_manifest = [c for c in manifest if c["layout_id"] != "tabular"]
    bad_path = tmp_path / "bad_manifest.json"
    with open(bad_path, "w") as f:
        json.dump(bad_manifest, f)
        
    with pytest.raises(ValueError, match="Could not find valid distribution"):
        select_pilot_cases(bad_path)

def test_run_pilot_v2_mocked(tmp_path):
    manifest_path = Path("research_v2/synthetic_documents/robustness_dataset_manifest.json")
    if not manifest_path.exists():
        pytest.skip("Dataset manifest not found")
        
    output_dir = tmp_path / "pilot_out_v2"
    
    def mock_generate_json(prompt):
        return {
            "monthly_income": 50000.0,
            "employer_name": "Example Pvt Ltd",
            "employee_name": "Asha Rao",
            "pay_period": "August 2026",
            "evidence": {
                "monthly_income": "INR 50,000.00",
                "employer_name": "EXAMPLE PVT LTD"
            },
            "missing_fields": [],
            "confidence": 0.95
        }
        
    res = run_pilot(manifest_path, output_dir, generator=mock_generate_json, skip_health=True)
    
    assert (output_dir / "pilot_selection_manifest.json").exists()
    assert (output_dir / "pilot_results.json").exists()
    
    assert res["summary"]["total_cases"] == 20
    assert "mocked_qwen_pilot" == res["run_metadata"]["evaluation_mode"]
    assert "git_commit" in res["run_metadata"]
    
    hex_pattern = re.compile(r'^[0-9a-f]{64}$')
    
    for r in res["results"]:
        assert r["pre_reassessment_blocked"] is True
        
        assert hex_pattern.match(r["state_hash_before"])
        assert hex_pattern.match(r["state_hash_after_ingestion"])
        
        if r["state_changed"]:
            assert r["state_hash_before"] != r["state_hash_after_ingestion"]
        else:
            assert r["state_hash_before"] == r["state_hash_after_ingestion"]
            
        assert "ocr_text" in r
        assert "llm_raw_response" in r
        assert "git_commit" in r
