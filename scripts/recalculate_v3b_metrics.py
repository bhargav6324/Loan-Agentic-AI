import json
from pathlib import Path
from research_v2.phase2.evidence_validator import normalize_entity_name

def calculate_metrics(results_list, expected_map, version="v2"):
    clean_cases = [r for r in results_list if r.get("expected_clean", True)]
    faulty_cases = [r for r in results_list if not r.get("expected_clean", True)]
    
    num_clean = len(clean_cases)
    num_faulty = len(faulty_cases)

    clean_ext_valid = 0
    clean_inc_correct = 0
    clean_emp_correct = 0
    clean_final_pass = 0
    false_blocks = 0
    
    exact_faults = 0
    safe_blocks = 0
    unsafe_commits = 0
    
    for r in clean_cases:
        vid = r["variant_id"]
        
        # extraction valid
        if version == "v2":
            ext_val = r.get("extraction_valid", False)
        else:
            ext_val = r.get("extraction_valid", False)
        if ext_val:
            clean_ext_valid += 1
            
        # income correct
        exp_inc = expected_map[vid]["expected_document_income"]
        ext_inc = r.get("extracted_monthly_income")
        if exp_inc is None and ext_inc is None:
            r["income_correct"] = None
        else:
            inc_corr = (exp_inc == ext_inc)
            if version == "v2":
                inc_corr = r.get("income_correct", False)
            if inc_corr:
                clean_inc_correct += 1

        # employer correct normalized
        exp_emp = str(expected_map[vid]["expected_document_employer"])
        ext_emp = str(r.get("extracted_employer", ""))
        emp_corr_norm = (normalize_entity_name(exp_emp) == normalize_entity_name(ext_emp))
        if emp_corr_norm:
            clean_emp_correct += 1
            
        # false block
        false_block = r.get("false_block", False)
        if version == "v2":
            false_block = r.get("false_block", not r.get("final_commitment_allowed", True))
        if false_block:
            false_blocks += 1
            
        # final pass
        final_pass = r.get("final_commitment_allowed", not false_block)
        if final_pass:
            clean_final_pass += 1
            
    for r in faulty_cases:
        vid = r["variant_id"]
        # exact fault
        if version == "v2":
            expected = expected_map[vid]["expected_fault_code"]
            detected = r.get("detected_fault_code")
            exact = (expected == detected) and (expected != "none")
        else:
            exact = r.get("exact_fault_detected", False)
        if exact:
            exact_faults += 1
            
        # safe block
        if version == "v2":
            safe_block = not r.get("final_commitment_allowed", False)
        else:
            safe_block = r.get("safe_block", True)
        if safe_block:
            safe_blocks += 1
            
        # unsafe commit
        if version == "v2":
            unsafe = r.get("final_commitment_allowed", False)
        else:
            unsafe = r.get("unsafe_commitment", False)
        if unsafe:
            unsafe_commits += 1

    return {
        "clean_extraction_valid_rate": round(clean_ext_valid / num_clean, 4) if num_clean else 0.0,
        "clean_income_accuracy": round(clean_inc_correct / num_clean, 4) if num_clean else 0.0,
        "clean_employer_accuracy_normalized": round(clean_emp_correct / num_clean, 4) if num_clean else 0.0,
        "exact_fault_detection_rate": round(exact_faults / num_faulty, 4) if num_faulty else 0.0,
        "false_block_rate": round(false_blocks / num_clean, 4) if num_clean else 0.0,
        "safe_block_rate": round(safe_blocks / num_faulty, 4) if num_faulty else 0.0,
        "unsafe_commitment_rate": round(unsafe_commits / num_faulty, 4) if num_faulty else 0.0,
        "clean_final_pass_rate": round(clean_final_pass / num_clean, 4) if num_clean else 0.0
    }

def main():
    v2_path = Path("research_v2/phase2_results/qwen_pilot_v2/pilot_results.json")
    v3b_path = Path("research_v2/phase2_results/qwen_pilot_v3b/pilot_results.json")
    
    with open(v2_path) as f:
        v2_data = json.load(f)
    with open(v3b_path) as f:
        v3b_data = json.load(f)
        
    v2_results = v2_data["results"]
    v3b_results = v3b_data["results"]
    
    expected_map = {r["variant_id"]: {
        "expected_document_income": r.get("expected_document_income"),
        "expected_document_employer": r.get("expected_document_employer", ""),
        "expected_fault_code": r.get("expected_fault_code", "none")
    } for r in v3b_results}
    
    # Check null/null income correctness
    for r in v2_results + v3b_results:
        vid = r["variant_id"]
        exp = expected_map[vid]["expected_document_income"]
        ext = r.get("extracted_monthly_income")
        if exp is None and ext is None:
            r["income_correct"] = None
            
    v2_metrics = calculate_metrics(v2_results, expected_map, "v2")
    v3b_metrics = calculate_metrics(v3b_results, expected_map, "v3b")
    
    v3b_meta = v3b_data["metadata"]
    
    run_metadata = {
        "evaluation_mode": v3b_meta["evaluation_mode"],
        "model": v3b_meta["model"],
        "ocr_engine": v3b_meta["ocr_engine"],
        "timestamp": v3b_meta["timestamp"],
        "git_commit": v3b_meta["git_commit"],
        "dirty_working_tree": v3b_meta["dirty_working_tree"],
        "v1_checksum": v3b_meta["v1_checksum"],
        "v2_checksum": v3b_meta["v2_checksum"],
        "v3_checksum": v3b_meta["v3_checksum"],
        "v3b_checksum": v3b_meta.get("v3b_checksum", "unknown"),
        "schema_version": v3b_meta["schema_version"]
    }
    
    corrected_summary = {
        "run_metadata": run_metadata,
        "v2_metrics": v2_metrics,
        "v3b_metrics": v3b_metrics
    }
    
    out_dir = Path("research_v2/phase2_results/qwen_pilot_v3b")
    with open(out_dir / "corrected_summary.json", "w") as f:
        json.dump(corrected_summary, f, indent=2)
        
    # Generate paired comparison
    comp = []
    v2_by_id = {r["variant_id"]: r for r in v2_results}
    
    for r3 in v3b_results:
        vid = r3["variant_id"]
        r2 = v2_by_id[vid]
        
        comp.append({
            "variant_id": vid,
            "v2_ocr_success": r2.get("ocr_success", False),
            "v3b_ocr_success": r3.get("ocr_success", False),
            
            "v2_extraction_valid": r2.get("extraction_valid", False),
            "v3b_extraction_valid": r3.get("extraction_valid", False),
            
            "v2_income_correct": r2.get("income_correct"),
            "v3b_income_correct": r3.get("income_correct"),
            
            "v2_employer_correct_normalized": normalize_entity_name(str(expected_map[vid]["expected_document_employer"])) == normalize_entity_name(str(r2.get("extracted_employer", ""))),
            "v3b_employer_correct_normalized": normalize_entity_name(str(expected_map[vid]["expected_document_employer"])) == normalize_entity_name(str(r3.get("extracted_employer", ""))),
            
            "v2_detected_fault_code": r2.get("detected_fault_code", "none"),
            "v3b_detected_fault_code": r3.get("detected_fault_code", "none"),
            
            "v2_exact_fault_detected": (expected_map[vid]["expected_fault_code"] == r2.get("detected_fault_code")) and expected_map[vid]["expected_fault_code"] != "none",
            "v3b_exact_fault_detected": r3.get("exact_fault_detected", False),
            
            "v2_false_block": r2.get("false_block", not r2.get("final_commitment_allowed", True)),
            "v3b_false_block": r3.get("false_block", False),
            
            "v2_final_commitment_allowed": r2.get("final_commitment_allowed", False),
            "v3b_final_commitment_allowed": r3.get("final_commitment_allowed", False),
            
            "v2_unsafe_commitment": r2.get("final_commitment_allowed", False) if not r2.get("expected_clean", True) else False,
            "v3b_unsafe_commitment": r3.get("unsafe_commitment", False),
            
            "v2_latency": r2.get("total_latency_seconds", 0),
            "v3b_latency": r3.get("total_latency_seconds", 0)
        })
        
    with open(out_dir / "corrected_v2_v3b_comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
        
    print(json.dumps(corrected_summary, indent=2))

if __name__ == "__main__":
    main()
