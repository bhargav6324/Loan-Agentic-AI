import json
import time
import shutil
import hashlib
from pathlib import Path
from copy import deepcopy

import sys
sys.path.append(str(Path(__file__).parent.parent))

from research_v2.phase2.schemas import LoanEvidenceCase
from research_v2.phase2.ocr_adapter import extract_and_ingest_salary_slip_image, extract_image_text
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.workflow_adapter import build_workflow_certificates, REQUIRED_CERTIFICATE_TYPES
from research_v2.phase2.certificates import compute_state_hash
from research_v2.phase2.evidence_validator import normalize_entity_name
from research_v2.phase2.state_lifecycle import mark_risk_reassessed
from agents.ollama_client import generate_json

def load_and_validate_v3_selection(v2_manifest_path: Path, v3_manifest_path: Path):
    with open(v2_manifest_path) as f:
        v2 = json.load(f)
    with open(v3_manifest_path) as f:
        v3 = json.load(f)
        
    if len(v2) != len(v3):
        raise ValueError("V3 selection does not perfectly match V2")
        
    for a, b in zip(v2, v3):
        if a["variant_id"] != b["variant_id"]:
            raise ValueError("V3 selection does not perfectly match V2")
            
    return v3

def run_v3_pilot():
    out_dir = Path("research_v2/phase2_results/qwen_pilot_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    v2_manifest_path = Path("research_v2/phase2_results/qwen_pilot_v2/pilot_selection_manifest.json")
    v3_manifest_path = out_dir / "pilot_selection_manifest.json"
    
    shutil.copy(v2_manifest_path, v3_manifest_path)
    
    v2_hash = hashlib.sha256(v2_manifest_path.read_bytes()).hexdigest()
    
    selected_cases = load_and_validate_v3_selection(v2_manifest_path, v3_manifest_path)
    
    results = []
    
    for case_data in selected_cases:
        t0 = time.time()
        
        expected_clean = case_data["expected_clean"]
        expected_fault_code = case_data["expected_fault_code"]
        
        case = LoanEvidenceCase(
            case_id=case_data["variant_id"],
            applicant_data={
                "monthly_income": case_data["declared_monthly_income"],
                "employer_name": case_data["declared_employer"],
                "average_salary_credit": case_data["declared_monthly_income"],
                "loan_amount": 100000.0,
                "credit_score": 750,
                "dti_ratio": 0.3,
                "loan_to_income": 2.0
            }
        )
        case.execution.policy_node_executed = True
        graph_result = {"policy_passed": True, "final_decision": "approve", "commit_status": "committed_approve"}
        case.state_version = 1
        case = mark_risk_reassessed(case)
        
        initial_certificates = build_workflow_certificates(case, graph_result)
        current_certificates = initial_certificates
        
        ocr_profile = "compact_grayscale_psm6" if case_data["layout_id"] == "compact" else "default"
        
        t_ocr_start = time.time()
        ocr = extract_image_text(Path(case_data["image_path"]), profile=ocr_profile)
        t_ocr_end = time.time()
        
        from research_v2.phase2.llm_document_extractor import extract_salary_slip_text
        from research_v2.phase2.state_lifecycle import ingest_salary_slip_extraction
        
        extraction = None
        if ocr.valid:
            extraction = extract_salary_slip_text("doc1", ocr.text, generator=generate_json)
            ingestion = ingest_salary_slip_extraction(case, extraction, current_certificates)
            case = ingestion.transition.case
            current_certificates = ingestion.transition.certificates
            
        t1 = time.time()
        
        case = mark_risk_reassessed(case)
        certs = build_workflow_certificates(case, graph_result)
        gate = evaluate_commitment(case, certs, REQUIRED_CERTIFICATE_TYPES, True)
        
        ext = extraction.document if extraction and extraction.valid else None
        
        emp_correct_raw = False
        emp_correct_norm = False
        
        doc_emp = case_data.get("document_employer", "")
        ext_emp = ext.fields["employer"].value if ext and "employer" in ext.fields else None
        
        if doc_emp and ext_emp:
            emp_correct_raw = ext_emp == doc_emp
            emp_correct_norm = normalize_entity_name(str(ext_emp)).lower() == normalize_entity_name(doc_emp).lower()
            
        doc_inc = case_data.get("document_monthly_income")
        ext_inc = ext.fields["monthly_income"].value if ext and "monthly_income" in ext.fields else None
        inc_correct = (doc_inc == ext_inc)
        
        has_salary_evidence = False
        if ocr.valid:
            from scripts.run_compact_ocr_diagnostic_v2 import contains_salary_evidence
            has_salary_evidence = contains_salary_evidence(ocr.text)
        
        detected_fault = "none"
        if "missing_salary_slip" in case.fault_labels:
            detected_fault = "missing_salary_slip"
        elif "income_contradiction" in case.fault_labels:
            detected_fault = "income_contradiction"
        elif "employer_contradiction" in case.fault_labels:
            detected_fault = "employer_contradiction"
            
        exact_fault_match = False
        if not expected_clean and detected_fault == expected_fault_code:
            exact_fault_match = True
            
        false_block = expected_clean and not gate.allowed
        unsafe_commitment = not expected_clean and gate.allowed
        
        results.append({
            "variant_id": case_data["variant_id"],
            "layout_id": case_data["layout_id"],
            "expected_clean": expected_clean,
            "expected_fault_code": expected_fault_code,
            "ocr_success": ocr.valid,
            "has_salary_evidence": has_salary_evidence,
            "extraction_valid": extraction.valid if extraction else False,
            "income_correct": inc_correct,
            "employer_correct_raw": emp_correct_raw,
            "employer_correct_normalized": emp_correct_norm,
            "detected_fault_code": detected_fault,
            "exact_fault_detected": exact_fault_match,
            "final_commitment": gate.allowed,
            "false_block": false_block,
            "unsafe_commitment": unsafe_commitment,
            "latency": t1 - t0
        })
        
    summary = {
        "v2_selection_checksum": v2_hash,
        "total_cases": len(results),
        "clean_extraction_valid_rate": sum(1 for r in results if r["expected_clean"] and r["extraction_valid"]) / sum(1 for r in results if r["expected_clean"]) if sum(1 for r in results if r["expected_clean"]) else 0,
        "clean_income_accuracy": sum(1 for r in results if r["expected_clean"] and r["income_correct"]) / sum(1 for r in results if r["expected_clean"]) if sum(1 for r in results if r["expected_clean"]) else 0,
        "clean_employer_accuracy_normalized": sum(1 for r in results if r["expected_clean"] and r["employer_correct_normalized"]) / sum(1 for r in results if r["expected_clean"]) if sum(1 for r in results if r["expected_clean"]) else 0,
        "exact_fault_detection_rate": sum(1 for r in results if not r["expected_clean"] and r["exact_fault_detected"]) / sum(1 for r in results if not r["expected_clean"]) if sum(1 for r in results if not r["expected_clean"]) else 0,
        "safe_block_rate": sum(1 for r in results if not r["expected_clean"] and not r["final_commitment"]) / sum(1 for r in results if not r["expected_clean"]) if sum(1 for r in results if not r["expected_clean"]) else 0,
        "false_block_rate": sum(1 for r in results if r["expected_clean"] and r["false_block"]) / sum(1 for r in results if r["expected_clean"]) if sum(1 for r in results if r["expected_clean"]) else 0,
        "unsafe_commitment_rate": sum(1 for r in results if not r["expected_clean"] and r["unsafe_commitment"]) / sum(1 for r in results if not r["expected_clean"]) if sum(1 for r in results if not r["expected_clean"]) else 0,
        "clean_final_pass_rate": sum(1 for r in results if r["expected_clean"] and r["final_commitment"]) / sum(1 for r in results if r["expected_clean"]) if sum(1 for r in results if r["expected_clean"]) else 0,
        "mean_latency": sum(r["latency"] for r in results) / len(results)
    }
    
    # Sort results
    results = sorted(results, key=lambda x: [c["variant_id"] for c in selected_cases].index(x["variant_id"]))
    
    with open(out_dir / "pilot_results.json", "w") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2)
        
    v2_results_path = Path("research_v2/phase2_results/qwen_pilot_v2/pilot_results.json")
    with open(v2_results_path) as f:
        v2_data = json.load(f)
        
    v2_res_map = {r["variant_id"]: r for r in v2_data["results"]}
    
    comparison = []
    for r3 in results:
        vid = r3["variant_id"]
        r2 = v2_res_map.get(vid, {})
        
        comparison.append({
            "variant_id": vid,
            "v2_ocr_success": r2.get("ocr_success", False),
            "v3_ocr_success": r3["ocr_success"],
            "v2_has_salary_evidence": r2.get("has_salary_evidence", False),
            "v3_has_salary_evidence": r3["has_salary_evidence"],
            "v2_false_block": r2.get("false_block", False),
            "v3_false_block": r3["false_block"],
            "v2_extraction_valid": r2.get("extraction_valid", False),
            "v3_extraction_valid": r3["extraction_valid"],
            "v2_income_correct": r2.get("income_correct", False),
            "v3_income_correct": r3["income_correct"],
            "v2_employer_correct_normalized": r2.get("employer_correct_normalized", False) or r2.get("employer_correct", False),
            "v3_employer_correct_normalized": r3["employer_correct_normalized"],
            "v2_detected_fault_code": r2.get("detected_fault_code", "none"),
            "v3_detected_fault_code": r3["detected_fault_code"],
            "v2_exact_fault_detected": r2.get("exact_fault_detected", False) or r2.get("fault_detected", False),
            "v3_exact_fault_detected": r3["exact_fault_detected"],
            "v2_final_commitment": r2.get("final_commitment", False) or r2.get("final_commitment_allowed", False),
            "v3_final_commitment": r3["final_commitment"],
            "v2_unsafe_commitment": r2.get("unsafe_commitment", False),
            "v3_unsafe_commitment": r3["unsafe_commitment"],
            "v2_latency": r2.get("total_latency", 0) or r2.get("latency", 0),
            "v3_latency": r3["latency"]
        })
        
    with open(out_dir / "v2_v3_paired_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

if __name__ == "__main__":
    run_v3_pilot()
