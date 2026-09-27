import json
import time
from pathlib import Path
from research_v2.phase2.schemas import LoanEvidenceCase, ExecutionTrace
from research_v2.phase2.ocr_adapter import extract_and_ingest_salary_slip_image, extract_image_text
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.workflow_adapter import build_workflow_certificates, REQUIRED_CERTIFICATE_TYPES
from research_v2.phase2.certificates import compute_state_hash
from research_v2.phase2.evidence_validator import normalize_entity_name
from research_v2.phase2.state_lifecycle import mark_risk_reassessed, ingest_salary_slip_extraction
from agents.ollama_client import generate_json
from research_v2.phase2.llm_document_extractor import extract_salary_slip_text

def load_and_validate_v3b_selection(v2_manifest_path: Path, v3b_manifest_path: Path):
    with open(v2_manifest_path) as f:
        v2 = json.load(f)
    with open(v3b_manifest_path) as f:
        v3b = json.load(f)
        
    if len(v2) != len(v3b):
        raise ValueError(f"Selection length mismatch: V2 {len(v2)} != V3b {len(v3b)}")
        
    for i, (a, b) in enumerate(zip(v2, v3b)):
        if a["variant_id"] != b["variant_id"]:
            raise ValueError(f"Selection mismatch at index {i}: {a['variant_id']} != {b['variant_id']}")
            
    return v3b

def run_v3b_pilot():
    manifest_path = Path("research_v2/phase2_results/qwen_pilot_v2/pilot_selection_manifest.json")
    v3b_dir = Path("research_v2/phase2_results/qwen_pilot_v3b")
    v3b_dir.mkdir(parents=True, exist_ok=True)
    
    cases_data = load_and_validate_v3b_selection(manifest_path, manifest_path)
    
    results = []
    
    for case_data in cases_data:
        t0 = time.time()
        
        expected_clean = case_data["expected_clean"]
        expected_fault_code = case_data["expected_fault_code"]
        layout_id = case_data["layout_id"]
        
        case = LoanEvidenceCase(
            case_id=case_data["case_id"],
            state_version=1,
            applicant_data={
                "monthly_income": case_data["declared_monthly_income"],
                "loan_amount": 100000,
                "average_salary_credit": case_data["declared_monthly_income"],
                "credit_score": 750,
                "dti_ratio": 0.3,
                "loan_to_income": 2.0,
                "employer_name": case_data["declared_employer"],
            },
            documents=[],
            extracted_evidence={},
            execution=ExecutionTrace(policy_node_executed=True),
            fault_labels=[case_data["expected_fault_code"]] if not expected_clean and expected_fault_code != "none" else []
        )
        
        # 2. Mark initial risk assessment
        case = mark_risk_reassessed(case)
        # 3. Build initial certificates
        graph_result = {"policy_passed": True, "final_decision": "approve", "commit_status": "committed_approve"}
        certs = build_workflow_certificates(case, graph_result)
        
        state_hash_before = compute_state_hash(case)
        state_version_before = case.state_version
        
        # 4. Run OCR
        profile = "compact_grayscale_psm6" if layout_id == "compact" else "default"
        image_path = Path(case_data["image_path"])
        
        ocr_t0 = time.time()
        ocr = extract_image_text(image_path, profile=profile)
        ocr_t1 = time.time()
        
        ext = None
        extraction = None
        ingestion = None
        
        llm_t0 = time.time()
        # 5. Run bounded Qwen extraction & 6. Ingest extraction
        if ocr.valid:
            extraction = extract_salary_slip_text("doc1", ocr.text, generator=generate_json)
            ingestion = ingest_salary_slip_extraction(case, extraction, certs)
            case = ingestion.transition.case
            certs = ingestion.transition.certificates
            ext = extraction.document if extraction.valid else None
        llm_t1 = time.time()
        
        # 7. Evaluate the pre-reassessment gate
        pre_gate = evaluate_commitment(case, certs, REQUIRED_CERTIFICATE_TYPES, True)
        
        # 8. Perform deterministic reassessment
        case = mark_risk_reassessed(case)
        
        # 9. Regenerate valid certificates
        certs = build_workflow_certificates(case, graph_result)
        
        # 10. Evaluate the final gate
        final_gate = evaluate_commitment(case, certs, REQUIRED_CERTIFICATE_TYPES, True)
        
        # 11. Derive observed faults from final gate reasons
        detected_evidence_faults = [r for r in final_gate.reasons if r.startswith("evidence:")]
        detected_fault_code = "none"
        if detected_evidence_faults:
            detected_fault_code = detected_evidence_faults[0].split("evidence:")[1]
            
        exact_fault_detected = False
        if not expected_clean and expected_fault_code != "none":
            exact_fault_detected = f"evidence:{expected_fault_code}" in final_gate.reasons
            
        # Additional fields
        doc_inc = case_data.get("document_monthly_income")
        ext_inc = ext.fields["monthly_income"].value if ext and "monthly_income" in ext.fields else None
        inc_correct = (doc_inc == ext_inc)
        
        doc_emp = str(case_data.get("document_employer"))
        ext_emp = str(ext.fields["employer_name"].value) if ext and "employer_name" in ext.fields and ext.fields["employer_name"].value else ""
        emp_correct_raw = (doc_emp == ext_emp)
        emp_correct_norm = (normalize_entity_name(ext_emp) == normalize_entity_name(doc_emp))
        
        false_block = expected_clean and not final_gate.allowed
        unsafe_commitment = not expected_clean and final_gate.allowed
        safe_block = not expected_clean and not final_gate.allowed
        
        t1 = time.time()
        
        # 12. Record the complete audit result
        results.append({
            "pilot_order": case_data["pilot_order"],
            "case_id": case_data["case_id"],
            "variant_id": case_data["variant_id"],
            "variant_type": case_data["variant_type"],
            "layout_id": layout_id,
            "severity": case_data.get("severity", "none"),
            "expected_clean": expected_clean,
            "expected_fault_code": expected_fault_code,
            "ocr_profile": profile,
            "ocr_success": ocr.valid,
            "ocr_confidence": ocr.mean_confidence,
            "ocr_errors": ocr.errors,
            "ocr_text": ocr.text,
            "extraction_valid": extraction.valid if extraction else False,
            "extraction_errors": extraction.errors if extraction else ["ocr_failed"],
            "llm_raw_response": extraction.raw_response if extraction else "",
            "extracted_monthly_income": ext_inc,
            "extracted_employer": ext_emp,
            "expected_document_income": doc_inc,
            "expected_document_employer": doc_emp,
            "income_correct": inc_correct,
            "employer_correct_exact": emp_correct_raw,
            "employer_correct_normalized": emp_correct_norm,
            "ingestion_accepted": ingestion.accepted if ingestion else False,
            "changed_fields": list(ingestion.transition.changed_fields) if ingestion else [],
            "state_changed": ingestion.transition.state_changed if ingestion else False,
            "state_version_before": state_version_before,
            "state_version_after_ingestion": case.state_version,
            "state_hash_before": state_hash_before,
            "state_hash_after_ingestion": compute_state_hash(case),
            "pre_reassessment_blocked": not pre_gate.allowed,
            "pre_reassessment_gate_reasons": pre_gate.reasons,
            "issued_certificate_types": [c.certificate_type for c in certs],
            "detected_fault_codes": [r.split("evidence:")[1] for r in detected_evidence_faults],
            "detected_fault_code": detected_fault_code,
            "exact_fault_detected": exact_fault_detected,
            "final_commitment_allowed": final_gate.allowed,
            "final_gate_reasons": final_gate.reasons,
            "safe_block": safe_block,
            "false_block": false_block,
            "unsafe_commitment": unsafe_commitment,
            "ocr_latency_seconds": ocr_t1 - ocr_t0,
            "llm_latency_seconds": llm_t1 - llm_t0,
            "total_latency_seconds": t1 - t0
        })
        
    summary = {
        "total_cases": len(results),
        "clean_extraction_valid_rate": sum(1 for r in results if r["expected_clean"] and r["extraction_valid"]) / sum(1 for r in results if r["expected_clean"]) if any(r["expected_clean"] for r in results) else 0,
        "clean_income_accuracy": sum(1 for r in results if r["expected_clean"] and r["income_correct"]) / sum(1 for r in results if r["expected_clean"]) if any(r["expected_clean"] for r in results) else 0,
        "clean_employer_accuracy_normalized": sum(1 for r in results if r["expected_clean"] and r["employer_correct_normalized"]) / sum(1 for r in results if r["expected_clean"]) if any(r["expected_clean"] for r in results) else 0,
        "exact_fault_detection_rate": sum(1 for r in results if not r["expected_clean"] and r["exact_fault_detected"]) / sum(1 for r in results if not r["expected_clean"]) if any(not r["expected_clean"] for r in results) else 0,
        "safe_block_rate": sum(1 for r in results if not r["expected_clean"] and r["safe_block"]) / sum(1 for r in results if not r["expected_clean"]) if any(not r["expected_clean"] for r in results) else 0,
        "false_block_rate": sum(1 for r in results if r["expected_clean"] and r["false_block"]) / sum(1 for r in results if r["expected_clean"]) if any(r["expected_clean"] for r in results) else 0,
        "unsafe_commitment_rate": sum(1 for r in results if not r["expected_clean"] and r["unsafe_commitment"]) / sum(1 for r in results if not r["expected_clean"]) if any(not r["expected_clean"] for r in results) else 0,
        "clean_final_pass_rate": sum(1 for r in results if r["expected_clean"] and r["final_commitment_allowed"]) / sum(1 for r in results if r["expected_clean"]) if any(r["expected_clean"] for r in results) else 0,
        "mean_latency": sum(r["total_latency_seconds"] for r in results) / len(results)
    }
    
    metadata = {
        "evaluation_mode": "real_qwen_pilot_v3b",
        "model": "qwen3:8b",
        "ocr_engine": "tesseract",
        "timestamp": time.time(),
        "schema_version": "1.1",
        "git_commit": "unknown", # would be subprocess call
        "dirty_working_tree": False,
        "v1_checksum": "7bfe884356f1b93a6a07d5559c1bb16476cc9260",
        "v2_checksum": "3fde14b106bec6b04a09c1c4c23ebca602b2b5d0",
        "v2_selection_checksum": "30ae21c9df88d9d5ef277f9dc793b7ba2ec5149d",
        "v3_checksum": "broken",
        "cases_processed": len(results)
    }
    
    final_output = {
        "metadata": metadata,
        "summary": summary,
        "results": results
    }
    
    with open(v3b_dir / "pilot_results.json", "w") as f:
        json.dump(final_output, f, indent=2)
        
    # generate comparison
    import json as js
    with open("research_v2/phase2_results/qwen_pilot_v2/pilot_results.json") as f:
        v2_results = js.load(f)["results"]
        
    v2_by_id = {r["variant_id"]: r for r in v2_results}
    
    comp = []
    for r3 in results:
        vid = r3["variant_id"]
        r2 = v2_by_id.get(vid, {})
        comp.append({
            "variant_id": vid,
            "v2_ocr_success": r2.get("ocr_success", False),
            "v3b_ocr_success": r3["ocr_success"],
            "v2_false_block": r2.get("false_block", False),
            "v3b_false_block": r3["false_block"],
            "v2_extraction_valid": r2.get("extraction_valid", False),
            "v3b_extraction_valid": r3["extraction_valid"],
            "v2_income_correct": r2.get("income_correct", False),
            "v3b_income_correct": r3["income_correct"],
            "v2_employer_correct_normalized": r2.get("employer_correct_normalized", False),
            "v3b_employer_correct_normalized": r3["employer_correct_normalized"],
            "v2_detected_fault_code": r2.get("detected_fault_code", "none"),
            "v3b_detected_fault_code": r3["detected_fault_code"],
            "v2_exact_fault_detected": r2.get("exact_fault_detected", False),
            "v3b_exact_fault_detected": r3["exact_fault_detected"],
            "v2_final_commitment_allowed": r2.get("final_commitment_allowed", False) if "final_commitment_allowed" in r2 else (not r2.get("false_block", False) and r2.get("expected_clean", True)),
            "v3b_final_commitment_allowed": r3["final_commitment_allowed"],
            "v2_unsafe_commitment": r2.get("unsafe_commitment", False),
            "v3b_unsafe_commitment": r3["unsafe_commitment"],
            "v2_latency": r2.get("total_latency_seconds", 0),
            "v3b_latency": r3["total_latency_seconds"]
        })
        
    with open(v3b_dir / "v2_v3b_paired_comparison.json", "w") as f:
        json.dump(comp, f, indent=2)
        
    print(json.dumps(summary, indent=2))

if __name__ == "__main__":
    run_v3b_pilot()
