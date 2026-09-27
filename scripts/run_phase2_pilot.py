import json
import random
import time
import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, List, Dict, Optional

import sys
sys.path.append(str(Path(__file__).parent.parent))

from research_v2.phase2.ocr_adapter import extract_image_text
from research_v2.phase2.llm_document_extractor import extract_salary_slip_text
from research_v2.phase2.state_lifecycle import ingest_salary_slip_extraction, mark_risk_reassessed
from research_v2.phase2.workflow_adapter import build_workflow_certificates, REQUIRED_CERTIFICATE_TYPES
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.certificates import compute_state_hash
from research_v2.phase2.schemas import LoanEvidenceCase
from agents.ollama_client import generate_json, check_health, OllamaConnectionError
from config import settings

def select_pilot_cases(manifest_path: Path, seed: int = 42) -> List[Dict]:
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    random.seed(seed)
    
    for attempt in range(1000):
        random.shuffle(manifest)
        
        selected = []
        selected_ids = set()
        
        def layout_count(l):
            return sum(1 for c in selected if c["layout_id"] == l)
            
        def add_case(cond):
            for c in manifest:
                if c["variant_id"] not in selected_ids and cond(c) and layout_count(c["layout_id"]) < 7:
                    selected.append(c)
                    selected_ids.add(c["variant_id"])
                    return c
            return None
            
        for layout in ["standard", "compact", "tabular"]:
            add_case(lambda c: c["layout_id"] == layout and c["variant_type"] == "clean_baseline")
            
        faults = ["missing_salary_slip", "income_contradiction", "employer_contradiction"]
        for f in faults:
            c1 = add_case(lambda c: c["expected_fault_code"] == f)
            if c1:
                add_case(lambda c: c["expected_fault_code"] == f and c["layout_id"] != c1["layout_id"])
            
        degraded = ["rotation", "blur", "noise", "low_contrast", "low_brightness", "partial_edge_crop"]
        for d in degraded:
            add_case(lambda c: c["variant_type"] == d and c["expected_clean"])
            
        for sev in ["low", "medium", "high"]:
            if not any(c["severity"] == sev for c in selected):
                add_case(lambda c: c["severity"] == sev and c["expected_clean"])
                
        for c in manifest:
            if len(selected) >= 20: break
            if c["variant_id"] in selected_ids: continue
            
            if layout_count(c["layout_id"]) < 7:
                selected.append(c)
                selected_ids.add(c["variant_id"])
                
        if len(selected) != 20: continue
        
        lc = {l: layout_count(l) for l in ["standard", "compact", "tabular"]}
        if any(count < 6 or count > 7 for count in lc.values()): continue
        
        faulty_count = sum(1 for c in selected if not c["expected_clean"])
        if faulty_count < 6: continue
        
        bad_fault = False
        for f in faults:
            f_cases = [c for c in selected if c["expected_fault_code"] == f]
            if len(f_cases) < 2: bad_fault = True
            f_layouts = set(c["layout_id"] for c in f_cases)
            if len(f_layouts) < 2: bad_fault = True
        if bad_fault: continue
        
        sem_cases = [c for c in selected if c["expected_fault_code"] in ["income_contradiction", "employer_contradiction"]]
        if all(c["layout_id"] == "compact" for c in sem_cases): continue
        
        if not all(any(c["variant_type"] == d for c in selected) for d in degraded): continue
        
        if not all(any(c["severity"] == s for c in selected) for s in ["low", "medium", "high"]): continue
        
        for i, c in enumerate(selected):
            c["pilot_order"] = i + 1
            
        return selected
        
    raise ValueError("Could not find valid distribution after 1000 attempts")

def _summary(results: List[Dict]) -> Dict:
    clean = [r for r in results if r["expected_clean"]]
    faulty = [r for r in results if not r["expected_clean"]]
    def rate(num, den): return round(num/den, 4) if den else 0.0

    layout_counts = {}
    variant_counts = {}
    severity_counts = {}
    fault_counts = {}
    for r in results:
        layout_counts[r["layout_id"]] = layout_counts.get(r["layout_id"], 0) + 1
        variant_counts[r["variant_type"]] = variant_counts.get(r["variant_type"], 0) + 1
        severity_counts[r["severity"]] = severity_counts.get(r["severity"], 0) + 1
        fault_counts[r["expected_fault_code"]] = fault_counts.get(r["expected_fault_code"], 0) + 1

    exact_faults_by_cat = {}
    for r in faulty:
        f = r["expected_fault_code"]
        if f not in exact_faults_by_cat:
            exact_faults_by_cat[f] = {"detected": 0, "total": 0}
        exact_faults_by_cat[f]["total"] += 1
        if r["fault_detected"]:
            exact_faults_by_cat[f]["detected"] += 1

    exact_faults_rates = {k: rate(v["detected"], v["total"]) for k, v in exact_faults_by_cat.items()}

    mean_ocr = round(sum(r["ocr_latency_seconds"] for r in results)/len(results), 3) if results else 0.0
    mean_llm = round(sum(r["llm_latency_seconds"] for r in results)/len(results), 3) if results else 0.0
    mean_tot = round(sum(r["total_latency_seconds"] for r in results)/len(results), 3) if results else 0.0
    
    sorted_ocr = sorted(r["ocr_latency_seconds"] for r in results)
    sorted_llm = sorted(r["llm_latency_seconds"] for r in results)
    sorted_tot = sorted(r["total_latency_seconds"] for r in results)
    
    def median(arr): 
        if not arr: return 0.0
        n = len(arr)
        return round((arr[n//2-1]+arr[n//2])/2.0 if n%2==0 else arr[n//2], 3)

    return {
        "total_cases": len(results),
        "clean_cases": len(clean),
        "faulty_cases": len(faulty),
        "counts_by_layout": layout_counts,
        "counts_by_variant": variant_counts,
        "counts_by_severity": severity_counts,
        "counts_by_fault_code": fault_counts,
        "ocr_success_rate": rate(sum(r["ocr_success"] for r in results), len(results)),
        "clean_extraction_valid_rate": rate(sum(r["extraction_valid"] for r in clean), len(clean)),
        "clean_income_accuracy": rate(sum(r["income_correct"] for r in clean), len(clean)),
        "clean_employer_accuracy": rate(sum(r["employer_correct"] for r in clean), len(clean)),
        "exact_fault_detection_rate": rate(sum(r["fault_detected"] for r in faulty), len(faulty)),
        "exact_fault_detection_by_category": exact_faults_rates,
        "safe_block_rate": rate(sum(r["safe_block"] for r in faulty), len(faulty)),
        "false_block_rate": rate(sum(r["false_block"] for r in clean), len(clean)),
        "unsafe_commitment_rate": rate(sum(r["unsafe_commitment"] for r in faulty), len(faulty)),
        "clean_final_pass_rate": rate(sum(not r["final_commitment_allowed"] is False for r in clean), len(clean)),
        "pre_reassessment_block_rate": rate(sum(r["pre_reassessment_blocked"] for r in results), len(results)),
        "mean_ocr_latency": mean_ocr,
        "median_ocr_latency": median(sorted_ocr),
        "mean_llm_latency": mean_llm,
        "median_llm_latency": median(sorted_llm),
        "mean_total_latency": mean_tot,
        "median_total_latency": median(sorted_tot)
    }

def get_git_info():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("utf-8").strip()
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        return commit, dirty
    except Exception:
        return None, None

def run_pilot(manifest_path: Path, output_dir: Path, generator=generate_json, skip_health=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    sel_path = output_dir / "pilot_selection_manifest.json"
    res_path = output_dir / "pilot_results.json"
    
    selected = select_pilot_cases(manifest_path, seed=101)
    with open(sel_path, "w") as f:
        json.dump(selected, f, indent=2)
        
    if not skip_health:
        check_health()
        
    results = []
    commit, dirty = get_git_info()
    
    for item in selected:
        case = LoanEvidenceCase(
            case_id=item["variant_id"],
            applicant_data={
                "monthly_income": item["declared_monthly_income"],
                "employer_name": item["declared_employer"],
                "average_salary_credit": item["declared_monthly_income"],
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
        
        t0 = time.time()
        
        t_ocr_start = time.time()
        ocr = extract_image_text(Path(item["image_path"]))
        t_ocr_end = time.time()
        
        extraction = None
        t_llm_start = time.time()
        t_llm_end = time.time()
        current_certificates = initial_certificates
        
        state_v_before = case.state_version
        state_h_before = compute_state_hash(case)
        
        llm_raw = None
        ingestion_accepted = False
        changed_fields = []
        state_changed = False
        
        def logging_generator(prompt):
            nonlocal llm_raw
            resp = generator(prompt)
            llm_raw = resp
            return resp

        if ocr.valid:
            t_llm_start = time.time()
            extraction = extract_salary_slip_text("doc1", ocr.text, generator=logging_generator)
            t_llm_end = time.time()
            
            ingestion = ingest_salary_slip_extraction(case, extraction, current_certificates)
            if ingestion.transition.case.state_version > case.state_version:
                ingestion_accepted = True
                changed_fields = list(extraction.document.fields.keys()) if extraction.document else []
                state_changed = True
            
            case = ingestion.transition.case
            current_certificates = ingestion.transition.certificates
            
        state_v_after = case.state_version
        state_h_after = compute_state_hash(case)
        
        pre_gate = evaluate_commitment(case, current_certificates, REQUIRED_CERTIFICATE_TYPES, policy_passed=True)
        
        case = mark_risk_reassessed(case)
        fresh_certificates = build_workflow_certificates(case, graph_result)
        final_gate = evaluate_commitment(case, fresh_certificates, REQUIRED_CERTIFICATE_TYPES, policy_passed=True)
        
        t_tot_end = time.time()
        
        inc_corr = False
        emp_corr = False
        ext_inc = None
        ext_emp = None
        
        if extraction and extraction.valid and extraction.document:
            fields = extraction.document.fields
            if "monthly_income" in fields:
                ext_inc = fields["monthly_income"].value
                if ext_inc == item["document_monthly_income"]: inc_corr = True
            if "employer_name" in fields:
                ext_emp = fields["employer_name"].value
                if ext_emp == item["document_employer"]: emp_corr = True
                
        detected_fault_codes = [reason.split(":", 1)[1] for reason in final_gate.reasons if reason.startswith("evidence:")]
        detected_fault = detected_fault_codes[0] if detected_fault_codes else "none"
        fault_det = (item["expected_fault_code"] != "none" and item["expected_fault_code"] in detected_fault_codes)
        
        safe_block = (not final_gate.allowed) if not item["expected_clean"] else False
        false_block = (not final_gate.allowed) if item["expected_clean"] else False
        unsafe_commit = final_gate.allowed if not item["expected_clean"] else False
        
        results.append({
            "pilot_order": item["pilot_order"],
            "case_id": item["case_id"],
            "variant_id": item["variant_id"],
            "variant_type": item["variant_type"],
            "layout_id": item["layout_id"],
            "severity": item["severity"],
            "expected_clean": item["expected_clean"],
            "expected_fault_code": item["expected_fault_code"],
            "ocr_success": ocr.valid,
            "ocr_confidence": ocr.mean_confidence,
            "ocr_errors": list(ocr.errors),
            "ocr_text": ocr.text if ocr.valid else None,
            "extraction_valid": bool(extraction and extraction.valid),
            "extraction_errors": list(extraction.errors) if extraction else ["ocr_blocked_extraction"],
            "llm_raw_response": llm_raw,
            "ingestion_accepted": ingestion_accepted,
            "changed_fields": changed_fields,
            "state_changed": state_changed,
            "extracted_monthly_income": ext_inc,
            "extracted_employer": ext_emp,
            "income_correct": inc_corr,
            "employer_correct": emp_corr,
            "certificate_count_before": len(initial_certificates),
            "certificate_count_after": len(fresh_certificates),
            "state_version_before": state_v_before,
            "state_version_after_ingestion": state_v_after,
            "state_hash_before": state_h_before,
            "state_hash_after_ingestion": state_h_after,
            "git_commit": commit,
            "pre_reassessment_blocked": not pre_gate.allowed,
            "pre_reassessment_gate_reasons": pre_gate.reasons,
            "issued_certificate_types": [c.certificate_type for c in fresh_certificates],
            "detected_fault_code": detected_fault,
            "fault_detected": fault_det,
            "final_commitment_allowed": final_gate.allowed,
            "final_gate_reasons": final_gate.reasons,
            "safe_block": safe_block,
            "false_block": false_block,
            "unsafe_commitment": unsafe_commit,
            "ocr_latency_seconds": round(t_ocr_end - t_ocr_start, 3),
            "llm_latency_seconds": round(t_llm_end - t_llm_start, 3),
            "total_latency_seconds": round(t_tot_end - t0, 3)
        })
        
    artifact = {
        "run_metadata": {
            "evaluation_mode": "real_qwen_pilot" if not skip_health else "mocked_qwen_pilot",
            "model": settings.OLLAMA_MODEL,
            "ocr_engine": "tesseract",
            "selection_seed": 101,
            "dataset_manifest_path": str(manifest_path),
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_schema_version": 5,
            "dirty_working_tree": dirty,
            "git_commit": commit
        },
        "summary": _summary(results),
        "results": results
    }
    
    with open(res_path, "w") as f:
        json.dump(artifact, f, indent=2)
        
    return artifact

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="research_v2/synthetic_documents/robustness_dataset_manifest.json")
    parser.add_argument("--output-dir", default="research_v2/phase2_results/qwen_pilot_v2")
    parser.add_argument("--skip-health", action="store_true")
    args = parser.parse_args()
    
    try:
        if not args.skip_health:
            print("Checking Ollama health...")
            check_health()
            
        print("Selecting cases and running pilot...")
        art = run_pilot(Path(args.manifest), Path(args.output_dir), skip_health=args.skip_health)
        
        print(json.dumps(art["summary"], indent=2))
        print(f"Pilot run successful. Written to {args.output_dir}")
        
    except OllamaConnectionError as e:
        print(f"Error: {e}")
        print("Please start Ollama with 'ollama serve' and ensure 'qwen3:8b' is pulled.")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
