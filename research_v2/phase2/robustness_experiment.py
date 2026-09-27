import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np

from research_v2.phase2.synthetic_salary_slips import SalarySlipSpec, create_salary_slip_image, default_specs
from research_v2.phase2.ocr_adapter import extract_image_text
from research_v2.phase2.schemas import LoanEvidenceCase
from research_v2.phase2.commitment_gate import evaluate_commitment
from research_v2.phase2.state_lifecycle import ingest_salary_slip_extraction, mark_risk_reassessed
from research_v2.phase2.llm_document_extractor import extract_salary_slip_text
from research_v2.phase2.workflow_adapter import build_workflow_certificates, REQUIRED_CERTIFICATE_TYPES
from agents.ollama_client import generate_json
from config import settings


def add_noise(image: Image.Image, level: str) -> Image.Image:
    img_array = np.array(image)
    noise_std = {"low": 10, "medium": 30, "high": 50}[level]
    noise = np.random.normal(0, noise_std, img_array.shape).astype(np.int16)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

def generate_robustness_variants(output_dir: Path, manifest_path: Path, dry_run: bool = False):
    random.seed(42)
    np.random.seed(42)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    base_specs = [s for s in default_specs() if s.variant == "clean"]
    if dry_run:
        base_specs = base_specs[:1]

    manifest = []
    
    def save_variant(img: Image.Image, spec: SalarySlipSpec, variant_name: str, expected_clean: bool, fault_type: str = "none"):
        filename = f"{spec.case_id}_{variant_name}.png"
        img_path = output_dir / filename
        img.save(img_path, format="PNG")
        manifest.append({
            "case_id": spec.case_id,
            "variant": variant_name,
            "image_path": str(img_path),
            "expected_clean": expected_clean,
            "fault_type": fault_type,
            "declared_income": spec.declared_monthly_income,
            "declared_employer": spec.declared_employer,
            "document_income": spec.document_monthly_income,
            "document_employer": spec.document_employer,
        })

    for spec in base_specs:
        temp_path = output_dir / f"{spec.case_id}_temp.png"
        create_salary_slip_image(spec, temp_path)
        base_img = Image.open(temp_path)
        
        save_variant(base_img, spec, "clean_baseline", True)

        for angle in [-5, -2, 2, 5]:
            rotated = base_img.rotate(angle, fillcolor="white")
            save_variant(rotated, spec, f"rotation_{angle}", True)
            if dry_run: break
            
        for level, radius in [("low", 1.0), ("medium", 2.5), ("high", 5.0)]:
            blurred = base_img.filter(ImageFilter.GaussianBlur(radius=radius))
            save_variant(blurred, spec, f"blur_{level}", True)
            if dry_run: break
            
        for level in ["low", "medium", "high"]:
            noisy = add_noise(base_img, level)
            save_variant(noisy, spec, f"noise_{level}", True)
            if dry_run: break
            
        enhancer = ImageEnhance.Contrast(base_img)
        save_variant(enhancer.enhance(0.3), spec, "low_contrast", True)
        
        enhancer = ImageEnhance.Brightness(base_img)
        save_variant(enhancer.enhance(0.4), spec, "low_brightness", True)
        
        w, h = base_img.size
        edge_crop = base_img.crop((50, 50, w, h))
        save_variant(edge_crop, spec, "partial_edge_crop", True)
        
        critical_crop = base_img.copy()
        draw = ImageDraw.Draw(critical_crop)
        draw.rectangle((90, 680, w - 90, 780), fill="white")
        save_variant(critical_crop, spec, "critical_field_crop", False, "missing_salary_slip")

        alt_income_spec = SalarySlipSpec(
            case_id=spec.case_id,
            variant="altered_income",
            employee_name=spec.employee_name,
            employee_id=spec.employee_id,
            declared_employer=spec.declared_employer,
            document_employer=spec.document_employer,
            declared_monthly_income=spec.declared_monthly_income,
            document_monthly_income=999999.0
        )
        create_salary_slip_image(alt_income_spec, temp_path)
        save_variant(Image.open(temp_path), alt_income_spec, "altered_income", False, "income_contradiction")

        alt_emp_spec = SalarySlipSpec(
            case_id=spec.case_id,
            variant="altered_employer",
            employee_name=spec.employee_name,
            employee_id=spec.employee_id,
            declared_employer=spec.declared_employer,
            document_employer="Fake Company Inc",
            declared_monthly_income=spec.declared_monthly_income,
            document_monthly_income=spec.document_monthly_income
        )
        create_salary_slip_image(alt_emp_spec, temp_path)
        save_variant(Image.open(temp_path), alt_emp_spec, "altered_employer", False, "employer_contradiction")
        
        temp_path.unlink(missing_ok=True)
        if dry_run: break

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    return manifest

def _default_result_path(dry_run: bool, evaluation_mode: str) -> Path:
    scope = "dryrun" if dry_run else "full"
    safe_mode = evaluation_mode.replace(" ", "_").lower()
    return Path(
        f"research_v2/synthetic_documents/robustness_results/"
        f"robustness_results_{scope}_{safe_mode}_v3.json"
    )


def _summary(results: List[Dict[str, object]]) -> Dict[str, object]:
    clean = [row for row in results if row["expected_clean"]]
    faulty = [row for row in results if not row["expected_clean"]]

    def rate(numerator: int, denominator: int) -> float:
        return round(numerator / denominator, 4) if denominator else 0.0

    return {
        "total_cases": len(results),
        "clean_cases": len(clean),
        "faulty_cases": len(faulty),
        "ocr_success_rate": rate(sum(bool(row["ocr_success"]) for row in results), len(results)),
        "all_case_income_field_match_rate": rate(
            sum(bool(row["income_extraction_accuracy"]) for row in results), len(results)
        ),
        "all_case_employer_field_match_rate": rate(
            sum(bool(row["employer_extraction_accuracy"]) for row in results), len(results)
        ),
        "clean_extraction_valid_rate": rate(
            sum(bool(row["extraction_valid"]) for row in clean), len(clean)
        ),
        "clean_income_extraction_accuracy": rate(
            sum(bool(row["income_extraction_accuracy"]) for row in clean), len(clean)
        ),
        "clean_employer_extraction_accuracy": rate(
            sum(bool(row["employer_extraction_accuracy"]) for row in clean), len(clean)
        ),
        "pre_reassessment_block_rate": rate(
            sum(bool(row["pre_reassessment_block"]) for row in results), len(results)
        ),
        "clean_final_pass_rate": rate(
            sum(bool(row["clean_final_pass"]) for row in clean), len(clean)
        ),
        "false_block_rate": rate(
            sum(bool(row["false_blocking"]) for row in clean), len(clean)
        ),
        "fault_detection_rate": rate(
            sum(bool(row["fault_detected"]) for row in faulty), len(faulty)
        ),
        "safe_block_rate": rate(
            sum(bool(row["safe_blocking"]) for row in faulty), len(faulty)
        ),
        "mean_ocr_latency_seconds": round(
            sum(float(row["ocr_latency"]) for row in results) / len(results), 3
        ),
        "mean_llm_latency_seconds": round(
            sum(float(row["llm_latency"]) for row in results) / len(results), 3
        ),
        "mean_total_latency_seconds": round(
            sum(float(row["total_latency"]) for row in results) / len(results), 3
        ),
    }


def run_experiment(
    dry_run: bool = True,
    generator: Callable[[str], dict] = generate_json,
    output_images_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    results_path: Optional[Path] = None,
    evaluation_mode: str = "real_qwen",
):
    output_images_dir = output_images_dir or Path(
        "output/images/synthetic_salary_slips/robustness"
    )
    manifest_path = manifest_path or Path(
        "research_v2/synthetic_documents/robustness_manifest.json"
    )
    results_path = results_path or _default_result_path(dry_run, evaluation_mode)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    manifest = generate_robustness_variants(output_images_dir, manifest_path, dry_run=dry_run)
    
    results = []
    
    for item in manifest:
        case_id = item["case_id"]
        variant = item["variant"]
        img_path = Path(item["image_path"])
        expected_clean = item["expected_clean"]
        expected_fault = item.get("fault_type", "none")
        
        case = LoanEvidenceCase(
            case_id=f"{case_id}_{variant}",
            applicant_data={
                "monthly_income": item["declared_income"],
                "employer_name": item["declared_employer"],
                "average_salary_credit": item["declared_income"],
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
        ocr = extract_image_text(img_path)
        t_ocr_end = time.time()
        ocr_latency = t_ocr_end - t_ocr_start
        
        llm_latency = 0.0
        extraction = None
        current_certificates = initial_certificates
        
        if ocr.valid:
            t_llm_start = time.time()
            extraction = extract_salary_slip_text("doc1", ocr.text, generator=generator)
            t_llm_end = time.time()
            llm_latency = t_llm_end - t_llm_start
            
            ingestion = ingest_salary_slip_extraction(case, extraction, current_certificates)
            case = ingestion.transition.case
            current_certificates = ingestion.transition.certificates
            
        pre_reassessment_gate = evaluate_commitment(case, current_certificates, REQUIRED_CERTIFICATE_TYPES, policy_passed=True)
        pre_reassessment_blocked = not pre_reassessment_gate.allowed
        
        case = mark_risk_reassessed(case)
        fresh_certificates = build_workflow_certificates(case, graph_result)
        final_gate = evaluate_commitment(case, fresh_certificates, REQUIRED_CERTIFICATE_TYPES, policy_passed=True)
        final_blocked = not final_gate.allowed
        
        t_total_end = time.time()
        total_latency = t_total_end - t0
        
        ocr_success = ocr.valid
        ocr_confidence = ocr.mean_confidence
        
        income_acc = False
        employer_acc = False
        if extraction and extraction.valid and extraction.document:
            fields = extraction.document.fields
            if "monthly_income" in fields and fields["monthly_income"].value == item["document_income"]:
                income_acc = True
            if "employer_name" in fields and fields["employer_name"].value == item["document_employer"]:
                employer_acc = True
                
        detected_fault_codes = [
            reason.split(":", 1)[1]
            for reason in final_gate.reasons
            if reason.startswith("evidence:")
        ]
        detected_fault_code = detected_fault_codes[0] if detected_fault_codes else "none"
        fault_detected = (
            expected_fault != "none" and expected_fault in detected_fault_codes
        )
            
        safe_blocking = final_blocked if not expected_clean else False
        false_blocking = final_blocked if expected_clean else False
        clean_final_pass = not final_blocked if expected_clean else False
        
        results.append({
            "case_id": case_id,
            "variant": variant,
            "expected_clean": expected_clean,
            "expected_fault_code": expected_fault,
            "ocr_success": ocr_success,
            "ocr_confidence": ocr_confidence,
            "ocr_errors": list(ocr.errors),
            "extraction_valid": bool(extraction and extraction.valid),
            "extraction_errors": list(extraction.errors) if extraction else ["ocr_blocked_extraction"],
            "income_extraction_accuracy": income_acc,
            "employer_extraction_accuracy": employer_acc,
            "pre_reassessment_block": pre_reassessment_blocked,
            "final_commitment_block": final_blocked,
            "detected_fault_code": detected_fault_code,
            "fault_detected": fault_detected,
            "safe_blocking": safe_blocking,
            "false_blocking": false_blocking,
            "clean_final_pass": clean_final_pass,
            "issued_certificate_types": [c.certificate_type for c in fresh_certificates],
            "ocr_latency": round(ocr_latency, 3),
            "llm_latency": round(llm_latency, 3),
            "total_latency": round(total_latency, 3),
            "pre_reassessment_gate_reasons": pre_reassessment_gate.reasons,
            "gate_reasons": final_gate.reasons
        })

    artifact = {
        "run_metadata": {
            "evaluation_mode": evaluation_mode,
            "model": settings.OLLAMA_MODEL if evaluation_mode == "real_qwen" else "mocked_generator",
            "ocr_engine": "tesseract",
            "random_seed": 42,
            "dry_run": dry_run,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "result_schema_version": 3,
        },
        "summary": _summary(results),
        "results": results,
    }
    with open(results_path, "w") as handle:
        json.dump(artifact, handle, indent=2)

    return artifact

if __name__ == "__main__":
    import sys
    dry_run = "--full" not in sys.argv
    print(f"Running robustness experiment (dry_run={dry_run})...")
    artifact = run_experiment(dry_run=dry_run)
    print(json.dumps(artifact["summary"], indent=2))
