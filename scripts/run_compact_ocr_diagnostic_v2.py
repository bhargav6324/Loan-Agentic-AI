import json
import time
import subprocess
import csv
import io
import re
import tempfile
from pathlib import Path
import statistics
from PIL import Image, ImageEnhance

import sys
sys.path.append(str(Path(__file__).parent.parent))
from research_v2.phase2.synthetic_salary_slips import get_critical_field_region
from research_v2.phase2.ocr_adapter import _parse_tsv
from research_v2.phase2.evidence_validator import normalize_entity_name

def run_tesseract(image_path: str, psm: str = "3") -> tuple[str, float, float]:
    t0 = time.time()
    completed = subprocess.run(
        ["tesseract", image_path, "stdout", "--psm", psm, "tsv"],
        capture_output=True,
        text=True,
        errors="replace",
        check=False
    )
    t1 = time.time()
    
    if completed.returncode != 0:
        return "", 0.0, t1 - t0
        
    rows = _parse_tsv(completed.stdout)
    line_order = []
    lines = {}
    for line_key, word, confidence in rows:
        if line_key not in lines:
            lines[line_key] = []
            line_order.append(line_key)
        lines[line_key].append(word)
    text = "\n".join(" ".join(lines[key]) for key in line_order)
    
    confidences = [confidence for _, _, confidence in rows]
    mean_confidence = round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    
    return text, mean_confidence, t1 - t0

def extract_salary_candidates(ocr_text: str) -> list[float]:
    candidates = []
    text = ocr_text.lower()
    
    pattern1 = r'(?:inr|rs\.?|₹)\s*([\d,]+(?:\.\d{2})?)'
    for m in re.findall(pattern1, text):
        try:
            val = float(m.replace(',', ''))
            if val > 1000 and val != 2026:
                candidates.append(val)
        except ValueError:
            pass
            
    pattern2 = r'net\s*pay(?:[\s:]*(?:inr|rs\.?|₹))?\s*[:\-]?\s*([\d,]+(?:\.\d{2})?)'
    for m in re.findall(pattern2, text):
        try:
            val = float(m.replace(',', ''))
            if val > 1000 and val != 2026:
                candidates.append(val)
        except ValueError:
            pass
            
    pattern3 = r'([\d,]+(?:\.\d{2})?)\s*(?:inr|rs\.?|₹)'
    for m in re.findall(pattern3, text):
        try:
            val = float(m.replace(',', ''))
            if val > 1000 and val != 2026:
                candidates.append(val)
        except ValueError:
            pass
            
    return sorted(list(set(candidates)))

def contains_salary_evidence(ocr_text: str) -> bool:
    return len(extract_salary_candidates(ocr_text)) > 0

def salary_matches_expected(ocr_text: str, expected_income: float) -> bool:
    if expected_income is None:
        return False
    return expected_income in extract_salary_candidates(ocr_text)

def main():
    manifest_path = Path("research_v2/synthetic_documents/robustness_dataset_manifest.json")
    out_dir = Path("research_v2/phase2_results/compact_ocr_diagnostic_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    compact_cases = [c for c in manifest if c["layout_id"] == "compact"]
    
    methods = [
        {"name": "default", "preprocess": lambda img: img, "psm": "3"},
        {"name": "grayscale_default", "preprocess": lambda img: img.convert('L'), "psm": "3"},
        {"name": "grayscale_psm6", "preprocess": lambda img: img.convert('L'), "psm": "6"},
        {"name": "grayscale_psm4", "preprocess": lambda img: img.convert('L'), "psm": "4"},
        {"name": "grayscale_psm11", "preprocess": lambda img: img.convert('L'), "psm": "11"},
        {"name": "grayscale_contrast", "preprocess": lambda img: ImageEnhance.Contrast(img.convert('L')).enhance(2.0), "psm": "6"},
        {"name": "roi_crop", "preprocess": lambda img: img.crop(get_critical_field_region("compact")), "psm": "6"}
    ]
    
    results = []
    
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        
        for case in compact_cases:
            orig_img_path = Path(case["image_path"])
            try:
                img = Image.open(orig_img_path)
            except Exception as e:
                print(f"Skipping {orig_img_path}: {e}")
                continue
                
            expected_income = case["document_monthly_income"]
            expected_emp = case.get("document_employer", "")
            
            for m in methods:
                proc_img = m["preprocess"](img.copy())
                t_file = tmp_path / f"img_{m['name']}.png"
                proc_img.save(t_file)
                
                latencies = []
                best_text = ""
                best_conf = 0.0
                
                for _ in range(3):
                    text, conf, lat = run_tesseract(str(t_file), psm=m["psm"])
                    latencies.append(lat)
                    if conf > best_conf or not best_text:
                        best_text = text
                        best_conf = conf
                        
                median_lat = statistics.median(latencies)
                
                salary_found = salary_matches_expected(best_text, expected_income)
                
                # Check for false recovery on critical crop (when expected_income is None, should not find ANY salary)
                has_salary_evidence = contains_salary_evidence(best_text)
                
                emp_found = False
                if expected_emp:
                    norm_exp = normalize_entity_name(expected_emp).lower()
                    if norm_exp in best_text.lower():
                        emp_found = True
                    # if expected_emp string is directly present case insensitive
                    elif expected_emp.lower() in best_text.lower():
                        emp_found = True
                        
                results.append({
                    "variant_id": case["variant_id"],
                    "variant_type": case["variant_type"],
                    "severity": case["severity"],
                    "applicant_id": case["applicant_id"],
                    "expected_fault_code": case["expected_fault_code"],
                    "expected_salary": expected_income,
                    "preprocessing_method": m["name"],
                    "tesseract_psm": m["psm"],
                    "ocr_text": best_text,
                    "net_pay_found": "net pay" in best_text.lower(),
                    "has_salary_evidence": has_salary_evidence,
                    "salary_matches_expected": salary_found,
                    "employer_found": emp_found,
                    "confidence": best_conf,
                    "median_latency_seconds": median_lat
                })

    with open(out_dir / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    method_stats = {}
    failures = []
    
    for m in methods:
        m_name = m["name"]
        m_results = [r for r in results if r["preprocessing_method"] == m_name]
        
        crop_cases = [r for r in m_results if r["variant_type"] == "critical_field_crop"]
        readable_cases = [r for r in m_results if r["variant_type"] != "critical_field_crop"]
        
        crop_false_rec = sum(1 for r in crop_cases if r["has_salary_evidence"]) / len(crop_cases) if crop_cases else 0.0
        
        salary_rec = sum(1 for r in readable_cases if r["has_salary_evidence"]) / len(readable_cases) if readable_cases else 0.0
        salary_acc = sum(1 for r in readable_cases if r["salary_matches_expected"]) / len(readable_cases) if readable_cases else 0.0
        
        emp_rec = sum(1 for r in readable_cases if r["employer_found"]) / len(readable_cases) if readable_cases else 0.0
        mean_lat = sum(r["median_latency_seconds"] for r in m_results) / len(m_results) if m_results else 0.0
        
        method_stats[m_name] = {
            "false_recovery_rate": crop_false_rec,
            "salary_recovery_rate": salary_rec,
            "exact_expected_accuracy": salary_acc,
            "employer_recovery_rate": emp_rec,
            "median_latency_seconds": statistics.median([r["median_latency_seconds"] for r in m_results]) if m_results else 0.0
        }
        
        # Track failures
        for r in m_results:
            failed = False
            if r["variant_type"] == "critical_field_crop" and r["has_salary_evidence"]:
                failed = True
            elif r["variant_type"] != "critical_field_crop" and not r["salary_matches_expected"]:
                failed = True
            elif r["variant_type"] != "critical_field_crop" and not r["employer_found"]:
                failed = True
                
            if failed:
                failures.append({
                    "method": m_name,
                    "variant_id": r["variant_id"],
                    "variant_type": r["variant_type"],
                    "reason": "false_recovery" if r["variant_type"] == "critical_field_crop" else "missing_evidence"
                })
                
    with open(out_dir / "method_comparison.json", "w") as f:
        json.dump(method_stats, f, indent=2)
        
    with open(out_dir / "failures.json", "w") as f:
        json.dump(failures, f, indent=2)
        
    # Rank methods
    ranked = list(method_stats.items())
    ranked.sort(key=lambda x: (
        x[1]["false_recovery_rate"] == 0, # prioritize 0 false recovery
        x[1]["salary_recovery_rate"],
        x[1]["exact_expected_accuracy"],
        x[1]["employer_recovery_rate"],
        # prioritize simplicity (default > psm6 > psm4 > psm11) by checking method name in list
        -methods.index(next(m for m in methods if m["name"] == x[0])),
        -x[1]["median_latency_seconds"]
    ), reverse=True)
    
    recommended = ranked[0][0]
    
    with open(out_dir / "recommended_method.json", "w") as f:
        json.dump({
            "recommended_method": recommended,
            "justification": f"Ranked #1 with false recovery={method_stats[recommended]['false_recovery_rate']}, salary_recovery={method_stats[recommended]['salary_recovery_rate']}"
        }, f, indent=2)

if __name__ == "__main__":
    main()
