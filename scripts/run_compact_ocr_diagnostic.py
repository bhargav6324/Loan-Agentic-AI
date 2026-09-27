import json
import time
import subprocess
import csv
import io
import shutil
from pathlib import Path
from PIL import Image, ImageEnhance

import sys
sys.path.append(str(Path(__file__).parent.parent))
from research_v2.phase2.synthetic_salary_slips import get_critical_field_region
from research_v2.phase2.ocr_adapter import _parse_tsv

def run_tesseract(image_path: Path, psm: str = "6") -> tuple[str, float, float]:
    t0 = time.time()
    completed = subprocess.run(
        ["tesseract", str(image_path), "stdout", "--psm", psm, "tsv"],
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

def select_diagnostic_sample(manifest_path: Path):
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    compact_cases = [c for c in manifest if c["layout_id"] == "compact"]
    
    categories = [
        "clean_baseline", "rotation", "blur", "noise", "low_contrast", 
        "low_brightness", "partial_edge_crop", "altered_income", 
        "altered_employer", "critical_field_crop"
    ]
    
    sample = []
    selected_cats = set()
    
    for cat in categories:
        for c in compact_cases:
            if c["variant_type"] == cat and cat not in selected_cats:
                sample.append(c)
                selected_cats.add(cat)
                break
                
    if len(selected_cats) != len(categories):
        missing = set(categories) - selected_cats
        raise ValueError(f"Missing categories in compact cases: {missing}")
        
    return sample

def main():
    manifest_path = Path("research_v2/synthetic_documents/robustness_dataset_manifest.json")
    out_dir = Path("research_v2/phase2_results/compact_ocr_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    sample = select_diagnostic_sample(manifest_path)
    
    tmp_dir = Path("research_v2/phase2_results/compact_ocr_diagnostic/tmp")
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    
    methods = [
        {"name": "original", "preprocess": lambda img: img, "psm": "6"},
        {"name": "grayscale_psm6", "preprocess": lambda img: img.convert('L'), "psm": "6"},
        {"name": "grayscale_psm4", "preprocess": lambda img: img.convert('L'), "psm": "4"},
        {"name": "grayscale_psm11", "preprocess": lambda img: img.convert('L'), "psm": "11"},
        {"name": "contrast_psm6", "preprocess": lambda img: ImageEnhance.Contrast(img).enhance(2.0), "psm": "6"},
        {"name": "upscale_psm6", "preprocess": lambda img: img.resize((img.width * 2, img.height * 2), Image.Resampling.LANCZOS), "psm": "6"},
        {"name": "roi_crop_psm6", "preprocess": lambda img: img.crop(get_critical_field_region("compact")), "psm": "6"}
    ]
    
    results = []
    
    for case in sample:
        orig_img_path = Path(case["image_path"])
        img = Image.open(orig_img_path)
        
        expected_income = case["document_monthly_income"]
        income_str1 = f"{expected_income:,.2f}" if expected_income else "N/A"
        income_str2 = f"{expected_income}" if expected_income else "N/A"
        
        expected_emp = str(case.get("document_employer", "")).lower()
        
        for m in methods:
            proc_img = m["preprocess"](img.copy())
            tmp_path = tmp_dir / f"{case['variant_id']}_{m['name']}.png"
            proc_img.save(tmp_path)
            
            text, conf, lat = run_tesseract(tmp_path, psm=m["psm"])
            
            text_lower = text.lower()
            
            net_pay_found = "net pay" in text_lower
            
            if expected_income is None:
                income_found = False
            else:
                income_found = income_str1 in text or income_str2 in text
                
            emp_found = expected_emp in text_lower if expected_emp else False
            
            results.append({
                "variant_id": case["variant_id"],
                "variant_type": case["variant_type"],
                "expected_salary": expected_income,
                "preprocessing_method": m["name"],
                "tesseract_psm": m["psm"],
                "ocr_text": text,
                "net_pay_found": net_pay_found,
                "salary_found": income_found,
                "employer_found": emp_found,
                "confidence": conf,
                "latency_seconds": lat
            })
            
    with open(out_dir / "diagnostic_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    method_stats = {}
    for m in methods:
        m_name = m["name"]
        m_results = [r for r in results if r["preprocessing_method"] == m_name]
        
        readable_cases = [r for r in m_results if r["variant_type"] != "critical_field_crop"]
        crop_cases = [r for r in m_results if r["variant_type"] == "critical_field_crop"]
        
        salary_rec = sum(1 for r in readable_cases if r["salary_found"]) / len(readable_cases) if readable_cases else 0.0
        emp_rec = sum(1 for r in readable_cases if r["employer_found"]) / len(readable_cases) if readable_cases else 0.0
        crop_false_rec = sum(1 for r in crop_cases if r["salary_found"]) / len(crop_cases) if crop_cases else 0.0
        mean_lat = sum(r["latency_seconds"] for r in m_results) / len(m_results) if m_results else 0.0
        
        method_stats[m_name] = {
            "salary_recovery_rate": round(salary_rec, 4),
            "employer_recovery_rate": round(emp_rec, 4),
            "critical_crop_false_recovery_rate": round(crop_false_rec, 4),
            "mean_latency_seconds": round(mean_lat, 4)
        }
        
    with open(out_dir / "method_comparison.json", "w") as f:
        json.dump(method_stats, f, indent=2)
        
    # Recommended method: highest salary recovery that doesn't hallucinate (false recovery = 0)
    valid_methods = [(name, stats) for name, stats in method_stats.items() if stats["critical_crop_false_recovery_rate"] == 0]
    
    # Sort by salary recovery desc, employer recovery desc, latency asc
    valid_methods.sort(key=lambda x: (x[1]["salary_recovery_rate"], x[1]["employer_recovery_rate"], -x[1]["mean_latency_seconds"]), reverse=True)
    
    if valid_methods:
        recommended = valid_methods[0][0]
    else:
        recommended = "original"
        
    with open(out_dir / "recommended_method.json", "w") as f:
        json.dump({
            "recommended_method": recommended,
            "justification": f"Selected because it achieved {method_stats[recommended]['salary_recovery_rate']*100}% salary recovery without any hallucinations on critically cropped documents."
        }, f, indent=2)
        
    print(f"Diagnostic finished. Recommended method: {recommended}")
    for name, stats in method_stats.items():
        print(f"{name}: Salary Rec: {stats['salary_recovery_rate']}, Emp Rec: {stats['employer_recovery_rate']}, False Rec: {stats['critical_crop_false_recovery_rate']}, Latency: {stats['mean_latency_seconds']}")

if __name__ == "__main__":
    main()
