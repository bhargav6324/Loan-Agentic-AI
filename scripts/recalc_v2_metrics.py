import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from research_v2.phase2.evidence_validator import normalize_entity_name

def main():
    pilot_dir = Path("research_v2/phase2_results/qwen_pilot_v2")
    results_path = pilot_dir / "pilot_results.json"
    manifest_path = pilot_dir / "pilot_selection_manifest.json"
    out_path = pilot_dir / "corrected_metric_analysis.json"
    
    with open(results_path) as f:
        pilot_data = json.load(f)
        
    with open(manifest_path) as f:
        manifest_data = json.load(f)
        
    manifest_map = {item["variant_id"]: item for item in manifest_data}
    
    clean_cases = [r for r in pilot_data["results"] if r["expected_clean"]]
    
    for r in pilot_data["results"]:
        variant = manifest_map[r["variant_id"]]
        expected_emp = variant["document_employer"]
        ext_emp = r.get("extracted_employer")
        
        r["document_employer"] = expected_emp
        r["employer_correct_raw"] = r["employer_correct"]
        
        if ext_emp is not None and expected_emp is not None:
            r["employer_correct_normalized"] = normalize_entity_name(ext_emp) == normalize_entity_name(expected_emp)
        else:
            r["employer_correct_normalized"] = False
            
    corrected_clean_employer_accuracy = sum(1 for r in clean_cases if r["employer_correct_normalized"]) / len(clean_cases) if clean_cases else 0.0
    
    analysis = {
        "note": "The original V2 execution is unchanged. Only metric interpretation (employer accuracy) was corrected using normalize_entity_name.",
        "original_clean_employer_accuracy": pilot_data["summary"]["clean_employer_accuracy"],
        "corrected_clean_employer_accuracy": round(corrected_clean_employer_accuracy, 4),
        "results_breakdown": []
    }
    
    for r in pilot_data["results"]:
        analysis["results_breakdown"].append({
            "case_id": r["case_id"],
            "variant_id": r["variant_id"],
            "layout_id": r["layout_id"],
            "extracted_employer": r.get("extracted_employer"),
            "document_employer": r["document_employer"],
            "employer_correct_raw": r["employer_correct_raw"],
            "employer_correct_normalized": r["employer_correct_normalized"]
        })
        
    with open(out_path, "w") as f:
        json.dump(analysis, f, indent=2)
        
    print(f"Corrected metric analysis written to {out_path}")
    print(f"Corrected clean employer accuracy: {corrected_clean_employer_accuracy}")

if __name__ == "__main__":
    main()
