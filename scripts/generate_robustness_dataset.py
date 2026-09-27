import json
import random
import argparse
from pathlib import Path
from PIL import Image, ImageEnhance, ImageFilter, ImageDraw
import numpy as np
import sys

# Ensure project root is in path if running as script
sys.path.append(str(Path(__file__).parent.parent))

from research_v2.phase2.synthetic_salary_slips import SalarySlipSpec, create_salary_slip_image

def add_noise(image: Image.Image, level: str) -> Image.Image:
    img_array = np.array(image)
    noise_std = {"low": 10, "medium": 30, "high": 50}[level]
    noise = np.random.normal(0, noise_std, img_array.shape).astype(np.int16)
    noisy_img = np.clip(img_array.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy_img)

from typing import Optional

def generate_dataset(
    output_dir: Path,
    manifest_path: Path,
    seed: int = 42,
    size: int = 200,
    inspection_dir: Optional[Path] = None,
):
    if size <= 0 or size % 10 != 0:
        raise ValueError("size must be a positive multiple of 10")

    random.seed(seed)
    np.random.seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for stale_image in output_dir.glob("case-*.png"):
        stale_image.unlink()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    
    samples_dir = inspection_dir or Path(
        "output/images/synthetic_salary_slips/inspection_samples"
    )
    samples_dir.mkdir(parents=True, exist_ok=True)
    samples_index = []
    
    manifest = []
    
    first_names = ["Asha", "Rohan", "Neha", "Kabir", "Mira", "Arjun", "Sara", "Raj", "Priya", "Amit"]
    last_names = ["Rao", "Mehta", "Shah", "Singh", "Joshi", "Nair", "Khan", "Patel", "Sharma", "Kumar"]
    employers = ["Example Pvt Ltd", "Acme Services Ltd", "GlobalTech Solutions", "Innovatech Corp", "Nexus Industries"]
    layouts = ["standard", "compact", "tabular"]
    
    base_cases = []
    num_bases = max(1, size // 10)
    for i in range(num_bases):
        name = f"{random.choice(first_names)} {random.choice(last_names)}"
        emp = random.choice(employers)
        income = random.choice([50000.0, 62500.0, 45000.0, 120000.0, 35000.0, 85000.0])
        layout = layouts[i % 3]
        base_cases.append({
            "applicant_id": f"APP-{1000+i}",
            "case_id": f"case-{1000+i}",
            "employee_name": name,
            "employer": emp,
            "income": income,
            "layout_id": layout
        })

    variant_types = [
        ("clean_baseline", "none"),
        ("rotation", "none"),
        ("blur", "none"),
        ("noise", "none"),
        ("low_contrast", "none"),
        ("low_brightness", "none"),
        ("partial_edge_crop", "none"),
        ("critical_field_crop", "missing_salary_slip"),
        ("altered_income", "income_contradiction"),
        ("altered_employer", "employer_contradiction"),
    ]

    from research_v2.phase2.synthetic_salary_slips import get_critical_field_region
    
    samples_collected = {layout: [] for layout in layouts}

    def save_variant(img: Image.Image, base: dict, v_type: str, severity: str, fault: str, doc_inc, doc_emp, variant_id: str,
                     rot_deg: Optional[float], blur_rad: Optional[float], noise_lvl: Optional[str]):
        expected_clean = (fault == "none")
        filename = f"{variant_id}.png"
        img_path = output_dir / filename
        img.save(img_path, format="PNG")
        manifest.append({
            "case_id": base["case_id"],
            "applicant_id": base["applicant_id"],
            "variant_id": variant_id,
            "variant_type": v_type,
            "image_path": str(img_path),
            "random_seed": seed,
            "expected_clean": expected_clean,
            "expected_fault_code": fault,
            "declared_monthly_income": base["income"],
            "document_monthly_income": doc_inc,
            "declared_employer": base["employer"],
            "document_employer": doc_emp,
            "layout_id": base["layout_id"],
            "severity": severity,
            "rotation_degrees": rot_deg,
            "blur_radius": blur_rad,
            "noise_level": noise_lvl,
            "critical_field_region": get_critical_field_region(base["layout_id"])
        })
        
        layout = base["layout_id"]
        sample_types = ["clean_baseline", "noise", "critical_field_crop", "altered_income", "altered_employer"]
        if v_type in sample_types and v_type not in samples_collected[layout]:
            samples_collected[layout].append(v_type)
            sample_filename = f"{layout}_{v_type}.png"
            sample_path = samples_dir / sample_filename
            img.save(sample_path, format="PNG")
            samples_index.append({
                "layout_id": layout,
                "variant_type": v_type,
                "severity": severity,
                "expected_fault_code": fault,
                "source_image": str(img_path),
                "inspection_image": str(sample_path)
            })

    for base in base_cases:
        spec = SalarySlipSpec(
            case_id=base["case_id"],
            variant="base",
            employee_name=base["employee_name"],
            employee_id=base["applicant_id"],
            declared_employer=base["employer"],
            document_employer=base["employer"],
            declared_monthly_income=base["income"],
            document_monthly_income=base["income"],
            layout_id=base["layout_id"]
        )
        temp_path = output_dir / f"{base['case_id']}_temp.png"
        create_salary_slip_image(spec, temp_path)
        base_img = Image.open(temp_path)
        
        for v_type, fault in variant_types:
            variant_id = f"{base['case_id']}_{v_type}"
            severity = "none"
            doc_inc = base["income"]
            doc_emp = base["employer"]
            rot_deg = None
            blur_rad = None
            noise_lvl = None
            
            if v_type == "clean_baseline":
                img = base_img.copy()
            elif v_type == "rotation":
                rot_deg = random.choice([-5, -2, 2, 5])
                severity = "low" if abs(rot_deg) <= 2 else "medium"
                img = base_img.rotate(rot_deg, fillcolor="white")
            elif v_type == "blur":
                severity, blur_rad = random.choice([("low", 1.0), ("medium", 2.5), ("high", 5.0)])
                img = base_img.filter(ImageFilter.GaussianBlur(radius=blur_rad))
            elif v_type == "noise":
                severity = random.choice(["low", "medium", "high"])
                noise_lvl = severity
                img = add_noise(base_img, severity)
            elif v_type == "low_contrast":
                severity = "medium"
                img = ImageEnhance.Contrast(base_img).enhance(0.3)
            elif v_type == "low_brightness":
                severity = "medium"
                img = ImageEnhance.Brightness(base_img).enhance(0.4)
            elif v_type == "partial_edge_crop":
                severity = "medium"
                w, h = base_img.size
                img = base_img.crop((50, 50, w, h))
            elif v_type == "critical_field_crop":
                severity = "high"
                img = base_img.copy()
                draw = ImageDraw.Draw(img)
                box = get_critical_field_region(base["layout_id"])
                draw.rectangle(box, fill="white")
                doc_inc = None
            elif v_type == "altered_income":
                severity = "high"
                doc_inc = 999999.0
                alt_spec = SalarySlipSpec(
                    case_id=base["case_id"],
                    variant=v_type,
                    employee_name=base["employee_name"],
                    employee_id=base["applicant_id"],
                    declared_employer=base["employer"],
                    document_employer=base["employer"],
                    declared_monthly_income=base["income"],
                    document_monthly_income=doc_inc,
                    layout_id=base["layout_id"]
                )
                create_salary_slip_image(alt_spec, temp_path)
                img = Image.open(temp_path).copy()
            elif v_type == "altered_employer":
                severity = "high"
                doc_emp = "Fake Company Inc"
                alt_spec = SalarySlipSpec(
                    case_id=base["case_id"],
                    variant=v_type,
                    employee_name=base["employee_name"],
                    employee_id=base["applicant_id"],
                    declared_employer=base["employer"],
                    document_employer=doc_emp,
                    declared_monthly_income=base["income"],
                    document_monthly_income=base["income"],
                    layout_id=base["layout_id"]
                )
                create_salary_slip_image(alt_spec, temp_path)
                img = Image.open(temp_path).copy()
            
            save_variant(img, base, v_type, severity, fault, doc_inc, doc_emp, variant_id, rot_deg, blur_rad, noise_lvl)
            
        temp_path.unlink(missing_ok=True)

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
        
    with open(samples_dir / "index.json", "w") as f:
        json.dump(samples_index, f, indent=2)

    return manifest

def validate_dataset(manifest_path: Path):
    with open(manifest_path) as f:
        manifest = json.load(f)
        
    errors = []
    total = len(manifest)
    clean = sum(1 for x in manifest if x["expected_clean"])
    faulty = total - clean
    
    variant_counts = {}
    severity_counts = {}
    fault_counts = {}
    layout_counts = {}
    
    ids = set()
    
    for row in manifest:
        v_type = row["variant_type"]
        sev = row["severity"]
        fault = row["expected_fault_code"]
        layout = row["layout_id"]
        
        variant_counts[v_type] = variant_counts.get(v_type, 0) + 1
        severity_counts[sev] = severity_counts.get(sev, 0) + 1
        fault_counts[fault] = fault_counts.get(fault, 0) + 1
        layout_counts[layout] = layout_counts.get(layout, 0) + 1
        
        if sev not in ["none", "low", "medium", "high"]:
            errors.append(f"Invalid severity {sev} for variant {row['variant_id']}")
            
        if "rotation_degrees" not in row or "blur_radius" not in row or "noise_level" not in row:
            errors.append(f"Missing numeric metadata for variant {row['variant_id']}")
            
        if "critical_field_region" not in row:
            errors.append(f"Missing critical_field_region for variant {row['variant_id']}")
        
        if row["variant_id"] in ids:
            errors.append(f"Duplicate variant_id: {row['variant_id']}")
        ids.add(row["variant_id"])
        
        if not Path(row["image_path"]).exists():
            errors.append(f"Missing image: {row['image_path']}")
            
        if row["expected_clean"]:
            if row["declared_monthly_income"] != row["document_monthly_income"]:
                errors.append(f"Clean case has income mismatch: {row['variant_id']}")
            if row["declared_employer"] != row["document_employer"]:
                errors.append(f"Clean case has employer mismatch: {row['variant_id']}")
        else:
            if row["expected_fault_code"] == "income_contradiction":
                if row["declared_monthly_income"] == row["document_monthly_income"]:
                    errors.append(f"Income contradiction case has matching income: {row['variant_id']}")
            elif row["expected_fault_code"] == "employer_contradiction":
                if row["declared_employer"] == row["document_employer"]:
                    errors.append(f"Employer contradiction case has matching employer: {row['variant_id']}")
            elif row["expected_fault_code"] == "missing_salary_slip":
                if row["document_monthly_income"] is not None:
                    errors.append(f"Missing salary slip case has income: {row['variant_id']}")

    print("--- Dataset Validation Report ---")
    print(f"Total documents: {total}")
    print(f"Clean/readable count: {clean}")
    print(f"Faulty count: {faulty}")
    
    print("\nCounts by layout:")
    for k, v in layout_counts.items(): print(f"  {k}: {v}")
    
    print("\nCounts by variant:")
    for k, v in variant_counts.items(): print(f"  {k}: {v}")
    
    print("\nCounts by severity:")
    for k, v in severity_counts.items(): print(f"  {k}: {v}")
    
    print("\nCounts by expected fault code:")
    for k, v in fault_counts.items(): print(f"  {k}: {v}")
    
    if len(layout_counts) < 3:
        errors.append("Expected at least 3 distinct layouts")
    
    if errors:
        print("\nValidation errors:")
        for e in errors: print(f"  - {e}")
        return False
    else:
        print("\nValidation successful. No errors.")
        return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--output-dir", default="output/images/synthetic_salary_slips/robustness_dataset")
    parser.add_argument("--manifest", default="research_v2/synthetic_documents/robustness_dataset_manifest.json")
    parser.add_argument("--size", type=int, default=200)
    args = parser.parse_args()
    
    if args.generate:
        generate_dataset(Path(args.output_dir), Path(args.manifest), size=args.size)
    if args.validate:
        validate_dataset(Path(args.manifest))
