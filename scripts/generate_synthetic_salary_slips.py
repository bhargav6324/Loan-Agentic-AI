"""Generate the deterministic Phase-2 synthetic salary-slip dataset."""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_v2.phase2.synthetic_salary_slips import (
    create_salary_slip_image,
    create_salary_slip_pdf,
    default_specs,
    manifest_entry,
)


def main():
    root = Path(__file__).resolve().parents[1]
    pdf_dir = root / "output" / "pdf" / "synthetic_salary_slips"
    image_dir = root / "output" / "images" / "synthetic_salary_slips"
    manifest_path = root / "research_v2" / "synthetic_documents" / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    entries = []
    for spec in default_specs():
        pdf_path = pdf_dir / f"{spec.case_id}.pdf"
        image_path = image_dir / f"{spec.case_id}.png"
        create_salary_slip_pdf(spec, pdf_path)
        create_salary_slip_image(spec, image_path)
        entries.append(
            manifest_entry(
                spec,
                pdf_path.relative_to(root),
                image_path.relative_to(root),
            )
        )

    manifest_path.write_text(json.dumps({
        "dataset": "synthetic_salary_slips_v1",
        "document_count": len(entries),
        "contains_real_personal_data": False,
        "documents": entries,
    }, indent=2), encoding="utf-8")
    print(f"Generated {len(entries)} PDFs and {len(entries)} PNGs")
    print(manifest_path)


if __name__ == "__main__":
    main()
