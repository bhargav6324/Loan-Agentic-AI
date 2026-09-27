"""Evaluate searchable synthetic PDFs with the bounded local LLM extractor."""

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from research_v2.phase2.document_io import extract_searchable_pdf_text
from research_v2.phase2.llm_document_extractor import extract_salary_slip_text


def main():
    root = Path(__file__).resolve().parents[1]
    manifest_path = root / "research_v2" / "synthetic_documents" / "manifest.json"
    output_path = root / "research_v2" / "synthetic_documents" / "llm_extraction_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    results = []
    for item in manifest["documents"]:
        print(f"Evaluating {item['case_id']}...", flush=True)
        text = extract_searchable_pdf_text(root / item["pdf_path"])
        started = time.perf_counter()
        extraction = extract_salary_slip_text(item["case_id"] + "-pdf", text)
        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        fields = extraction.document.fields if extraction.document else {}
        income = fields.get("monthly_income")
        employer = fields.get("employer_name")
        expected_income = item["document_monthly_income"]
        results.append({
            "case_id": item["case_id"],
            "variant": item["variant"],
            "extraction_valid": extraction.valid,
            "errors": extraction.errors,
            "latency_ms": latency_ms,
            "extracted_monthly_income": income.value if income else None,
            "expected_monthly_income": expected_income,
            "income_match": (
                income is None if expected_income is None
                else income is not None and abs(float(income.value) - float(expected_income)) < 0.005
            ),
            "extracted_employer": employer.value if employer else None,
            "expected_employer": item["document_employer"],
            "employer_match": bool(
                employer and str(employer.value).casefold() == item["document_employer"].casefold()
            ),
            "declared_employer_consistent": (
                item["declared_employer"].casefold() == item["document_employer"].casefold()
            ),
        })

    summary = {
        "total_documents": len(results),
        "valid_extractions": sum(row["extraction_valid"] for row in results),
        "income_matches": sum(row["income_match"] for row in results),
        "employer_matches": sum(row["employer_match"] for row in results),
        "mean_latency_ms": round(sum(row["latency_ms"] for row in results) / len(results), 2),
    }
    output_path.write_text(
        json.dumps({"summary": summary, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))
    print(output_path)


if __name__ == "__main__":
    main()
