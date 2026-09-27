from pathlib import Path

from research_v2.phase2.document_io import extract_searchable_pdf_text
from research_v2.phase2.synthetic_salary_slips import (
    create_salary_slip_pdf,
    default_specs,
)


def test_default_dataset_covers_required_variants():
    variants = {spec.variant for spec in default_specs()}
    assert {
        "clean", "missing_income", "wrong_income", "conflicting_employer",
        "prompt_injection", "blurred_scan",
    }.issubset(variants)


def test_generated_pdf_is_searchable_and_marked_synthetic(tmp_path: Path):
    spec = default_specs()[0]
    output = tmp_path / "salary.pdf"
    create_salary_slip_pdf(spec, output)
    text = extract_searchable_pdf_text(output)
    assert "Asha Rao" in text
    assert "INR 50,000.00" in text
    assert "SYNTHETIC DOCUMENT" in text


def test_missing_income_pdf_does_not_print_salary_amount(tmp_path: Path):
    spec = next(spec for spec in default_specs() if spec.variant == "missing_income")
    output = tmp_path / "missing.pdf"
    create_salary_slip_pdf(spec, output)
    text = extract_searchable_pdf_text(output)
    assert "NET PAY" in text
    assert "NOT PROVIDED" in text
