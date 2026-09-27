"""Safe text extraction helpers for Phase-2 synthetic documents."""

from pathlib import Path

import pdfplumber


def extract_searchable_pdf_text(path: Path, max_pages: int = 3) -> str:
    if path.suffix.lower() != ".pdf":
        raise ValueError("Only PDF files are supported")
    if max_pages <= 0:
        raise ValueError("max_pages must be positive")
    with pdfplumber.open(path) as document:
        if len(document.pages) > max_pages:
            raise ValueError("PDF exceeds maximum page count")
        return "\n".join((page.extract_text() or "") for page in document.pages).strip()

