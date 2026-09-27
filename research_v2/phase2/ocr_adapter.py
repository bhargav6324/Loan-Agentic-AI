"""Deterministic Tesseract OCR adapter and governed image ingestion."""

from __future__ import annotations

import csv
import hashlib
import io
import subprocess
from pathlib import Path
from typing import Callable, Iterable, List, Optional

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field

from .certificates import EvidenceCertificate
from .llm_document_extractor import SalarySlipExtractionResult, extract_salary_slip_text
from .schemas import LoanEvidenceCase
from .state_lifecycle import SalarySlipIngestionResult, ingest_salary_slip_extraction


SUPPORTED_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}


class OCRResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    valid: bool
    text: str = ""
    mean_confidence: float = 0.0
    word_count: int = 0
    image_sha256: str = ""
    width: int = 0
    height: int = 0
    errors: List[str] = Field(default_factory=list)


class SalarySlipImageIngestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    accepted: bool
    ocr: OCRResult
    ingestion: SalarySlipIngestionResult


def _parse_tsv(tsv: str):
    rows = []
    for row in csv.DictReader(io.StringIO(tsv), delimiter="\t"):
        text = (row.get("text") or "").strip()
        if not text:
            continue
        try:
            confidence = float(row.get("conf", "-1"))
        except ValueError:
            confidence = -1.0
        if confidence < 0:
            continue
        line_key = (
            row.get("page_num", ""), row.get("block_num", ""),
            row.get("par_num", ""), row.get("line_num", ""),
        )
        rows.append((line_key, text, confidence))
    return rows


import tempfile
import os

def extract_image_text(
    path: Path,
    tesseract_command: str = "tesseract",
    timeout_seconds: float = 30.0,
    minimum_confidence: float = 40.0,
    profile: str = "default",
) -> OCRResult:
    """Validate an image and extract text plus OCR confidence using TSV output."""

    path = Path(path)
    if path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        return OCRResult(valid=False, errors=["unsupported_image_type"])
    if not path.is_file():
        return OCRResult(valid=False, errors=["image_not_found"])

    raw_bytes = path.read_bytes()
    digest = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
    except (UnidentifiedImageError, OSError):
        return OCRResult(
            valid=False, image_sha256=digest, errors=["invalid_image"]
        )
    if width * height > 25_000_000:
        return OCRResult(
            valid=False, image_sha256=digest, width=width, height=height,
            errors=["image_too_large"],
        )

    psm = "6"
    target_path = str(path)
    tmp_path = None
    
    try:
        if profile == "compact_grayscale_psm6":
            with Image.open(path) as image:
                gray = image.convert("L")
                fd, tmp_path = tempfile.mkstemp(suffix=".png")
                os.close(fd)
                gray.save(tmp_path)
            target_path = tmp_path
            psm = "6"
            
        completed = subprocess.run(
            [tesseract_command, target_path, "stdout", "--psm", psm, "tsv"],
            capture_output=True,
            text=True,
            errors="replace",
            timeout=timeout_seconds,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError) as exc:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
        return OCRResult(
            valid=False, image_sha256=digest, width=width, height=height,
            errors=["ocr_unavailable:" + type(exc).__name__],
        )
        
    if tmp_path and os.path.exists(tmp_path):
        os.remove(tmp_path)
        
    if completed.returncode != 0:
        return OCRResult(
            valid=False, image_sha256=digest, width=width, height=height,
            errors=["ocr_process_failed"],
        )

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
    mean_confidence = (
        round(sum(confidences) / len(confidences), 2) if confidences else 0.0
    )
    errors = []
    if len(rows) < 4:
        errors.append("insufficient_ocr_text")
    if mean_confidence < minimum_confidence:
        errors.append("low_ocr_confidence")
    return OCRResult(
        valid=not errors,
        text=text,
        mean_confidence=mean_confidence,
        word_count=len(rows),
        image_sha256=digest,
        width=width,
        height=height,
        errors=errors,
    )


def extract_and_ingest_salary_slip_image(
    case: LoanEvidenceCase,
    image_path: Path,
    document_id: str,
    certificates: Iterable[EvidenceCertificate] = (),
    generator: Optional[Callable[[str], dict]] = None,
    tesseract_command: str = "tesseract",
    ocr_profile: str = "default",
) -> SalarySlipImageIngestionResult:
    """OCR an image, run bounded LLM extraction, and update governed state."""

    ocr = extract_image_text(image_path, tesseract_command=tesseract_command, profile=ocr_profile)
    if not ocr.valid:
        extraction = SalarySlipExtractionResult(
            valid=False,
            errors=["ocr:" + error for error in ocr.errors],
        )
    elif generator is None:
        extraction = extract_salary_slip_text(document_id, ocr.text)
    else:
        extraction = extract_salary_slip_text(document_id, ocr.text, generator)
    ingestion = ingest_salary_slip_extraction(case, extraction, certificates)
    return SalarySlipImageIngestionResult(
        accepted=ingestion.accepted,
        ocr=ocr,
        ingestion=ingestion,
    )

