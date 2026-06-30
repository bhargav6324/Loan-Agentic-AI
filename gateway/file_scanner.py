"""
Zero Trust Gateway — File Scanner
──────────────────────────────────
Validates uploaded files: type, size, content checking.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from fastapi import UploadFile, HTTPException, status

from config import settings

# Allowed MIME types
ALLOWED_MIME_TYPES = {
    "application/pdf",
}

# PDF magic bytes
PDF_MAGIC = b"%PDF"


async def validate_upload(file: UploadFile) -> None:
    """
    Validate an uploaded file:
    1. Check MIME type
    2. Check file extension
    3. Check file size
    4. Check magic bytes (content sniffing)
    """
    # 1. Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid file type: {file.content_type}. Only PDF files are accepted.",
        )

    # 2. Check extension
    if file.filename:
        ext = Path(file.filename).suffix.lower()
        if ext != ".pdf":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file extension: {ext}. Only .pdf files are accepted.",
            )

    # 3. Check file size (read first chunk to estimate)
    contents = await file.read()
    if len(contents) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"File too large. Maximum size: {settings.MAX_UPLOAD_SIZE_MB}MB",
        )

    if len(contents) == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Empty file uploaded",
        )

    # 4. Check magic bytes — must start with %PDF
    if not contents[:4].startswith(PDF_MAGIC):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="File content is not a valid PDF",
        )

    # Reset file position for downstream readers
    await file.seek(0)


async def save_upload(
    file: UploadFile,
    application_id: str,
) -> str:
    """
    Save an uploaded file to disk.
    Returns the file path relative to project root.
    """
    # Create application-specific upload directory
    app_upload_dir = settings.upload_path / application_id
    app_upload_dir.mkdir(parents=True, exist_ok=True)

    # Use original filename (sanitised)
    safe_name = "".join(
        c if c.isalnum() or c in ".-_" else "_"
        for c in (file.filename or "upload.pdf")
    )
    file_path = app_upload_dir / safe_name

    # Write to disk
    contents = await file.read()
    with open(file_path, "wb") as f:
        f.write(contents)

    return str(file_path)
