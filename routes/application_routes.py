"""
Application Routes — Submit, List, Detail, Upload
──────────────────────────────────────────────────
User-facing endpoints for loan applications.
"""


from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Request, status
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.middleware import get_current_user, CurrentUser
from gateway.rate_limiter import limiter
from gateway.file_scanner import validate_upload, save_upload
from models.database import Application, Document, ApplicationStatus, get_db
from models.schemas import (
    LoanApplicationCreate,
    LoanApplicationResponse,
    ApplicationListResponse,
    DocumentResponse,
)

router = APIRouter(prefix="/applications", tags=["Loan Applications"])


@router.post("/", response_model=LoanApplicationResponse, status_code=201)
@limiter.limit("5/minute")
async def submit_application(
    request: Request,
    body: LoanApplicationCreate,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Submit a new loan application."""
    application = Application(
        user_id=current_user.user_id,
        status=ApplicationStatus.SUBMITTED,
        loan_amount=body.loan_amount,
        loan_purpose=body.loan_purpose,
        loan_tenure_months=body.loan_tenure_months,
        applicant_name=body.applicant_name,
        applicant_age=body.applicant_age,
        applicant_gender=body.applicant_gender,
        applicant_email=body.applicant_email,
        applicant_phone=body.applicant_phone,
        applicant_address=body.applicant_address,
        applicant_pan=body.applicant_pan,
        applicant_aadhaar=body.applicant_aadhaar,
        monthly_income=body.monthly_income,
        employment_type=body.employment_type,
        employer_name=body.employer_name,
        shared_state={},
    )

    db.add(application)
    await db.commit()
    await db.refresh(application)

    return application


@router.get("/", response_model=ApplicationListResponse)
async def list_applications(
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """List all applications for the current user."""
    result = await db.execute(
        select(Application)
        .where(Application.user_id == current_user.user_id)
        .order_by(Application.created_at.desc())
    )
    apps = result.scalars().all()

    return ApplicationListResponse(
        total=len(apps),
        applications=[LoanApplicationResponse.model_validate(a) for a in apps],
    )


@router.get("/{application_id}", response_model=LoanApplicationResponse)
async def get_application(
    application_id: str,
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Get details of a specific application."""
    result = await db.execute(
        select(Application).where(
            Application.id == application_id,
            Application.user_id == current_user.user_id,
        )
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    return application


@router.post("/{application_id}/upload", response_model=DocumentResponse, status_code=201)
@limiter.limit("10/minute")
async def upload_document(
    request: Request,
    application_id: str,
    file: UploadFile = File(...),
    current_user: CurrentUser = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Upload a PDF document (bank statement) for an application."""
    # Verify application exists and belongs to user
    result = await db.execute(
        select(Application).where(
            Application.id == application_id,
            Application.user_id == current_user.user_id,
        )
    )
    application = result.scalar_one_or_none()

    if not application:
        raise HTTPException(status_code=404, detail="Application not found")

    # Validate file
    await validate_upload(file)

    # Save file to disk
    file_path = await save_upload(file, application_id)

    # Create document record
    doc = Document(
        application_id=application_id,
        file_name=file.filename or "upload.pdf",
        file_path=file_path,
        file_type=file.content_type or "application/pdf",
        file_size_bytes=file.size,
    )
    db.add(doc)
    await db.commit()
    await db.refresh(doc)

    return doc
