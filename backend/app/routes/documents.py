"""
Document upload, listing, retrieval, and human-correction routes.

Input validation lives at the top of `upload_document` and returns clear
4xx errors (wrong content type, file too large, unreadable/corrupt file)
instead of letting those cases fall through into a 500.
"""
from __future__ import annotations

import json
import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from app.config import get_settings
from app.db import Correction, Document, get_db
from app.models import CorrectionRequest, DocumentSummary, ProcessingResult
from app.ocr import OCRError
from app.pipeline import process_document

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/documents", tags=["documents"])
settings = get_settings()


def _persist_upload(file_bytes: bytes, original_filename: str) -> str:
    """Saves the raw uploaded file to UPLOAD_DIR under a random name (so two
    uploads named 'invoice.png' never collide) and returns the path. This is
    what lets the review UI show reviewers the actual source image next to
    the extracted fields, instead of just the flattened OCR text."""
    upload_dir = Path(settings.upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    suffix = Path(original_filename).suffix or ""
    stored_name = f"{uuid.uuid4().hex}{suffix}"
    stored_path = upload_dir / stored_name
    stored_path.write_bytes(file_bytes)
    return str(stored_path)


@router.post("/upload", response_model=ProcessingResult, status_code=201)
async def upload_document(file: UploadFile = File(...), db: Session = Depends(get_db)) -> ProcessingResult:
    # --- input validation -------------------------------------------------
    if file.content_type not in settings.allowed_content_types:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported content type '{file.content_type}'. "
            f"Allowed: {', '.join(settings.allowed_content_types)}.",
        )

    file_bytes = await file.read()
    size_mb = len(file_bytes) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(
            status_code=413,
            detail=f"File is {size_mb:.1f}MB, exceeds the {settings.max_upload_mb}MB limit.",
        )
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty (0 bytes).")

    # --- run pipeline --------------------------------------------------
    try:
        result = process_document(file_bytes, file.content_type)
    except OCRError as exc:
        raise HTTPException(status_code=422, detail=f"Could not process file: {exc}") from exc

    # --- persist original file (best-effort: a disk failure shouldn't lose
    # an otherwise-successful extraction) --------------------------------
    stored_path: str | None = None
    try:
        stored_path = _persist_upload(file_bytes, file.filename or "unnamed")
    except OSError as exc:
        logger.warning("Could not persist uploaded file to disk: %s", exc)

    # --- persist extraction result ---------------------------------------
    doc = Document(
        filename=file.filename or "unnamed",
        content_type=file.content_type,
        status=result.status,
        flagged_for_review=result.flagged_for_review,
        raw_ocr_text=result.raw_ocr_text,
        ocr_confidence=result.ocr_confidence,
        detected_language=result.detected_language,
        stored_file_path=stored_path,
        extracted_json=result.extracted.model_dump_json() if result.extracted else None,
        issues_json=json.dumps([i.model_dump() for i in result.issues]),
        vendor_name=result.extracted.vendor_name.value if result.extracted else None,
        total_amount=result.extracted.total_amount.value if result.extracted else None,
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)

    result.id = doc.id
    return result


@router.get("", response_model=list[DocumentSummary])
def list_documents(
    flagged_only: bool = False, limit: int = 50, db: Session = Depends(get_db)
) -> list[DocumentSummary]:
    query = db.query(Document)
    if flagged_only:
        query = query.filter(Document.flagged_for_review.is_(True))
    docs = query.order_by(Document.created_at.desc()).limit(limit).all()
    return [
        DocumentSummary(
            id=d.id,
            filename=d.filename,
            status=d.status,
            flagged_for_review=d.flagged_for_review,
            vendor_name=d.vendor_name,
            total_amount=d.total_amount,
            created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.get("/{document_id}")
def get_document(document_id: int, db: Session = Depends(get_db)) -> dict:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    return {
        "id": doc.id,
        "filename": doc.filename,
        "status": doc.status,
        "flagged_for_review": doc.flagged_for_review,
        "ocr_confidence": doc.ocr_confidence,
        "detected_language": doc.detected_language,
        "raw_ocr_text": doc.raw_ocr_text,
        "extracted": doc.extracted_dict(),
        "issues": doc.issues_list(),
        "has_source_file": doc.stored_file_path is not None,
    }


@router.get("/{document_id}/file")
def get_document_file(document_id: int, db: Session = Depends(get_db)) -> FileResponse:
    """Serves the original uploaded image/PDF so the review UI can show a
    reviewer the source document next to the extracted fields, instead of
    asking them to trust flattened OCR text alone."""
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")
    if not doc.stored_file_path or not Path(doc.stored_file_path).exists():
        raise HTTPException(status_code=404, detail="Original file is not available for this document.")
    return FileResponse(doc.stored_file_path, media_type=doc.content_type, filename=doc.filename)


@router.post("/{document_id}/corrections")
def submit_corrections(
    document_id: int, correction: CorrectionRequest, db: Session = Depends(get_db)
) -> dict:
    doc = db.query(Document).filter(Document.id == document_id).first()
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {document_id} not found.")

    extracted = doc.extracted_dict() or {}
    logged = 0
    for field_name, corrected_value in correction.corrected_fields.items():
        original = extracted.get(field_name, {}).get("value") if isinstance(extracted.get(field_name), dict) else None
        db.add(
            Correction(
                document_id=doc.id,
                field_name=field_name,
                original_value=str(original) if original is not None else None,
                corrected_value=str(corrected_value) if corrected_value is not None else None,
                reviewer_note=correction.reviewer_note,
            )
        )
        if field_name in extracted and isinstance(extracted[field_name], dict):
            extracted[field_name]["value"] = corrected_value
            extracted[field_name]["confidence"] = 1.0  # human-verified
        logged += 1

    doc.extracted_json = json.dumps(extracted)
    doc.status = "corrected"
    doc.flagged_for_review = False
    db.commit()

    return {"document_id": doc.id, "fields_corrected": logged, "status": doc.status}


@router.get("/stats/corrections")
def correction_stats(db: Session = Depends(get_db)) -> dict:
    """Aggregate corrections by field -- the feedback signal the brief calls
    for ('corrections feed back into your validation rules'). A field that
    is corrected disproportionately often is a strong candidate for a
    lowered confidence threshold or an extra validation rule."""
    corrections = db.query(Correction).all()
    by_field: dict[str, int] = {}
    for c in corrections:
        by_field[c.field_name] = by_field.get(c.field_name, 0) + 1
    total_docs = db.query(Document).count()
    return {
        "total_documents": total_docs,
        "total_corrections": len(corrections),
        "corrections_by_field": dict(sorted(by_field.items(), key=lambda kv: -kv[1])),
    }
