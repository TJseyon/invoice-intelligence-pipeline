"""
The end-to-end pipeline: bytes in -> ProcessingResult out.

This is the single function both the API route and the eval harness call,
so "what the API does" and "what we evaluate" can never silently drift apart.

Edge cases handled explicitly (named, not just implicitly swallowed):
  - empty_document: OCR produced ~no text -> skip the LLM call entirely.
  - unsupported_language: OCR text is in a language the configured
    tesseract language pack doesn't cover -> flag, skip extraction.
  - extraction_failed: LLM never returned valid JSON after retries.
  - malformed/corrupt file: OCRError from a bad upload -> caller (routes.py)
    turns this into a 422 with a clear message.
"""
from __future__ import annotations

import logging

from app.config import get_settings
from app.extraction import ExtractionError, extract_fields
from app.models import ProcessingResult, ValidationIssue
from app.ocr import OCRError, run_ocr
from app.validation import should_flag_for_review, validate_invoice

logger = logging.getLogger(__name__)
settings = get_settings()


def process_document(file_bytes: bytes, content_type: str) -> ProcessingResult:
    ocr_result = run_ocr(file_bytes, content_type)  # raises OCRError upstream on corrupt files

    if ocr_result.is_empty:
        return ProcessingResult(
            status="empty_document",
            flagged_for_review=True,
            issues=[
                ValidationIssue(
                    code="empty_document",
                    message="OCR extracted fewer than the configured minimum characters; "
                    "the document is likely blank, unreadable, or scanned upside down.",
                    severity="error",
                )
            ],
            raw_ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            detected_language=None,
        )

    if not ocr_result.language_supported:
        return ProcessingResult(
            status="unsupported_language",
            flagged_for_review=True,
            issues=[
                ValidationIssue(
                    code="unsupported_language",
                    message=f"Detected language '{ocr_result.detected_language}' is not covered by the "
                    "configured OCR language pack; extraction was skipped to avoid garbage output.",
                    severity="error",
                )
            ],
            raw_ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            detected_language=ocr_result.detected_language,
        )

    try:
        invoice, rejected_fields = extract_fields(ocr_result.text)
    except ExtractionError as exc:
        return ProcessingResult(
            status="extraction_failed",
            flagged_for_review=True,
            issues=[
                ValidationIssue(
                    code="extraction_failed",
                    message=str(exc),
                    severity="error",
                )
            ],
            raw_ocr_text=ocr_result.text,
            ocr_confidence=ocr_result.confidence,
            detected_language=ocr_result.detected_language,
        )

    issues = validate_invoice(invoice)

    if ocr_result.is_low_confidence:
        issues.append(
            ValidationIssue(
                code="low_ocr_confidence",
                message=f"Mean OCR word confidence ({ocr_result.confidence:.1f}) is below the "
                f"{settings.ocr_min_confidence} threshold; extracted fields may be unreliable "
                "regardless of the LLM's own confidence.",
                severity="warning",
            )
        )

    if rejected_fields and settings.flag_if_source_text_unverified:
        for name in rejected_fields:
            issues.append(
                ValidationIssue(
                    code="unsupported_field_value",
                    field=name,
                    message=f"Field '{name}' was extracted but its claimed source text could not be "
                    "verified against the OCR output; treated as unsupported and nulled out "
                    "rather than trusted.",
                    severity="warning",
                )
            )

    flagged = should_flag_for_review(issues)

    return ProcessingResult(
        status="flagged" if flagged else "ok",
        flagged_for_review=flagged,
        issues=issues,
        extracted=invoice,
        raw_ocr_text=ocr_result.text,
        ocr_confidence=ocr_result.confidence,
        detected_language=ocr_result.detected_language,
    )
