"""
Pydantic schemas: the strict structured-output contract the LLM must fill in,
plus the API request/response shapes built on top of it.

Every extracted field is wrapped in `ExtractedField`, which carries not just
the value but the model's confidence *and* the verbatim source snippet it
claims to have pulled the value from. That source snippet is what the
hallucination check in validation.py verifies against the raw OCR text.
"""
from __future__ import annotations

from datetime import date
from enum import Enum
from typing import Generic, Optional, TypeVar

from pydantic import BaseModel, Field, field_validator

T = TypeVar("T")


class DocumentType(str, Enum):
    INVOICE = "invoice"
    RECEIPT = "receipt"
    UNKNOWN = "unknown"


class ExtractedField(BaseModel, Generic[T]):
    """A single extracted value plus the evidence for trusting it."""

    value: Optional[T] = None
    confidence: float = Field(ge=0.0, le=1.0, default=0.0)
    source_text: Optional[str] = Field(
        default=None,
        description="Verbatim substring of the OCR text that supports this value. "
        "Required whenever value is not null.",
    )


class LineItem(BaseModel):
    description: ExtractedField[str]
    quantity: ExtractedField[float]
    unit_price: ExtractedField[float]
    line_total: ExtractedField[float]


class ExtractedInvoice(BaseModel):
    """The strict schema the LLM must return. This is what gets validated,
    scored against the eval set, and stored."""

    document_type: DocumentType = DocumentType.UNKNOWN
    vendor_name: ExtractedField[str]
    invoice_number: ExtractedField[str]
    invoice_date: ExtractedField[str]  # kept as string; parsed/validated separately
    currency: ExtractedField[str]
    subtotal_amount: ExtractedField[float]
    tax_amount: ExtractedField[float]
    total_amount: ExtractedField[float]
    line_items: list[LineItem] = Field(default_factory=list)

    @field_validator("line_items")
    @classmethod
    def cap_line_items(cls, v: list[LineItem]) -> list[LineItem]:
        if len(v) > 200:
            raise ValueError("Implausible number of line items (>200); likely a bad extraction.")
        return v


class ValidationIssue(BaseModel):
    code: str
    field: Optional[str] = None
    message: str
    severity: str = Field(description="'error' (blocks) or 'warning' (flags for review)")


class ProcessingResult(BaseModel):
    id: Optional[int] = None  # populated by the upload route after the DB insert
    status: str  # "ok" | "flagged" | "empty_document" | "unsupported_language" | "extraction_failed"
    flagged_for_review: bool
    issues: list[ValidationIssue] = Field(default_factory=list)
    extracted: Optional[ExtractedInvoice] = None
    raw_ocr_text: Optional[str] = None
    ocr_confidence: Optional[float] = None
    detected_language: Optional[str] = None


class DocumentSummary(BaseModel):
    id: int
    filename: str
    status: str
    flagged_for_review: bool
    vendor_name: Optional[str] = None
    total_amount: Optional[float] = None
    created_at: str


class CorrectionRequest(BaseModel):
    """A human reviewer's corrected values for a flagged document."""

    corrected_fields: dict[str, str | float | None]
    reviewer_note: Optional[str] = None
