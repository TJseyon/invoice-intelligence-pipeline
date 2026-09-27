"""
Unit tests for the anti-hallucination guard in extraction.py: any field
whose claimed source_text can't be found in the OCR text must be nulled out,
regardless of what confidence the model reported for it.
"""
from __future__ import annotations

from app.extraction import _similar, _verify_sources
from app.models import ExtractedField as EF
from app.models import DocumentType, ExtractedInvoice


def _invoice_with(vendor_value, vendor_source):
    return ExtractedInvoice(
        document_type=DocumentType.INVOICE,
        vendor_name=EF(value=vendor_value, confidence=0.95, source_text=vendor_source),
        invoice_number=EF(value=None, confidence=0.0, source_text=None),
        invoice_date=EF(value=None, confidence=0.0, source_text=None),
        currency=EF(value=None, confidence=0.0, source_text=None),
        subtotal_amount=EF(value=None, confidence=0.0, source_text=None),
        tax_amount=EF(value=None, confidence=0.0, source_text=None),
        total_amount=EF(value=None, confidence=0.0, source_text=None),
        line_items=[],
    )


def test_similar_exact_substring_matches():
    assert _similar("VENDOR: Acme Supply Co  INVOICE #: 123", "VENDOR: Acme Supply Co")


def test_similar_rejects_unrelated_text():
    assert not _similar("VENDOR: Acme Supply Co", "TOTAL: $9,999,999.00")


def test_verify_sources_keeps_grounded_field():
    ocr_text = "VENDOR: Acme Supply Co   INVOICE #: INV-42"
    invoice = _invoice_with("Acme Supply Co", "VENDOR: Acme Supply Co")
    rejected = _verify_sources(invoice, ocr_text)
    assert invoice.vendor_name.value == "Acme Supply Co"
    assert invoice.vendor_name.confidence == 0.95
    assert rejected == []


def test_verify_sources_nulls_ungrounded_field():
    ocr_text = "VENDOR: Acme Supply Co   INVOICE #: INV-42"
    # model claims a source_text that does not appear anywhere in the OCR text
    invoice = _invoice_with("Globex Corporation", "VENDOR: Globex Corporation")
    rejected = _verify_sources(invoice, ocr_text)
    assert invoice.vendor_name.value is None
    assert invoice.vendor_name.confidence == 0.0
    assert invoice.vendor_name.source_text is None
    assert rejected == ["vendor_name"]


def test_verify_sources_nulls_field_with_missing_source_text():
    ocr_text = "VENDOR: Acme Supply Co"
    invoice = _invoice_with("Acme Supply Co", None)  # value given but no source_text at all
    rejected = _verify_sources(invoice, ocr_text)
    assert invoice.vendor_name.value is None
    assert rejected == ["vendor_name"]
