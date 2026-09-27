"""
Unit tests for deterministic code: validation.py's cross-field rules.

These never touch the network, an LLM, or a database -- pure function in,
list of issues out. This is the kind of test that should run in milliseconds
on every commit, as distinct from the eval tests in test_eval.py which
exercise the full stochastic pipeline against a scored dataset.
"""
from __future__ import annotations

from app.models import ExtractedField as EF
from app.models import ExtractedInvoice, LineItem
from app.validation import should_flag_for_review, validate_invoice


def _blank_field(value=None, confidence=0.0):
    return EF(value=value, confidence=confidence, source_text=str(value) if value is not None else None)


def make_invoice(**overrides) -> ExtractedInvoice:
    defaults = dict(
        vendor_name=EF(value="Acme Co", confidence=0.95, source_text="Acme Co"),
        invoice_number=EF(value="INV-1", confidence=0.95, source_text="INV-1"),
        invoice_date=EF(value="2026-01-15", confidence=0.95, source_text="2026-01-15"),
        currency=EF(value="USD", confidence=0.95, source_text="USD"),
        subtotal_amount=EF(value=100.0, confidence=0.95, source_text="100.00"),
        tax_amount=EF(value=8.0, confidence=0.95, source_text="8.00"),
        total_amount=EF(value=108.0, confidence=0.95, source_text="108.00"),
        line_items=[],
    )
    defaults.update(overrides)
    return ExtractedInvoice(**defaults)


def test_clean_invoice_produces_no_issues():
    invoice = make_invoice()
    issues = validate_invoice(invoice)
    assert issues == []
    assert should_flag_for_review(issues) is False


def test_missing_required_field_is_an_error():
    invoice = make_invoice(vendor_name=_blank_field())
    issues = validate_invoice(invoice)
    codes = [i.code for i in issues]
    assert "missing_required_field" in codes
    assert should_flag_for_review(issues) is True


def test_line_items_matching_subtotal_passes():
    items = [
        LineItem(
            description=EF(value="Widget", confidence=0.9, source_text="Widget"),
            quantity=EF(value=2, confidence=0.9, source_text="2"),
            unit_price=EF(value=50.0, confidence=0.9, source_text="50.00"),
            line_total=EF(value=100.0, confidence=0.9, source_text="100.00"),
        )
    ]
    invoice = make_invoice(line_items=items, subtotal_amount=EF(value=100.0, confidence=0.9, source_text="100.00"),
                            tax_amount=EF(value=8.0, confidence=0.9, source_text="8.00"),
                            total_amount=EF(value=100.0, confidence=0.9, source_text="100.00"))
    issues = validate_invoice(invoice)
    assert not any(i.code == "line_items_total_mismatch" for i in issues)


def test_line_items_sum_mismatch_is_flagged():
    items = [
        LineItem(
            description=EF(value="Widget", confidence=0.9, source_text="Widget"),
            quantity=EF(value=2, confidence=0.9, source_text="2"),
            unit_price=EF(value=50.0, confidence=0.9, source_text="50.00"),
            line_total=EF(value=100.0, confidence=0.9, source_text="100.00"),
        )
    ]
    invoice = make_invoice(line_items=items, subtotal_amount=EF(value=500.0, confidence=0.9, source_text="500.00"))
    issues = validate_invoice(invoice)
    assert any(i.code == "line_items_total_mismatch" for i in issues)
    assert should_flag_for_review(issues) is True


def test_line_items_compare_against_subtotal_not_tax_inclusive_total():
    """Regression test for a real bug caught during development: line items
    sum to the pre-tax subtotal, not the tax-inclusive total. A correctly
    extracted invoice with real tax must NOT be flagged just because
    line_sum != total_amount."""
    items = [
        LineItem(
            description=EF(value="Widget", confidence=0.9, source_text="Widget"),
            quantity=EF(value=1, confidence=0.9, source_text="1"),
            unit_price=EF(value=100.0, confidence=0.9, source_text="100.00"),
            line_total=EF(value=100.0, confidence=0.9, source_text="100.00"),
        )
    ]
    # subtotal (100) matches line items exactly; total (108) includes 8% tax
    # and legitimately differs from the line-item sum by more than the
    # tolerance -- that must NOT trigger a false mismatch.
    invoice = make_invoice(line_items=items)  # defaults: subtotal=100, tax=8, total=108
    issues = validate_invoice(invoice)
    assert not any(i.code == "line_items_total_mismatch" for i in issues)


def test_line_items_within_tolerance_does_not_flag():
    # 1.5% default tolerance: line items sum to 100, subtotal says 101 -- a
    # 1% diff, should pass without flagging.
    items = [
        LineItem(
            description=EF(value="Widget", confidence=0.9, source_text="Widget"),
            quantity=EF(value=1, confidence=0.9, source_text="1"),
            unit_price=EF(value=100.0, confidence=0.9, source_text="100.00"),
            line_total=EF(value=100.0, confidence=0.9, source_text="100.00"),
        )
    ]
    invoice = make_invoice(line_items=items, subtotal_amount=EF(value=101.0, confidence=0.9, source_text="101.00"))
    issues = validate_invoice(invoice)
    assert not any(i.code == "line_items_total_mismatch" for i in issues)


def test_low_confidence_field_is_a_warning_not_error():
    invoice = make_invoice(vendor_name=EF(value="Acme Co", confidence=0.3, source_text="Acme Co"))
    issues = validate_invoice(invoice)
    low_conf = [i for i in issues if i.code == "low_confidence_field"]
    assert len(low_conf) == 1
    assert low_conf[0].severity == "warning"
    # warnings still trigger review by default (flag_if_any_field_low_confidence=True)
    assert should_flag_for_review(issues) is True


def test_unparseable_date_is_flagged():
    invoice = make_invoice(invoice_date=EF(value="not-a-date", confidence=0.9, source_text="not-a-date"))
    issues = validate_invoice(invoice)
    assert any(i.code == "unparseable_date" for i in issues)


def test_no_line_items_does_not_itself_trigger_sum_mismatch():
    invoice = make_invoice(line_items=[])
    issues = validate_invoice(invoice)
    assert not any(i.code == "line_items_total_mismatch" for i in issues)
