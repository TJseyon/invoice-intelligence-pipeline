"""
Validation & review-flagging.

Answers the interview question "how do you know when to trust the
extraction vs. flag it for a human?" with a concrete, inspectable rule set
rather than a black-box score:

  1. Required fields present.
  2. Line items sum to subtotal_amount (the pre-tax figure -- summing
     against the tax-inclusive total would false-flag almost every invoice
     that carries nonzero tax), within tolerance.
  3. Per-field LLM confidence above threshold.
  4. Every non-null field's source_text was verified against the OCR text
     (done upstream in extraction.py; a nulled-out field here means the
     hallucination check already rejected it).
  5. Date parses as a real calendar date.

A document is flagged for review if ANY check produces an "error" or (per
config) a "warning" severity issue. Each issue is returned with a field
name so the review UI can highlight exactly what needs a human look.
"""
from __future__ import annotations

import datetime as dt

from app.config import get_settings
from app.models import ExtractedInvoice, ValidationIssue

settings = get_settings()


def validate_invoice(invoice: ExtractedInvoice) -> list[ValidationIssue]:
    issues: list[ValidationIssue] = []

    issues.extend(_check_required_fields(invoice))
    issues.extend(_check_line_item_sum(invoice))
    issues.extend(_check_field_confidences(invoice))
    issues.extend(_check_date(invoice))

    return issues


def _check_required_fields(invoice: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    for name in settings.required_fields:
        field = getattr(invoice, name, None)
        if field is None or field.value in (None, ""):
            issues.append(
                ValidationIssue(
                    code="missing_required_field",
                    field=name,
                    message=f"Required field '{name}' could not be extracted (or failed the source-text check).",
                    severity="error",
                )
            )
    return issues


def _check_line_item_sum(invoice: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    if not invoice.line_items:
        return issues  # nothing to cross-check; not itself an error

    line_sum = 0.0
    all_present = True
    for item in invoice.line_items:
        if item.line_total.value is None:
            all_present = False
            break
        line_sum += item.line_total.value

    if not all_present:
        issues.append(
            ValidationIssue(
                code="incomplete_line_items",
                field="line_items",
                message="One or more line items are missing a line_total; cannot cross-check against the invoice total.",
                severity="warning",
            )
        )
        return issues

    expected_subtotal = invoice.subtotal_amount.value
    # Line items should sum to the pre-tax subtotal, not the tax-inclusive
    # total. Comparing against total_amount directly would false-flag almost
    # every real invoice that carries nonzero tax. Fall back to total_amount
    # only when subtotal wasn't extracted, on the assumption of no tax.
    target_field = "subtotal_amount"
    expected_value = expected_subtotal
    if expected_value is None:
        target_field = "total_amount"
        expected_value = invoice.total_amount.value

    if expected_value is None:
        return issues

    if expected_value == 0:
        return issues

    pct_diff = abs(line_sum - expected_value) / abs(expected_value) * 100
    if pct_diff > settings.line_item_sum_tolerance_pct:
        issues.append(
            ValidationIssue(
                code="line_items_total_mismatch",
                field=target_field,
                message=(
                    f"Line items sum to {line_sum:.2f}, but {target_field} is "
                    f"{expected_value:.2f} ({pct_diff:.1f}% difference, "
                    f"tolerance is {settings.line_item_sum_tolerance_pct}%)."
                ),
                severity="error",
            )
        )
    return issues


def _check_field_confidences(invoice: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    top_level_fields = [
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "currency",
        "subtotal_amount",
        "tax_amount",
        "total_amount",
    ]
    for name in top_level_fields:
        field = getattr(invoice, name)
        if field.value is not None and field.confidence < settings.llm_field_confidence_threshold:
            issues.append(
                ValidationIssue(
                    code="low_confidence_field",
                    field=name,
                    message=f"Field '{name}' extracted with confidence {field.confidence:.2f}, "
                    f"below the {settings.llm_field_confidence_threshold} threshold.",
                    severity="warning",
                )
            )
    return issues


def _check_date(invoice: ExtractedInvoice) -> list[ValidationIssue]:
    issues = []
    raw = invoice.invoice_date.value
    if raw is None:
        return issues
    try:
        dt.date.fromisoformat(raw)
    except ValueError:
        issues.append(
            ValidationIssue(
                code="unparseable_date",
                field="invoice_date",
                message=f"invoice_date value '{raw}' is not a valid ISO-8601 date.",
                severity="warning",
            )
        )
    return issues


def should_flag_for_review(issues: list[ValidationIssue]) -> bool:
    has_error = any(i.severity == "error" for i in issues)
    has_warning = any(i.severity == "warning" for i in issues)
    if has_error:
        return True
    if has_warning and settings.flag_if_any_field_low_confidence:
        return True
    return False
