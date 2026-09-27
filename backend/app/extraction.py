"""
LLM-based field extraction.

Design choices that matter for the "how do you know when to trust the
extraction" interview question:

1. The model is required to return, for every field, a `source_text`
   snippet: the exact substring of the OCR text it claims supports the
   value. We verify that snippet actually appears in the OCR text
   (fuzzy, to tolerate whitespace/OCR noise) before we ever mark a field
   as trustworthy. A field whose source can't be located is treated as
   unsupported, not just low-confidence -- this is the hallucination
   guard called for in the brief.
2. The model also self-reports a 0-1 confidence per field. That's a
   second, independent signal from the source-text check, combined in
   validation.py.
3. Malformed JSON gets exactly one retry with an explicit "fix your JSON"
   follow-up before we give up and mark the document extraction_failed.
   Retrying indefinitely would hide a systematic prompting problem instead
   of surfacing it.
"""
from __future__ import annotations

import json
import logging
from difflib import SequenceMatcher

from pydantic import ValidationError

from app.config import get_settings
from app.models import ExtractedInvoice

logger = logging.getLogger(__name__)
settings = get_settings()

SYSTEM_PROMPT = """You are a meticulous invoice/receipt data-extraction engine.
You will be given raw OCR text from a scanned invoice or receipt. Extract the
requested fields and return ONLY a single JSON object matching this schema
(no markdown fences, no commentary):

{
  "document_type": "invoice" | "receipt" | "unknown",
  "vendor_name": {"value": str|null, "confidence": 0-1, "source_text": str|null},
  "invoice_number": {"value": str|null, "confidence": 0-1, "source_text": str|null},
  "invoice_date": {"value": str|null (ISO 8601 YYYY-MM-DD if possible), "confidence": 0-1, "source_text": str|null},
  "currency": {"value": str|null (ISO 4217 code, e.g. USD), "confidence": 0-1, "source_text": str|null},
  "subtotal_amount": {"value": float|null, "confidence": 0-1, "source_text": str|null},
  "tax_amount": {"value": float|null, "confidence": 0-1, "source_text": str|null},
  "total_amount": {"value": float|null, "confidence": 0-1, "source_text": str|null},
  "line_items": [
    {
      "description": {"value": str, "confidence": 0-1, "source_text": str},
      "quantity": {"value": float, "confidence": 0-1, "source_text": str},
      "unit_price": {"value": float, "confidence": 0-1, "source_text": str},
      "line_total": {"value": float, "confidence": 0-1, "source_text": str}
    }
  ]
}

CRITICAL RULES:
- `source_text` MUST be a verbatim substring copied from the OCR text you were given.
  Never invent a source_text. If you cannot find explicit support for a value in the
  text, set value to null, confidence to 0, and source_text to null.
- Never fabricate numbers or dates that do not appear anywhere in the OCR text.
- confidence reflects YOUR certainty that the value is correct and clearly stated
  in the text, not a guess about what a typical invoice would contain.
"""


class ExtractionError(Exception):
    pass


def _similar(a: str, b: str, threshold: float = 0.8) -> bool:
    """Fuzzy substring check: is `b` (source_text) actually present in `a` (OCR text)?
    Tolerates minor OCR/whitespace noise instead of requiring an exact match."""
    a_norm, b_norm = a.lower(), b.lower().strip()
    if not b_norm:
        return False
    if b_norm in a_norm:
        return True
    # sliding-window fuzzy match for near-misses (OCR noise, stray punctuation)
    window = len(b_norm)
    step = max(1, window // 4)
    for i in range(0, max(1, len(a_norm) - window), step):
        if SequenceMatcher(None, a_norm[i : i + window], b_norm).ratio() >= threshold:
            return True
    return False


def _mock_extract(ocr_text: str) -> dict:
    """Deterministic offline extraction used when llm_provider == 'mock'.
    Lets the whole pipeline (and the eval harness) run with zero API cost/
    network access, e.g. in CI. Uses simple regex/heuristics over the OCR
    text rather than a real model."""
    import re

    def find(pattern, text, group=1):
        m = re.search(pattern, text, re.IGNORECASE)
        return m.group(group).strip() if m else None

    # NOTE on these patterns: OCR text is joined into a single space-separated
    # line (see ocr.py), so "TOTAL:" would also match inside "LINETOTAL:" or
    # "SUBTOTAL:" without a lookbehind excluding a preceding letter, and a bare
    # "INVOICE" (from the "INVOICE" title) would out-compete "INVOICE #:" for
    # invoice_number without requiring the literal "#". Both are covered below
    # and pinned down with regression tests in tests/test_extraction.py.
    vendor = find(r"VENDOR:\s*([A-Za-z0-9 &.,'-]+?)(?:\s{2,}|INVOICE|$)", ocr_text)
    inv_num = find(r"INVOICE\s*#\s*:?\s*([A-Za-z0-9-]+)", ocr_text)
    inv_date = find(r"DATE:\s*(\d{4}-\d{2}-\d{2})", ocr_text)
    currency = find(r"\b(USD|EUR|GBP|INR)\b", ocr_text)
    total = find(r"(?<![A-Za-z])TOTAL:\s*\$?([\d,]+\.\d{2})", ocr_text)
    subtotal = find(r"SUBTOTAL:\s*\$?([\d,]+\.\d{2})", ocr_text)
    tax = find(r"(?<![A-Za-z])TAX:\s*\$?([\d,]+\.\d{2})", ocr_text)

    def field(value, src_pattern):
        if value is None:
            return {"value": None, "confidence": 0.0, "source_text": None}
        src = find(src_pattern, ocr_text) or value
        return {"value": value, "confidence": 0.9, "source_text": str(src)}

    line_items = []
    for m in re.finditer(
        r"ITEM:\s*([A-Za-z0-9 .-]+?)\s+QTY:\s*(\d+)\s+PRICE:\s*\$?([\d.]+)\s+LINETOTAL:\s*\$?([\d.]+)",
        ocr_text,
        re.IGNORECASE,
    ):
        desc, qty, price, ltotal = m.groups()
        line_items.append(
            {
                "description": {"value": desc.strip(), "confidence": 0.9, "source_text": desc.strip()},
                "quantity": {"value": float(qty), "confidence": 0.9, "source_text": qty},
                "unit_price": {"value": float(price), "confidence": 0.9, "source_text": price},
                "line_total": {"value": float(ltotal), "confidence": 0.9, "source_text": ltotal},
            }
        )

    def numval(v):
        return float(v.replace(",", "")) if v else None

    return {
        "document_type": "invoice",
        "vendor_name": field(vendor, r"(VENDOR:\s*[A-Za-z0-9 &.,'-]+)"),
        "invoice_number": field(inv_num, r"(INVOICE\s*#\s*:?\s*[A-Za-z0-9-]+)"),
        "invoice_date": field(inv_date, r"(DATE:\s*\d{4}-\d{2}-\d{2})"),
        "currency": field(currency, r"(USD|EUR|GBP|INR)"),
        "subtotal_amount": field(numval(subtotal), r"(SUBTOTAL:\s*\$?[\d,]+\.\d{2})"),
        "tax_amount": field(numval(tax), r"(?<![A-Za-z])(TAX:\s*\$?[\d,]+\.\d{2})"),
        "total_amount": field(numval(total), r"(?<![A-Za-z])(TOTAL:\s*\$?[\d,]+\.\d{2})"),
        "line_items": line_items,
    }


def _call_anthropic(ocr_text: str, retry_note: str | None = None) -> str:
    import anthropic

    if not settings.anthropic_api_key:
        raise ExtractionError("ANTHROPIC_API_KEY is not set but llm_provider='anthropic'.")

    client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    user_content = f"OCR TEXT:\n---\n{ocr_text}\n---"
    if retry_note:
        user_content += f"\n\nYour previous response was invalid: {retry_note}\nReturn corrected JSON only."

    response = client.messages.create(
        model=settings.llm_model,
        max_tokens=2000,
        temperature=settings.llm_temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


def _call_gemini(ocr_text: str, retry_note: str | None = None) -> str:
    """Google's Gemini API free tier (via AI Studio, no credit card) is the
    option for a public deployment nobody has to pay for -- see README
    'Deploy for free'. Uses response_mime_type='application/json' rather
    than parsing markdown fences out of a text reply, which Gemini supports
    natively for structured output."""
    from google import genai
    from google.genai import types

    if not settings.gemini_api_key:
        raise ExtractionError("GEMINI_API_KEY is not set but llm_provider='gemini'.")

    client = genai.Client(api_key=settings.gemini_api_key)
    user_content = f"OCR TEXT:\n---\n{ocr_text}\n---"
    if retry_note:
        user_content += f"\n\nYour previous response was invalid: {retry_note}\nReturn corrected JSON only."

    response = client.models.generate_content(
        model=settings.gemini_model,
        contents=user_content,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=settings.llm_temperature,
        ),
    )
    return response.text or ""


def extract_fields(ocr_text: str) -> tuple[ExtractedInvoice, list[str]]:
    """Runs LLM (or mock) extraction, validates against the strict schema,
    and verifies every non-null field's source_text against the OCR text.
    Retries once on malformed JSON before raising ExtractionError.

    Returns (invoice, rejected_fields): rejected_fields names every field
    the model claimed a value for that failed the source-text check (and
    was consequently nulled out). The pipeline surfaces these explicitly
    rather than letting a rejected field look identical to one the model
    simply never attempted -- "we caught and discarded a hallucination" and
    "the model found nothing" are different situations for a reviewer."""
    last_error: str | None = None

    for attempt in range(settings.llm_max_retries + 1):
        if settings.llm_provider == "mock":
            raw = json.dumps(_mock_extract(ocr_text))
        elif settings.llm_provider == "gemini":
            raw = _call_gemini(ocr_text, retry_note=last_error)
        else:
            raw = _call_anthropic(ocr_text, retry_note=last_error)

        cleaned = raw.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            payload = json.loads(cleaned)
            invoice = ExtractedInvoice.model_validate(payload)
        except (json.JSONDecodeError, ValidationError) as exc:
            last_error = str(exc)
            logger.warning("Extraction attempt %d failed to parse/validate: %s", attempt, exc)
            continue

        rejected = _verify_sources(invoice, ocr_text)
        return invoice, rejected

    raise ExtractionError(f"LLM extraction failed after {settings.llm_max_retries + 1} attempts: {last_error}")


def _verify_sources(invoice: ExtractedInvoice, ocr_text: str) -> list[str]:
    """Mutates `invoice` in place: any field whose source_text isn't actually
    found in the OCR text gets its confidence zeroed and value nulled. This
    is the concrete anti-hallucination check requested in the brief.
    Returns the list of field names that were rejected this way."""
    rejected: list[str] = []
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
        if field.value is not None:
            if not field.source_text or not _similar(ocr_text, field.source_text):
                logger.info("Unsupported field '%s' (value=%r): source_text not found in OCR text.", name, field.value)
                field.confidence = 0.0
                field.value = None
                field.source_text = None
                rejected.append(name)

    for idx, item in enumerate(invoice.line_items):
        for sub_name in ("description", "quantity", "unit_price", "line_total"):
            field = getattr(item, sub_name)
            if field.value is not None and (not field.source_text or not _similar(ocr_text, field.source_text)):
                field.confidence = 0.0
                field.value = None
                field.source_text = None
                rejected.append(f"line_items[{idx}].{sub_name}")

    return rejected
