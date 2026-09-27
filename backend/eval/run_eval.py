"""
Eval harness: runs every document in the held-out eval set through the exact
same `process_document` function the API uses, scores field-level accuracy
and the false-flag rate, and logs every input/output pair to CSV.

Metrics produced (measured, not invented -- see docs/architecture.md for
how the eval set itself was built):

  field_level_accuracy
      Of all (field, document) pairs where we have a manually-verified
      ground truth value, what fraction did the pipeline extract correctly?
      Computed only over documents where a ground-truth value exists (i.e.
      "normal" and "noisy" categories; blank/non-English docs are scored
      separately as a status-detection problem, see below).

  status_detection_accuracy
      For documents whose *expected outcome* is a specific pipeline status
      (blank pages -> "empty_document", French text -> "unsupported_language"),
      did the pipeline assign that status?

  flag_catch_rate (recall on "bad extractions")
      Of the documents that actually contain an extraction error (a field
      mismatch, or the wrong status), what fraction did the pipeline flag
      for human review? This is the "false-flag rate" the brief asks for,
      reported as a catch rate: 1.0 = never silently ships a bad extraction.

  flag_false_positive_rate
      Of the documents that were extracted correctly, what fraction got
      needlessly flagged for review? High values mean reviewers waste time
      on documents that didn't need a human.

Run directly:
    python -m eval.run_eval
or via the API:
    POST /eval/run
"""
from __future__ import annotations

import csv
import datetime as dt
import json
import math
from pathlib import Path
from typing import Any

from app.config import get_settings
from app.pipeline import process_document

settings = get_settings()

GROUND_TRUTH_FIELDS = [
    "vendor_name", "invoice_number", "invoice_date", "currency",
    "subtotal_amount", "tax_amount", "total_amount",
]
NUMERIC_FIELDS = {"subtotal_amount", "tax_amount", "total_amount"}
NUMERIC_TOLERANCE = 0.01  # absolute dollar tolerance for float field matches


def _norm_str(v: Any) -> str:
    return str(v).strip().lower() if v not in (None, "") else ""


def _field_matches(expected: Any, actual: Any, field: str) -> bool:
    if expected in (None, "") and actual in (None, ""):
        return True
    if expected in (None, "") or actual in (None, ""):
        return False
    if field in NUMERIC_FIELDS:
        try:
            return math.isclose(float(expected), float(actual), abs_tol=NUMERIC_TOLERANCE)
        except (TypeError, ValueError):
            return False
    return _norm_str(expected) == _norm_str(actual)


def _load_eval_set(csv_path: Path) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


def run_eval(sample_limit: int | None = None, log_results: bool = True) -> dict:
    eval_csv = Path(settings.eval_set_path)
    if not eval_csv.exists():
        raise FileNotFoundError(
            f"Eval set not found at {eval_csv}. Generate it with: "
            f"python scripts/generate_sample_data.py --out backend/eval/data"
        )
    rows = _load_eval_set(eval_csv)
    if sample_limit:
        rows = rows[:sample_limit]

    data_dir = eval_csv.parent
    field_total, field_correct = 0, 0
    status_total, status_correct = 0, 0

    bad_extraction_flagged, bad_extraction_total = 0, 0  # catch rate numerator/denominator
    good_extraction_flagged, good_extraction_total = 0, 0  # false-positive numerator/denominator

    log_rows: list[dict] = []

    for row in rows:
        image_path = data_dir / row["filename"]
        if not image_path.exists():
            continue
        file_bytes = image_path.read_bytes()
        content_type = "image/png"

        result = process_document(file_bytes, content_type)
        extracted = result.extracted.model_dump() if result.extracted else None

        category = row["category"]
        is_status_case = category in ("blank", "non_english")
        doc_has_error = False

        if is_status_case:
            status_total += 1
            correct = result.status == row["expected_status"]
            status_correct += int(correct)
            doc_has_error = not correct
        else:
            for field in GROUND_TRUTH_FIELDS:
                expected_raw = row.get(field, "")
                if expected_raw == "":
                    continue  # this field wasn't hand-verified for this row; skip
                actual_val = extracted[field]["value"] if extracted else None
                field_total += 1
                match = _field_matches(expected_raw, actual_val, field)
                field_correct += int(match)
                if not match:
                    doc_has_error = True

        if doc_has_error:
            bad_extraction_total += 1
            bad_extraction_flagged += int(result.flagged_for_review)
        else:
            good_extraction_total += 1
            good_extraction_flagged += int(result.flagged_for_review)

        log_rows.append({
            "filename": row["filename"],
            "category": category,
            "expected_status": row.get("expected_status", ""),
            "actual_status": result.status,
            "flagged_for_review": result.flagged_for_review,
            "doc_had_error": doc_has_error,
            "ocr_confidence": result.ocr_confidence,
            "num_issues": len(result.issues),
            "extracted_json": json.dumps(extracted) if extracted else "",
        })

    field_accuracy = field_correct / field_total if field_total else None
    status_accuracy = status_correct / status_total if status_total else None
    catch_rate = bad_extraction_flagged / bad_extraction_total if bad_extraction_total else None
    false_positive_rate = good_extraction_flagged / good_extraction_total if good_extraction_total else None

    summary = {
        "run_at": dt.datetime.utcnow().isoformat(),
        "documents_evaluated": len(log_rows),
        "field_level_accuracy": field_accuracy,
        "field_pairs_scored": field_total,
        "status_detection_accuracy": status_accuracy,
        "status_cases_scored": status_total,
        "flag_catch_rate": catch_rate,
        "bad_extractions_total": bad_extraction_total,
        "flag_false_positive_rate": false_positive_rate,
        "good_extractions_total": good_extraction_total,
    }

    if log_results:
        results_dir = data_dir.parent / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        stamp = dt.datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        with open(results_dir / f"run_{stamp}.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()) if log_rows else [])
            writer.writeheader()
            writer.writerows(log_rows)

        with open(results_dir / f"run_{stamp}_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        with open(results_dir / "latest_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

    return summary


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save-baseline",
        action="store_true",
        help="Write this run's scores to eval/baseline_score.json for eval/test_eval.py "
        "to regression-test future runs against. Run this once after generating real "
        "(or synthetic) eval data, and again deliberately whenever an improvement should "
        "become the new floor.",
    )
    args = parser.parse_args()

    result = run_eval()
    print(json.dumps(result, indent=2))

    if args.save_baseline:
        with open(settings.eval_baseline_path, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nSaved as baseline -> {settings.eval_baseline_path}")
