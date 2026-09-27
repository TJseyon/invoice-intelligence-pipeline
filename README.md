# Invoice & Receipt Intelligence Pipeline

Turns messy invoice/receipt images into clean, validated, structured data —
OCR + LLM-based field extraction with automated cross-field validation,
a source-text hallucination check, and a human review queue for anything
that isn't trustworthy enough to ship unattended.

```
Upload → OCR (Tesseract + preprocessing) → LLM extraction (strict JSON schema)
       → hallucination check (source_text verified against OCR text)
       → validation (required fields, line-item sum, confidence, date)
       → flag for review, or accept
```

See [`docs/architecture.md`](docs/architecture.md) for the full diagram and
the reasoning behind three key design decisions (Tesseract vs. cloud OCR,
Postgres vs. SQLite, sync vs. async).

## Why flag vs. trust an extraction

Every extracted field carries a **confidence score** and a **verbatim
source_text snippet** the model claims to have read the value from. Before
anything is trusted:

1. The source_text is checked against the raw OCR text. If it isn't
   actually there, the field is nulled out and logged as a distinct
   `unsupported_field_value` issue — a confident-sounding but unsupported
   field is treated as *rejected*, not silently indistinguishable from a
   field the model never attempted.
2. Required fields (vendor, invoice #, date, total) must be present.
3. Line items must sum to the pre-tax subtotal within a configurable
   tolerance.
4. Per-field confidence must clear a configurable threshold.
5. Dates must parse.

Any failure (or borderline confidence) sends the document to
`/review` for a human, who corrects it in one click. Corrections are logged
to an append-only table and aggregated at `GET /documents/stats/corrections`
— a field that gets corrected disproportionately often is the concrete
signal to tighten that field's validation rule.

## Quickstart

```bash
git clone <this-repo>
cd invoice-intelligence-pipeline
docker compose up --build
```

- **App (for anyone to use):** http://localhost:8000/
- **Reviewer queue:** http://localhost:8000/review
- **API docs:** http://localhost:8000/docs
- **Health check:** http://localhost:8000/health

**Two frontends, two audiences:**
- **`/` — the app.** A plain-language, black-and-white "ledger" interface
  anyone can use: drop in a document, get a readable result (vendor,
  amounts, dates, line items) with a plain-English explanation whenever
  something needs a second look — styled around the actual subject matter
  (perforated tear-lines, a ledger table, an ink stamp for status) rather
  than a generic dashboard. `backend/static/user_ui/index.html`.
- **`/review` — reviewer tools.** The technical queue for correcting
  flagged extractions, described above. `backend/static/review_ui/index.html`.

Both are static, dependency-free HTML/CSS/JS — no build step, no framework
— served directly by FastAPI.

By default the pipeline runs with `LLM_PROVIDER=mock` — a deterministic,
regex-based stand-in for the LLM call (see `_mock_extract` in
`backend/app/extraction.py`) so the **entire system runs offline, for free,
out of the box**. To use real LLM extraction, copy `.env.example` to `.env`,
set `LLM_PROVIDER=anthropic` and `ANTHROPIC_API_KEY=...`, and re-run
`docker compose up --build`.

Try it immediately: generate sample invoices and upload one through
http://localhost:8000/docs (`POST /documents/upload`):

```bash
python3 scripts/generate_sample_data.py --out backend/eval/data --count 30
```

## Project layout

```
backend/
  app/
    config.py        # every threshold/setting, sourced from env vars
    models.py         # strict Pydantic schema (ExtractedField w/ confidence + source_text)
    db.py              # SQLAlchemy models: documents, corrections
    ocr.py              # Tesseract + OpenCV preprocessing, language detection
    extraction.py        # LLM call, JSON-schema validation, hallucination check
    validation.py         # cross-field rules, review-flagging logic
    pipeline.py            # orchestrates OCR -> extraction -> validation
    routes/                # FastAPI routers (documents, eval)
    main.py                 # app entrypoint
  eval/
    run_eval.py         # scores the pipeline against eval/data/eval_set.csv
    baseline_score.json  # regression baseline (populated by --save-baseline, not hand-edited)
  tests/
    test_validation.py  # unit tests: deterministic validation logic
    test_extraction.py  # unit tests: hallucination guard
    test_api.py          # integration tests: routes, input validation, DB
    test_eval.py          # eval test: full pipeline vs. eval set, regression-checked
  static/
    user_ui/                 # the black-and-white app anyone uploads through ("/")
    review_ui/                 # reviewer queue ("/review")
  pytest.ini                   # pythonpath = . -- see "Running tests" for why
scripts/
  generate_sample_data.py  # synthetic invoices + ground truth (no network needed)
docs/architecture.md        # diagram + reasoned design decisions
```

## Handling edge cases (not just the happy path)

| Case | Behavior |
|---|---|
| Blank / unreadable document | OCR text below `OCR_MIN_CHARS` → `status: empty_document`, LLM never called, flagged for review |
| Non-English text | `langdetect` disagrees with configured `OCR_LANGUAGES` → `status: unsupported_language`, flagged, LLM skipped |
| Malformed LLM output | Invalid JSON / schema violation → one retry with an explicit correction prompt, then `status: extraction_failed` |
| Hallucinated field | `source_text` not found in OCR text → field nulled, confidence zeroed, logged as `unsupported_field_value` |
| Low OCR image quality | Mean word confidence below `OCR_MIN_CONFIDENCE` → `low_ocr_confidence` warning, flagged even if the LLM was confident |
| Wrong file type / oversized / 0-byte upload | Rejected at the API boundary with a specific 4xx and message, before OCR runs |
| Corrupt image / unreadable PDF | `OCRError` → `422` with a clear message |
| Line items don't sum to subtotal | `error`-severity issue, flagged (see `docs/architecture.md` for why subtotal, not tax-inclusive total) |

## Running tests

**Inside Docker (recommended — no local Python setup needed):**
```bash
docker compose exec backend pytest -v
docker compose exec backend pytest -v -k "not eval"   # unit + API tests only
docker compose exec backend pytest -v tests/test_eval.py
```

**Locally instead:**
```bash
cd backend
pip install -r requirements.txt
pytest -v
```

`backend/pytest.ini` sets `pythonpath = .` so `from app.db import ...` etc.
resolve correctly no matter how pytest is invoked. Without it, running the
bare `pytest` command (as opposed to `python -m pytest`) raises
`ModuleNotFoundError: No module named 'app'` — `uvicorn app.main:app`
doesn't have this problem because uvicorn inserts the current directory
into `sys.path` itself when resolving its import string, but the `pytest`
console script doesn't do that automatically. If you see that error, you
have an older copy of this project without `pytest.ini`.

**Unit tests** (`test_validation.py`, `test_extraction.py`) check pure,
deterministic functions — same input always produces the same output, so a
plain assert is meaningful. **API tests** (`test_api.py`) run the real
FastAPI app end-to-end (mocked LLM, real OCR, real DB) against a TestClient.
**The eval test** (`test_eval.py`) is a different kind of test: it runs the
full pipeline against a scored dataset and checks that accuracy hasn't
*regressed* below a saved baseline, which is a statistical claim over a
sample rather than a single-input equality check — this is why it's kept in
its own file and treated as a separate CI stage from the unit suite.

## Running the eval

```bash
python3 scripts/generate_sample_data.py --out backend/eval/data --count 30
cd backend
python3 -m eval.run_eval                    # print metrics, log every input/output pair
python3 -m eval.run_eval --save-baseline     # also save as the regression baseline
```

**How the eval set was built:** `scripts/generate_sample_data.py` renders
synthetic invoices with known ground truth (exact vendor/date/amounts,
since we drew the text ourselves), split across normal, blank, non-English,
and rotated/low-contrast categories. This is a legitimate way to get a
scored dataset with zero licensing/PII concerns and no manual data entry —
but it is honestly a stand-in for real manually-verified invoices, not a
replacement. **Before citing these numbers on a resume as a production
claim, swap in 20-30 real, manually-verified held-out invoices** (drop
images + a filled-in `eval_set.csv` into `backend/eval/data/`, same
format).

**Metrics measured** (see `docs/architecture.md` for definitions):
`field_level_accuracy`, `status_detection_accuracy`, `flag_catch_rate`
(did we catch bad extractions?), `flag_false_positive_rate` (did we
needlessly flag good ones?). Every run writes a full per-document log to
`backend/eval/results/run_<timestamp>.csv` for manual/spreadsheet review —
"score 30 rows yourself" is a legitimate way to validate the automated
scoring logic before trusting it.

I have not pre-populated `baseline_score.json` with numbers from the
official harness, because I can't run the real FastAPI/pydantic stack
in the sandbox this was built in (no network, couldn't install
`fastapi`/`pydantic`/`langdetect`) — see the note below. I did, however,
write a standalone reimplementation of the core logic (OCR call + mock
extraction regex + field scoring, no pydantic/FastAPI) and ran it against
all 28 generated sample documents to sanity-check the algorithm before
shipping it:

```
field_level_accuracy:        164/168 = 0.976   (normal + noisy categories, 7 fields x 24 docs)
status_detection_accuracy:     4/4   = 1.000   (2 blank docs -> empty_document, 2 French docs -> unsupported_language)
```

The 4 misses were all on the deliberately rotated/low-contrast "noisy"
images, where Tesseract itself misread the literal anchor text (e.g.
`"VENDOR:"` came back as `"ENDOR:"` after a bad rotation), so the
regex-based **mock** extractor couldn't find its anchor. This is expected
and is exactly what that category is designed to stress-test — a real LLM
(`LLM_PROVIDER=anthropic`) infers "ENDOR: Cascade Office Supplies" is a
vendor name from context and would very likely get these right where the
regex mock can't; that gap is itself a legitimate, concrete example of
LLM extraction outperforming rule-based extraction on noisy OCR, worth
mentioning in an interview. Run `make eval-baseline` after your first
`docker compose up` to get the *official*, code-path-verified numbers
(under a minute) — that's what should go on a resume, not this smoke test.

## What was actually verified before you got this

Being upfront about testing depth, since "no errors, working" was the ask:

- **Verified live, in a sandbox with Tesseract/OpenCV/Pillow but no network:**
  OCR + preprocessing (denoise/threshold/deskew) against real generated
  sample invoices — clean invoices OCR near-perfectly, blank pages produce
  0 characters, rotated + low-contrast images still OCR well after
  deskewing, French text extracts cleanly for the language-mismatch path.
  Also ran a standalone reimplementation of the scoring logic across all
  28 generated documents end-to-end (see "Running the eval" above for the
  actual numbers this produced). This process caught and fixed **two real
  regex bugs** in the mock extractor (invoice_number matching the word
  "VENDOR"; TOTAL matching inside LINETOTAL/SUBTOTAL) and **one real logic
  bug** in validation (line items were being compared against the
  tax-inclusive total instead of the pre-tax subtotal, which would have
  false-flagged every invoice with tax).
- **Verified by static analysis:** every Python file parses cleanly
  (`ast.parse`); imports, route paths, and Pydantic/SQLAlchemy model shapes
  were manually traced for consistency.
- **Not yet run end-to-end:** the full FastAPI + Postgres stack via Docker,
  since this environment couldn't install `fastapi`/`pydantic`/`sqlalchemy`
  or reach a package index. Run `docker compose up --build` followed by
  `pytest -v` (30 tests) as your first step — if anything fails, it's a
  fast, well-isolated fix from here rather than a rewrite.

## Resume bullet

> Built an invoice-extraction pipeline combining OCR and LLM-based field
> extraction with automated cross-field validation and a source-verified
> hallucination check, flagging N% of documents for human review
> (measured on a held-out eval set: field-level accuracy X%, review-flag
> catch rate Y%). *(Fill in N/X/Y from your own `eval-baseline` run.)*
