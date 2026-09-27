"""
API/integration tests: routes, input validation, DB persistence. Runs the
real pipeline end-to-end but with llm_provider='mock' (set in conftest.py),
so these exercise real OCR + real validation logic without any network call
or API cost.
"""
from __future__ import annotations

import io

from PIL import Image


def test_health_check(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_upload_rejects_unsupported_content_type(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("notes.txt", b"hello world", "text/plain")},
    )
    assert resp.status_code == 415


def test_upload_rejects_empty_file(client):
    resp = client.post(
        "/documents/upload",
        files={"file": ("empty.png", b"", "image/png")},
    )
    assert resp.status_code == 400


def test_upload_rejects_oversized_file(client, monkeypatch):
    from app import config

    # Shrink the limit rather than generating a 15MB fixture.
    monkeypatch.setattr(config.get_settings(), "max_upload_mb", 0.0001)
    big_bytes = b"0" * 2000
    resp = client.post(
        "/documents/upload",
        files={"file": ("big.png", big_bytes, "image/png")},
    )
    assert resp.status_code == 413


def test_upload_blank_image_returns_empty_document_status(client):
    img = Image.new("RGB", (200, 200), "white")
    buf = io.BytesIO()
    img.save(buf, format="PNG")

    resp = client.post(
        "/documents/upload",
        files={"file": ("blank.png", buf.getvalue(), "image/png")},
    )
    assert resp.status_code == 201
    body = resp.json()
    assert body["status"] == "empty_document"
    assert body["flagged_for_review"] is True


def test_upload_valid_invoice_gets_persisted_and_listed(client, sample_invoice_png):
    resp = client.post(
        "/documents/upload",
        files={"file": ("invoice.png", sample_invoice_png, "image/png")},
    )
    assert resp.status_code == 201

    listing = client.get("/documents")
    assert listing.status_code == 200
    assert len(listing.json()) >= 1


def test_get_nonexistent_document_returns_404(client):
    resp = client.get("/documents/999999")
    assert resp.status_code == 404


def test_corrections_endpoint_updates_document(client, sample_invoice_png):
    upload = client.post(
        "/documents/upload",
        files={"file": ("invoice.png", sample_invoice_png, "image/png")},
    )
    doc_id = client.get("/documents").json()[0]["id"]

    resp = client.post(
        f"/documents/{doc_id}/corrections",
        json={"corrected_fields": {"vendor_name": "Corrected Vendor Inc"}, "reviewer_note": "typo fix"},
    )
    assert resp.status_code == 200
    assert resp.json()["fields_corrected"] == 1

    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["status"] == "corrected"
    assert detail["extracted"]["vendor_name"]["value"] == "Corrected Vendor Inc"


def test_correction_stats_endpoint(client, sample_invoice_png):
    client.post("/documents/upload", files={"file": ("invoice.png", sample_invoice_png, "image/png")})
    doc_id = client.get("/documents").json()[0]["id"]
    client.post(
        f"/documents/{doc_id}/corrections",
        json={"corrected_fields": {"total_amount": 999.0}},
    )
    stats = client.get("/documents/stats/corrections")
    assert stats.status_code == 200
    body = stats.json()
    assert body["total_corrections"] == 1
    assert body["corrections_by_field"]["total_amount"] == 1


def test_uploaded_file_is_persisted_and_servable(client, sample_invoice_png, tmp_path, monkeypatch):
    from app import config

    # get_settings() is an lru_cache singleton, so this patches the same
    # instance every module in the app (including routes/documents.py) holds.
    monkeypatch.setattr(config.get_settings(), "upload_dir", str(tmp_path))

    client.post(
        "/documents/upload",
        files={"file": ("invoice.png", sample_invoice_png, "image/png")},
    )
    doc_id = client.get("/documents").json()[0]["id"]
    detail = client.get(f"/documents/{doc_id}").json()
    assert detail["has_source_file"] is True

    file_resp = client.get(f"/documents/{doc_id}/file")
    assert file_resp.status_code == 200
    assert file_resp.content == sample_invoice_png


def test_get_file_404_when_never_uploaded_to_disk(client):
    resp = client.get("/documents/999999/file")
    assert resp.status_code == 404
