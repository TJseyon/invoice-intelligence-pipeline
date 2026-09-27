"""Shared pytest fixtures.

Uses an in-memory SQLite DB for tests so the suite never touches the dev/
prod database, and forces llm_provider='mock' so tests run with zero
network access and zero API cost -- required for CI.
"""
from __future__ import annotations

import os

os.environ.setdefault("LLM_PROVIDER", "mock")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("ENVIRONMENT", "test")

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.db import Base, get_db
from app.main import app


@pytest.fixture()
def db_session():
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    Base.metadata.create_all(bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as c:
        yield c
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_invoice_png() -> bytes:
    """A minimal synthetic invoice image generated on the fly, so tests
    don't depend on files existing outside the repo."""
    import io

    from PIL import Image, ImageDraw, ImageFont

    img = Image.new("RGB", (700, 500), "white")
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    except OSError:
        font = ImageFont.load_default()

    lines = [
        "INVOICE",
        "VENDOR: Test Vendor Co",
        "INVOICE #: INV-0001",
        "DATE: 2026-01-15",
        "CURRENCY: USD",
        "ITEM: Widget A  QTY: 2  PRICE: $10.00  LINETOTAL: $20.00",
        "SUBTOTAL: $20.00",
        "TAX: $1.60",
        "TOTAL: $21.60",
    ]
    y = 20
    for line in lines:
        d.text((20, y), line, font=font, fill="black")
        y += 30

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()
