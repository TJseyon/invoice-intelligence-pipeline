"""
SQLAlchemy models + session handling.

Two tables:
  - documents: one row per uploaded file, with the full extraction result
    stashed as JSON (Postgres JSONB / SQLite TEXT) so we never lose the raw
    LLM output even as the Pydantic schema evolves.
  - corrections: every human correction to a flagged field, kept as an
    append-only log. This is the feedback data the "standout add-on" (review
    UI -> corrections feed back into validation rules) is built on.
"""
from __future__ import annotations

import datetime as dt
import json
from typing import Generator

from sqlalchemy import JSON, DateTime, Float, ForeignKey, String, Boolean, create_engine, Text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker

from app.config import get_settings

settings = get_settings()

connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


class Base(DeclarativeBase):
    pass


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(512))
    content_type: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(64), default="pending")
    flagged_for_review: Mapped[bool] = mapped_column(Boolean, default=False)

    raw_ocr_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    ocr_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    detected_language: Mapped[str | None] = mapped_column(String(16), nullable=True)
    stored_file_path: Mapped[str | None] = mapped_column(String(1024), nullable=True)

    extracted_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # ExtractedInvoice.model_dump_json()
    issues_json: Mapped[str | None] = mapped_column(Text, nullable=True)  # list[ValidationIssue] as JSON

    vendor_name: Mapped[str | None] = mapped_column(String(256), nullable=True)
    total_amount: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    corrections: Mapped[list["Correction"]] = relationship(back_populates="document")

    def extracted_dict(self) -> dict | None:
        return json.loads(self.extracted_json) if self.extracted_json else None

    def issues_list(self) -> list:
        return json.loads(self.issues_json) if self.issues_json else []


class Correction(Base):
    __tablename__ = "corrections"

    id: Mapped[int] = mapped_column(primary_key=True, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id"))
    field_name: Mapped[str] = mapped_column(String(128))
    original_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    corrected_value: Mapped[str | None] = mapped_column(Text, nullable=True)
    reviewer_note: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[dt.datetime] = mapped_column(DateTime, default=dt.datetime.utcnow)

    document: Mapped["Document"] = relationship(back_populates="corrections")


def init_db() -> None:
    Base.metadata.create_all(bind=engine)


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
