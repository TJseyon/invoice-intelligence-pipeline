"""
Centralized configuration.

Everything that used to be a hardcoded constant scattered across the codebase
lives here instead, sourced from environment variables (with sane defaults for
local dev). This is the "small settings/config layer" called for in the
project brief: one place to change the OCR language, confidence thresholds,
DB location, or LLM model without touching business logic.
"""
from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- App ---
    app_name: str = "Invoice Intelligence Pipeline"
    environment: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"

    # --- Database ---
    database_url: str = Field(
        default="sqlite:///./invoice_pipeline.db",
        description="SQLAlchemy connection string. Postgres in docker-compose, "
        "SQLite for quick local/dev/test runs.",
    )

    # --- Upload / input validation ---
    max_upload_mb: int = 15
    allowed_content_types: tuple[str, ...] = (
        "image/png",
        "image/jpeg",
        "image/tiff",
        "application/pdf",
    )
    upload_dir: str = "./uploaded_documents"

    # --- OCR ---
    ocr_engine: Literal["tesseract"] = "tesseract"
    ocr_languages: str = "eng"  # tesseract lang codes, e.g. "eng+fra"
    ocr_min_confidence: float = 40.0  # tesseract word-level confidence (0-100)
    ocr_min_chars: int = 20  # below this -> treated as an empty/unreadable document
    auto_detect_language: bool = True

    # --- LLM extraction ---
    # "gemini" is the option to reach for a free public deployment: Google's
    # AI Studio free tier needs no credit card and no billing account, which
    # anthropic's API does not offer -- see README "Deploy for free" section.
    llm_provider: Literal["anthropic", "gemini", "mock"] = "mock"
    anthropic_api_key: str | None = None
    llm_model: str = "claude-sonnet-5"
    gemini_api_key: str | None = None
    gemini_model: str = "gemini-3.8-flash"
    llm_max_retries: int = 2
    llm_field_confidence_threshold: float = 0.75  # per-field LLM self-reported confidence
    llm_temperature: float = 0.0

    # --- Validation / review-flagging ---
    line_item_sum_tolerance_pct: float = 1.5  # allowed % drift between sum(line items) and total
    required_fields: tuple[str, ...] = (
        "vendor_name",
        "invoice_number",
        "invoice_date",
        "total_amount",
    )
    flag_if_any_field_low_confidence: bool = True
    flag_if_source_text_unverified: bool = True

    # --- Eval ---
    eval_set_path: str = "./eval/data/eval_set.csv"
    eval_baseline_path: str = "./eval/baseline_score.json"
    eval_regression_tolerance: float = 0.05  # allowed drop in accuracy before a test fails


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton; FastAPI dependencies pull from here."""
    return Settings()