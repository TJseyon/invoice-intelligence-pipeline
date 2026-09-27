"""
OCR layer: image preprocessing + Tesseract text extraction.

Preprocessing is PIL + numpy only (no OpenCV): OpenCV adds real memory
overhead (opencv-python-headless alone is a 60-90MB import with a heavy
resident footprint), which matters when this runs on a free-tier host with
a hard RAM ceiling (e.g. Render's free web service caps a container at
512MB). Deskewing uses a small projection-profile search instead of
OpenCV's minAreaRect -- verified to match or beat the OpenCV version's
accuracy on rotated scans (see docs/architecture.md), at a fraction of the
dependency weight.

Handles two of the named edge cases directly:
  - empty / unreadable documents: if extracted text is below a character
    threshold, we short-circuit before ever calling the LLM (saves cost and
    gives a precise status instead of a confusing empty extraction).
  - non-English text: langdetect runs on the OCR output; if the detected
    language isn't in the configured tesseract language set, we flag it
    rather than silently mis-extracting.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass

import numpy as np
import pytesseract
from langdetect import DetectorFactory, LangDetectException, detect
from PIL import Image, ImageOps, UnidentifiedImageError

from app.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# langdetect is otherwise non-deterministic across runs
DetectorFactory.seed = 0


class OCRError(Exception):
    """Raised for unrecoverable OCR failures (corrupt file, unsupported format)."""


@dataclass
class OCRResult:
    text: str
    confidence: float  # mean word-level confidence, 0-100
    is_empty: bool
    is_low_confidence: bool
    detected_language: str | None
    language_supported: bool


def _load_image(file_bytes: bytes, content_type: str) -> Image.Image:
    if content_type == "application/pdf":
        # Local import: pdf2image needs poppler installed (handled in Dockerfile).
        from pdf2image import convert_from_bytes

        pages = convert_from_bytes(file_bytes, dpi=300)
        if not pages:
            raise OCRError("PDF contained no pages.")
        return pages[0].convert("RGB")  # MVP: first page only

    try:
        return Image.open(io.BytesIO(file_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise OCRError(f"File could not be read as an image: {exc}") from exc


def _estimate_skew_angle(grayscale: Image.Image) -> float:
    """Projection-profile skew estimation: rotate a small thumbnail across a
    +/-10 degree sweep and keep the angle whose horizontal row-sum profile
    has the highest variance -- text lines snap into sharp, high-contrast
    bands exactly at the correct angle, blurring together at any other.
    Pure PIL + numpy (no OpenCV): a ~40-iteration search over a downsampled
    thumbnail runs in a fraction of a second and needs no heavy CV library,
    which matters on memory-capped free-tier hosting."""
    small = grayscale.copy()
    small.thumbnail((500, 500))
    arr = np.array(small, dtype=np.float64)

    threshold = arr.mean() - arr.std() * 0.5
    ink_pixels = int((arr < threshold).sum())
    if ink_pixels < 50:
        return 0.0  # near-blank page/thumbnail: nothing to align to

    best_angle, best_score = 0.0, -1.0
    for angle in np.arange(-10, 10.25, 0.25):
        rotated = small.rotate(float(angle), expand=False, fillcolor=255, resample=Image.BILINEAR)
        rotated_arr = np.array(rotated, dtype=np.float64)
        row_sums = (rotated_arr < threshold).sum(axis=1)
        score = row_sums.var()
        if score > best_score:
            best_score, best_angle = score, float(angle)
    return best_angle


def _preprocess(image: Image.Image) -> Image.Image:
    """Grayscale -> contrast stretch -> deskew. Cheap operations that
    measurably help Tesseract on phone-camera photos and rotated scans."""
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    angle = _estimate_skew_angle(contrasted)
    if abs(angle) < 0.4:  # not worth rotating for negligible skew
        return contrasted
    return contrasted.rotate(angle, expand=True, fillcolor="white", resample=Image.BICUBIC)


def run_ocr(file_bytes: bytes, content_type: str) -> OCRResult:
    if settings.ocr_engine != "tesseract":
        # Only one engine is wired up today; this guard is the extension
        # point for a cloud-OCR backend (see docs/architecture.md decision
        # #1) rather than silently ignoring an unsupported config value.
        raise OCRError(f"Unsupported OCR engine '{settings.ocr_engine}'.")

    image = _load_image(file_bytes, content_type)
    processed = _preprocess(image)

    data = pytesseract.image_to_data(
        processed, lang=settings.ocr_languages, output_type=pytesseract.Output.DICT
    )
    words = [w for w in data["text"] if w.strip()]
    confidences = [float(c) for c, w in zip(data["conf"], data["text"]) if w.strip() and float(c) >= 0]

    text = " ".join(words).strip()
    mean_conf = sum(confidences) / len(confidences) if confidences else 0.0
    is_empty = len(text) < settings.ocr_min_chars

    detected_lang, supported = None, True
    if not is_empty and settings.auto_detect_language:
        try:
            detected_lang = detect(text)
        except LangDetectException:
            detected_lang = None
        configured_langs = {c[:2] for c in settings.ocr_languages.split("+")}
        # "eng" -> "en" for comparison with langdetect's ISO-639-1 codes
        configured_iso = {"eng": "en", "fra": "fr", "deu": "de", "spa": "es"}.get(
            settings.ocr_languages.split("+")[0], settings.ocr_languages[:2]
        )
        supported = detected_lang is None or detected_lang == configured_iso

    return OCRResult(
        text=text,
        confidence=mean_conf,
        is_empty=is_empty,
        is_low_confidence=(not is_empty) and mean_conf < settings.ocr_min_confidence,
        detected_language=detected_lang,
        language_supported=supported,
    )
