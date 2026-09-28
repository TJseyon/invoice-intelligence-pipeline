FROM python:3.12-slim

# System dependencies: tesseract for OCR, poppler for PDF->image, fonts for
# the sample-data generator. No OpenCV system libs (libgl1/libglib2.0-0)
# needed -- preprocessing is PIL + numpy only (see app/ocr.py), which keeps
# both the image and the running container meaningfully smaller. That
# matters on memory-capped free hosting (e.g. Render's free web service is
# capped at 512MB).
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-eng \
    tesseract-ocr-fra \
    poppler-utils \
    fonts-dejavu-core \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p uploaded_documents eval/results

# Tesseract's OpenMP threads thrash on a tiny shared CPU; one thread is faster there.
ENV OMP_THREAD_LIMIT=1

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s \
    CMD curl -f http://localhost:${PORT:-8000}/health || exit 1

# Shell form (not exec-form JSON array) so ${PORT:-8000} is expanded at
# container start. Render injects PORT (default 10000) and requires the
# app to bind to it; running locally via docker-compose without PORT set
# falls back to 8000, matching the "8000:8000" port mapping there.
CMD uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}