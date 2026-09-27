"""FastAPI application entrypoint."""
from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_settings
from app.db import init_db
from app.routes import documents, eval as eval_routes

settings = get_settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.app_name,
    description="OCR + LLM invoice/receipt extraction with automated cross-field "
    "validation and human-in-the-loop review.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # local/demo deployment; tighten for real production use
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    logger.info("%s starting up (environment=%s, llm_provider=%s)", settings.app_name, settings.environment, settings.llm_provider)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception on %s", request.url)
    return JSONResponse(status_code=500, content={"detail": "Internal server error. Check server logs."})


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "environment": settings.environment}


app.include_router(documents.router)
app.include_router(eval_routes.router)

app.mount("/review", StaticFiles(directory="static/review_ui", html=True), name="review_ui")

# Mounted last, and specifically at "/": Starlette checks routes in
# registration order and uses the first match, so every API route and the
# /review mount above are matched first. This mount only ever catches
# requests nothing else claimed (i.e. "/" itself, serving user_ui/index.html)
# -- it must stay the final line in this file, or it will shadow routes
# registered after it.
app.mount("/", StaticFiles(directory="static/user_ui", html=True), name="user_ui")
