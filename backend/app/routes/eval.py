"""Expose the eval harness over HTTP so a fresh clone's docker-compose
deployment can produce metrics without shelling into the container."""
from __future__ import annotations

from fastapi import APIRouter

from eval.run_eval import run_eval

router = APIRouter(prefix="/eval", tags=["eval"])


@router.post("/run")
def trigger_eval(sample_limit: int | None = None) -> dict:
    """Runs the held-out eval set through the full pipeline and returns
    field-level accuracy + false-flag rate. See eval/run_eval.py for the
    scoring logic and eval/eval_set.csv for the ground truth."""
    return run_eval(sample_limit=sample_limit)
