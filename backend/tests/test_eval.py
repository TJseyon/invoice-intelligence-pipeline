"""
Eval test: NOT a unit test. This runs the full, stochastic pipeline against
the held-out eval set and checks that scored accuracy hasn't regressed
below the saved baseline (minus a small tolerance for noise).

Why this is a separate category from test_validation.py / test_api.py:
those check deterministic code paths (a given input always produces the
same output, so a plain assert is meaningful). This test's job is
different -- it checks that the *behavior of the AI component* hasn't
drifted, which is a statistical claim over a sample, not a single
input/output equality. Unit tests should run on every commit in seconds;
this one is slower and is the one you'd gate a release on, not every push.

This test SKIPS (not fails) if no baseline has been saved yet, since a
regression check against a placeholder would be meaningless. Populate the
baseline once with:
    python -m eval.run_eval --save-baseline
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.config import get_settings

settings = get_settings()


def _load_baseline() -> dict | None:
    path = Path(settings.eval_baseline_path)
    if not path.exists():
        return None
    with open(path) as f:
        data = json.load(f)
    if data.get("field_level_accuracy") is None:
        return None  # still the placeholder
    return data


@pytest.mark.skipif(
    not Path(get_settings().eval_set_path).exists(),
    reason="Eval set not generated yet. Run: python scripts/generate_sample_data.py",
)
def test_field_level_accuracy_has_not_regressed():
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip(
            "No eval baseline saved yet. Run: python -m eval.run_eval --save-baseline "
            "(after generating sample data) to establish one."
        )

    from eval.run_eval import run_eval

    current = run_eval(log_results=False)

    assert current["field_level_accuracy"] is not None, "Current run scored zero field pairs."
    floor = baseline["field_level_accuracy"] - settings.eval_regression_tolerance
    assert current["field_level_accuracy"] >= floor, (
        f"Field-level accuracy regressed: {current['field_level_accuracy']:.3f} "
        f"is below baseline {baseline['field_level_accuracy']:.3f} "
        f"minus tolerance {settings.eval_regression_tolerance}."
    )


@pytest.mark.skipif(
    not Path(get_settings().eval_set_path).exists(),
    reason="Eval set not generated yet. Run: python scripts/generate_sample_data.py",
)
def test_flag_catch_rate_has_not_regressed():
    """The catch rate (recall on genuinely bad extractions) is arguably more
    important than raw accuracy: silently shipping a wrong number is worse
    than flagging a correct one for a human to rubber-stamp."""
    baseline = _load_baseline()
    if baseline is None:
        pytest.skip("No eval baseline saved yet.")

    from eval.run_eval import run_eval

    current = run_eval(log_results=False)

    if baseline.get("flag_catch_rate") is None or current.get("flag_catch_rate") is None:
        pytest.skip("No bad-extraction cases in this eval set to score a catch rate on.")

    floor = baseline["flag_catch_rate"] - settings.eval_regression_tolerance
    assert current["flag_catch_rate"] >= floor, (
        f"Flag catch rate regressed: {current['flag_catch_rate']:.3f} is below "
        f"baseline {baseline['flag_catch_rate']:.3f} minus tolerance."
    )
