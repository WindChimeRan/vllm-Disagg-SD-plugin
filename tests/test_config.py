import pytest

from disagg_sd.config import RuntimeConfig


@pytest.mark.parametrize(
    "values",
    [
        {"polciy": "overlap"},
        {"policy": "unknown"},
        {"max_ahead": 0},
        {"draft_device": True},
        {"draft_timeout": float("inf")},
        {"draft_timeout": -1},
        {"trace_path": 7},
    ],
)
def test_invalid_options_fail_at_configuration(values):
    with pytest.raises(ValueError):
        RuntimeConfig.parse(values)
