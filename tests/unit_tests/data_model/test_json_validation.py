"""Tests for validation of finite JSON-compatible values."""

import math

import pytest

from simtools.data_model.json_validation import validate_finite_json_values


def test_validate_finite_json_values_accepts_nested_finite_values():
    """Finite numbers and booleans are valid JSON-compatible values."""
    validate_finite_json_values({"records": [{"value": 1.0, "enabled": True}]})


def test_validate_finite_json_values_reports_nested_path():
    """The error identifies the nested location of a non-finite number."""
    with pytest.raises(ValueError, match=r"\$\.records\[0\]\.value"):
        validate_finite_json_values({"records": [{"value": math.nan}]})
