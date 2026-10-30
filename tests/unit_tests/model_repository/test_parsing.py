"""Tests for model-parameter parsing helpers."""

from simtools.model_repository.parsing import normalize_model_parameter


def test_normalize_model_parameter_preserves_input_and_normalizes_unit():
    """Normalize an actual integer parameter while preserving its document."""
    data = {"type": "int64", "unit": "cm", "value": "42"}

    result = normalize_model_parameter(data)

    assert result == {"type": "int64", "unit": "cm", "value": 42}
    assert data == {"type": "int64", "unit": "cm", "value": "42"}
