"""Tests for model-parameter parsing helpers."""

import pytest

from simtools.model_repository import parsing


@pytest.mark.parametrize(
    ("parameter_type", "is_integer"),
    [("int64", True), ("float64", False), ("", False)],
)
def test_normalize_model_parameter_normalizes_value_and_unit(mocker, parameter_type, is_integer):
    """Normalize a copy while preserving the input document."""
    split = mocker.patch.object(
        parsing.value_conversion, "split_value_and_unit", return_value=(42, "cm")
    )
    get_value = mocker.patch.object(
        parsing.value_conversion, "get_value_unit_type", return_value=(42, "m", "float")
    )
    normalize_unit = mocker.patch.object(
        parsing.value_conversion, "normalize_model_parameter_unit", return_value="m"
    )
    data = {"unit": "cm", "value": "42 cm"}
    if parameter_type:
        data["type"] = parameter_type

    result = parsing.normalize_model_parameter(data)

    expected = {"unit": "m", "value": 42}
    original = {"unit": "cm", "value": "42 cm"}
    if parameter_type:
        expected["type"] = parameter_type
        original["type"] = parameter_type
    assert result == expected
    assert data == original
    split.assert_called_once_with("42 cm", is_integer)
    get_value.assert_called_once_with(value=42, unit_str="cm")
    normalize_unit.assert_called_once_with(42, "m")
