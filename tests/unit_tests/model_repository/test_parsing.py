"""Tests for model-parameter parsing helpers."""

import math

from simtools.model_repository.parsing import normalize_model_parameter


def test_normalize_model_parameter_preserves_input_and_normalizes_unit():
    """Normalize an actual integer parameter while preserving its document."""
    data = {"type": "int64", "unit": "cm", "value": "42"}

    result = normalize_model_parameter(data)

    assert result == {"type": "int64", "unit": "cm", "value": 42}
    assert data == {"type": "int64", "unit": "cm", "value": "42"}


def test_normalize_model_parameter_preserves_file_name_case():
    """File-backed values retain the case needed to locate their assets."""
    data = {"type": "string", "file": True, "unit": None, "value": "Model.DAT"}

    assert normalize_model_parameter(data)["value"] == "Model.DAT"


def test_normalize_model_parameter_preserves_mixed_list_values():
    """Preserve strings, numbers and units in a mixed parameter."""
    data = {
        "type": ["string", "float64", "float64"],
        "file": False,
        "unit": [None, "ns", "ns"],
        "value": ["Gauss-Exponential", 0.4, 1.59],
    }

    result = normalize_model_parameter(data)

    assert result["value"][0] == "Gauss-Exponential"
    assert math.isclose(result["value"][1], data["value"][1], abs_tol=1e-12)
    assert math.isclose(result["value"][2], data["value"][2], abs_tol=1e-12)
    assert result["unit"] == data["unit"]
    assert data["value"][0] == "Gauss-Exponential"


def test_normalize_model_parameter_preserves_string_lists():
    """Preserve each string in a list-valued parameter."""
    data = {"type": "string", "file": False, "unit": None, "value": ["Majority", "SUM"]}

    assert normalize_model_parameter(data)["value"] == data["value"]


def test_normalize_model_parameter_preserves_file_list_case():
    """Preserve file names even when they are supplied as a list."""
    data = {"type": "file", "file": True, "unit": None, "value": ["First.DAT", "Second.DAT"]}

    assert normalize_model_parameter(data)["value"] == data["value"]
