"""Tests for model-parameter output validation."""

import json
from pathlib import Path

from simtools.testing.output_validation import model_parameters


def test_validate_model_parameter_applies_scaling(tmp_test_directory, mocker):
    """Compare a generated parameter with a scaled database value."""
    output = Path(tmp_test_directory) / "parameter.json"
    output.write_text(json.dumps({"value": [1.0, 2.0]}), encoding="utf-8")
    database = mocker.patch.object(model_parameters.db_handler, "DatabaseHandler")
    database.return_value.get_model_parameter.return_value = {"parameter": {"value": [2.0, 4.0]}}

    model_parameters.validate(
        output,
        {"reference_parameter_name": "parameter", "tolerance": 1.0e-5, "scaling": 2.0},
        {"site": "North", "telescope": "LSTN-01", "model_version": "7.0.0"},
    )

    database.return_value.get_model_parameter.assert_called_once_with(
        parameter="parameter",
        site="North",
        array_element_name="LSTN-01",
        model_version="7.0.0",
    )
