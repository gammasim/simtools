#!/usr/bin/python3

import pytest

from simtools.model.legacy_model_parameter import (
    UPDATE_HANDLERS,
    _convert_column_data_to_row_data,
    _get_unsupported_update_message,
    _update_dsum_threshold,
    _update_fadc_pulse_shape,
    _update_flasher_pulse_shape,
    apply_legacy_updates_to_parameters,
    register_update,
    update_parameter,
)


def test_register_update():
    """Test register_update decorator."""

    @register_update("test_parameter")
    def test_handler(*_args, **_kwargs):
        return {"test": "value"}

    assert "test_parameter" in UPDATE_HANDLERS
    assert UPDATE_HANDLERS["test_parameter"] is test_handler

    # Clean up
    del UPDATE_HANDLERS["test_parameter"]


def test_get_unsupported_update_message():
    """Test _get_unsupported_update_message function."""

    para_data = {"parameter": "test_param", "model_parameter_schema_version": "0.1.0"}
    schema_version = "0.2.0"

    message = _get_unsupported_update_message(para_data, schema_version)

    assert "Unsupported update for legacy parameter test_param" in message
    assert "0.1.0 to 0.2.0" in message


def test_update_flasher_pulse_shape():
    """Test _update_flasher_pulse_shape function."""
    parameters = {
        "flasher_pulse_shape": {
            "parameter": "flasher_pulse_shape",
            "value": "gaussian",
            "model_parameter_schema_version": "0.1.0",
        },
        "flasher_pulse_width": {"parameter": "flasher_pulse_width", "value": 5.0},
        "flasher_pulse_exp_decay": {"parameter": "flasher_pulse_exp_decay", "value": 10.0},
    }

    result = _update_flasher_pulse_shape(parameters, "0.2.0")

    assert "flasher_pulse_shape" in result
    assert result["flasher_pulse_shape"]["value"] == ["gaussian", 5.0, 10.0]
    assert result["flasher_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"
    assert result["flasher_pulse_shape"]["unit"] == [None, "ns", "ns"]
    assert result["flasher_pulse_shape"]["type"] == ["string", "float64", "float64"]
    assert result["flasher_pulse_width"]["remove_parameter"] is True
    assert result["flasher_pulse_exp_decay"]["remove_parameter"] is True


def test_update_dsum_threshold():
    """Test _update_dsum_threshold function."""
    parameters = {
        "dsum_threshold": {
            "parameter": "dsum_threshold",
            "value": "42",
            "model_parameter_schema_version": "0.1.0",
        }
    }

    result = _update_dsum_threshold(parameters, "0.2.0")

    assert "dsum_threshold" in result
    assert result["dsum_threshold"]["value"] == 42
    assert isinstance(result["dsum_threshold"]["value"], int)
    assert result["dsum_threshold"]["model_parameter_schema_version"] == "0.2.0"


def test_update_dsum_threshold_unsupported_version():
    """Test _update_dsum_threshold with unsupported version."""
    parameters = {
        "dsum_threshold": {
            "parameter": "dsum_threshold",
            "value": "42",
            "model_parameter_schema_version": "0.2.0",
        }
    }

    with pytest.raises(
        ValueError, match=r"Unsupported update for legacy parameter dsum_threshold.*0.2.0 to 0.3.0"
    ):
        _update_dsum_threshold(parameters, "0.3.0")


def test_update_parameter_with_registered_handler():
    """Test update_parameter with a registered handler."""
    parameters = {
        "dsum_threshold": {
            "parameter": "dsum_threshold",
            "value": "42",
            "model_parameter_schema_version": "0.1.0",
        }
    }

    result = _update_dsum_threshold(parameters, "0.2.0")

    assert "dsum_threshold" in result
    assert result["dsum_threshold"]["value"] == 42


def test_update_parameter_with_unregistered_handler():
    """Test update_parameter with an unregistered handler."""

    parameters = {
        "unknown_parameter": {
            "parameter": "unknown_parameter",
            "value": "test",
            "model_parameter_schema_version": "0.1.0",
        }
    }

    with pytest.raises(
        ValueError, match="Unsupported update for legacy parameter unknown_parameter"
    ):
        update_parameter("unknown_parameter", parameters, "0.2.0")


def test_apply_legacy_updates_to_parameters():
    """Test apply_legacy_updates_to_parameters function."""

    parameters = {"param1": {"value": 10}, "param2": {"value": 20}, "param3": {"value": 30}}

    legacy_updates = {
        "param1": {"value": 100},
        "param2": {"remove_parameter": True},
        "param4": {"value": 40},
    }

    apply_legacy_updates_to_parameters(parameters, legacy_updates)

    assert parameters["param1"]["value"] == 100
    assert "param2" not in parameters
    assert parameters["param3"]["value"] == 30


def test_update_flasher_pulse_shape_unsupported_version():
    """Test _update_flasher_pulse_shape with unsupported version."""
    parameters = {
        "flasher_pulse_shape": {
            "parameter": "flasher_pulse_shape",
            "value": "gaussian",
            "model_parameter_schema_version": "0.2.0",
        }
    }

    with pytest.raises(
        ValueError,
        match=r"Unsupported update for legacy parameter flasher_pulse_shape.*0.2.0 to 0.3.0",
    ):
        _update_flasher_pulse_shape(parameters, "0.3.0")


def test_update_parameter_returns_handler_result():
    """Test update_parameter returns the result from the registered handler."""
    parameters = {
        "flasher_pulse_shape": {
            "parameter": "flasher_pulse_shape",
            "value": "gaussian",
            "model_parameter_schema_version": "0.1.0",
        },
        "flasher_pulse_width": {"parameter": "flasher_pulse_width", "value": 5.0},
        "flasher_pulse_exp_decay": {"parameter": "flasher_pulse_exp_decay", "value": 10.0},
    }

    result = update_parameter("flasher_pulse_shape", parameters, "0.2.0")

    assert isinstance(result, dict)
    assert "flasher_pulse_shape" in result
    assert result["flasher_pulse_shape"]["value"] == ["gaussian", 5.0, 10.0]
    assert result["flasher_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"


def test_convert_column_data_to_row_data():
    """Test conversion from column-oriented legacy data to row-oriented data."""
    row_data = _convert_column_data_to_row_data(
        {
            "columns": ["time", "amplitude", "amplitude (low gain)"],
            "data": {
                "time": [0.0, 1.0],
                "amplitude": [0.1, 0.2],
                "amplitude (low gain)": [0.01, 0.02],
            },
        }
    )

    assert row_data == {
        "columns": ["time", "amplitude", "amplitude (low gain)"],
        "rows": [[0.0, 0.1, 0.01], [1.0, 0.2, 0.02]],
    }


@pytest.mark.parametrize("legacy_type", ["file", "string"])
def test_update_fadc_pulse_shape_from_file_to_embedded_row_data(mocker, legacy_type):
    """Test file-backed fadc_pulse_shape migration to embedded row data."""
    expected_value = {
        "columns": ["time", "amplitude"],
        "rows": [[0.0, 0.1], [1.0, 0.2]],
    }
    parameters = {
        "fadc_pulse_shape": {
            "parameter": "fadc_pulse_shape",
            "value": "pulse.dat",
            "model_parameter_schema_version": "0.1.0",
            "type": legacy_type,
            "file": True,
        }
    }
    resolver = mocker.Mock(return_value=expected_value)

    result = _update_fadc_pulse_shape(parameters, "0.2.0", value_resolver=resolver)

    resolver.assert_called_once_with("fadc_pulse_shape", "pulse.dat")
    assert result["fadc_pulse_shape"]["value"] == expected_value
    assert result["fadc_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"
    assert result["fadc_pulse_shape"]["type"] == "dict"
    assert result["fadc_pulse_shape"]["file"] is False


@pytest.mark.parametrize(
    ("legacy_value", "expected_value"),
    [
        (
            {
                "columns": ["time", "amplitude", "amplitude (low gain)"],
                "dtype": ["float64", "float64", "float64"],
                "unit": ["ns", "dimensionless", "dimensionless"],
                "data": {
                    "time": [0.0, 1.0],
                    "amplitude": [0.1, 0.2],
                    "amplitude (low gain)": [0.01, 0.02],
                },
            },
            {
                "columns": ["time", "amplitude", "amplitude (low gain)"],
                "rows": [[0.0, 0.1, 0.01], [1.0, 0.2, 0.02]],
            },
        ),
        (
            {
                "columns": ["time", "amplitude", "amplitude (low gain)"],
                "rows": [[0.0, 0.1, 0.01], [1.0, 0.2, 0.02]],
            },
            {
                "columns": ["time", "amplitude", "amplitude (low gain)"],
                "rows": [[0.0, 0.1, 0.01], [1.0, 0.2, 0.02]],
            },
        ),
    ],
)
def test_update_fadc_pulse_shape_embedded_formats(legacy_value, expected_value):
    """Test embedded fadc_pulse_shape migration for legacy and canonical dict layouts."""
    parameters = {
        "fadc_pulse_shape": {
            "parameter": "fadc_pulse_shape",
            "value": legacy_value,
            "model_parameter_schema_version": "0.2.0",
            "type": "dict",
            "file": False,
        }
    }

    result = _update_fadc_pulse_shape(parameters, "0.2.0")

    assert result["fadc_pulse_shape"]["value"] == expected_value
    assert result["fadc_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"


def test_update_parameter_passes_value_resolver(mocker):
    """Test update_parameter forwards the optional value_resolver to the handler."""
    parameters = {
        "fadc_pulse_shape": {
            "parameter": "fadc_pulse_shape",
            "value": "pulse.dat",
            "model_parameter_schema_version": "0.1.0",
            "type": "file",
            "file": True,
        }
    }
    resolver = mocker.Mock(return_value={"columns": ["time", "amplitude"], "rows": [[0.0, 0.1]]})

    result = update_parameter(
        "fadc_pulse_shape",
        parameters,
        "0.2.0",
        value_resolver=resolver,
    )

    resolver.assert_called_once_with("fadc_pulse_shape", "pulse.dat")
    assert result["fadc_pulse_shape"]["model_parameter_schema_version"] == "0.2.0"
