"""Tests for the telescope sphere derivation command."""

from unittest.mock import Mock, patch

import astropy.units as u
import pytest

from simtools.applications.derive_telescope_sphere_radius import main

_PARAMETER_VERSION = "parameter_version"
_APPLICATION_MODULE = "simtools.applications.derive_telescope_sphere_radius"
_APPLICATION_START = "simtools.application.definition.ApplicationDefinition.start"
_WRITER_TARGET = f"{_APPLICATION_MODULE}.ModelDataWriter"
_DERIVE_TARGET = f"{_APPLICATION_MODULE}.derive_telescope_sphere_radius"
_INITIALIZE_TARGET = f"{_APPLICATION_MODULE}.initialize_simulation_models"


def _args(telescope="LSTN-01", site="North"):
    return {
        "site": site,
        "telescope": telescope,
        "model_version": "7.0.0",
        _PARAMETER_VERSION: "1.0.0",
        "application_label": "test",
        "output_path": None,
    }


@patch(_WRITER_TARGET)
@patch(_DERIVE_TARGET)
@patch(_INITIALIZE_TARGET)
@patch(_APPLICATION_START)
def test_main_writes_radius_and_metadata(start, initialize, derive, writer):
    context = Mock()
    context.args = _args()
    start.return_value = context
    model = Mock()
    initialize.return_value = (model, Mock(), Mock())
    derive.return_value = 12 * u.m

    main()

    derive.assert_called_once_with(model)
    kwargs = writer.write_model_parameter.call_args.kwargs
    assert kwargs["value"] == 12
    assert kwargs["unit"] == "m"
    assert kwargs[_PARAMETER_VERSION] == context.args[_PARAMETER_VERSION]
    assert kwargs["metadata_input_dict"] == context.args


@patch(_WRITER_TARGET)
@patch(_DERIVE_TARGET)
@patch(_INITIALIZE_TARGET)
@patch(_APPLICATION_START)
def test_main_does_not_write_invalid_geometry(start, initialize, derive, writer):
    context = Mock()
    context.args = _args(telescope="SCTS-design", site="South")
    start.return_value = context
    initialize.return_value = (Mock(), Mock(), Mock())
    derive.side_effect = ValueError("Required mirror geometry is absent.")

    with pytest.raises(ValueError, match="Required mirror geometry"):
        main()

    writer.write_model_parameter.assert_not_called()
