"""Test native configuration writing without modifying resolved models."""

import copy
from unittest.mock import Mock

import astropy.units as u
import pytest

from simtools import settings
from simtools.model.array_model import ArrayModel
from simtools.model.telescope_model import TelescopeModel
from simtools.simtel.model_writer import SimtelModelWriter
from simtools.simulation.configuration import get_model_writer


def test_write_sim_telarray_config_file(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    exporter = get_model_writer(telescope_copy)

    mock_writer = mocker.Mock()
    mock_writer.write_telescope_config_file = mocker.Mock()

    mock_export = mocker.patch.object(TelescopeModel, "export_model_files")
    mock_load_writer = mocker.patch.object(
        exporter,
        "_load_simtel_config_writer",
        side_effect=lambda *args, **kwargs: setattr(exporter, "config_writer", mock_writer),
    )

    telescope_copy.write_sim_telarray_config_file()
    mock_export.assert_called_once_with(update_if_necessary=True)
    mock_load_writer.assert_called_once_with(label="test-telescope-model-lst")
    mock_writer.write_telescope_config_file.assert_called_once()

    mock_export.reset_mock()
    mock_load_writer.reset_mock()
    mock_writer.write_telescope_config_file.reset_mock()

    add_model = copy.deepcopy(telescope_model_lst)
    add_model.parameters = {"test_param": "test_value"}

    telescope_copy.write_sim_telarray_config_file(additional_models=add_model)
    assert mock_export.call_count == 2  # Called for both models
    mock_load_writer.assert_called_once_with(label="test-telescope-model-lst")
    assert "test_param" not in telescope_copy.parameters
    assert (
        mock_writer.write_telescope_config_file.call_args.kwargs["parameters"]["test_param"]
        == "test_value"
    )
    mock_writer.write_telescope_config_file.assert_called_once()

    mock_export.assert_any_call(telescope_copy.config_file_directory, update_if_necessary=True)


def test_write_sim_telarray_config_file_exports_nsb_correction_file(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    exporter = get_model_writer(telescope_copy)
    telescope_copy._simulation_config_parameters["sim_telarray"][
        "correct_nsb_spectrum_to_telescope_altitude"
    ] = {"value": "correction.ecsv"}

    mocker.patch.object(TelescopeModel, "export_model_files")
    mock_export_nsb = mocker.patch.object(
        exporter, "export_nsb_spectrum_to_telescope_altitude_correction_file"
    )
    mock_writer = mocker.Mock()
    mocker.patch.object(
        exporter,
        "_load_simtel_config_writer",
        side_effect=lambda *args, **kwargs: setattr(exporter, "config_writer", mock_writer),
    )

    telescope_copy.write_sim_telarray_config_file()

    mock_export_nsb.assert_called_once_with(model_directory=telescope_copy.config_file_directory)


def test_export_nsb_correction_file_preserves_parameter_metadata(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    parameter_name = "correct_nsb_spectrum_to_telescope_altitude"
    parameter = {
        "value": "correction-1.0.0.ecsv",
        "parameter_version": "1.0.0",
        "instrument": "LSTS-design",
        "site": "North",
    }
    telescope_copy._simulation_config_parameters["sim_telarray"][parameter_name] = parameter
    mock_export = mocker.patch.object(telescope_copy.model_reader, "export_model_files")
    mock_table = mocker.patch.object(
        telescope_copy.model_reader, "get_parameter_table", return_value=mocker.Mock()
    )
    mock_write_table = mocker.patch(
        "simtools.simtel.model_writer.table_serializers.write_simtel_table"
    )

    telescope_copy.export_nsb_spectrum_to_telescope_altitude_correction_file(
        model_directory=telescope_copy.config_file_directory
    )

    exported = mock_export.call_args.kwargs["parameters"][parameter_name]
    assert exported["parameter"] == parameter_name
    assert exported["parameter_version"] == "1.0.0"
    assert exported["instrument"] == "LSTS-design"
    assert exported["site"] == "North"
    assert exported["file"] is True
    mock_table.assert_called_once_with(exported)
    mock_write_table.assert_called_once()
    assert mock_write_table.call_args.args[:2] == (
        mock_table.return_value,
        telescope_copy.config_file_directory,
    )
    assert mock_write_table.call_args.kwargs["contract"]["table_format"] == (
        "atmospheric_transmission"
    )


def test_add_additional_models(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    exporter = get_model_writer(telescope_copy)

    # Test case 1: None input
    parameters = telescope_copy.parameters.copy()
    exporter._add_additional_models(None, parameters)
    # Should not change anything

    # Test case 2: Single model
    mock_model = mocker.Mock()
    mock_model.parameters = {"new_param": "new_value"}
    mock_model.get_simulation_software_parameters.return_value = {
        "iobuf_maximum": {"value": 1000},
        "min_photons": {"value": 3},
    }
    mock_model.export_model_files = mocker.Mock()

    exporter._add_additional_models(mock_model, parameters)
    assert "new_param" in parameters
    assert parameters["new_param"] == "new_value"
    assert parameters["iobuf_maximum"] == {"value": 1000}
    assert "min_photons" not in parameters
    mock_model.export_model_files.assert_called_once()

    # Test case 3: Dictionary of models
    mock_model2 = mocker.Mock()
    mock_model2.parameters = {"param2": "value2"}
    mock_model2.get_simulation_software_parameters.return_value = {}
    mock_model2.export_model_files = mocker.Mock()

    models_dict = {"model1": mock_model, "model2": mock_model2}
    exporter._add_additional_models(models_dict, parameters)
    assert parameters["param2"] == "value2"


def test_get_additional_simtel_metadata(array_model_north, mocker):
    array_model_north_cp = copy.deepcopy(array_model_north)
    mocker.patch.object(
        array_model_north_cp.site_model, "get_nsb_integrated_flux", return_value=42.0
    )
    mocker.patch.object(
        settings.config,
        "_args",
        {
            "primary": "gamma",
            "azimuth_angle": 180.0 * u.deg,
            "zenith_angle": 20.0 * u.deg,
            "ha": 0.0 * u.deg,
            "dec": 30.0 * u.deg,
        },
    )

    metadata = get_model_writer(array_model_north_cp)._get_additional_simtel_metadata()

    assert metadata["nsb_integrated_flux"] == pytest.approx(42.0)
    assert metadata["primary"] == "gamma"
    assert metadata["azimuth_angle"] == pytest.approx(180.0)
    assert metadata["zenith_angle"] == pytest.approx(20.0)
    assert metadata["ha_angle"] == pytest.approx(0.0)
    assert metadata["dec_angle"] == pytest.approx(30.0)


def test_export_all_simtel_config_files(mocker):
    exporter = SimtelModelWriter(mocker.Mock())
    telescopes = mocker.patch.object(exporter, "export_simtel_telescope_config_files")
    array = mocker.patch.object(exporter, "export_sim_telarray_config_file")
    exporter.export_config_files()
    telescopes.assert_called_once()
    array.assert_called_once()
    telescopes.reset_mock()
    array.reset_mock()
    exporter._telescope_model_files_exported = True
    exporter._array_model_file_exported = True
    exporter.export_config_files()
    telescopes.assert_not_called()
    array.assert_not_called()


def test_export_simtel_telescope_config_files(array_model_north):
    am = array_model_north

    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file = Mock()

    am.export_simtel_telescope_config_files()

    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file.assert_called_once()

    assert get_model_writer(am)._telescope_model_files_exported is True


def test_export_simtel_telescope_config_files_skips_duplicates(mocker):
    am = Mock(spec=ArrayModel)
    am._logger = Mock()
    am.configuration_writers = {}
    am.calibration_models = {}

    # Create two telescope objects with the same name
    tel_model_1 = Mock()
    tel_model_1.name = "LST_1"
    tel_model_1.write_sim_telarray_config_file = Mock()

    tel_model_2 = Mock()
    tel_model_2.name = "LST_1"  # Same name as tel_model_1
    tel_model_2.write_sim_telarray_config_file = Mock()

    am.telescope_models = {"LSTN-01": tel_model_1, "LSTN-02": tel_model_2}

    ArrayModel.export_simtel_telescope_config_files(am)

    # Verify write was called only once (for the first telescope with this name)
    tel_model_1.write_sim_telarray_config_file.assert_called_once()
    tel_model_2.write_sim_telarray_config_file.assert_not_called()

    # Verify the logger was called for the second telescope
    # Both telescope entries share one configuration file.

    assert get_model_writer(am)._telescope_model_files_exported is True


def test_export_sim_telarray_config_file(array_model_north, mocker):
    am = array_model_north
    mocker.patch.object(am.site_model, "export_model_files")

    mock_simtel_writer = mocker.MagicMock()
    mocker.patch(
        "simtools.simtel.model_writer.SimtelConfigWriter",
        return_value=mock_simtel_writer,
    )

    mock_metadata = {"nsb_integrated_flux": 42.0}
    mocker.patch.object(
        get_model_writer(am), "_get_additional_simtel_metadata", return_value=mock_metadata
    )

    am.export_sim_telarray_config_file()

    # Verify site model export was called
    am.site_model.export_model_files.assert_called_once()

    # Verify SimtelConfigWriter was instantiated with correct parameters
    mock_simtel_writer.write_array_config_file.assert_called_once()
    call_args = mock_simtel_writer.write_array_config_file.call_args
    assert call_args[1]["config_file_path"] == am.config_file_path
    assert call_args[1]["telescope_model"] == am.telescope_models
    assert call_args[1]["site_model"] == am.site_model
    assert call_args[1]["additional_metadata"] == mock_metadata

    # Verify the flag is set
    assert get_model_writer(am)._array_model_file_exported is True
