#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.table import QTable

from simtools import settings
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()


@pytest.fixture
def array_model_north_from_list(model_version):
    return ArrayModel(
        label="test",
        site="North",
        model_version=model_version,
        array_elements=["LSTN-01", "MSTN-01"],
    )


def test_array_model_north_from_file(model_version, tmp_test_directory):
    position_table = QTable()
    position_table["telescope_name"] = [
        "LSTN-01",
        *[f"MSTN-{index:02d}" for index in range(1, 13)],
    ]
    position_table["position_x"] = np.arange(13) * u.m
    position_table["position_y"] = np.arange(13) * u.m
    position_table["position_z"] = np.zeros(13) * u.m
    position_file = tmp_test_directory / "north-positions.ecsv"
    position_table.write(position_file, format="ascii.ecsv", overwrite=True)

    am = ArrayModel(
        label="test",
        site="North",
        model_version=model_version,
        array_elements=Path(position_file),
    )
    assert am.number_of_telescopes == 13


def test_array_model_run_specific_config_directory(model_version, io_handler):
    am = ArrayModel(
        site="North",
        model_version=model_version,
        array_elements=["LSTN-01"],
        model_directory_subdir="run000010",
    )

    expected = io_handler.get_model_configuration_directory(model_version, "run000010")
    assert am.get_config_directory() == expected
    assert am.site_model.config_file_directory == expected
    assert am.telescope_models["LSTN-01"].config_file_directory == expected
    assert am.telescope_models["LSTN-01"].config_file_path.parent == expected


def test_array_model_north_init_without_layout_or_telescope_list(model_version):
    with pytest.raises(ValueError, match=r"No array elements found."):
        ArrayModel(label="test", site="North", model_version=model_version)


def test_input_validation(array_model_north):
    am = array_model_north
    am.print_telescope_list()
    assert am.number_of_telescopes == 13


def test_site(array_model_north):
    am = array_model_north
    assert am.site == "North"


def test_get_config_file(array_model_north):
    am = array_model_north
    assert am.config_file_path.name == "CTAO-North-test_layout.cfg"


def test_export_array_elements_as_table(array_model_north):
    am = array_model_north
    table_ground = am.export_array_elements_as_table(coordinate_system="ground")
    assert isinstance(table_ground, QTable)
    assert "position_z" in table_ground.colnames
    assert len(table_ground) > 0

    table_utm = am.export_array_elements_as_table(coordinate_system="utm")
    assert isinstance(table_utm, QTable)
    assert "altitude" in table_utm.colnames
    assert len(table_utm) > 0


def test_export_array_elements_as_table_with_calibration_elements_flag():
    array_model = Mock(spec=ArrayModel)
    array_model.layout_name = "test_layout"
    array_model.site_model = Mock()
    array_model.site_model.site = "South"
    array_model.model_version = "6.0.2"
    array_model.label = "test"
    array_model.overwrite_model_parameter_dict = None
    array_model.model_reader = Mock()
    array_model.array_elements = {"LSTS-01": None, "ILLS": None}

    telescope_model = Mock()
    telescope_model.position.return_value = [1.0 * u.m, 2.0 * u.m, 3.0 * u.m]
    telescope_model.get_parameter_value_with_unit.return_value = 10.0 * u.m
    array_model.telescope_models = {"LSTS-01": telescope_model}

    with (
        patch(
            "simtools.model.array_model.names.get_collection_name_from_array_element_name"
        ) as mock_collection,
        patch("simtools.model.array_model.CalibrationModel") as mock_calibration_model,
    ):
        mock_collection.side_effect = lambda name: (
            "calibration_devices" if name == "ILLS" else "telescopes"
        )

        calibration_model = Mock()

        def _mock_calibration_parameter(parameter):
            if parameter == "array_element_position_ground":
                return [100.0 * u.m, 200.0 * u.m, 300.0 * u.m]
            if parameter == "array_element_sphere_radius":
                raise KeyError("array_element_sphere_radius")
            raise KeyError(parameter)

        calibration_model.get_parameter_value_with_unit.side_effect = _mock_calibration_parameter
        mock_calibration_model.return_value = calibration_model

        table_default = ArrayModel.export_array_elements_as_table(array_model)
        assert "ILLS" not in table_default["telescope_name"]

        table_with_calibration = ArrayModel.export_array_elements_as_table(
            array_model, include_calibration_array_elements=True
        )
        assert "ILLS" in table_with_calibration["telescope_name"]
        assert np.isnan(
            table_with_calibration["sphere_radius"][
                table_with_calibration["telescope_name"] == "ILLS"
            ][0].to_value("m")
        )


def test_get_array_elements_from_list(array_model_north, site_model_north):
    am = array_model_north
    assert am._get_array_elements_from_list(["LSTN-01", "MSTN-01"]) == {
        "LSTN-01": None,
        "MSTN-01": None,
    }
    all_msts_plus_lst = am._get_array_elements_from_list(["LSTN-01", "MSTN"], site_model_north)
    assert "MSTN-01" in all_msts_plus_lst
    assert "MSTN-05" in all_msts_plus_lst
    assert "LSTN-01" in all_msts_plus_lst


def test_get_all_array_elements_of_type(array_model_north, site_model_north):
    am = array_model_north
    assert am._get_all_array_elements_of_type("LSTS", site_model_north) == {
        "LSTS-01": None,
        "LSTS-02": None,
        "LSTS-03": None,
        "LSTS-04": None,
    }
    # simple check that more than 10 MSTS are there
    assert len(am._get_all_array_elements_of_type("MSTS", site_model_north)) > 10

    assert len(am._get_all_array_elements_of_type("MSTE", site_model_north)) == 0


def test_pack_model_files(array_model_north, io_handler, tmp_path, model_version):
    mock_tarfile = MagicMock()
    mock_tarfile_open = MagicMock()
    # Create a context manager wrapper so `with tarfile.open(...) as tar:` yields mock_tarfile
    mock_cm = MagicMock()
    mock_cm.__enter__.return_value = mock_tarfile
    # ensure exiting the context calls close() on the mock tarfile to match real behavior
    mock_cm.__exit__.side_effect = lambda *args: mock_tarfile.close()
    mock_tarfile_open.return_value = mock_cm
    # Return files under the mocked config directory so relative_to(base) works
    mock_output_dir = tmp_path / "output" / "directory" / "model" / model_version
    mock_rglob = MagicMock(return_value=[mock_output_dir / "file1", mock_output_dir / "file2"])
    mock_get_output_directory = MagicMock(return_value=mock_output_dir)

    with (
        patch("tarfile.open", mock_tarfile_open),  # NOSONAR
        patch("pathlib.Path.rglob", mock_rglob),
        patch.object(io_handler, "get_output_directory", mock_get_output_directory),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        archive_path = array_model_north.pack_model_files()

        assert archive_path == mock_output_dir.joinpath(f"model_files_{model_version}.tar.gz")
        assert mock_tarfile.add.call_count == 2

    mock_rglob = MagicMock(return_value=[])
    with (
        patch("tarfile.open", mock_tarfile_open),  # NOSONAR
        patch("pathlib.Path.rglob", mock_rglob),
        patch.object(io_handler, "get_output_directory", mock_get_output_directory),
    ):
        assert array_model_north.pack_model_files() is None


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

    metadata = array_model_north_cp._get_additional_simtel_metadata()

    assert metadata["nsb_integrated_flux"] == pytest.approx(42.0)
    assert metadata["primary"] == "gamma"
    assert metadata["azimuth_angle"] == pytest.approx(180.0)
    assert metadata["zenith_angle"] == pytest.approx(20.0)
    assert metadata["ha_angle"] == pytest.approx(0.0)
    assert metadata["dec_angle"] == pytest.approx(30.0)


def test_build_calibration_models():

    array_model_north = Mock(spec=ArrayModel)
    array_model_north._build_calibration_models = ArrayModel._build_calibration_models
    telescope_model = Mock()
    telescope_model.get_calibration_device_name = Mock()

    site_model = Mock()
    site_model.site = "North"

    # Test case 1: No calibration device types provided
    result = array_model_north._build_calibration_models(
        array_model_north, telescope_model, site_model, None
    )
    assert result == {}

    # Test case 2: Empty calibration device types list
    result = array_model_north._build_calibration_models(
        array_model_north, telescope_model, site_model, []
    )
    assert result == {}

    # Test case 3: Calibration device types provided but device name not found
    telescope_model.get_calibration_device_name.return_value = None
    result = array_model_north._build_calibration_models(
        array_model_north, telescope_model, site_model, ["flasher"]
    )
    assert result == {}
    telescope_model.get_calibration_device_name.assert_called_with("flasher")

    # Test case 4: Calibration device types provided and device names found
    def mock_device_name(device_type):
        return f"device_{device_type}" if device_type in ["flasher", "illuminator"] else None

    telescope_model.get_calibration_device_name.side_effect = mock_device_name

    with patch("simtools.model.array_model.CalibrationModel") as mock_calibration_model:
        mock_calibration_instance = Mock()
        mock_calibration_model.return_value = mock_calibration_instance

        array_model_north.model_version = "6.0.0"
        array_model_north.label = "test_label"
        array_model_north.overwrite_model_parameter_dict = None
        array_model_north.model_reader = Mock()

        result = array_model_north._build_calibration_models(
            array_model_north,
            telescope_model,
            site_model,
            ["flasher", "illuminator", "nonexistent"],
        )

        assert len(result) == 2
        assert "device_flasher" in result
        assert "device_illuminator" in result
        assert result["device_flasher"] == mock_calibration_instance
        assert result["device_illuminator"] == mock_calibration_instance

        # Check that CalibrationModel was called twice with correct parameters
        assert mock_calibration_model.call_count == 2


def test_export_all_simtel_config_files():

    array_model_north = Mock()
    array_model_north._telescope_model_files_exported = False
    array_model_north._array_model_file_exported = False

    ArrayModel.export_all_simtel_config_files(array_model_north)

    array_model_north.export_simtel_telescope_config_files.assert_called_once()
    array_model_north.export_sim_telarray_config_file.assert_called_once()


def test_build_telescope_models():

    array_model_north = Mock()
    array_model_north.model_version = "6.0.0"
    array_model_north.label = "test"

    site_model = Mock()
    site_model.site = "North"

    array_elements = {"LSTN-01": None, "non_telescope": None}

    with (
        patch(
            "simtools.model.array_model.names.get_collection_name_from_array_element_name"
        ) as mock_names,
        patch("simtools.model.array_model.TelescopeModel") as mock_tel_model,
    ):
        mock_names.side_effect = lambda name: "telescopes" if name == "LSTN-01" else "other"

        telescope_models, _ = ArrayModel._build_telescope_models(
            array_model_north, site_model, array_elements, None
        )

        assert "LSTN-01" in telescope_models
        assert "non_telescope" not in telescope_models
        mock_tel_model.assert_called_once()


def test_export_simtel_telescope_config_files(array_model_north):
    am = array_model_north

    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file = Mock()

    am.export_simtel_telescope_config_files()

    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file.assert_called_once()

    assert am._telescope_model_files_exported is True


def test_export_simtel_telescope_config_files_skips_duplicates(mocker):
    am = Mock(spec=ArrayModel)
    am._logger = Mock()
    am._telescope_model_files_exported = False
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
    am._logger.debug.assert_called_once()
    assert "already exists" in am._logger.debug.call_args[0][0]

    assert am._telescope_model_files_exported is True


def test_export_sim_telarray_config_file(array_model_north, mocker):
    am = array_model_north
    mocker.patch.object(am.site_model, "export_model_files")

    mock_simtel_writer = mocker.MagicMock()
    mocker.patch(
        "simtools.model.array_model.simtel_config_writer.SimtelConfigWriter",
        return_value=mock_simtel_writer,
    )

    mock_metadata = {"nsb_integrated_flux": 42.0}
    mocker.patch.object(am, "_get_additional_simtel_metadata", return_value=mock_metadata)

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
    assert am._array_model_file_exported is True
