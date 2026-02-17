#!/usr/bin/python3

import copy
import logging
from unittest.mock import MagicMock, Mock, patch

import pytest
from astropy import units as u
from astropy.table import QTable

from simtools.data_model import schema
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


def test_array_model_north_from_file(model_version, telescope_north_test_file):
    am = ArrayModel(
        label="test",
        site="North",
        model_version=model_version,
        array_elements=telescope_north_test_file,
    )
    assert am.number_of_telescopes == 13


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


def test_load_array_element_positions_from_file(array_model_north, telescope_north_test_file):
    am = array_model_north
    telescopes = am._load_array_element_positions_from_file(telescope_north_test_file, "North")
    assert len(telescopes) > 0


def test_get_telescope_position_parameter(array_model_north):
    am = array_model_north
    assert am._get_telescope_position_parameter(
        "LSTN-01", "North", 10.0 * u.m, 200.0 * u.cm, 30.0 * u.m, "2.0.0"
    ) == {
        "schema_version": schema.get_model_parameter_schema_version(),
        "parameter": "array_element_position_ground",
        "instrument": "LSTN-01",
        "site": "North",
        "parameter_version": "2.0.0",
        "unique_id": None,
        "value": [10.0, 2.0, 30.0],
        "unit": "m",
        "type": "float64",
        "file": False,
        "meta_parameter": False,
        "model_parameter_schema_version": "0.1.0",
    }


def test_get_config_file(array_model_north):
    am = array_model_north
    assert am.config_file_path.name == "CTA-test_layout-North_test-lst-array.cfg"


def test_get_config_directory(array_model_north):
    am = array_model_north
    assert am.get_config_directory().is_dir()


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


def test_update_array_element_position(array_model_north_from_list):
    am = array_model_north_from_list
    assert "LSTN-01" in am.array_elements
    assert "LSTN-01" in am.telescope_models
    assert am.array_elements["LSTN-01"] is None


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

    assert "nsb_integrated_flux" in array_model_north_cp._get_additional_simtel_metadata()


def test_build_calibration_models():
    """Test _build_calibration_models method with mocked dependencies."""

    # Create a mock array model instance
    array_model_north = Mock(spec=ArrayModel)
    array_model_north._build_calibration_models = ArrayModel._build_calibration_models

    # Mock telescope model
    telescope_model = Mock()
    telescope_model.get_calibration_device_name = Mock()

    # Mock site model
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

    # Mock the CalibrationModel constructor
    with patch("simtools.model.array_model.CalibrationModel") as mock_calibration_model:
        mock_calibration_instance = Mock()
        mock_calibration_model.return_value = mock_calibration_instance

        # Set up array model attributes for CalibrationModel initialization
        array_model_north.model_version = "6.0.0"
        array_model_north.label = "test_label"
        array_model_north.overwrite_model_parameter_dict = None

        result = array_model_north._build_calibration_models(
            array_model_north,
            telescope_model,
            site_model,
            ["flasher", "illuminator", "nonexistent"],
        )

        # Should create calibration models for flasher and illuminator, but not nonexistent
        assert len(result) == 2
        assert "device_flasher" in result
        assert "device_illuminator" in result
        assert result["device_flasher"] == mock_calibration_instance
        assert result["device_illuminator"] == mock_calibration_instance

        # Check that CalibrationModel was called twice with correct parameters
        assert mock_calibration_model.call_count == 2


def test_export_all_simtel_config_files():
    """Test export_all_simtel_config_files method calls both export methods when needed."""

    array_model_north = Mock()
    array_model_north._telescope_model_files_exported = False
    array_model_north._array_model_file_exported = False

    ArrayModel.export_all_simtel_config_files(array_model_north)

    array_model_north.export_simtel_telescope_config_files.assert_called_once()
    array_model_north.export_sim_telarray_config_file.assert_called_once()


def test_build_telescope_models():
    """Test _build_telescope_models method with mocked dependencies."""

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


def test_sim_telarray_seeds_property(array_model_north):
    assert array_model_north.sim_telarray_seed is None


def test_export_simtel_telescope_config_files(array_model_north):
    """Test export_simtel_telescope_config_files exports config for each telescope."""
    am = array_model_north

    # Mock the telescope models' write method
    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file = Mock()

    am.export_simtel_telescope_config_files()

    # Verify write_sim_telarray_config_file was called for each telescope
    for tel_model in am.telescope_models.values():
        tel_model.write_sim_telarray_config_file.assert_called_once()

    assert am._telescope_model_files_exported is True


def test_export_simtel_telescope_config_files_skips_duplicates(mocker):
    """Test export_simtel_telescope_config_files skips duplicate telescope models."""
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

    # Call the actual method
    ArrayModel.export_simtel_telescope_config_files(am)

    # Verify write was called only once (for the first telescope with this name)
    tel_model_1.write_sim_telarray_config_file.assert_called_once()
    tel_model_2.write_sim_telarray_config_file.assert_not_called()

    # Verify the logger was called for the second telescope
    am._logger.debug.assert_called_once()
    assert "already exists" in am._logger.debug.call_args[0][0]

    assert am._telescope_model_files_exported is True


def test_export_simtel_telescope_config_files_with_calibration_models(mocker):
    """Test export_simtel_telescope_config_files passes calibration models to write method."""
    am = Mock(spec=ArrayModel)
    am._logger = Mock()
    am._telescope_model_files_exported = False

    # Create telescope and calibration models
    tel_model = Mock()
    tel_model.name = "LSTN-01"
    tel_model.write_sim_telarray_config_file = Mock()

    calibration_model = Mock()
    am.calibration_models = {"LSTN-01": {"flasher": calibration_model}}
    am.telescope_models = {"LSTN-01": tel_model}

    ArrayModel.export_simtel_telescope_config_files(am)

    # Verify write was called with the calibration models
    tel_model.write_sim_telarray_config_file.assert_called_once_with(
        additional_models={"flasher": calibration_model}
    )


def test_export_simtel_telescope_config_files_with_empty_calibration_models(mocker):
    """Test export_simtel_telescope_config_files with telescopes that have no calibration models."""
    am = Mock(spec=ArrayModel)
    am._logger = Mock()
    am._telescope_model_files_exported = False

    tel_model = Mock()
    tel_model.name = "MSTN-01"
    tel_model.write_sim_telarray_config_file = Mock()

    am.calibration_models = {}
    am.telescope_models = {"MSTN-01": tel_model}

    ArrayModel.export_simtel_telescope_config_files(am)

    # Verify write was called with None (no calibration models)
    tel_model.write_sim_telarray_config_file.assert_called_once_with(additional_models=None)


def test_export_sim_telarray_config_file(array_model_north, mocker):
    """Test export_sim_telarray_config_file exports array config and site model files."""
    am = array_model_north

    # Mock the site model's export method
    mocker.patch.object(am.site_model, "export_model_files")

    # Mock the SimtelConfigWriter
    mock_simtel_writer = mocker.MagicMock()
    mocker.patch(
        "simtools.model.array_model.simtel_config_writer.SimtelConfigWriter",
        return_value=mock_simtel_writer,
    )

    # Mock the metadata method
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


def test_export_sim_telarray_config_file_creates_writer_correctly(array_model_north, mocker):
    """Test that SimtelConfigWriter is created with correct parameters."""
    am = array_model_north

    mocker.patch.object(am.site_model, "export_model_files")
    mock_simtel_writer_class = mocker.patch(
        "simtools.model.array_model.simtel_config_writer.SimtelConfigWriter"
    )
    mock_simtel_writer = mocker.MagicMock()
    mock_simtel_writer_class.return_value = mock_simtel_writer

    mocker.patch.object(am, "_get_additional_simtel_metadata", return_value={})

    am.export_sim_telarray_config_file()

    # Verify SimtelConfigWriter was instantiated with correct parameters
    mock_simtel_writer_class.assert_called_once_with(
        site=am.site_model.site,
        layout_name=am.layout_name,
        model_version=am.model_version,
        label=am.label,
    )


def test_export_sim_telarray_config_file_idempotent(array_model_north, mocker):
    """Test that calling export multiple times sets flag correctly."""
    am = array_model_north

    mocker.patch.object(am.site_model, "export_model_files")
    mocker.patch("simtools.model.array_model.simtel_config_writer.SimtelConfigWriter")
    mocker.patch.object(am, "_get_additional_simtel_metadata", return_value={})

    assert am._array_model_file_exported is False

    am.export_sim_telarray_config_file()
    assert am._array_model_file_exported is True

    # Calling again should still have the flag set
    am.export_sim_telarray_config_file()
    assert am._array_model_file_exported is True
