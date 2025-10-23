#!/usr/bin/python3

import copy
import logging
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from astropy import units as u
from astropy.table import QTable

from simtools.data_model import schema
from simtools.model.array_model import ArrayModel

logger = logging.getLogger()


@pytest.fixture
def array_model_from_list(db_config, model_version):
    return ArrayModel(
        label="test",
        site="North",
        mongo_db_config=db_config,
        model_version=model_version,
        array_elements=["LSTN-01", "MSTN-01"],
    )


def test_array_model_from_file(db_config, model_version, telescope_north_test_file):
    am = ArrayModel(
        label="test",
        site="North",
        mongo_db_config=db_config,
        model_version=model_version,
        array_elements=telescope_north_test_file,
    )
    assert am.number_of_telescopes == 13


def test_array_model_init_without_layout_or_telescope_list(db_config, model_version):
    with pytest.raises(ValueError, match=r"No array elements found."):
        ArrayModel(
            label="test",
            site="North",
            mongo_db_config=db_config,
            model_version=model_version,
        )


def test_input_validation(array_model):
    am = array_model
    am.print_telescope_list()
    assert am.number_of_telescopes == 13


def test_site(array_model):
    am = array_model
    assert am.site == "North"


def test_exporting_config_files(db_config, model_version):
    am = ArrayModel(
        label="test",
        site="North",
        layout_name="test_layout",
        mongo_db_config=db_config,
        model_version=model_version,
    )

    am.export_simtel_telescope_config_files()
    am.export_sim_telarray_config_file()

    test_cfg = "_test.cfg"
    list_of_export_files = [
        "CTA-LST_lightguide_eff_2020-04-12_average.dat",
        "CTA-North-LSTN-01-" + model_version + test_cfg,
        "CTA-North-MSTN-01-" + model_version + test_cfg,
        "CTA-test_layout-North-" + model_version + test_cfg,
        "NectarCAM_lightguide_efficiency_POP_131019.dat",
        "Pulse_template_nectarCam_17042020-noshift.dat",
        "array_triggers.dat",
        "atm_trans_2156_1_3_2_0_0_0.1_0.1.dat",
        "atmprof_ecmwf_north_winter_fixed.dat",
        "camera_CTA-LST-1_analogsum21_v2020-04-14.dat",
        "camera_CTA-MST-NectarCam_20191120_majority-3nn.dat",
        "mirror_CTA-100_1.20-86-0.04.dat",
        "mirror_CTA-N-LST1_v2019-03-31_rotated.dat",
        "pulse_LST_8dynode_pix6_20200204.dat",
        "qe_R12992-100-05c.dat",
        "qe_lst1_20200318_high+low.dat",
        "ref_MST-North-MLT_2022_06_28.dat",
        "ref_LST1_2022_04_01.dat",
        "spe_LST_2022-04-27_AP2.0e-4.dat",
        "spe_afterpulse_pdf_NectarCam_18122019.dat",
        "transmission_lst_window_No7-10_ave.dat",
    ]

    for model_file in list_of_export_files:
        logger.info("Checking file: %s", model_file)
        assert Path(am.get_config_directory()).joinpath(model_file).exists()


def test_load_array_element_positions_from_file(array_model, telescope_north_test_file):
    am = array_model
    telescopes = am._load_array_element_positions_from_file(telescope_north_test_file, "North")
    assert len(telescopes) > 0


def test_get_telescope_position_parameter(array_model):
    am = array_model
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


def test_get_config_file(model_version, array_model):
    am = array_model
    assert am.config_file_path.name == "CTA-test_layout-North-" + model_version + "_test.cfg"


def test_get_config_directory(array_model):
    am = array_model
    assert am.get_config_directory().is_dir()


def test_export_array_elements_as_table(array_model):
    am = array_model
    table_ground = am.export_array_elements_as_table(coordinate_system="ground")
    assert isinstance(table_ground, QTable)
    assert "position_z" in table_ground.colnames
    assert len(table_ground) > 0

    table_utm = am.export_array_elements_as_table(coordinate_system="utm")
    assert isinstance(table_utm, QTable)
    assert "altitude" in table_utm.colnames
    assert len(table_utm) > 0


def test_get_array_elements_from_list(array_model):
    am = array_model
    assert am._get_array_elements_from_list(["LSTN-01", "MSTN-01"]) == {
        "LSTN-01": None,
        "MSTN-01": None,
    }
    all_msts_plus_lst = am._get_array_elements_from_list(["LSTN-01", "MSTN"])
    assert "MSTN-01" in all_msts_plus_lst
    assert "MSTN-05" in all_msts_plus_lst
    assert "LSTN-01" in all_msts_plus_lst


def test_get_all_array_elements_of_type(array_model):
    am = array_model
    assert am._get_all_array_elements_of_type("LSTS") == {
        "LSTS-01": None,
        "LSTS-02": None,
        "LSTS-03": None,
        "LSTS-04": None,
    }
    # simple check that more than 10 MSTS are there
    assert len(am._get_all_array_elements_of_type("MSTS")) > 10

    assert len(am._get_all_array_elements_of_type("MSTE")) == 0


def test_update_array_element_position(array_model_from_list):
    am = array_model_from_list
    assert "LSTN-01" in am.array_elements
    assert "LSTN-01" in am.telescope_models
    assert am.array_elements["LSTN-01"] is None


def test_model_version_setter_with_valid_string(array_model):
    am = array_model
    am.model_version = "6.0.0"
    assert am.model_version == "6.0.0"


def test_model_version_setter_with_valid_list(array_model):
    am = array_model
    error_message = "Only one model version can be passed to ArrayModel, not a list."
    with pytest.raises(ValueError, match=error_message):
        am.model_version = ["6.0.0"]

    with pytest.raises(ValueError, match=error_message):
        am.model_version = ["6.0.0", "7.0.0"]


def test_pack_model_files(array_model, io_handler, tmp_path, model_version):
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
        patch("tarfile.open", mock_tarfile_open),
        patch("pathlib.Path.rglob", mock_rglob),
        patch.object(io_handler, "get_output_directory", mock_get_output_directory),
        patch("pathlib.Path.is_file", return_value=True),
    ):
        archive_path = array_model.pack_model_files()

        assert archive_path == mock_output_dir.joinpath(f"model_files_{model_version}.tar.gz")
        assert mock_tarfile.add.call_count == 2

    mock_rglob = MagicMock(return_value=[])
    with (
        patch("tarfile.open", mock_tarfile_open),
        patch("pathlib.Path.rglob", mock_rglob),
        patch.object(io_handler, "get_output_directory", mock_get_output_directory),
    ):
        assert array_model.pack_model_files() is None


def test_get_additional_simtel_metadata(array_model, mocker):
    array_model_cp = copy.deepcopy(array_model)
    array_model_cp.sim_telarray_seeds = {"seeds": 1234}
    mocker.patch.object(array_model_cp.site_model, "get_nsb_integrated_flux", return_value=42.0)

    assert "nsb_integrated_flux" in array_model_cp._get_additional_simtel_metadata()
    assert "seeds" in array_model_cp._get_additional_simtel_metadata()


def test_build_calibration_models():
    """Test _build_calibration_models method with mocked dependencies."""
    from unittest.mock import Mock

    from simtools.model.array_model import ArrayModel

    # Create a mock array model instance
    array_model = Mock(spec=ArrayModel)
    array_model._build_calibration_models = ArrayModel._build_calibration_models

    # Mock telescope model
    telescope_model = Mock()
    telescope_model.get_calibration_device_name = Mock()

    # Mock site model
    site_model = Mock()
    site_model.site = "North"

    # Test case 1: No calibration device types provided
    result = array_model._build_calibration_models(array_model, telescope_model, site_model, None)
    assert result == {}

    # Test case 2: Empty calibration device types list
    result = array_model._build_calibration_models(array_model, telescope_model, site_model, [])
    assert result == {}

    # Test case 3: Calibration device types provided but device name not found
    telescope_model.get_calibration_device_name.return_value = None
    result = array_model._build_calibration_models(
        array_model, telescope_model, site_model, ["flasher"]
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
        array_model.mongo_db_config = {"test": "config"}
        array_model.model_version = "6.0.0"
        array_model.label = "test_label"

        result = array_model._build_calibration_models(
            array_model, telescope_model, site_model, ["flasher", "illuminator", "nonexistent"]
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
    from unittest.mock import Mock

    array_model = Mock()
    array_model._telescope_model_files_exported = False
    array_model._array_model_file_exported = False

    ArrayModel.export_all_simtel_config_files(array_model)

    array_model.export_simtel_telescope_config_files.assert_called_once()
    array_model.export_sim_telarray_config_file.assert_called_once()


def test_build_telescope_models():
    """Test _build_telescope_models method with mocked dependencies."""
    from unittest.mock import Mock, patch

    array_model = Mock()
    array_model.model_version = "6.0.0"
    array_model.mongo_db_config = {"test": "config"}
    array_model.label = "test"

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
            array_model, site_model, array_elements, None
        )

        assert "LSTN-01" in telescope_models
        assert "non_telescope" not in telescope_models
        mock_tel_model.assert_called_once()
