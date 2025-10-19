#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import pytest
from astropy import units as u

import simtools.utils.general as gen
from simtools.db.db_handler import DatabaseHandler
from simtools.model.model_parameter import InvalidModelParameterError
from simtools.model.telescope_model import TelescopeModel

logger = logging.getLogger()


def test_get_parameter_type(telescope_model_lst, caplog):
    assert telescope_model_lst.get_parameter_type("num_gains") == "int64"
    telescope_model_copy = copy.deepcopy(telescope_model_lst)
    telescope_model_copy.parameters["num_gains"].pop("type")
    with caplog.at_level(logging.DEBUG):
        assert telescope_model_copy.get_parameter_type("num_gains") is None
    assert "Parameter num_gains does not have a type." in caplog.text


def test_get_parameter_file_flag(telescope_model_lst, caplog):
    assert telescope_model_lst.get_parameter_file_flag("num_gains") is False
    telescope_model_copy = copy.deepcopy(telescope_model_lst)
    telescope_model_copy.parameters["num_gains"].pop("file")
    with caplog.at_level(logging.DEBUG):
        assert telescope_model_copy.get_parameter_file_flag("num_gains") is False
    assert "Parameter num_gains does not have a file associated with it." in caplog.text


def test_get_parameter_dict(telescope_model_lst):
    tel_model = telescope_model_lst
    assert isinstance(tel_model._get_parameter_dict("num_gains"), dict)
    assert isinstance(tel_model._get_parameter_dict("num_gains")["value"], int)
    assert isinstance(tel_model._get_parameter_dict("telescope_axis_height")["value"], float)
    assert tel_model._get_parameter_dict("telescope_axis_height")["unit"] == "m"

    with pytest.raises(InvalidModelParameterError):
        tel_model._get_parameter_dict("not_a_parameter")


def test_get_parameter_value(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)
    assert isinstance(tel_model.get_parameter_value("num_gains"), int)

    with pytest.raises(InvalidModelParameterError):
        tel_model.get_parameter_value("not_num_gains")

    tel_model.parameters["num_gains"]["value"] = "2 3 4"
    t_int = tel_model.get_parameter_value("num_gains")
    assert len(t_int) == 3
    assert isinstance(t_int, list)
    assert isinstance(t_int[2], int)

    t_1 = tel_model.get_parameter_value("telescope_transmission")
    assert isinstance(t_1, list)
    assert len(t_1) == 6

    # single value floats should return as floats
    _tmp_dict = {
        "value": "0.8",
        "type": "float64",
    }
    tel_model.parameters["t_2"] = _tmp_dict
    t_2 = tel_model.get_parameter_value("t_2")
    assert t_2 == pytest.approx(0.8)
    # string-type lists
    _tmp_dict["value"] = "0.8 0.9"
    tel_model.parameters["t_2"] = _tmp_dict
    t_2 = tel_model.get_parameter_value("t_2")
    assert len(t_2) == 2
    assert t_2[0] == pytest.approx(0.8)
    assert t_2[1] == pytest.approx(0.9)
    # mixed strings should become list of strings
    _tmp_dict["value"] = "0.8 abc"
    tel_model.parameters["t_2"] = _tmp_dict
    t_2 = tel_model.get_parameter_value("t_2")
    assert t_2 == ["0.8", "abc"]


def test_get_parameter_value_with_unit(telescope_model_lst):
    tel_model = telescope_model_lst

    # check handling of list of values and units including null units
    t_1 = tel_model.get_parameter_value_with_unit("focus_offset")
    assert isinstance(t_1, list)
    assert isinstance(t_1[0], u.Quantity)  # list of quantities returned
    assert t_1[0].unit == u.cm
    assert t_1[2].unit == u.dimensionless_unscaled

    # check handling of list of values with a single unit
    t_2 = tel_model.get_parameter_value_with_unit("array_element_position_utm")
    assert isinstance(t_2, u.Quantity)  # returns Quantity [a,b,c] with a shared unit 'm'
    assert t_2.unit == "m"

    # check handling of units with spaces in them
    t_3 = tel_model.get_parameter_value_with_unit("teltrig_min_sigsum")
    assert t_3.unit == "mV ns"

    assert isinstance(tel_model.get_parameter_value_with_unit("fadc_mhz"), u.Quantity)
    assert not isinstance(tel_model.get_parameter_value_with_unit("num_gains"), u.Quantity)


def test_handling_parameters(telescope_model_lst):
    tel_model = telescope_model_lst

    logger.info(
        "Old mirror_reflection_random_angle: "
        f"{tel_model.get_parameter_value('mirror_reflection_random_angle')}"
    )
    logger.info("Changing mirror_reflection_random_angle")
    new_mrra = "0.0080 0 0"
    tel_model.change_parameter("mirror_reflection_random_angle", new_mrra)
    assert tel_model.get_parameter_value("mirror_reflection_random_angle")[0] == pytest.approx(
        0.0080
    )

    tel_model.change_parameter(
        "mirror_reflection_random_angle", gen.convert_string_to_list(new_mrra)
    )
    assert tel_model.get_parameter_value("mirror_reflection_random_angle")[0] == pytest.approx(
        0.0080
    )

    with pytest.raises(InvalidModelParameterError):
        tel_model._get_parameter_dict("bla_bla")


def test_set_config_file_directory_and_name(telescope_model_lst, caplog):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    telescope_copy.name = None
    telescope_copy.site = None
    with caplog.at_level(logging.DEBUG):
        telescope_copy._set_config_file_directory_and_name()
    assert "Config file path" not in caplog.text


def test_get_simulation_software_parameters(telescope_model_lst):
    assert isinstance(telescope_model_lst.get_simulation_software_parameters("corsika"), dict)


def test_load_simulation_software_parameter(telescope_model_lst, caplog):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    telescope_copy._simulation_config_parameters = {"not_corsika": {}, "not_simtel": {}}
    with caplog.at_level(logging.WARNING):
        telescope_copy._load_simulation_software_parameter()

    assert len(caplog.records) == 0


def test_load_parameters_from_db(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_db = mocker.patch.object(DatabaseHandler, "get_model_parameters")
    telescope_copy._load_parameters_from_db()
    assert mock_db.call_count == 3

    telescope_copy.db = None
    telescope_copy._load_parameters_from_db()
    assert mock_db.call_count == 3


def test_change_parameter(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('camera_pixels')}")
    tel_model.change_parameter("camera_pixels", 9999)
    assert tel_model.get_parameter_value("camera_pixels") == 9999

    logger.info("Testing changing camera_pixels to a float (now allowed)")
    with pytest.raises(ValueError, match=r"^Could not cast 9999.9 of type"):
        tel_model.change_parameter("camera_pixels", 9999.9)

    logger.info("Testing changing camera_pixels to a nonsense string")
    with pytest.raises(ValueError, match=r"^Could not cast bla_bla of type"):
        tel_model.change_parameter("camera_pixels", "bla_bla")

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('mirror_focal_length')}")
    tel_model.change_parameter("mirror_focal_length", 55.0)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")
    tel_model.change_parameter("mirror_focal_length", 55)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")

    tel_model.change_parameter("mirror_focal_length", "9999.9 0.")
    assert pytest.approx(9999.9) == tel_model.get_parameter_value("mirror_focal_length")[0]

    logger.info("Testing changing mirror_focal_length to a nonsense string")
    with pytest.raises(ValueError, match=r"^Could not cast bla_bla of type"):
        tel_model.change_parameter("mirror_focal_length", "bla_bla")

    with pytest.raises(InvalidModelParameterError, match="Parameter bla_bla not in the model"):
        tel_model.change_parameter("bla_bla", 9999.9)


def test_change_multiple_parameters_from_file(telescope_model_lst, caplog, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mocker_gen = mocker.patch("simtools.io.ascii_handler.collect_data_from_file", return_value={})
    with caplog.at_level(logging.WARNING):
        telescope_copy.change_multiple_parameters_from_file(file_name="test_file")
    assert "Changing multiple parameters from file is a feature for developers." in caplog.text
    mocker_gen.assert_called_once()


def test_change_multiple_parameters(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_change = mocker.patch.object(TelescopeModel, "change_parameter")
    telescope_copy.change_multiple_parameters(**{"camera_pixels": 9999, "mirror_focal_length": 55})
    mock_change.assert_any_call("camera_pixels", 9999)
    mock_change.assert_any_call("mirror_focal_length", 55)
    assert not telescope_copy._is_config_file_up_to_date


def test_flen_type(telescope_model_lst):
    tel_model = telescope_model_lst
    flen_info = tel_model._get_parameter_dict("focal_length")
    logger.info(f"Focal Length = {flen_info['value']}, type = {flen_info['type']}")

    assert isinstance(flen_info["value"], float)


def test_updating_export_model_files(db_config, model_version):
    tel = TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        label="test-telescope-model-2",
        db_config=db_config,
    )

    logger.debug(
        "tel._is_exported_model_files should be False because write_sim_telarray_config_file"
        " was not called yet."
    )
    assert False is tel._is_exported_model_files_up_to_date

    # Exporting config file
    tel.write_sim_telarray_config_file()
    logger.debug(
        "tel._is_exported_model_files should be True because "
        "write_sim_telarray_config_file was called."
    )
    assert tel._is_exported_model_files_up_to_date

    # Changing a non-file parameter
    logger.info("Changing a parameter that IS NOT a file - mirror_reflection_random_angle")
    tel.change_parameter("mirror_reflection_random_angle", "0.0080 0 0")
    logger.debug(
        "tel._is_exported_model_files should still be True because the changed "
        "parameter was not a file"
    )
    assert tel._is_exported_model_files_up_to_date

    # Testing the DB connection
    logger.info("DB should NOT be read next.")
    tel.write_sim_telarray_config_file()

    # Changing a parameter that is a file
    logger.debug("Changing a parameter that IS a file - camera_config_file")
    tel.change_parameter("camera_config_file", tel.get_parameter_value("camera_config_file"))
    logger.debug(
        "tel._is_exported_model_files should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._is_exported_model_files_up_to_date


def test_export_parameter_file(telescope_model_lst, mocker):
    parameter = "array_coordinates_UTM"
    file_path = "tests/resources/telescope_positions-North-ground.ecsv"
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_copy = mocker.patch("shutil.copy")
    telescope_copy.export_parameter_file(par_name=parameter, file_path=file_path)
    mock_copy.assert_called_once_with(file_path, telescope_copy.config_file_directory)


def test_export_model_files(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_db = mocker.patch.object(DatabaseHandler, "export_model_files")
    telescope_copy.export_model_files()
    assert telescope_copy._is_exported_model_files_up_to_date
    mock_db.assert_called_once()

    telescope_copy._added_parameter_files = ["test_file"]
    with pytest.raises(KeyError):
        telescope_copy.export_model_files()


def test_config_file_path(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    telescope_copy._config_file_path = None
    mock_config = mocker.patch.object(TelescopeModel, "_set_config_file_directory_and_name")
    telescope_copy.config_file_path
    mock_config.assert_called_once()

    telescope_copy._config_file_path = Path("test_path")
    assert telescope_copy.config_file_path == Path("test_path")
    not mock_config.assert_called_once()


def test_export_nsb_spectrum_to_telescope_altitude_correction_file(telescope_model_lst, mocker):
    model_directory = Path("test_model_directory")
    telescope_copy = copy.deepcopy(telescope_model_lst)

    mock_db_export = mocker.patch.object(DatabaseHandler, "export_model_files")
    mock_simulation_config_parameters = {
        "sim_telarray": {"correct_nsb_spectrum_to_telescope_altitude": {"value": "test_value"}}
    }
    telescope_copy._simulation_config_parameters = mock_simulation_config_parameters

    telescope_copy.export_nsb_spectrum_to_telescope_altitude_correction_file(model_directory)

    mock_db_export.assert_called_once_with(
        parameters={
            "nsb_spectrum_at_2200m": {
                "value": "test_value",
                "file": True,
            }
        },
        dest=model_directory,
    )


def test_write_sim_telarray_config_file(telescope_model_lst, mocker):
    """Test writing sim_telarray config file with and without additional model."""
    telescope_copy = copy.deepcopy(telescope_model_lst)

    mock_writer = mocker.Mock()
    mock_writer.write_telescope_config_file = mocker.Mock()

    mock_export = mocker.patch.object(TelescopeModel, "export_model_files")
    mock_load_writer = mocker.patch.object(
        TelescopeModel,
        "_load_simtel_config_writer",
        side_effect=lambda: setattr(telescope_copy, "simtel_config_writer", mock_writer),
    )

    telescope_copy.write_sim_telarray_config_file()
    mock_export.assert_called_once_with(update_if_necessary=True)
    mock_load_writer.assert_called_once()
    mock_writer.write_telescope_config_file.assert_called_once()

    mock_export.reset_mock()
    mock_load_writer.reset_mock()
    mock_writer.write_telescope_config_file.reset_mock()

    add_model = copy.deepcopy(telescope_model_lst)
    add_model.parameters = {"test_param": "test_value"}

    telescope_copy.write_sim_telarray_config_file(additional_models=add_model)
    assert mock_export.call_count == 2  # Called for both models
    mock_load_writer.assert_called_once()
    assert telescope_copy.parameters.get("test_param") == "test_value"
    mock_writer.write_telescope_config_file.assert_called_once()

    mock_export.assert_any_call(telescope_copy.config_file_directory, update_if_necessary=True)


def test_add_additional_models(telescope_model_lst, mocker):
    """Test _add_additional_models method."""
    telescope_copy = copy.deepcopy(telescope_model_lst)

    # Test case 1: None input
    telescope_copy._add_additional_models(None)
    # Should not change anything

    # Test case 2: Single model
    mock_model = mocker.Mock()
    mock_model.parameters = {"new_param": "new_value"}
    mock_model.export_model_files = mocker.Mock()

    telescope_copy._add_additional_models(mock_model)
    assert "new_param" in telescope_copy.parameters
    assert telescope_copy.parameters["new_param"] == "new_value"
    mock_model.export_model_files.assert_called_once()

    # Test case 3: Dictionary of models
    mock_model2 = mocker.Mock()
    mock_model2.parameters = {"param2": "value2"}
    mock_model2.export_model_files = mocker.Mock()

    models_dict = {"model1": mock_model, "model2": mock_model2}
    telescope_copy._add_additional_models(models_dict)
    assert telescope_copy.parameters["param2"] == "value2"
