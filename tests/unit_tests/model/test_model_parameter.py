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
    tel_model.overwrite_model_parameter("mirror_reflection_random_angle", new_mrra)
    assert tel_model.get_parameter_value("mirror_reflection_random_angle")[0] == pytest.approx(
        0.0080
    )

    tel_model.overwrite_model_parameter(
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


def test_overwrite_model_parameter(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('camera_pixels')}")
    tel_model.overwrite_model_parameter("camera_pixels", 9999)
    assert tel_model.get_parameter_value("camera_pixels") == 9999

    logger.info("Testing changing camera_pixels to a float (now allowed)")
    with pytest.raises(ValueError, match=r"^Could not cast 9999.9 of type"):
        tel_model.overwrite_model_parameter("camera_pixels", 9999.9)

    logger.info("Testing changing camera_pixels to a nonsense string")
    with pytest.raises(ValueError, match=r"^Could not cast bla_bla of type"):
        tel_model.overwrite_model_parameter("camera_pixels", "bla_bla")

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('mirror_focal_length')}")
    tel_model.overwrite_model_parameter("mirror_focal_length", 55.0)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")
    tel_model.overwrite_model_parameter("mirror_focal_length", 55)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")

    tel_model.overwrite_model_parameter("mirror_focal_length", "9999.9 0.")
    assert pytest.approx(9999.9) == tel_model.get_parameter_value("mirror_focal_length")[0]

    logger.info("Testing changing mirror_focal_length to a nonsense string")
    with pytest.raises(ValueError, match=r"^Could not cast bla_bla of type"):
        tel_model.overwrite_model_parameter("mirror_focal_length", "bla_bla")

    with pytest.raises(InvalidModelParameterError, match="Parameter bla_bla not in the model"):
        tel_model.overwrite_model_parameter("bla_bla", 9999.9)


def test_overwrite_parameters(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_change = mocker.patch.object(TelescopeModel, "overwrite_model_parameter")
    telescope_copy.overwrite_parameters(
        {"camera_pixels": {"value": 9999}, "mirror_focal_length": {"value": 55}}
    )
    mock_change.assert_any_call("camera_pixels", 9999, None)
    mock_change.assert_any_call("mirror_focal_length", 55, None)


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
    tel.overwrite_model_parameter("mirror_reflection_random_angle", "0.0080 0 0")
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
    tel.overwrite_model_parameter(
        "camera_config_file", tel.get_parameter_value("camera_config_file")
    )
    logger.debug(
        "tel._is_exported_model_files should be False because a parameter that "
        "is a file was changed."
    )
    assert False is tel._is_exported_model_files_up_to_date


def test_overwrite_model_file(telescope_model_lst, mocker):
    parameter = "array_coordinates_UTM"
    file_path = "tests/resources/telescope_positions-North-ground.ecsv"
    telescope_copy = copy.deepcopy(telescope_model_lst)
    mock_copy = mocker.patch("shutil.copy")
    telescope_copy.overwrite_model_file(par_name=parameter, file_path=file_path)
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


def test_get_parameter_value_with_none_type(telescope_model_lst, caplog):
    """Test get_parameter_value when get_parameter_type returns None."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a parameter with a string value but no type
    # This tests the AttributeError exception handling
    tel_model.parameters["test_param"] = {"value": "1.5"}

    with caplog.at_level(logging.DEBUG):
        result = tel_model.get_parameter_value("test_param")
    # Should handle None type gracefully and return the parsed value
    assert abs(result - 1.5) < 0.001


def test_get_parameter_value_no_value(telescope_model_lst):
    """Test get_parameter_value when parameter has no value key."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a parameter without a value
    tel_model.parameters["no_value_param"] = {"type": "int64"}

    with pytest.raises(InvalidModelParameterError, match="does not have a value"):
        tel_model.get_parameter_value("no_value_param")


def test_get_parameter_version(telescope_model_lst):
    """Test get_parameter_version method."""
    assert isinstance(telescope_model_lst.get_parameter_version("num_gains"), str)
    # Check that version format is correct (e.g., "1.0.0")
    version = telescope_model_lst.get_parameter_version("num_gains")
    assert len(version.split(".")) == 3


def test_check_model_parameter_versions_no_schema(telescope_model_lst, mocker):
    """Test _check_model_parameter_versions when parameter not in schema."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Mock names.model_parameters to return empty dict
    mocker.patch("simtools.model.model_parameter.names.model_parameters", return_value={})

    # Should not raise any error
    tel_model._check_model_parameter_versions(["num_gains"])


def test_overwrite_model_parameter_with_parameter_version(telescope_model_lst, mocker):
    """Test overwrite_model_parameter with parameter_version but no value."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Mock get_model_parameter to return parameter dict
    mock_param_dict = {
        "num_gains": {
            "value": 999,
            "parameter_version": "2.0.0",
            "type": "int64",
            "file": False,
        }
    }
    mocker.patch.object(tel_model.db, "get_model_parameter", return_value=mock_param_dict)

    # Call with only parameter_version (no value)
    tel_model.overwrite_model_parameter("num_gains", value=None, parameter_version="2.0.0")

    # Verify the parameter was updated
    assert tel_model.parameters["num_gains"]["value"] == 999
    assert tel_model.parameters["num_gains"]["parameter_version"] == "2.0.0"


def test_overwrite_model_parameter_not_in_model(telescope_model_lst):
    """Test overwrite_model_parameter with parameter not in model."""
    tel_model = copy.deepcopy(telescope_model_lst)

    with pytest.raises(InvalidModelParameterError, match="not in the model"):
        tel_model.overwrite_model_parameter("nonexistent_param", value=123)


def test_overwrite_model_parameter_updates_exported_files_flag(telescope_model_lst):
    """Test that overwriting a file parameter sets _is_exported_model_files_up_to_date to False."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Find a parameter that is a file
    file_param = None
    for par_name, par_dict in tel_model.parameters.items():
        if par_dict.get("file", False):
            file_param = par_name
            break

    if file_param:
        tel_model._is_exported_model_files_up_to_date = True
        tel_model.overwrite_model_parameter(file_param, value="new_file.dat")
        assert tel_model._is_exported_model_files_up_to_date is False


def test_overwrite_parameters_from_file_no_changes(telescope_model_lst, tmp_path):
    """Test overwrite_parameters_from_file when no changes for this model."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a valid file with changes for a different model using proper telescope name pattern
    changes_file = tmp_path / "changes.yml"
    changes_file.write_text(
        "model_version: 6.0.0\n"
        "model_update: patch_update\n"
        "model_version_history: [5.0.0]\n"
        "description: Test changes\n"
        "schema_version: 0.1.0\n"
        "changes:\n"
        "  MSTN-01:\n"  # Valid telescope name, but different from tel_model.name
        "    num_gains:\n"
        "      version: 1.0.0\n"
        "      value: 123\n",
        encoding="utf-8",
    )

    # Should not raise error, just not apply any changes since MSTN-01 != tel_model.name
    original_params = copy.deepcopy(tel_model.parameters)
    tel_model.overwrite_parameters_from_file(str(changes_file))

    # Parameters should be unchanged (unless tel_model.name happens to be MSTN-01)
    if tel_model.name != "MSTN-01":
        assert tel_model.parameters == original_params


def test_overwrite_parameters_with_version_dict(telescope_model_lst):
    """Test overwrite_parameters with dict containing version key."""
    tel_model = copy.deepcopy(telescope_model_lst)

    changes = {"num_gains": {"value": 4, "version": "2.0.0"}}

    tel_model.overwrite_parameters(changes)

    assert tel_model.parameters["num_gains"]["value"] == 4
    assert tel_model.parameters["num_gains"]["parameter_version"] == "2.0.0"


def test_overwrite_parameters_with_simple_value(telescope_model_lst):
    """Test overwrite_parameters with simple value (not a dict)."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Simple value (not a dict with 'value' or 'version' keys)
    changes = {"num_gains": 5}

    tel_model.overwrite_parameters(changes)

    assert tel_model.parameters["num_gains"]["value"] == 5


def test_overwrite_parameters_from_file_with_changes(telescope_model_lst, tmp_path):
    """Test overwrite_parameters_from_file when changes exist."""
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a valid file with changes for this model
    changes_file = tmp_path / "changes.yml"
    changes_file.write_text(
        f"model_version: 6.0.0\n"
        f"model_update: patch_update\n"
        f"model_version_history: [5.0.0]\n"
        f"description: Test changes\n"
        f"schema_version: 0.1.0\n"
        f"changes:\n"
        f"  {tel_model.name}:\n"
        f"    num_gains:\n"
        f"      version: 1.0.0\n"
        f"      value: 999\n",
        encoding="utf-8",
    )

    tel_model.overwrite_parameters_from_file(str(changes_file))

    # Parameter should be changed
    assert tel_model.parameters["num_gains"]["value"] == 999


def test_check_model_parameter_with_overwrite_file(
    db_config, io_handler, model_version, tmp_path, mocker
):
    """Test _check_model_parameter_versions with overwrite_model_parameters."""

    # Create a temporary overwrite file
    overwrite_file = tmp_path / "overwrite.yml"
    overwrite_file.write_text(
        "model_version: 6.0.0\n"
        "model_update: patch_update\n"
        "model_version_history: [5.0.0]\n"
        "description: Test\n"
        "schema_version: 0.1.0\n"
        "changes:\n"
        "  LSTN-01:\n"
        "    num_gains:\n"
        "      version: 1.0.0\n"
        "      value: 10\n",
        encoding="utf-8",
    )

    # Mock names.model_parameters to avoid actual schema loading
    mocker.patch("simtools.model.model_parameter.names.model_parameters", return_value={})

    # Create telescope model with overwrite_model_parameters
    tel_model = TelescopeModel(
        site="North",
        telescope_name="LSTN-01",
        model_version=model_version,
        db_config=db_config,
        label="test-telescope-model",
        overwrite_model_parameters=str(overwrite_file),
    )

    # The overwrite file should have been applied during initialization
    assert tel_model.parameters["num_gains"]["value"] == 10


def test__get_key_for_parameter_changes(telescope_model_lst):
    assert telescope_model_lst._get_key_for_parameter_changes("North", None, {}) == "OBS-North"

    lst = "LSTN-01"

    assert telescope_model_lst._get_key_for_parameter_changes("North", lst, {}) is None

    assert telescope_model_lst._get_key_for_parameter_changes("North", lst, {lst: "abc"}) == lst

    assert (
        telescope_model_lst._get_key_for_parameter_changes("North", lst, {"LSTN-design": "abc"})
        == "LSTN-design"
    )

    assert (
        telescope_model_lst._get_key_for_parameter_changes("North", "LSTN-design", {lst: "abc"})
        is None
    )
