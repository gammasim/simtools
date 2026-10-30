#!/usr/bin/python3

import copy
import logging
from pathlib import Path

import pytest
from astropy import units as u

from simtools.data_model import parameter_catalog as catalog
from simtools.model.model_parameter import (
    InvalidModelParameterError,
    ModelParameter,
    _normalise_overrides,
)
from simtools.model.telescope_model import TelescopeModel
from simtools.utils import value_conversion

# Capture the original static method at import time, before autouse fixtures can mock it
_check_model_parameter_versions = ModelParameter.__dict__[
    "_check_model_parameter_versions"
].__func__

logger = logging.getLogger()


def test_model_parameter_without_array_element_does_not_load_design_model(mocker):
    model_reader = mocker.Mock()

    ModelParameter(model_version="1.0.0", model_reader=model_reader)

    model_reader.get_design_model.assert_not_called()


@pytest.fixture
def num_gains_dict():
    """Fixture for num_gains parameter dictionary."""
    return {
        "num_gains": {
            "value": 999,
            "parameter_version": "2.0.0",
            "type": "int64",
            "file": False,
        }
    }


def test_get_parameter_type(telescope_model_lst, caplog):
    assert telescope_model_lst.get_parameter_type("num_gains") == "int64"
    telescope_model_copy = copy.deepcopy(telescope_model_lst)
    telescope_model_copy.parameters["num_gains"].pop("type")
    with caplog.at_level(logging.DEBUG):
        assert telescope_model_copy.get_parameter_type("num_gains") is None
    assert "Parameter num_gains does not have a type." in caplog.text


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
        "mirror_reflection_random_angle",
        value_conversion.convert_string_to_list(new_mrra),
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


def test_set_config_file_directory_and_name_initializes_directory(
    telescope_model_lst, mocker, tmp_test_directory
):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    telescope_copy._config_file_directory = None
    mocker.patch.object(
        telescope_copy.io_handler,
        "get_model_configuration_directory",
        return_value=Path(tmp_test_directory),
    )
    mocker.patch(
        "simtools.model.model_parameter.names.sim_telarray_config_file_name",
        return_value="config.cfg",
    )

    telescope_copy._set_config_file_directory_and_name()

    assert telescope_copy.config_file_path == Path(tmp_test_directory) / "config.cfg"


def test_get_simulation_software_parameters(telescope_model_lst):
    assert isinstance(telescope_model_lst.get_simulation_software_parameters("corsika"), dict)


def test_overwrite_model_parameter(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('camera_pixels')}")
    tel_model.overwrite_model_parameter("camera_pixels", 9999)
    assert tel_model.get_parameter_value("camera_pixels") == 9999

    logger.info("Testing changing camera_pixels to a float (now allowed)")
    with pytest.raises(TypeError, match=r"^Error validating dictionary"):
        tel_model.overwrite_model_parameter("camera_pixels", 9999.9)

    logger.info("Testing changing camera_pixels to a nonsense string")
    with pytest.raises(TypeError, match=r"^Error validating dictionary"):
        tel_model.overwrite_model_parameter("camera_pixels", "bla_bla")

    logger.info(f"Old camera_pixels:{tel_model.get_parameter_value('mirror_focal_length')}")
    tel_model.overwrite_model_parameter("mirror_focal_length", 55.0)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")
    tel_model.overwrite_model_parameter("mirror_focal_length", 55)
    assert pytest.approx(55.0) == tel_model.get_parameter_value("mirror_focal_length")

    tel_model.overwrite_model_parameter("mirror_focal_length", "9999.9 0.")
    assert pytest.approx(9999.9) == tel_model.get_parameter_value("mirror_focal_length")[0]

    logger.info("Testing changing mirror_focal_length to a nonsense string")
    with pytest.raises(TypeError, match=r"^Error validating dictionary using"):
        tel_model.overwrite_model_parameter("mirror_focal_length", "bla_bla")

    with pytest.raises(InvalidModelParameterError, match="Parameter bla_bla not in the model"):
        tel_model.overwrite_model_parameter("bla_bla", 9999.9)


def test_overwrite_parameter_with_schema_metadata(telescope_model_lst, tmp_test_directory, mocker):
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create mock schema file
    schema_dir = Path(tmp_test_directory) / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_file = schema_dir / "mirror_focal_length.schema.yml"
    schema_content = """---
schema_version: "0.1.0"
name: mirror_focal_length
data:
  - type: float64
    unit: m
"""
    schema_file.write_text(schema_content)

    mocker.patch("simtools.data_model.schema.MODEL_PARAMETER_SCHEMA_PATH", schema_dir)

    tel_model.overwrite_model_parameter(
        "mirror_focal_length",
        55.0,
        metadata={
            "unit": "m",
            "model_parameter_schema_version": "0.1.0",
        },
    )

    # Verify metadata was applied
    param_dict = tel_model._get_parameter_dict("mirror_focal_length")
    assert param_dict["value"] == pytest.approx(55.0)
    assert param_dict["unit"] == "m"
    assert param_dict["model_parameter_schema_version"] == "0.1.0"
    # Type should remain scalar (not converted to list)
    assert isinstance(param_dict["type"], str)


def test_overwrite_heterogeneous_list_with_schema_metadata(
    telescope_model_lst, tmp_test_directory, mocker
):
    tel_model = copy.deepcopy(telescope_model_lst)

    # Add a mock heterogeneous list parameter to the telescope model
    tel_model.parameters["test_hetero_param"] = {
        "value": ["old", 1.0, 2.0],
        "type": "list",
        "unit": ["none", "m", "m"],
        "model_parameter_schema_version": "0.1.0",
    }

    # Create mock schema file for heterogeneous list parameter
    schema_dir = Path(tmp_test_directory) / "schemas"
    schema_dir.mkdir(parents=True, exist_ok=True)
    schema_file = schema_dir / "test_hetero_param.schema.yml"
    schema_content = """---
schema_version: "0.2.0"
name: test_hetero_param
data:
  - type: str
  - type: float64
  - type: float64
"""
    schema_file.write_text(schema_content)

    mocker.patch("simtools.data_model.schema.MODEL_PARAMETER_SCHEMA_PATH", schema_dir)
    mocker.patch(
        "simtools.model.model_parameter.DataValidator.validate_model_parameter",
        side_effect=lambda par_dict, par_name=None: par_dict,
    )

    # Overwrite with heterogeneous list value and schema metadata
    tel_model.overwrite_model_parameter(
        "test_hetero_param",
        ["new_string", 5.0, 10.0],
        metadata={
            "unit": ["none", "ns", "ns"],
            "model_parameter_schema_version": "0.2.0",
        },
    )

    # Verify metadata was applied
    param_dict = tel_model._get_parameter_dict("test_hetero_param")
    assert param_dict["value"][0] == "new_string"
    assert param_dict["value"][1] == pytest.approx(5.0)
    assert param_dict["value"][2] == pytest.approx(10.0)
    assert param_dict["unit"] == ["none", "ns", "ns"]
    assert param_dict["model_parameter_schema_version"] == "0.2.0"
    # Type should be a list of types for heterogeneous list
    assert isinstance(param_dict["type"], list)
    assert param_dict["type"] == ["str", "float64", "float64"]


def test_get_parameter_value_with_none_type(telescope_model_lst, caplog):
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a parameter with a string value but no type
    # This tests the AttributeError exception handling
    tel_model.parameters["test_param"] = {"value": "1.5"}

    with caplog.at_level(logging.DEBUG):
        result = tel_model.get_parameter_value("test_param")
    # Should handle None type gracefully and return the parsed value
    assert abs(result - 1.5) < 0.001


def test_get_parameter_value_no_value(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)

    # Create a parameter without a value
    tel_model.parameters["no_value_param"] = {"type": "int64"}

    with pytest.raises(InvalidModelParameterError, match="does not have a value"):
        tel_model.get_parameter_value("no_value_param")


def test_get_parameter_version(telescope_model_lst):
    assert isinstance(telescope_model_lst.get_parameter_version("num_gains"), str)
    # Check that version format is correct (e.g., "1.0.0")
    version = telescope_model_lst.get_parameter_version("num_gains")
    assert len(version.split(".")) == 3


def test_overwrite_model_parameter_with_parameter_version(
    telescope_model_lst, mocker, num_gains_dict
):
    tel_model = copy.deepcopy(telescope_model_lst)

    # Mock get_model_parameter to return parameter dict
    mocker.patch.object(tel_model.model_reader, "get_model_parameter", return_value=num_gains_dict)

    # Call with only parameter_version (no value)
    tel_model.overwrite_model_parameter("num_gains", value=None, parameter_version="2.0.0")

    # Verify the parameter was updated
    assert tel_model.parameters["num_gains"]["value"] == 999
    assert tel_model.parameters["num_gains"]["parameter_version"] == "2.0.0"


def test_check_model_parameter_with_overwrite(model_version, run_context):

    # Create a temporary telescope model to get the current value of num_gains of the telescope,
    # to make sure we change it to a different value in the test
    site = "North"
    telescope_name = "LSTN-01"
    temp_tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label="test-telescope-model",
        context=run_context,
    )

    changes = {
        telescope_name: {
            "num_gains": {
                "version": "1.0.0",
                "value": 1 if temp_tel_model.parameters["num_gains"]["value"] != 1 else 2,
            }
        }
    }

    # Create telescope model with overwrite_model_parameters
    tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label="test-telescope-model",
        overwrite_model_parameter_dict=changes,
        context=run_context,
    )

    # The overwrite should have been applied during initialization
    assert (
        tel_model.parameters["num_gains"]["value"] == changes[telescope_name]["num_gains"]["value"]
    )


def test_check_simulation_software_parameter_with_overwrite(model_version, run_context):

    site = "North"
    telescope_name = "LSTN-01"
    changes = {
        telescope_name: {
            "correct_nsb_spectrum_to_telescope_altitude": {"value": "nsb_spectrum_overwritten.dat"},
            "min_photoelectrons": {"value": 0},
            "min_photons": {"value": 0},
        }
    }

    tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label="test-telescope-model",
        overwrite_model_parameter_dict=changes,
        context=run_context,
    )

    assert (
        tel_model.get_simulation_software_parameters("sim_telarray")[
            "correct_nsb_spectrum_to_telescope_altitude"
        ]["value"]
        == "nsb_spectrum_overwritten.dat"
    )
    assert (
        tel_model.get_simulation_software_parameters("sim_telarray")["min_photoelectrons"]["value"]
        == 0
    )
    assert tel_model.get_simulation_software_parameters("sim_telarray")["min_photons"]["value"] == 0


def test_check_corsika_simulation_software_parameter_with_overwrite(
    model_version, mock_model_reader, run_context
):

    site = "North"
    telescope_name = "LSTN-01"
    corsika_parameter = "corsika_particle_kinetic_energy_cutoff"
    mock_model_reader.get_simulation_configuration_parameters.side_effect = lambda **kwargs: (
        {corsika_parameter: {"value": 1.0, "type": "float64", "file": False}}
        if kwargs.get("simulation_software") == "corsika"
        else {
            "correct_nsb_spectrum_to_telescope_altitude": {
                "value": "nsb.dat",
                "type": "str",
                "file": True,
            }
        }
    )

    changes = {
        "configuration_corsika": {
            corsika_parameter: {
                "value": 2.5,
            }
        }
    }

    tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label="test-telescope-model",
        overwrite_model_parameter_dict=changes,
        context=run_context.replace(model_reader=mock_model_reader),
    )

    assert tel_model.get_simulation_software_parameters("corsika")[corsika_parameter][
        "value"
    ] == pytest.approx(2.5)


def test_check_corsika_overwrite_with_configuration_key(
    model_version, mock_model_reader, run_context
):

    site = "North"
    telescope_name = "LSTN-01"
    corsika_parameter = "corsika_cherenkov_photon_bunch_size"
    mock_model_reader.get_simulation_configuration_parameters.side_effect = lambda **kwargs: (
        {corsika_parameter: {"value": 5.0, "type": "float64", "file": False}}
        if kwargs.get("simulation_software") == "corsika"
        else {
            "correct_nsb_spectrum_to_telescope_altitude": {
                "value": "nsb.dat",
                "type": "str",
                "file": True,
            }
        }
    )

    changes = {
        "configuration_corsika": {
            corsika_parameter: {
                "version": "2.0.0",
                "value": 50.0,
            }
        }
    }

    tel_model = TelescopeModel(
        site=site,
        telescope_name=telescope_name,
        model_version=model_version,
        label="test-telescope-model",
        overwrite_model_parameter_dict=changes,
        context=run_context.replace(model_reader=mock_model_reader),
    )

    assert tel_model.get_simulation_software_parameters("corsika")[corsika_parameter][
        "value"
    ] == pytest.approx(50.0)


def test_check_model_parameter_versions(mocker):
    parameters = {
        "num_gains": {
            "value": 1,
            "model_parameter_schema_version": "1.0.0",
        },
        "unknown_param": {  # Not in schema - should be silently skipped
            "value": 42,
            "model_parameter_schema_version": "1.0.0",
        },
    }

    mocker.patch(
        "simtools.data_model.parameter_catalog.model_parameters",
        return_value={"num_gains": {"schema_version": "1.0.0"}},
    )
    mock_validate = mocker.patch(
        "simtools.model.model_parameter.schema.validate_deprecation_and_version"
    )
    _check_model_parameter_versions(parameters, ignore_software_version=False)

    # validate called only for "num_gains" (known in schema); "unknown_param" is skipped
    mock_validate.assert_called_once_with(
        data={"schema_version": "1.0.0"},
        software_name=None,
        ignore_software_version=False,
    )
    assert parameters["unknown_param"]["value"] == 42


def test_check_model_parameter_versions_allows_older_scalar_schema(mocker):
    parameters = {
        "num_gains": {
            "value": 1,
            "model_parameter_schema_version": "0.9.0",
        }
    }

    mocker.patch(
        "simtools.data_model.parameter_catalog.model_parameters",
        return_value={"num_gains": {"schema_version": "1.0.0"}},
    )
    validate_schema = mocker.patch(
        "simtools.model.model_parameter.schema.validate_deprecation_and_version"
    )
    _check_model_parameter_versions(parameters, ignore_software_version=False)

    validate_schema.assert_called_once_with(
        data={"schema_version": "1.0.0"},
        software_name=None,
        ignore_software_version=False,
    )


def test_normalise_overrides_keys_by_collection_element_and_parameter():
    overrides = {
        "LSTN-01": {"num_gains": {"value": 2, "version": "2.0.0"}, "min_photons": 3},
        "global": {"min_photoelectrons": 8},
        "not_a_mapping": 1,
    }

    result = _normalise_overrides(overrides)

    assert {(element, par) for (_, element, par) in result} == {
        ("LSTN-01", "num_gains"),
        ("LSTN-01", "min_photons"),
        ("global", "min_photoelectrons"),
    }
    collection = catalog.get_collection_name_from_parameter_name("num_gains")
    assert result[(collection, "LSTN-01", "num_gains")] == {
        "value": 2,
        "version": "2.0.0",
        "metadata": None,
    }


def test_overwrite_model_parameter_with_parameter_version_only_raises_when_unknown(
    telescope_model_lst,
):
    tel_model = copy.deepcopy(telescope_model_lst)

    with pytest.raises(InvalidModelParameterError, match="Parameter unknown_parameter not in"):
        tel_model.overwrite_model_parameter("unknown_parameter", 1)


@pytest.mark.parametrize(
    ("value", "version", "expected_version"),
    [(2, None, None), (2, "2.0.0", "2.0.0")],
)
def test_overwrite_model_parameter_updates_value_and_optional_version(
    telescope_model_lst, value, version, expected_version
):
    tel_model = copy.deepcopy(telescope_model_lst)
    tel_model.overwrite_model_parameter("num_gains", value, parameter_version=version)

    assert tel_model.parameters["num_gains"]["value"] == 2
    if expected_version is not None:
        assert tel_model.parameters["num_gains"]["parameter_version"] == expected_version


def test_overwrite_model_parameter_schema_file_error_names_parameter(telescope_model_lst):
    tel_model = copy.deepcopy(telescope_model_lst)
    tel_model.parameters["no_such_parameter"] = {"value": 1}

    with pytest.raises(
        FileNotFoundError, match="Schema file for parameter no_such_parameter not found"
    ):
        tel_model.overwrite_model_parameter("no_such_parameter", 1)


def test_load_simulation_software_parameters_ignores_repository_value_error(
    telescope_model_lst, mocker
):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    existing = {"min_photons": {"value": 3}}
    telescope_copy._simulation_config_parameters["corsika"] = existing.copy()
    mocker.patch.object(
        telescope_copy.model_reader,
        "get_simulation_configuration_parameters",
        side_effect=ValueError("missing configuration"),
    )

    telescope_copy._load_simulation_software_parameters()

    assert telescope_copy.get_simulation_software_parameters("corsika") == existing


def test_load_simulation_software_parameters_keeps_routed_overrides(telescope_model_lst, mocker):
    telescope_copy = copy.deepcopy(telescope_model_lst)
    telescope_copy._simulation_config_parameters["sim_telarray"] = {
        "min_photoelectrons": {"value": 7}
    }
    mocker.patch.object(
        telescope_copy.model_reader,
        "get_simulation_configuration_parameters",
        side_effect=lambda **_kwargs: {"min_photons": {"value": 3}},
    )

    telescope_copy._load_simulation_software_parameters()

    assert telescope_copy.get_simulation_software_parameters("sim_telarray") == {
        "min_photoelectrons": {"value": 7},
        "min_photons": {"value": 3},
    }


def test_route_to_sim_telarray_keeps_value_and_metadata(telescope_model_lst):
    telescope_copy = copy.deepcopy(telescope_model_lst)

    telescope_copy._route_to_sim_telarray(
        "min_photoelectrons", {"value": 8, "version": None, "metadata": None}
    )
    telescope_copy._route_to_sim_telarray(
        "min_photons",
        {
            "value": 4,
            "version": None,
            "metadata": {"unit": "dimensionless", "model_parameter_schema_version": "1.0.0"},
        },
    )

    parameters = telescope_copy.get_simulation_software_parameters("sim_telarray")
    assert parameters["min_photoelectrons"] == {"value": 8}
    assert parameters["min_photons"] == {
        "value": 4,
        "unit": "dimensionless",
        "model_parameter_schema_version": "1.0.0",
    }
