#!/usr/bin/python3

import io
import logging
from pathlib import Path
from unittest import mock

import astropy.units as u
import numpy as np
import pytest

import simtools.simtel.simtel_table_writer as simtel_table_writer
from simtools.constants import SIM_TELARRAY_INCLUDE_FILENAME_MAX_LENGTH
from simtools.simtel.simtel_config_writer import SimtelConfigWriter

logger = logging.getLogger()


@pytest.fixture
def simtel_config_writer(model_version):
    return SimtelConfigWriter(
        site="North",
        model_version=model_version,
        label="test-simtel-config-writer",
        telescope_model_name="test_telescope",
    )


# Helper functions to reduce code duplication in tests
def create_trigger_dict(
    name,
    multiplicity=2,
    width=120.0,
    width_unit="ns",
    min_separation=None,
    minsep_unit="m",
    hard_stereo=True,
):
    """Create a standardized trigger dictionary for testing."""
    minsep_unit_value = minsep_unit if min_separation else None
    return {
        "name": name,
        "multiplicity": {"value": multiplicity},
        "width": {"value": width, "unit": width_unit},
        "min_separation": {"value": min_separation, "unit": minsep_unit_value},
        "hard_stereo": {"value": hard_stereo},
    }


def create_lsts_trigger(multiplicity=2, hard_stereo=True):
    """Create standard LSTS trigger for testing."""
    return create_trigger_dict("LSTS_array", multiplicity, 120.0, "ns", None, None, hard_stereo)


def create_msts_trigger(multiplicity=2, hard_stereo=False):
    """Create standard MSTS trigger for testing."""
    return create_trigger_dict("MSTS_array", multiplicity, 300.0, "ns", 25.0, "m", hard_stereo)


def create_ssts_trigger(multiplicity=2, hard_stereo=False):
    """Create standard SSTS trigger for testing."""
    return create_trigger_dict("SSTS_array", multiplicity, 300.0, "ns", 25.0, "m", hard_stereo)


def create_standard_telescope_mapping():
    """Create standard telescope mapping for testing."""
    return {"LSTS": [1, 2], "MSTS": [3, 4], "SSTS": [5, 6]}


def create_mock_telescope_model():
    """Create standard mock telescope model for testing."""
    return {
        "LSTS-01": mock.Mock(),
        "LSTS-02": mock.Mock(),
        "MSTS-01": mock.Mock(),
        "MSTS-02": mock.Mock(),
        "SSTS-01": mock.Mock(),
        "SSTS-02": mock.Mock(),
    }


def create_mock_array_triggers():
    """Create standard mock array triggers for testing."""
    return {
        "multiplicity": {"value": 2},
        "width": {"value": 400.0, "unit": "ns"},
        "min_separation": {"value": 30.0, "unit": "m"},
        "hard_stereo": {"value": False},
    }


def create_lsts_mock_trigger():
    """Create LSTS mock trigger response."""
    return {
        "multiplicity": {"value": 2},
        "width": {"value": 120.0, "unit": "ns"},
        "min_separation": {"value": None, "unit": None},
        "hard_stereo": {"value": True},
    }


def create_msts_different_params_mock_trigger():
    """Create MSTS mock trigger with different parameters."""
    return {
        "multiplicity": {"value": 2},
        "width": {"value": 300.0, "unit": "ns"},
        "min_separation": {"value": 25.0, "unit": "m"},
        "hard_stereo": {"value": False},
    }


def setup_mixed_trigger_test(simtel_config_writer, tmp_test_directory, mock_function):
    """Set up and execute a mixed trigger test scenario."""
    telescope_model = create_mock_telescope_model()
    array_triggers = create_mock_array_triggers()

    with mock.patch.object(
        simtel_config_writer,
        "_get_array_triggers_for_telescope_type",
        side_effect=mock_function,
    ):
        result_file = simtel_config_writer._write_array_triggers_file(
            array_triggers, tmp_test_directory, telescope_model
        )

    # Check file was created
    assert result_file == "array_triggers.dat"
    file_path = tmp_test_directory / result_file
    assert file_path.exists()

    # Read and return content
    with open(file_path, encoding="utf-8") as f:
        content = f.read()

    return content.strip().split("\n")


def create_msts_hardstereo_trigger(multiplicity=2):
    """Create MSTS hardstereo trigger for testing."""
    return create_trigger_dict("MSTS_array", multiplicity, 100.0, "ns", 20.0, "m", True)


def create_mixed_trigger_scenario():
    """Create a mixed hardstereo/non-hardstereo scenario for comprehensive testing."""
    return [
        create_lsts_trigger(2, True),
        create_msts_hardstereo_trigger(2),
        create_ssts_trigger(2, False),
    ]


def create_all_hardstereo_scenario():
    """Create all hardstereo scenario for testing."""
    return [
        create_lsts_trigger(2, True),
        create_msts_hardstereo_trigger(2),
    ]


def create_all_non_hardstereo_same_params_scenario():
    """Create all non-hardstereo with same parameters scenario."""
    return [
        create_msts_trigger(2, False),
        create_ssts_trigger(2, False),
    ]


# Common trigger line strings to reduce duplication
LSTS_HARDSTEREO_LINE = "Trigger 2 of 1, 2 width 120.0 hardstereo"
MSTS_HARDSTEREO_LINE = "Trigger 2 of 3, 4 width 100.0 hardstereo minsep 20.0"
MSTS_NON_HARDSTEREO_LINE = "Trigger 2 of 3, 4 width 300.0 minsep 25.0"
SSTS_NON_HARDSTEREO_LINE = "Trigger 2 of 5, 6 width 300.0 minsep 25.0"
COMBINED_NON_HARDSTEREO_LINE = "Trigger 2 of 5, 6, 7, 8 width 300.0 minsep 25.0"
COMBINED_ALL_NON_HARDSTEREO_LINE = "Trigger 2 of 3, 4, 5, 6 width 300.0 minsep 25.0"
TRIGGER_1_2_WIDTH_300_LINE = "Trigger 3 of 1, 2 width 300.0"
TRIGGER_3_4_WIDTH_400_LINE = "Trigger 3 of 3, 4 width 400.0"
TRIGGER_1234_WIDTH_300_LINE = "Trigger 3 of 1, 2, 3, 4 width 300.0"


def test_write_array_config_file(
    simtel_config_writer, telescope_model_lst, io_handler, file_has_text, site_model_north
):
    _file = io_handler.get_output_file(file_name="simtel-config-writer_array.txt")
    telescope_model = {
        "LSTN-01": telescope_model_lst,
        "LSTN-02": telescope_model_lst,
        "LSTN-03": telescope_model_lst,
        "LSTN-04": telescope_model_lst,
    }
    site_model_north._simulation_config_parameters["sim_telarray"].update(
        {
            "iobuf_maximum": {"value": 1000000000},
            "random_generator": {"value": "mt19937"},
        }
    )
    simtel_config_writer.write_array_config_file(
        config_file_path=_file,
        telescope_model=telescope_model,
        site_model=site_model_north,
    )
    assert file_has_text(_file, "TELESCOPE == 1")
    with open(_file, encoding="utf-8") as file:
        array_config = file.read()
    assert array_config.count("iobuf_maximum = 1000000000") == 1
    assert array_config.count("random_generator = mt19937") == 1

    # sim_telarray configuration files need to end with two new lines
    with open(_file) as f:
        lines = f.readlines()
        assert lines[-2].endswith("\n")
        assert lines[-1] == "\n"


def test_write_array_config_file_raises_for_too_long_include_filename(
    simtel_config_writer, io_handler, site_model_north, telescope_model_lst
):
    output_file = io_handler.get_output_file(file_name="simtel-config-writer_array-long-name.txt")
    too_long_name = "a" * (SIM_TELARRAY_INCLUDE_FILENAME_MAX_LENGTH - 3) + ".cfg"
    telescope_model = {
        "LSTN-01": mock.Mock(
            config_file_path=Path(too_long_name),
            parameters=telescope_model_lst.parameters,
        )
    }

    with pytest.raises(ValueError, match=r"^sim_telarray include filename exceeds parser limit"):
        simtel_config_writer.write_array_config_file(
            config_file_path=output_file,
            telescope_model=telescope_model,
            site_model=site_model_north,
        )


def test_write_tel_config_file(simtel_config_writer, io_handler, file_has_text):
    _file = io_handler.get_output_file(file_name="simtel-config-writer_telescope.txt")
    simtel_config_writer.write_telescope_config_file(
        config_file_path=_file,
        parameters={
            "num_gains": {
                "parameter": "num_gains",
                "value": 1,
                "unit": None,
            }
        },
    )
    assert file_has_text(_file, "num_gains = 1")

    simtel_config_writer.write_telescope_config_file(
        config_file_path=_file,
        parameters={
            "array_triggers": {
                "parameter": "array_triggers",
                "value": "array_triggers.dat",
                "unit": None,
            }
        },
    )
    assert not file_has_text(_file, "array_triggers = array_triggers.dat")

    simtel_config_writer.write_telescope_config_file(
        config_file_path=_file,
        parameters={
            "reference_point_longitude": {
                "parameter": "reference_point_longitude",
                "value": -70.316345,
                "unit": "deg",
            }
        },
    )
    assert not file_has_text(_file, "longitude = -70.316345")
    assert file_has_text(_file, "metaparam telescope set longitude=-70.316345")


def test_get_value_string_for_simtel(simtel_config_writer):
    assert simtel_config_writer._get_value_string_for_simtel(None) == "none"
    assert simtel_config_writer._get_value_string_for_simtel(True) == 1
    assert simtel_config_writer._get_value_string_for_simtel(False) == 0
    assert simtel_config_writer._get_value_string_for_simtel([1, 2, 3]) == "1 2 3"
    assert simtel_config_writer._get_value_string_for_simtel(np.array([1, 2, 3])) == "1 2 3"
    assert simtel_config_writer._get_value_string_for_simtel(5) == 5


def test_get_array_triggers_for_telescope_type(simtel_config_writer):
    array_triggers = [
        {"name": "LSTN_array", "multiplicity": {"value": 2}, "width": {"value": 10, "unit": "ns"}},
        {"name": "MSTN_single_telescope", "multiplicity": {"value": 1}},
    ]

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "LSTN", 2)
    assert result is not None
    assert result["name"] == "LSTN_array"
    assert result["multiplicity"]["value"] == 2
    assert result["width"]["value"] == 10
    assert result["width"]["unit"] == "ns"

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "MSTN", 1)
    assert result["multiplicity"]["value"] == 1

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "MSTN", 2)
    assert result is None

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "SST", 2)
    assert result is None


def test_convert_model_parameters_to_simtel_format(
    simtel_config_writer, tmp_test_directory, telescope_model_lst
):
    model_path = Path(tmp_test_directory) / "model"
    model_path.mkdir(exist_ok=True)

    simtel_name, value = simtel_config_writer._convert_model_parameters_to_simtel_format(
        "some_parameter", "some_value", model_path, {"LSTN-01": telescope_model_lst}
    )
    assert simtel_name == "some_parameter"
    assert value == "some_value"

    array_triggers = [
        {
            "name": "LSTN_single_telescope",
            "multiplicity": {"value": 1},
            "width": {"value": 10, "unit": "ns"},
            "min_separation": {"value": 40, "unit": "m"},
            "hard_stereo": {"value": True, "unit": None},
        },
    ]
    simtel_name, value = simtel_config_writer._convert_model_parameters_to_simtel_format(
        "array_triggers", array_triggers, model_path, {"LSTN-01": telescope_model_lst}
    )
    assert simtel_name == "array_triggers"
    assert value == "array_triggers.dat"

    with open(Path(model_path) / value) as f:
        content = f.read()
        assert "Trigger 1 of 1" in content
        assert "hardstereo" in content
        assert "minsep 40" in content
        assert "width 10" in content


def test_convert_model_parameters_to_simtel_format_hard_stereo_false(
    simtel_config_writer, tmp_test_directory, telescope_model_lst
):
    model_path = Path(tmp_test_directory) / "model"
    model_path.mkdir(exist_ok=True)

    array_triggers = [
        {
            "name": "MSTS_single_telescope",
            "multiplicity": {"value": 1},
            "width": {"value": 10, "unit": "ns"},
            "min_separation": {"value": 40, "unit": "m"},
            "hard_stereo": {"value": False, "unit": None},
        },
    ]
    simtel_name, value = simtel_config_writer._convert_model_parameters_to_simtel_format(
        "array_triggers", array_triggers, model_path, {"MSTS-01": telescope_model_lst}
    )

    assert simtel_name == "array_triggers"
    assert value == "array_triggers.dat"

    with open(Path(model_path) / value) as f:
        content = f.read()
        assert "Trigger 1 of 1" in content
        assert "hardstereo" not in content
        assert "minsep 40" in content
        assert "width 10" in content


def test_write_table_parameter_file_passes_through_non_dict_value(
    simtel_config_writer, tmp_test_directory
):
    result = simtel_config_writer._write_table_parameter_file(
        "fadc_pulse_shape",
        "already_a_file.dat",
        Path(tmp_test_directory) / "dummy.cfg",
        None,
    )

    assert result == "already_a_file.dat"


def test_get_sim_telarray_metadata_with_model_parameters(simtel_config_writer):
    model_parameters = {"test_param": {"value": 42}}

    def mock_get_name(key, software_name):
        return "test_param" if software_name == "sim_telarray" else None

    with (
        mock.patch(
            "simtools.utils.names.get_simulation_software_name_from_parameter_name",
            side_effect=mock_get_name,
        ),
        mock.patch(
            "simtools.utils.names.get_simulation_software_meta_parameter_mode",
            return_value="add",
        ),
        mock.patch("simtools.simtel.simtel_validate_metadata.validate_metadata"),
    ):
        tel_meta = simtel_config_writer._get_sim_telarray_metadata(
            "telescope", model_parameters, "test_telescope"
        )
        assert "metaparam telescope add test_param" in tel_meta

        site_meta = simtel_config_writer._get_sim_telarray_metadata("site", model_parameters, None)
        assert "metaparam global add test_param" in site_meta


def test_get_sim_telarray_metadata_without_model_parameters(simtel_config_writer):
    _tel = simtel_config_writer._get_sim_telarray_metadata(
        "telescope", None, simtel_config_writer._telescope_model_name
    )
    assert len(_tel) == 8
    assert f"camera_config_variant = {simtel_config_writer._telescope_model_name}" in _tel
    assert f"optics_config_variant = {simtel_config_writer._telescope_model_name}" in _tel
    # Check that variant fields have default value when telescope_design_model is not provided
    assert "camera_config_name = design_model_not_set" in _tel
    assert "optics_config_name = design_model_not_set" in _tel

    _site = simtel_config_writer._get_sim_telarray_metadata("site", None, None)
    assert f"site_config_name = {simtel_config_writer._site}" in _site
    assert f"array_config_name = {simtel_config_writer._layout_name}" in _site

    with pytest.raises(ValueError, match=r"^Unknown metadata type"):
        simtel_config_writer._get_sim_telarray_metadata("unknown", None, None)


def test_get_sim_telarray_metadata_includes_falsey_additional_metadata(simtel_config_writer):
    metadata = simtel_config_writer._get_sim_telarray_metadata(
        "site",
        None,
        None,
        {"primary": "gamma", "azimuth_angle": 0.0, "ha_angle": 0.0},
    )

    assert "metaparam global set primary=gamma" in metadata
    assert "metaparam global set azimuth_angle=0.0" in metadata
    assert "metaparam global set ha_angle=0.0" in metadata


def test_get_sim_telarray_metadata_raises_for_unknown_additional_metadata(simtel_config_writer):
    with pytest.raises(KeyError, match=r"Unknown sim_telarray metadata key emitted by writer"):
        simtel_config_writer._get_sim_telarray_metadata(
            "site", None, None, {"unknown_metadata_key": 1}
        )


def test_write_simtools_parameters_validates_metadata_lines(simtel_config_writer):
    file_obj = io.StringIO()

    with (
        mock.patch(
            "simtools.simtel.simtel_config_writer.dependencies.get_build_options",
            return_value={"corsika_build_id": "invalid-int"},
        ),
        pytest.raises(ValueError, match=r"invalid literal for int"),
    ):
        simtel_config_writer._write_simtools_parameters(file_obj)


def test_write_simtools_parameters(simtel_config_writer, tmp_test_directory, file_has_text):
    # Create a mock file to write to
    test_file = tmp_test_directory / "test_simtools_params.txt"
    with open(test_file, "w") as f:
        simtel_config_writer._write_simtools_parameters(f)

    # Check basic parameters are written
    assert file_has_text(test_file, "% Simtools parameters")
    assert file_has_text(test_file, "metaparam global set simtools_version")
    assert file_has_text(
        test_file,
        "metaparam global set simtools_model_production_version = "
        f"{simtel_config_writer._model_version}",
    )

    # Test with sim_telarray_path and build_opts.yml
    build_opts_file = tmp_test_directory / "build_opts.yml"
    with open(build_opts_file, "w") as f:
        f.write("build_date: 2023-01-01\nversion: 1.0.0")

    with mock.patch("simtools.simtel.simtel_config_writer.settings") as mock_settings:
        mock_settings.config.sim_telarray_path = tmp_test_directory
        with open(test_file, "w") as f:
            simtel_config_writer._write_simtools_parameters(f)

        # Check build_opts parameters are included
        assert file_has_text(test_file, "metaparam global set simtools_")

    # Test with invalid sim_telarray_path
    with mock.patch("simtools.simtel.simtel_config_writer.settings") as mock_settings:
        mock_settings.config.sim_telarray_path = tmp_test_directory / "nonexistent"
        with open(test_file, "w") as f:
            simtel_config_writer._write_simtools_parameters(f)
        # Should still write basic parameters without build_opts
        assert file_has_text(test_file, "% Simtools parameters")
        assert file_has_text(test_file, "metaparam global set simtools_version")


def test_write_single_mirror_list_file(simtel_config_writer, tmp_test_directory, file_has_text):
    mirror_number = 1
    mirrors = mock.Mock()
    mirrors.get_single_mirror_parameters.return_value = (
        None,
        None,
        1.2 * u.m,
        16.0 * u.m,
        0,
    )
    single_mirror_list_file = tmp_test_directory / "single_mirror_list.dat"

    simtel_config_writer.write_single_mirror_list_file(
        mirror_number, mirrors, single_mirror_list_file, set_focal_length_to_zero=False
    )

    assert single_mirror_list_file.exists()
    assert file_has_text(single_mirror_list_file, "0. 0. 120.0 1600.0 0 0.")

    simtel_config_writer.write_single_mirror_list_file(
        mirror_number, mirrors, single_mirror_list_file, set_focal_length_to_zero=True
    )

    assert file_has_text(single_mirror_list_file, "0. 0. 120.0 0 0 0.")


@pytest.mark.parametrize(
    ("shape", "width", "exp", "expected_sigtime", "expected_twidth", "expected_exptime"),
    [
        ("gauss", 2.5, 0.0, 2.5, 0.0, 0.0),
        ("tophat", 5.0, 0.0, 0.0, 5.0, 0.0),
        ("exponential", 0.0, 3.2, 0.0, 0.0, 3.2),
        ("gauss-exponential", 3.2, 3.2, 3.2, 0.0, 3.2),
        ("GAUSS", 1.5, 0.0, 1.5, 0.0, 0.0),  # case insensitive
    ],
)
def test_get_flasher_parameters_for_sim_telarray_valid_shapes(
    simtel_config_writer, shape, width, exp, expected_sigtime, expected_twidth, expected_exptime
):
    parameters = {
        "flasher_pulse_shape": {"value": [shape, width, exp]},
    }
    result = simtel_config_writer._get_flasher_parameters_for_sim_telarray(parameters, {})

    assert result["laser_pulse_sigtime"] == pytest.approx(expected_sigtime)
    assert result["laser_pulse_twidth"] == pytest.approx(expected_twidth)
    assert result["laser_pulse_exptime"] == pytest.approx(expected_exptime)


@pytest.mark.parametrize("shape", ["unknown_shape", ""])
def test_get_flasher_parameters_for_sim_telarray_invalid_shapes(
    simtel_config_writer, caplog, shape
):
    parameters = {
        # Provide unified list but with an invalid shape token
        "flasher_pulse_shape": {"value": [shape, 0.0, 0.0]},
    }

    with caplog.at_level(logging.WARNING):
        result = simtel_config_writer._get_flasher_parameters_for_sim_telarray(parameters, {})

    assert all(
        result[key] == pytest.approx(0.0)
        for key in ["laser_pulse_sigtime", "laser_pulse_twidth", "laser_pulse_exptime"]
    )
    assert f"Flasher pulse shape '{shape}' without width definition" in caplog.text


def test_write_array_triggers_file_mixed_hardstereo(simtel_config_writer, tmp_test_directory):

    # Mock the method to return different values for different telescope types
    def mock_get_array_triggers(array_triggers, tel_type, num_tels):
        if tel_type == "LSTS":
            return create_lsts_mock_trigger()
        return create_mock_array_triggers()

    lines = setup_mixed_trigger_test(
        simtel_config_writer, tmp_test_directory, mock_get_array_triggers
    )

    # Should have comment line, hardstereo line for LSTs, individual and combined lines
    assert "# Array trigger definition" in lines[0]

    # Check that LST line has hardstereo
    lst_line = next(line for line in lines if "hardstereo" in line)
    assert "Trigger 2 of 1, 2 width 120.0 hardstereo" in lst_line

    # Check that there's a combined line with all non-hardstereo telescopes
    combined_line = next(line for line in lines if "3, 4, 5, 6" in line)
    assert "Trigger 2 of 3, 4, 5, 6 width 400.0 minsep 30.0" in combined_line


def test_write_array_triggers_file_different_parameters(simtel_config_writer, tmp_test_directory):

    # Mock the method to return different values for different telescope types
    def mock_get_array_triggers(array_triggers, tel_type, num_tels):
        if tel_type == "LSTS":
            return create_lsts_mock_trigger()
        if tel_type == "MSTS":
            return create_msts_different_params_mock_trigger()
        # SSTS
        return create_mock_array_triggers()

    lines = setup_mixed_trigger_test(
        simtel_config_writer, tmp_test_directory, mock_get_array_triggers
    )

    # Should have comment, hardstereo line for LSTs, individual and combined lines
    assert "# Array trigger definition" in lines[0]

    # Check that LST line has hardstereo
    lst_line = next(line for line in lines if "hardstereo" in line)
    assert "Trigger 2 of 1, 2 width 120.0 hardstereo" in lst_line

    # Check that MSTs have their own line (different parameters)
    mst_line = next(line for line in lines if "3, 4" in line and "width 300.0" in line)
    assert "Trigger 2 of 3, 4 width 300.0 minsep 25.0" in mst_line

    # Check that SSTs have their own line (different parameters)
    sst_line = next(
        line for line in lines if "5, 6" in line and "width 400.0" in line and "30" in line
    )
    assert "Trigger 2 of 5, 6 width 400.0 minsep 30.0" in sst_line

    # Check that there's a combined line with all non-hardstereo telescopes using shortest values
    combined_line = next(line for line in lines if "3, 4, 5, 6" in line)
    assert "Trigger 2 of 3, 4, 5, 6 width 300.0 minsep 25.0" in combined_line  # shortest values


def test_build_trigger_line(simtel_config_writer):
    trigger_dict = {"multiplicity": {"value": 2}}
    tel_list = [1, 2, 3]
    width = 120.0 * u.ns
    minsep = 30.0 * u.m

    # Test hardstereo line
    line = simtel_config_writer._build_trigger_line(
        trigger_dict, tel_list, width, minsep, hardstereo=True
    )
    expected = "Trigger 2 of 1, 2, 3 width 120.0 ns hardstereo minsep 30.0 m"
    assert line == expected

    # Test non-hardstereo line
    line = simtel_config_writer._build_trigger_line(
        trigger_dict, tel_list, width, minsep, hardstereo=False
    )
    expected = "Trigger 2 of 1, 2, 3 width 120.0 ns minsep 30.0 m"
    assert line == expected

    # Test line without minsep
    line = simtel_config_writer._build_trigger_line(
        trigger_dict, tel_list, width, None, hardstereo=True
    )
    expected = "Trigger 2 of 1, 2, 3 width 120.0 ns hardstereo"
    assert line == expected


def test_get_minimum_minsep(simtel_config_writer):
    # Test with minsep values - use plain numbers as keys like the actual implementation
    non_hardstereo_groups = {
        (300.0, 25.0): [3, 4],
        (400.0, 30.0): [5, 6],
    }

    min_minsep = simtel_config_writer._get_minimum_minsep(non_hardstereo_groups)
    assert min_minsep == pytest.approx(25.0)  # Test with None values
    non_hardstereo_groups = {
        (300.0, None): [3, 4],
        (400.0, None): [5, 6],
    }

    min_minsep = simtel_config_writer._get_minimum_minsep(non_hardstereo_groups)
    assert min_minsep is None

    # Test with mixed values
    non_hardstereo_groups = {
        (300.0, 25.0): [3, 4],
        (400.0, None): [5, 6],
    }

    min_minsep = simtel_config_writer._get_minimum_minsep(non_hardstereo_groups)
    assert min_minsep == pytest.approx(25.0)


def _read_pulse_table(path: Path):
    """Helper to read pulse table two-column file into arrays."""
    with open(path, encoding="utf-8") as fh:
        lines = [ln.strip() for ln in fh.readlines() if not ln.startswith("#")]
    t_vals = []
    y_vals = []
    for ln in lines:
        if not ln:
            continue
        t_str, y_str = ln.split()
        t_vals.append(float(t_str))
        y_vals.append(float(y_str))
    return np.array(t_vals), np.array(y_vals)


def test_write_light_pulse_table_gauss_exp_conv_creates_normalized_file(tmp_test_directory):
    out = Path(tmp_test_directory) / "pulse_shape_test.dat"
    result = simtel_table_writer.write_light_pulse_table_gauss_exp_conv(
        file_path=out,
        width_ns=2.5,
        exp_decay_ns=5.0,
        dt_ns=0.2,
        fadc_sum_bins=40,
        time_margin_ns=5.0,
    )
    assert isinstance(result, Path)
    assert out.exists()
    t, y = _read_pulse_table(out)
    assert y.size == t.size
    assert y.size > 2
    assert np.isclose(y.max(), 1.0, atol=1e-2)
    # With centering enabled in the writer, the time axis is shifted so the peak is at ~0.
    margin = 5.0
    bins = 40.0
    dt = 0.2
    assert np.allclose(np.diff(t), dt, atol=1e-9)
    # Peak near t=0
    i_max = int(np.argmax(y))
    assert abs(t[i_max]) <= dt
    # Coverage at least bins + 2*margin (symmetric window after centering)
    expected_span = bins + 2 * margin
    assert (t[-1] - t[0]) >= (expected_span - dt)


def test_write_light_pulse_table_gauss_exp_conv_missing_params_raises(tmp_test_directory):
    out = Path(tmp_test_directory) / "pulse_missing_params.dat"
    with pytest.raises(ValueError, match="width_ns"):
        simtel_table_writer.write_light_pulse_table_gauss_exp_conv(
            file_path=out,
            width_ns=None,
            exp_decay_ns=5.0,
            fadc_sum_bins=10,
        )


def test_write_trigger_lines_no_hardstereo_no_minsep(simtel_config_writer):

    hardstereo_lines = []
    non_hardstereo_groups = {
        (300.0, None): [1, 2],
        (400.0, None): [3, 4],
    }
    all_non_hardstereo_tels = [1, 2, 3, 4]
    multiplicity = 3

    output = io.StringIO()
    simtel_config_writer._write_trigger_lines(
        output, hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity
    )

    content = output.getvalue()
    lines = content.strip().split("\n")

    # Should have individual lines + combined line (no minsep in combined line)
    assert len(lines) == 3
    assert TRIGGER_1_2_WIDTH_300_LINE in lines
    assert TRIGGER_3_4_WIDTH_400_LINE in lines
    assert TRIGGER_1234_WIDTH_300_LINE in lines  # Min width, no minsep


def test_write_simtools_parameters_attribute_error(simtel_config_writer, tmp_test_directory):
    # Create a mock file to write to
    test_file = tmp_test_directory / "test_simtools_params.txt"
    # Patch settings.config.corsika_exe to None to trigger AttributeError
    with mock.patch("simtools.simtel.simtel_config_writer.settings") as mock_settings:
        mock_settings.config.corsika_exe = None
        with pytest.raises(
            AttributeError, match=r"CORSIKA executable path is not set in settings."
        ):
            with open(test_file, "w") as f:
                simtel_config_writer._write_simtools_parameters(f)


def test_write_angular_distribution_table_lambertian(tmp_test_directory):
    file_path = Path(tmp_test_directory) / "lambertian.dat"

    # Test default parameters
    simtel_table_writer.write_angular_distribution_table_lambertian(
        file_path=file_path,
        max_angle_deg=90.0,
        n_samples=100,
    )

    assert file_path.exists()
    content = file_path.read_text().splitlines()

    # Check header
    assert content[0].startswith("# angle[deg] relative_intensity")

    # Check number of lines (header + n_samples)
    assert len(content) == 101

    # Check first and last values
    first_line = content[1].split()
    last_line = content[-1].split()

    # Angle 0 -> cos(0) = 1
    assert float(first_line[0]) == pytest.approx(0.0)
    assert float(first_line[1]) == pytest.approx(1.0)

    # Angle 90 -> cos(90) = 0
    assert float(last_line[0]) == pytest.approx(90.0)
    assert float(last_line[1]) == pytest.approx(0.0, abs=1e-7)

    # Test with max_angle > 90 (should be clipped to 0)
    file_path_large = Path(tmp_test_directory) / "lambertian_large.dat"
    simtel_table_writer.write_angular_distribution_table_lambertian(
        file_path=file_path_large,
        max_angle_deg=180.0,
        n_samples=181,
    )

    content_large = file_path_large.read_text().splitlines()
    # Angle 180 -> cos(180) = -1 -> clipped to 0
    last_line_large = content_large[-1].split()
    assert float(last_line_large[0]) == pytest.approx(180.0)
    assert float(last_line_large[1]) == pytest.approx(0.0)
