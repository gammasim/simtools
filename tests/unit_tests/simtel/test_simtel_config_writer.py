#!/usr/bin/python3

import io
import logging
from pathlib import Path
from unittest import mock

import astropy.units as u
import numpy as np
import pytest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter, sim_telarray_random_seeds

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
    simtel_config_writer.write_array_config_file(
        config_file_path=_file,
        telescope_model=telescope_model,
        site_model=site_model_north,
    )
    assert file_has_text(_file, "TELESCOPE == 1")

    # sim_telarray configuration files need to end with two new lines
    with open(_file) as f:
        lines = f.readlines()
        assert lines[-2].endswith("\n")
        assert lines[-1] == "\n"

    with mock.patch.object(
        SimtelConfigWriter,
        "_write_random_seeds_file",
    ) as write_random_seeds_file_mock:
        simtel_config_writer.write_array_config_file(
            config_file_path=_file,
            telescope_model=telescope_model,
            site_model=site_model_north,
            additional_metadata={
                "seed": 12345,
                "seed_file_name": "sim_telarray_instrument_seeds.txt",
                "random_instrument_instances": 5,
            },
        )
        write_random_seeds_file_mock.assert_called_once()


def test_write_tel_config_file(simtel_config_writer, io_handler, file_has_text):
    _file = io_handler.get_output_file(file_name="simtel-config-writer_telescope.txt")
    simtel_config_writer.write_telescope_config_file(
        config_file_path=_file,
        parameters={
            "num_gains": {
                "parameter": "num_gains",
                "value": 1,
                "unit": None,
                "meta_parameter": False,
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
                "meta_parameter": False,
            }
        },
    )
    assert not file_has_text(_file, "array_triggers = array_triggers.dat")


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


def test_get_sim_telarray_metadata_with_model_parameters(simtel_config_writer):
    model_parameters = {"test_param": {"value": 42, "meta_parameter": True}}

    def mock_get_name(key, software_name, set_meta_parameter):
        if software_name == "sim_telarray":
            if set_meta_parameter:
                return "test_set_param_meta"
            return "test_add_param"
        return None

    with mock.patch(
        "simtools.utils.names.get_simulation_software_name_from_parameter_name",
        side_effect=mock_get_name,
    ):
        tel_meta = simtel_config_writer._get_sim_telarray_metadata(
            "telescope", model_parameters, "test_telescope"
        )
        assert "metaparam telescope add test_add_param" in tel_meta
        assert "metaparam telescope set test_set_param_meta=42" in tel_meta

        site_meta = simtel_config_writer._get_sim_telarray_metadata("site", model_parameters, None)
        assert "metaparam global add test_add_param" in site_meta
        assert "metaparam global set test_set_param_meta=42" in site_meta


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


def test_write_dummy_telescope_configuration_file(
    simtel_config_writer, io_handler, tmp_test_directory, file_has_text
):
    config_file_path = Path(tmp_test_directory) / "dummy_config.cfg"
    telescope_name = "DummyTel"
    parameters = {
        "camera_config_file": {
            "parameter": "camera_config_file",
            "value": "camera.dat",
            "unit": None,
            "meta_parameter": False,
        },
        "discriminator_pulse_shape": {
            "parameter": "discriminator_pulse_shape",
            "value": "pulse.dat",
            "unit": None,
            "meta_parameter": False,
        },
        "mirror_list": {
            "parameter": "mirror_list",
            "value": "mirror.dat",
            "unit": None,
            "meta_parameter": False,
        },
        "camera_pixels": {
            "parameter": "camera_pixels",
            "value": 1024,
            "unit": None,
            "meta_parameter": False,
        },
        # this needs to be a parameter which is not overwritten by the
        # dummy telescope configuration
        "fadc_pedestal": {
            "parameter": "fadc_pedestal",
            "value": 10.0,
            "unit": None,
            "meta_parameter": False,
        },
    }

    simtel_config_writer.write_dummy_telescope_configuration_file(
        parameters, config_file_path, telescope_name
    )

    assert config_file_path.exists()
    assert file_has_text(config_file_path, "camera_config_file = DummyTel_single_pixel_camera.dat")
    assert file_has_text(config_file_path, "mirror_list = DummyTel_single_12m_mirror.dat")
    assert file_has_text(config_file_path, "camera_pixels = 1")

    mirror_file = Path(tmp_test_directory) / f"{telescope_name}_single_12m_mirror.dat"
    assert mirror_file.exists()
    assert file_has_text(mirror_file, "0 0 1200 0.0 0")

    camera_file = Path(tmp_test_directory) / f"{telescope_name}_single_pixel_camera.dat"
    assert camera_file.exists()
    assert file_has_text(camera_file, f'"{telescope_name}_funnels.dat"')

    # ensure that non-dummy telescopes are not lost
    assert file_has_text(config_file_path, "fadc_pedestal = 10.0")

    # ensure that the dummy configuration is not adding extra parameters
    assert not file_has_text(config_file_path, "trigger_pixels = 1")


def test_write_random_seeds_file(simtel_config_writer, tmp_test_directory):
    seed_file_name = "sim_telarray_instrument_seeds.txt"
    config_file_directory = Path(tmp_test_directory) / "model"
    config_file_directory.mkdir(exist_ok=True)
    sim_telarray_seeds = {
        "seed": 12345,
        "seed_file_name": seed_file_name,
        "random_instrument_instances": 5,
    }

    simtel_config_writer._write_random_seeds_file(sim_telarray_seeds, config_file_directory)

    seed_file_path = config_file_directory / sim_telarray_seeds["seed_file_name"]
    assert seed_file_path.exists()

    with open(seed_file_path, encoding="utf-8") as file:
        lines = file.readlines()
        assert len(lines) == sim_telarray_seeds["random_instrument_instances"] + 1
        for line in lines:
            if line[0] == "#":
                continue
            assert line.strip().isdigit()

    sim_telarray_seeds = {
        "seed": 12345,
        "seed_file_name": seed_file_name,
        "random_instrument_instances": 1025,
    }
    with pytest.raises(
        ValueError, match="Number of random instances of instrument must be less than 1024"
    ):
        simtel_config_writer._write_random_seeds_file(sim_telarray_seeds, config_file_directory)


def test_sim_telarray_random_seeds():
    seed = 12345
    number = 5
    seeds = sim_telarray_random_seeds(seed, number)
    assert len(seeds) == number
    assert all(isinstance(s, np.int32) for s in seeds)
    assert all(s >= 1 for s in seeds)  # sim_telarray seeds needs to be >0

    seed = 54321
    number = 10
    seeds = sim_telarray_random_seeds(seed, number)
    assert len(seeds) == number
    assert all(isinstance(s, np.int32) for s in seeds)

    # Test with zero number of seeds
    seed = 12345
    number = 0
    seeds = sim_telarray_random_seeds(seed, number)
    assert len(seeds) == number


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

    # Test with simtel_path and build_opts.yml
    build_opts_file = tmp_test_directory / "build_opts.yml"
    with open(build_opts_file, "w") as f:
        f.write("build_date: 2023-01-01\nversion: 1.0.0")

    simtel_config_writer._simtel_path = tmp_test_directory
    with open(test_file, "w") as f:
        simtel_config_writer._write_simtools_parameters(f)

    # Check build_opts parameters are included
    assert file_has_text(test_file, "metaparam global set simtools_build_date = 2023-01-01")
    assert file_has_text(test_file, "metaparam global set simtools_version = 1.0.0")

    # Test with invalid simtel_path
    simtel_config_writer._simtel_path = tmp_test_directory / "nonexistent"
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
    """Test _get_flasher_parameters_for_sim_telarray with valid pulse shapes (unified list)."""
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
    """Test _get_flasher_parameters_for_sim_telarray with invalid shapes - covers warning case."""
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


def test_get_flasher_parameters_for_sim_telarray_missing_params(simtel_config_writer, caplog):
    """Test _get_flasher_parameters_for_sim_telarray with missing parameters and existing ones."""
    simtel_par = {"existing_param": "existing_value"}

    result = simtel_config_writer._get_flasher_parameters_for_sim_telarray({}, simtel_par)

    assert result == simtel_par

    simtel_par = {"existing_param": "existing_value"}
    # Provide unified list with invalid shape token; expect warning and zeroed flasher params
    parameters = {"flasher_pulse_shape": {"value": ["bad_shape", 0.0, 0.0]}}

    with caplog.at_level(logging.WARNING):
        result = simtel_config_writer._get_flasher_parameters_for_sim_telarray(
            parameters, simtel_par
        )

    assert "Flasher pulse shape 'bad_shape' without width definition" in caplog.text

    # All flasher parameters should be 0.0, existing parameter preserved
    assert all(
        result[key] == pytest.approx(0.0)
        for key in ["laser_pulse_sigtime", "laser_pulse_twidth", "laser_pulse_exptime"]
    )
    assert result["existing_param"] == "existing_value"


def test_write_array_triggers_file_mixed_hardstereo(simtel_config_writer, tmp_test_directory):
    """Test array triggers file generation with mixed hardstereo settings."""

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
    """Test array triggers file generation with different width and min_separation values."""

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


def test_group_telescopes_by_type(simtel_config_writer):
    """Test the _group_telescopes_by_type helper method."""
    telescope_model = {
        "LSTS-01": mock.Mock(),
        "LSTS-02": mock.Mock(),
        "MSTS-01": mock.Mock(),
        "SSTS-01": mock.Mock(),
    }

    result = simtel_config_writer._group_telescopes_by_type(telescope_model)

    expected = {
        "LSTS": [1, 2],
        "MSTS": [3],
        "SSTS": [4],
    }
    assert result == expected


def test_extract_trigger_parameters(simtel_config_writer):
    """Test the _extract_trigger_parameters helper method."""
    trigger_dict = {
        "width": {"value": 120.0, "unit": "ns"},
        "min_separation": {"value": 30.0, "unit": "m"},
    }

    width, minsep = simtel_config_writer._extract_trigger_parameters(trigger_dict)

    # The .to() method returns the numerical value, not a Quantity object
    assert width == pytest.approx(120.0)  # nanoseconds
    assert minsep == pytest.approx(30.0)  # meters


def test_extract_trigger_parameters_no_minsep(simtel_config_writer):
    """Test _extract_trigger_parameters when min_separation is None."""
    trigger_dict = {
        "width": {"value": 120.0, "unit": "ns"},
        "min_separation": {"value": None, "unit": None},
    }

    width, minsep = simtel_config_writer._extract_trigger_parameters(trigger_dict)

    assert width == pytest.approx(120.0)  # nanoseconds
    assert minsep is None


def test_build_trigger_line(simtel_config_writer):
    """Test the _build_trigger_line helper method."""
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
    """Test the _get_minimum_minsep helper method."""
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


def test_process_telescope_triggers(simtel_config_writer):
    """Test the _process_telescope_triggers helper method."""
    array_triggers = [
        create_lsts_trigger(2, True),
        create_msts_trigger(2, False),
    ]

    trigger_per_telescope_type = {
        "LSTS": [1, 2],
        "MSTS": [3, 4],
    }

    hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity = (
        simtel_config_writer._process_telescope_triggers(array_triggers, trigger_per_telescope_type)
    )

    # Check hardstereo lines
    assert len(hardstereo_lines) == 1
    assert LSTS_HARDSTEREO_LINE in hardstereo_lines[0]

    # Check non-hardstereo groups - keys are plain numbers, not Quantity objects
    assert len(non_hardstereo_groups) == 1
    key = (300.0, 25.0)  # Plain numbers as used in the actual implementation
    assert key in non_hardstereo_groups
    assert non_hardstereo_groups[key] == [3, 4]

    # Check all non-hardstereo telescopes
    assert all_non_hardstereo_tels == [3, 4]

    # Check multiplicity
    assert multiplicity == 2


def test_extract_trigger_parameters_unit_conversion(simtel_config_writer):
    """Test _extract_trigger_parameters with unit conversion."""
    trigger_dict = {
        "width": {"value": 0.12, "unit": "us"},  # microseconds -> nanoseconds
        "min_separation": {"value": 0.03, "unit": "km"},  # kilometers -> meters
    }

    width, minsep = simtel_config_writer._extract_trigger_parameters(trigger_dict)

    assert width == pytest.approx(120.0)  # 0.12 us = 120 ns
    assert minsep == pytest.approx(30.0)  # 0.03 km = 30 m


def test_build_trigger_line_edge_cases(simtel_config_writer):
    """Test _build_trigger_line with edge cases."""
    trigger_dict = {"multiplicity": {"value": 1}}

    # Test single telescope
    line = simtel_config_writer._build_trigger_line(
        trigger_dict, [5], 100.0, None, hardstereo=False
    )
    expected = "Trigger 1 of 5 width 100.0"
    assert line == expected

    # Test large telescope list
    tel_list = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    line = simtel_config_writer._build_trigger_line(
        trigger_dict, tel_list, 200.0, 50.0, hardstereo=False
    )
    expected = "Trigger 1 of 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 width 200.0 minsep 50.0"
    assert line == expected


def test_get_minimum_minsep_edge_cases(simtel_config_writer):
    """Test _get_minimum_minsep with edge cases."""
    # Test empty dictionary
    min_minsep = simtel_config_writer._get_minimum_minsep({})
    assert min_minsep is None

    # Test single group
    non_hardstereo_groups = {(300.0, 25.0): [3, 4]}
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


def test_write_lightpulse_table_gauss_expconv_creates_normalized_file(tmp_test_directory):
    """Writer should create a pulse table with peak amplitude ~1 and expected window."""
    out = Path(tmp_test_directory) / "pulse_shape_test.dat"
    result = SimtelConfigWriter.write_lightpulse_table_gauss_expconv(
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


def test_write_lightpulse_table_gauss_expconv_time_spacing(tmp_test_directory):
    """Time column should be spaced by dt_ns consistently."""
    out = Path(tmp_test_directory) / "pulse_shape_spacing.dat"
    dt = 0.5
    SimtelConfigWriter.write_lightpulse_table_gauss_expconv(
        file_path=out,
        width_ns=3.0,
        exp_decay_ns=6.0,
        dt_ns=dt,
        fadc_sum_bins=20,
        time_margin_ns=5.0,
    )
    t, _ = _read_pulse_table(out)
    assert np.allclose(np.diff(t), dt, atol=1e-9)


def test_write_lightpulse_table_gauss_expconv_missing_params_raises(tmp_test_directory):
    """Missing width/decay should raise ValueError."""
    out = Path(tmp_test_directory) / "pulse_missing_params.dat"
    with pytest.raises(ValueError, match="width_ns"):
        SimtelConfigWriter.write_lightpulse_table_gauss_expconv(
            file_path=out,
            width_ns=None,
            exp_decay_ns=5.0,
            fadc_sum_bins=10,
        )


def test_write_ascii_pulse_table_writes_header_and_values(tmp_test_directory):
    """_write_ascii_pulse_table should create a two-column file with a header and values."""
    out = Path(tmp_test_directory) / "pulse_ascii.dat"
    t = np.array([0.0, 0.1, 0.2])
    y = np.array([0.25, 1.0, 0.5])

    result = SimtelConfigWriter._write_ascii_pulse_table(out, t, y)
    assert result == out
    assert out.exists()

    # First line is a header starting with '#'
    with open(out, encoding="utf-8") as fh:
        first_line = fh.readline()
    assert first_line.startswith("#")

    # Data lines parse back to the same arrays
    t_read, y_read = _read_pulse_table(out)
    assert np.allclose(t_read, t)
    assert np.allclose(y_read, y)


def test_process_telescope_triggers_multiple_hardstereo(simtel_config_writer):
    """Test _process_telescope_triggers with multiple hardstereo telescope types."""
    array_triggers = create_mixed_trigger_scenario()
    trigger_per_telescope_type = create_standard_telescope_mapping()

    hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity = (
        simtel_config_writer._process_telescope_triggers(array_triggers, trigger_per_telescope_type)
    )

    # Check multiple hardstereo lines
    assert len(hardstereo_lines) == 2
    assert any(LSTS_HARDSTEREO_LINE in line for line in hardstereo_lines)
    assert any(MSTS_HARDSTEREO_LINE in line for line in hardstereo_lines)

    # Check single non-hardstereo group
    assert len(non_hardstereo_groups) == 1
    assert all_non_hardstereo_tels == [5, 6]
    assert multiplicity == 2


def test_process_telescope_triggers_all_hardstereo(simtel_config_writer):
    """Test _process_telescope_triggers when all telescopes are hardstereo."""
    array_triggers = create_all_hardstereo_scenario()
    trigger_per_telescope_type = {
        "LSTS": [1, 2],
        "MSTS": [3, 4],
    }

    hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity = (
        simtel_config_writer._process_telescope_triggers(array_triggers, trigger_per_telescope_type)
    )

    # Check all are hardstereo
    assert len(hardstereo_lines) == 2
    assert len(non_hardstereo_groups) == 0
    assert len(all_non_hardstereo_tels) == 0
    assert multiplicity == 2


def test_process_telescope_triggers_all_non_hardstereo_same_params(simtel_config_writer):
    """Test _process_telescope_triggers when all non-hardstereo have same parameters."""
    array_triggers = create_all_non_hardstereo_same_params_scenario()
    trigger_per_telescope_type = {
        "MSTS": [1, 2],
        "SSTS": [3, 4],
    }

    hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, _ = (
        simtel_config_writer._process_telescope_triggers(array_triggers, trigger_per_telescope_type)
    )

    # Check no hardstereo lines
    assert len(hardstereo_lines) == 0

    # Check single group with all telescopes (same parameters)
    assert len(non_hardstereo_groups) == 1
    key = (300.0, 25.0)
    assert key in non_hardstereo_groups
    assert non_hardstereo_groups[key] == [1, 2, 3, 4]
    assert all_non_hardstereo_tels == [1, 2, 3, 4]


def test_write_trigger_lines_comprehensive(simtel_config_writer, tmp_test_directory):
    """Test _write_trigger_lines with comprehensive scenarios."""

    # Test scenario with hardstereo, multiple non-hardstereo groups, and combined line
    hardstereo_lines = [
        LSTS_HARDSTEREO_LINE,
        MSTS_HARDSTEREO_LINE,
    ]

    non_hardstereo_groups = {
        (300.0, 25.0): [5, 6],
        (400.0, 30.0): [7, 8],
    }

    all_non_hardstereo_tels = [5, 6, 7, 8]
    multiplicity = 2

    # Use StringIO to capture file output
    output = io.StringIO()
    simtel_config_writer._write_trigger_lines(
        output, hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity
    )

    content = output.getvalue()
    lines = content.strip().split("\n")

    # Should have hardstereo lines + individual group lines + combined line
    expected_line_count = len(hardstereo_lines) + len(non_hardstereo_groups) + 1
    assert len(lines) == expected_line_count

    # Check hardstereo lines are written first
    assert lines[0] == LSTS_HARDSTEREO_LINE
    assert lines[1] == MSTS_HARDSTEREO_LINE

    # Check individual group lines
    assert SSTS_NON_HARDSTEREO_LINE in lines
    assert "Trigger 2 of 7, 8 width 400.0 minsep 30.0" in lines

    # Check combined line uses minimum values
    assert COMBINED_NON_HARDSTEREO_LINE in lines


def test_write_trigger_lines_single_group_no_individual_lines(simtel_config_writer):
    """Test _write_trigger_lines when there's only one non-hardstereo group."""

    hardstereo_lines = [LSTS_HARDSTEREO_LINE]
    non_hardstereo_groups = {(300.0, 25.0): [3, 4, 5, 6]}  # Single group
    all_non_hardstereo_tels = [3, 4, 5, 6]
    multiplicity = 2

    output = io.StringIO()
    simtel_config_writer._write_trigger_lines(
        output, hardstereo_lines, non_hardstereo_groups, all_non_hardstereo_tels, multiplicity
    )

    content = output.getvalue()
    lines = content.strip().split("\n")

    # Should have hardstereo line + combined line only (no individual group lines)
    assert len(lines) == 2
    assert lines[0] == LSTS_HARDSTEREO_LINE
    assert lines[1] == COMBINED_ALL_NON_HARDSTEREO_LINE


def test_write_trigger_lines_no_hardstereo_no_minsep(simtel_config_writer):
    """Test _write_trigger_lines with no hardstereo and no min_separation."""

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
