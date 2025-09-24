#!/usr/bin/python3

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
        assert "minsep" in content
        assert "width" in content


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
    assert f"camera_config_name = {simtel_config_writer._telescope_model_name}" in _tel
    assert f"optics_config_name = {simtel_config_writer._telescope_model_name}" in _tel

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


def test_write_simtools_parameters(simtel_config_writer, tmp_path, file_has_text):
    # Create a mock file to write to
    test_file = tmp_path / "test_simtools_params.txt"
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
    build_opts_file = tmp_path / "build_opts.yml"
    with open(build_opts_file, "w") as f:
        f.write("build_date: 2023-01-01\nversion: 1.0.0")

    simtel_config_writer._simtel_path = tmp_path
    with open(test_file, "w") as f:
        simtel_config_writer._write_simtools_parameters(f)

    # Check build_opts parameters are included
    assert file_has_text(test_file, "metaparam global set simtools_build_date = 2023-01-01")
    assert file_has_text(test_file, "metaparam global set simtools_version = 1.0.0")

    # Test with invalid simtel_path
    simtel_config_writer._simtel_path = tmp_path / "nonexistent"
    with open(test_file, "w") as f:
        simtel_config_writer._write_simtools_parameters(f)
    # Should still write basic parameters without build_opts
    assert file_has_text(test_file, "% Simtools parameters")
    assert file_has_text(test_file, "metaparam global set simtools_version")


def test_write_single_mirror_list_file(simtel_config_writer, tmp_path, file_has_text):
    mirror_number = 1
    mirrors = mock.Mock()
    mirrors.get_single_mirror_parameters.return_value = (
        None,
        None,
        1.2 * u.m,
        16.0 * u.m,
        0,
    )
    single_mirror_list_file = tmp_path / "single_mirror_list.dat"

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
    ("shape", "width", "expected_sigtime", "expected_twidth", "expected_exptime"),
    [
        ("gauss", 2.5, 2.5, 0.0, 0.0),
        ("tophat", 5.0, 0.0, 5.0, 0.0),
        ("exponential", 3.2, 0.0, 0.0, 3.2),
        ("GAUSS", 1.5, 1.5, 0.0, 0.0),  # case insensitive
    ],
)
def test_get_flasher_parameters_for_sim_telarray_valid_shapes(
    simtel_config_writer, shape, width, expected_sigtime, expected_twidth, expected_exptime
):
    """Test _get_flasher_parameters_for_sim_telarray with valid pulse shapes."""
    parameters = {
        "flasher_pulse_shape": {"value": shape},
        "flasher_pulse_width": {"value": width},
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
        "flasher_pulse_shape": {"value": shape},
        "flasher_pulse_width": {"value": 1.0},
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

    parameters = {
        "flasher_pulse_width": {"value": 0.0},
        "flasher_pulse_shape": {"value": "bad_shape"},
    }

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
