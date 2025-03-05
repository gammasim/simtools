#!/usr/bin/python3

import logging
from pathlib import Path

import numpy as np
import pytest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter

logger = logging.getLogger()


@pytest.fixture
def simtel_config_writer(model_version):
    return SimtelConfigWriter(
        site="North",
        model_version=model_version,
        label="test-simtel-config-writer",
        telescope_model_name="test_telecope",
    )


def test_write_array_config_file(
    simtel_config_writer, telescope_model_lst, io_handler, file_has_text, site_model_north
):
    file = io_handler.get_output_file(file_name="simtel-config-writer_array.txt")
    telescope_model = {
        "LSTN-01": telescope_model_lst,
        "LSTN-02": telescope_model_lst,
        "LSTN-03": telescope_model_lst,
        "LSTN-04": telescope_model_lst,
    }
    simtel_config_writer.write_array_config_file(
        config_file_path=file,
        telescope_model=telescope_model,
        site_model=site_model_north,
    )
    assert file_has_text(file, "TELESCOPE == 1")

    # simtel configuration files need to end with two new lines
    with open(file) as f:
        lines = f.readlines()
        assert lines[-2].endswith("\n")
        assert lines[-1] == "\n"


def test_write_tel_config_file(simtel_config_writer, io_handler, file_has_text):
    file = io_handler.get_output_file(file_name="simtel-config-writer_telescope.txt")
    simtel_config_writer.write_telescope_config_file(
        config_file_path=file, parameters={"num_gains": 1}
    )
    assert file_has_text(file, "num_gains = 1")

    simtel_config_writer.write_telescope_config_file(
        config_file_path=file, parameters={"array_triggers": "array_triggers.dat"}
    )
    assert not file_has_text(file, "array_triggers = array_triggers.dat")


def test_get_simtel_metadata(simtel_config_writer):
    _tel = simtel_config_writer._get_simtel_metadata("telescope")
    assert len(_tel) == 8
    assert _tel["camera_config_name"] == simtel_config_writer._telescope_model_name
    assert _tel["optics_config_name"] == simtel_config_writer._telescope_model_name

    _site = simtel_config_writer._get_simtel_metadata("site")
    assert len(_site) == 8
    assert _site["site_config_name"] == simtel_config_writer._site
    assert _site["array_config_name"] == simtel_config_writer._layout_name

    with pytest.raises(ValueError, match=r"^Unknown metadata type"):
        simtel_config_writer._get_simtel_metadata("unknown")


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

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "LSTN")
    assert result is not None
    assert result["name"] == "LSTN_array"
    assert result["multiplicity"]["value"] == 2
    assert result["width"]["value"] == 10
    assert result["width"]["unit"] == "ns"

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "MSTN")
    assert result is None

    result = simtel_config_writer._get_array_triggers_for_telescope_type(array_triggers, "SST")
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
            "name": "LSTN_array",
            "multiplicity": {"value": 2},
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
        assert "Trigger 2 of 1" in content
