#!/usr/bin/python3

import logging

import pytest

from simtools.simtel.simtel_config_writer import SimtelConfigWriter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture()
def simtel_config_writer():
    simtel_config_writer = SimtelConfigWriter(
        site="North",
        model_version="Released",
        label="test-simtel-config-writer",
        telescope_model_name="test_telecope",
    )
    return simtel_config_writer


# @pytest.mark.skip(reason="TODO :test_write_array_config_file - KeyError: 'Released'")
def test_write_array_config_file(
    simtel_config_writer, telescope_model_lst, io_handler, file_has_text, site_model_north
):
    file = io_handler.get_output_file(file_name="simtel-config-writer_array.txt", dir_type="test")
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


def test_write_tel_config_file(simtel_config_writer, io_handler, file_has_text):
    file = io_handler.get_output_file(
        file_name="simtel-config-writer_telescope.txt", dir_type="test"
    )
    simtel_config_writer.write_telescope_config_file(
        config_file_path=file, parameters={"num_gains": 1}
    )
    assert file_has_text(file, "num_gains = 1")


def test_get_simtel_metadata(simtel_config_writer):

    _tel = simtel_config_writer._get_simtel_metadata("telescope")
    assert len(_tel) == 8
    assert _tel["camera_config_name"] == simtel_config_writer._telescope_model_name
    assert _tel["optics_config_name"] == simtel_config_writer._telescope_model_name

    _site = simtel_config_writer._get_simtel_metadata("site")
    assert len(_site) == 8
    assert _site["site_config_name"] == simtel_config_writer._site
    assert _site["array_config_name"] == simtel_config_writer._layout_name

    with pytest.raises(ValueError):
        simtel_config_writer._get_simtel_metadata("unknown")
