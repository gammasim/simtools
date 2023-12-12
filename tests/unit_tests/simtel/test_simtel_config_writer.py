#!/usr/bin/python3

import logging

import pytest

from simtools.layout.array_layout import ArrayLayout
from simtools.simtel.simtel_config_writer import SimtelConfigWriter

logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


@pytest.fixture
def simtel_config_writer():
    simtel_config_writer = SimtelConfigWriter(
        site="North",
        model_version="Released",
        label="test-simtel-config-writer",
        telescope_model_name="test_telecope",
    )
    return simtel_config_writer


@pytest.fixture
def layout(io_handler, db_config):
    layout = ArrayLayout.from_array_layout_name(
        mongo_db_config=db_config, array_layout_name="South-4LST"
    )
    return layout


# @pytest.mark.skip(reason="TODO :test_write_array_config_file - KeyError: 'Released'")
def test_write_array_config_file(
    simtel_config_writer, layout, telescope_model_lst, io_handler, file_has_text
):
    file = io_handler.get_output_file(file_name="simtel-config-writer_array.txt", dir_type="test")
    simtel_config_writer.write_array_config_file(
        config_file_path=file,
        layout=layout,
        telescope_model=[telescope_model_lst] * 4,
        site_parameters={},
    )
    assert file_has_text(file, "TELESCOPE == 1")


def test_write_tel_config_file(simtel_config_writer, io_handler, file_has_text):
    file = io_handler.get_output_file(
        file_name="simtel-config-writer_telescope.txt", dir_type="test"
    )
    simtel_config_writer.write_telescope_config_file(
        config_file_path=file, parameters={"par": {"Value": 1}}
    )
    assert file_has_text(file, "par = 1")
