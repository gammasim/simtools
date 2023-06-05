#!/usr/bin/python3

import pytest

import simtools.util.general as gen
from simtools.corsika.corsika_output import CorsikaOutput

test_file_name = "tel_output_10GeV-2-gamma-20deg-CTAO-South.dat"
# test_file_name = "tel_output.dat"


@pytest.fixture
def corsika_output_file(io_handler):
    corsika_output = gen.find_file(
        test_file_name,
        io_handler.get_output_directory(dir_type="corsika_output", test=True),
    )
    return corsika_output


@pytest.fixture
def corsika_output_instance(db, io_handler, corsika_output_file):
    # db.export_file_db(
    #    db_name="test-data",
    #    dest=io_handler.get_output_directory(dir_type="corsika_output", test=True),
    #    file_name=test_file_name,
    # )
    # return CorsikaOutput(corsika_output_file)
    return CorsikaOutput(test_file_name)


def test_file_exists(corsika_output_file):
    assert corsika_output_file.exists()


def test_init(corsika_output_instance):
    assert corsika_output_instance.input_file.name == test_file_name
    with pytest.raises(FileNotFoundError):
        CorsikaOutput("wrong_file_name")
    assert len(corsika_output_instance.event_information) > 15
    assert "zenith" in corsika_output_instance.event_information
    assert len(corsika_output_instance.header) > 10
    assert corsika_output_instance.version == 7.741
