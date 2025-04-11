#!/usr/bin/python3

import pytest

from simtools.simtel.simtel_io_metadata import (
    get_sim_telarray_telescope_id,
    read_sim_telarray_metadata,
)


@pytest.fixture
def test_sim_telarray_file():
    return "tests/resources/run000010_gamma_za20deg_azm000deg_North_test_layout_6.0.0_test-production-North.simtel.zst"


def test_read_sim_telarray_metadata(test_sim_telarray_file):
    global_meta, telescope_meta = read_sim_telarray_metadata(test_sim_telarray_file)
    assert global_meta is not None
    assert len(telescope_meta) > 0
    assert isinstance(telescope_meta, dict)
    assert all(isinstance(k, int) for k in telescope_meta.keys())
    assert all(isinstance(v, dict) for v in telescope_meta.values())

    for key in global_meta.keys():
        assert key[0] != "*"
        assert key.strip() == key
        assert key.lower() == key


def test_get_sim_telarray_telescope_id(test_sim_telarray_file):
    assert get_sim_telarray_telescope_id("LSTN-01", test_sim_telarray_file) == 1
    assert get_sim_telarray_telescope_id("MSTN-01", test_sim_telarray_file) == 5
    assert get_sim_telarray_telescope_id("MSTS-01", test_sim_telarray_file) is None
