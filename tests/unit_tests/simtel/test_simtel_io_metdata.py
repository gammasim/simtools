#!/usr/bin/python3

from unittest.mock import MagicMock

import pytest

from simtools.simtel.simtel_io_metadata import (
    get_corsika_run_number,
    get_sim_telarray_telescope_id,
    read_sim_telarray_metadata,
)


@pytest.fixture
def test_sim_telarray_file():
    return (
        "tests/resources/"
        "run000010_gamma_za20deg_azm000deg_North_test_layout_6.0.0"
        "_test-production-North.simtel.zst"
    )


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

    assert (float)(global_meta["latitude"]) > 0.0
    assert global_meta["array_config_name"] == "test_layout"


def test_get_sim_telarray_telescope_id(test_sim_telarray_file):
    assert get_sim_telarray_telescope_id("LSTN-01", test_sim_telarray_file) == 1
    assert get_sim_telarray_telescope_id("MSTN-01", test_sim_telarray_file) == 5
    assert get_sim_telarray_telescope_id("MSTS-01", test_sim_telarray_file) is None


def test_get_corsika_run_number_with_run_header():
    assert (
        get_corsika_run_number(
            "tests/resources/run000010_gamma_za20deg_azm000deg_North_test_layout_"
            "6.0.0_test-production-North.simtel.zst"
        )
        == 10
    )
    # The following file was actually created with the LightEmission package,
    # but it should still return the run number correctly.
    assert get_corsika_run_number("tests/resources/xyzls_layout.simtel.gz") == 1


def test_get_corsika_run_number_without_run_header_real_file():
    assert (
        get_corsika_run_number(
            "tests/resources/gamma_20deg_0deg_run1___cta-prod5-lapalma_"
            "desert-2158m-LaPalma-dark.hdata.zst"
        )
        is None
    )


def test_get_corsika_run_number_without_run_header_mocking():
    mock_eventio_file = MagicMock()
    mock_eventio_file.__enter__.return_value = []

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr("simtools.simtel.simtel_io_metadata.EventIOFile", lambda x: mock_eventio_file)
        run_number = get_corsika_run_number("test_file.simtel.gz")
        assert run_number is None
