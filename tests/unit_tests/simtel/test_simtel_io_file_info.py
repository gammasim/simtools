#!/usr/bin/python3

from simtools.simtel.simtel_io_file_info import get_corsika_run_header, get_corsika_run_number


def test_get_corsika_run_number_with_run_header(sim_telarray_file_gamma):
    assert get_corsika_run_number(sim_telarray_file_gamma) == 10

    # The following file was actually created with the LightEmission package,
    # but it should still return the run number correctly.
    assert get_corsika_run_number("tests/resources/xyzls_layout.simtel.gz") == 1


def test_get_corsika_run_number_without_run_header_real_file(sim_telarray_hdata_file_gamma):
    assert get_corsika_run_number(sim_telarray_hdata_file_gamma) is None


def test_get_corsika_run_header(sim_telarray_file_gamma):
    run_header = get_corsika_run_header(sim_telarray_file_gamma)
    assert isinstance(run_header, dict)
    assert run_header["run"] == 10
    assert run_header["primary_id"] == 0
