#!/usr/bin/python3

import warnings

from simtools.sim_events.file_info import (
    get_combined_eventio_run_header,
    get_corsika_run_and_event_headers,
    get_corsika_run_number,
    get_simulated_events,
)


def test_get_corsika_run_number_with_run_header(sim_telarray_file_gamma):
    assert get_corsika_run_number(sim_telarray_file_gamma) == 10

    # The following file was actually created with the LightEmission package,
    # but it should still return the run number correctly.
    assert get_corsika_run_number("tests/resources/xyzls_layout.simtel.gz") == 1


def test_get_corsika_run_number_without_run_header_real_file(sim_telarray_hdata_file_gamma):
    assert get_corsika_run_number(sim_telarray_hdata_file_gamma) is None


def test_get_combined_eventio_run_header(sim_telarray_file_gamma):
    run_header = get_combined_eventio_run_header(sim_telarray_file_gamma)
    assert isinstance(run_header, dict)
    assert run_header["run"] == 10
    assert run_header["primary_id"] == 0


def test_get_simulated_events(sim_telarray_file_gamma):
    n_showers, n_events = get_simulated_events(sim_telarray_file_gamma)
    assert n_showers == 50
    assert n_events == 1000


def test_get_simulated_events_corsika_iact(corsika_file_gamma):
    # Test CORSIKA IACT file
    n_showers, n_events = get_simulated_events(corsika_file_gamma)
    assert n_showers > 0
    assert n_events == 0  # CORSIKA IACT files don't have MCEvent objects


def test_get_corsika_run_and_event_headers(corsika_file_gamma):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_header, event_header = get_corsika_run_and_event_headers(corsika_file_gamma)
        # Headers are numpy structured arrays
        assert hasattr(run_header, "dtype")  # numpy structured array
        assert hasattr(event_header, "dtype")  # numpy structured array
        assert "run_number" in run_header.dtype.names  # Check field exists
        assert run_header["run_number"] == 7  # Run number from file name


def test_get_combined_eventio_run_header_incomplete(sim_telarray_hdata_file_gamma):
    # This file typically has incomplete header information
    run_header = get_combined_eventio_run_header(sim_telarray_hdata_file_gamma)
    assert run_header is None or isinstance(run_header, dict)
