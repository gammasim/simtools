#!/usr/bin/python3

import warnings

from simtools.sim_events.file_info import (
    get_combined_eventio_run_header,
    get_corsika_run_and_event_headers,
    get_corsika_run_number,
    get_simulated_events,
)

from ..conftest import get_test_data_file


def test_get_corsika_run_number_with_run_header():
    assert get_corsika_run_number(get_test_data_file("sim_telarray", "gamma")) == 10

    # The following file was actually created with the LightEmission package,
    # but it should still return the run number correctly.
    assert get_corsika_run_number("tests/resources/xyzls_layout.simtel.gz") == 1


def test_get_corsika_run_number_without_run_header_real_file():
    assert get_corsika_run_number(get_test_data_file("sim_telarray_hdata", "gamma")) is None


def test_get_combined_eventio_run_header():
    run_header = get_combined_eventio_run_header(get_test_data_file("sim_telarray", "gamma"))
    assert isinstance(run_header, dict)
    assert run_header["run"] == 10
    assert run_header["primary_id"] == 0


def test_get_simulated_events():
    n_showers, n_events = get_simulated_events(get_test_data_file("sim_telarray", "gamma"))
    assert n_showers == 50
    assert n_events == 1000


def test_get_simulated_events_corsika_iact():
    # Test CORSIKA IACT file
    n_showers, n_events = get_simulated_events(get_test_data_file("corsika", "gamma"))
    assert n_showers == 10
    assert n_events == 0  # CORSIKA IACT files don't have MCEvent objects


def test_get_corsika_run_and_event_headers():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        run_header, event_header = get_corsika_run_and_event_headers(
            get_test_data_file("corsika", "gamma")
        )
        # Headers are numpy structured arrays
        assert hasattr(run_header, "dtype")  # numpy structured array
        assert hasattr(event_header, "dtype")  # numpy structured array
        assert "run_number" in run_header.dtype.names  # Check field exists
        assert run_header["run_number"] == 7  # Run number from file name


def test_get_combined_eventio_run_header_incomplete():
    # This file typically has incomplete header information
    run_header = get_combined_eventio_run_header(get_test_data_file("sim_telarray_hdata", "gamma"))
    assert run_header is None or isinstance(run_header, dict)
