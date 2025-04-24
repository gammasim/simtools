#!/usr/bin/python3

import pytest

from simtools.simtel.simtel_io_file_info import get_corsika_run_header, get_corsika_run_number


@pytest.fixture
def test_file():
    return (
        "tests/resources/"
        "run000010_gamma_za20deg_azm000deg_North_test_layout_6.0.0"
        "_test-production-North.simtel.zst"
    )


def test_get_corsika_run_number_with_run_header(test_file):
    assert get_corsika_run_number(test_file) == 10

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


def test_get_corsika_run_header(test_file):
    run_header = get_corsika_run_header(test_file)
    assert isinstance(run_header, dict)
    assert run_header["run"] == 10
