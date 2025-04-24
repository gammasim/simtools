#!/usr/bin/python3


from simtools.simtel.simtel_io_file_info import get_corsika_run_number


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
