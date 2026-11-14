"""Tests for the atmosphere of the simulation chain (obdeect propagation inputs and command line)."""

import math
from pathlib import Path

import astropy.units as u
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.simulation_chain.atmosphere import (
    AtmosphereRequest,
    build_atmosphere_request,
    in_telescope_pathlength_m,
    propagation_command,
    propagation_program,
)


class _TelescopeStandIn:
    """Telescope model stand-in with the parameters used for the in-telescope path."""

    def __init__(self, mirror_class, focal_length_m=28.0):
        self._mirror_class = mirror_class
        self._focal_length_m = focal_length_m

    def get_parameter_value(self, name):
        assert name == "mirror_class"
        return self._mirror_class

    def get_parameter_value_with_unit(self, name):
        assert name == "focal_length"
        return self._focal_length_m * u.m


def _request(**kwargs):
    values = {
        "site": "North",
        "observation_level_m": 2158.0,
        "profile_file": Path("atmprof.dat"),
        "transmission_file": Path("transmission.dat"),
        "seed": 1,
    }
    values.update(kwargs)
    return AtmosphereRequest(**values)


def test_request_defaults():
    request = _request()

    assert request.mode == "thinned"
    assert request.geometry == "plane_parallel"
    assert request.wavelength_range_nm is None
    assert request.telescope_pathlen_m == {}


def test_request_is_hashable_without_pathlengths():
    assert hash(_request()) == hash(_request())


def test_request_rejects_unsupported_geometry():
    with pytest.raises(ValueError, match=r"Unsupported atmosphere geometry 'curved'"):
        _request(geometry="curved")


def test_request_rejects_unknown_mode():
    with pytest.raises(ValueError, match=r"Invalid atmosphere output mode 'bogus'"):
        _request(mode="bogus")


def test_propagation_command_default_stdio():
    request = _request(telescope_pathlen_m={1: 28.0, 2: 0.0})

    assert propagation_command(request) == [
        "obdeect-propagation-run",
        "--transmission",
        "transmission.dat",
        "--site-altitude",
        "2158.0",
        "--profile",
        "atmprof.dat",
        "--seed",
        "1",
        "--mode",
        "thinned",
        "--telescope-pathlen",
        "1=28.0",
        "--telescope-pathlen",
        "2=0.0",
        "-i",
        "-",
        "-o",
        "-",
    ]


def test_propagation_command_wavelength_range_and_weighted_mode():
    request = _request(mode="weighted", wavelength_range_nm=(250.0, 700.0))

    command = propagation_command(request, program="/opt/obdeect-propagation-run")

    assert command[0] == "/opt/obdeect-propagation-run"
    assert command[command.index("--mode") + 1] == "weighted"
    range_index = command.index("--wavelength-range")
    assert command[range_index + 1 : range_index + 3] == ["250.0", "700.0"]
    assert "--telescope-pathlen" not in command


def test_propagation_command_file_input_and_output():
    request = _request()

    command = propagation_command(request, input_file=Path("in.pst"), output_file=Path("out.pst"))

    assert command[-4:] == ["-i", "in.pst", "-o", "out.pst"]


def test_propagation_command_has_no_extinction_geometry_options():
    command = propagation_command(_request())

    assert "--no-extinction" not in command
    assert "--scattering" not in command
    assert "--envelope" not in command


def test_propagation_program_is_installed_binary():
    program = propagation_program()

    assert Path(program).name == "obdeect-propagation-run"
    assert Path(program).parent.name == "bin"


def test_in_telescope_pathlength_single_mirror_is_focal_length():
    telescope = _TelescopeStandIn(mirror_class=1, focal_length_m=28.0)

    assert in_telescope_pathlength_m(telescope) == pytest.approx(28.0)


def test_in_telescope_pathlength_dual_mirror_is_zero():
    telescope = _TelescopeStandIn(mirror_class=2, focal_length_m=5.6)

    assert math.isclose(in_telescope_pathlength_m(telescope), 0.0, abs_tol=1e-12)


def test_build_request_from_site_and_telescope_models(
    site_model_north, telescope_model_lst, telescope_model_mst, tmp_test_directory
):
    request = build_atmosphere_request(
        site_model_north,
        {"LSTN-01": telescope_model_lst, "MSTS-05": telescope_model_mst},
        seed=5,
        mode="weighted",
        wavelength_range_nm=[250.0, 700.0],
        directory=tmp_test_directory,
    )

    assert request.site == "North"
    assert request.seed == 5
    assert request.mode == "weighted"
    assert request.wavelength_range_nm == (250.0, 700.0)
    assert_quantity_allclose(request.observation_level_m * u.m, 2158.0 * u.m)
    assert request.profile_file == site_model_north.config_file_directory / (
        "atmospheric_profile_north_dummy.dat"
    )
    assert request.transmission_file == site_model_north.config_file_directory / (
        "atmospheric_transmission_north_dummy.dat"
    )
    assert set(request.telescope_pathlen_m) == {1, 2}
    assert request.telescope_pathlen_m[1] == pytest.approx(
        in_telescope_pathlength_m(telescope_model_lst)
    )
    assert request.telescope_pathlen_m[2] == pytest.approx(
        in_telescope_pathlength_m(telescope_model_mst)
    )


def test_build_request_defaults(site_model_north, telescope_model_lst, tmp_test_directory):
    request = build_atmosphere_request(
        site_model_north, {"LSTN-01": telescope_model_lst}, directory=tmp_test_directory
    )

    assert request.seed == 1
    assert request.mode == "thinned"
    assert request.wavelength_range_nm is None
