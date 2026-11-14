#!/usr/bin/python3

import math
import subprocess
import sys

import astropy.units as u
import numpy as np
import pytest

from simtools.simulation_chain import light_sources
from simtools.simulation_chain.light_sources import (
    AngularDistribution,
    FlasherRequest,
    FlasherSource,
    IlluminatorRequest,
    IlluminatorSource,
    PulseShape,
)


class _Model:
    """Model with the parameter access of the simtools models (values of the tests)."""

    def __init__(self, **parameters):
        self.parameters = parameters

    def get_parameter_value(self, name):
        value = self.parameters.get(name)
        return value.value if hasattr(value, "unit") else value

    def get_parameter_value_with_unit(self, name):
        return self.parameters.get(name)


@pytest.fixture
def flasher_model():
    return _Model(
        flasher_position=[2.0 * u.cm, 0.0 * u.cm, 0.0 * u.cm, None],
        flasher_wavelength=355.0 * u.nm,
        flasher_photons=1_000_000,
        flasher_bunch_size=1,
        flasher_type="flat_fielding",
        flasher_angular_distribution="FilledCone",
        flasher_angular_distribution_width=2.5 * u.deg,
        flasher_pulse_shape=["Gauss-Exponential", 0.4, 1.59],
    )


@pytest.fixture
def telescope_model():
    return _Model(
        focal_length=16.0 * u.m,
        focus_offset=[0.0 * u.cm, 0.0 * u.deg, 0.0 * u.one, 0.0 * u.one],
        mirror_offset=50.0 * u.cm,
        camera_body_diameter=240.0 * u.cm,
        camera_body_shape=0,
        array_element_position_ground=[10.0 * u.m, 0.0 * u.m, 5.0 * u.m],
        axes_offsets=[0.0 * u.cm, 0.0 * u.cm],
        telescope_sphere_radius=12.0 * u.m,
    )


@pytest.fixture
def site_model():
    return _Model(corsika_observation_level=2158.0 * u.m)


@pytest.fixture
def illuminator_model():
    return _Model(
        flasher_wavelength=[300.0 * u.nm, 355.0 * u.nm],
        flasher_photons=10_000_000_000,
        flasher_bunch_size=1,
        flasher_type="illuminator",
        flasher_angular_distribution="Lambertian",
        flasher_angular_distribution_width=20.0 * u.deg,
        flasher_pulse_shape=["Gauss", 2.0, 0.0],
        array_element_position_ground=[110.0 * u.m, 0.0 * u.m, 0.0 * u.m],
        illuminator_tower_height=5.0 * u.m,
    )


def test_flasher_request(flasher_model, telescope_model, site_model):
    request = light_sources.flasher_request(
        flasher_model,
        telescope_model,
        site_model,
        {
            "number_of_events": 3,
            "run_number": 7,
            "telescope": "LSTN-01",
            "site": "North",
            "model_version": "6.0.2",
            "instrument_seed": 5,
        },
        11,
    )
    assert request.position_m == pytest.approx((0.02, 0.0, -0.5))  # mirror_offset shift
    assert request.camera_radius_m == pytest.approx(1.2)
    assert request.events == (3,)
    assert request.photons == (1_000_000,)
    assert request.wavelength_nm == pytest.approx(355.0)
    assert request.angular == AngularDistribution("FilledCone", 2.5)
    assert request.pulse == PulseShape("Gauss-Exponential", 0.4, 1.59)
    assert request.observation_level_m == pytest.approx(2158.0)
    assert (request.run_number, request.seed, request.instrument_seed) == (7, 11, 5)
    assert (request.telescope, request.site, request.model_version) == ("LSTN-01", "North", "6.0.2")
    assert request.photons_per_pulse == (1_000_000,) * 3


def test_flasher_request_photon_override_and_levels(flasher_model, telescope_model, site_model):
    config = {"flasher_photons": "1000,2000", "number_of_events": [4, 5], "wavelength": 355 * u.nm}
    request = light_sources.flasher_request(flasher_model, telescope_model, site_model, config)
    assert request.photons == (1000, 2000)
    assert request.events == (4, 5)


def test_flasher_request_camera_shapes(flasher_model, telescope_model, site_model):
    telescope_model.parameters["camera_body_shape"] = 2
    request = light_sources.flasher_request(flasher_model, telescope_model, site_model, {})
    assert request.camera_radius_m == pytest.approx(2.4 / math.sqrt(2.0))
    telescope_model.parameters["camera_body_shape"] = 9
    with pytest.raises(ValueError, match="Unknown shape value"):
        light_sources.flasher_request(flasher_model, telescope_model, site_model, {})


@pytest.mark.parametrize(
    ("events", "photons", "expected"),
    [
        (1, 5, ((1,), (5,))),
        (2, [5, 6, 7], ((2, 2, 2), (5, 6, 7))),
        ([1, 2], "5,6", ((1, 2), (5, 6))),
    ],
)
def test_event_photon_sequences(events, photons, expected):
    assert light_sources.event_photon_sequences(events, photons) == expected


def test_event_photon_sequences_errors():
    with pytest.raises(ValueError, match="Invalid number_of_events list length"):
        light_sources.event_photon_sequences([1, 2, 3], [5, 6])
    with pytest.raises(ValueError, match="No photon yield"):
        light_sources.event_photon_sequences(1, None)


def test_select_wavelength(illuminator_model):
    assert light_sources.select_wavelength(illuminator_model, 355.3 * u.nm) == pytest.approx(355.0)
    with pytest.raises(ValueError, match="not supported by the light source"):
        light_sources.select_wavelength(illuminator_model, 400.0 * u.nm)
    with pytest.raises(ValueError, match="several wavelengths"):
        light_sources.select_wavelength(illuminator_model)


def test_illuminator_request_from_model(illuminator_model, telescope_model, site_model):
    config = {"wavelength": 355.0 * u.nm, "number_of_events": 2}
    request = light_sources.illuminator_request(
        illuminator_model, telescope_model, site_model, config, seed=5
    )
    assert request.position_m == pytest.approx((110.0, 0.0, 5.0))  # Including the tower.
    assert request.telescope_position_m == pytest.approx((10.0, 0.0, 5.0))
    assert request.pointing == pytest.approx((-1.0, 0.0, 0.0))
    assert request.telescope_pointing_deg == pytest.approx((90.0, 0.0))
    assert request.telescope_radius_m == pytest.approx(12.0)
    assert (request.events, request.photons) == (2, (10_000_000_000,))
    assert request.wavelength_nm == pytest.approx(355.0)
    assert (request.run_number, request.seed) == (1, 5)


def test_illuminator_request_configured_geometry(illuminator_model, telescope_model, site_model):
    config = {
        "wavelength": 300.0 * u.nm,
        "light_source_position": [0.0, 0.0, 100.0],
        "light_source_pointing": [0.0, 0.0, -1.0],
    }
    request = light_sources.illuminator_request(
        illuminator_model, telescope_model, site_model, config
    )
    # Downward beam: artificial telescope at the origin, vertical pointing.
    assert request.position_m == pytest.approx((0.0, 0.0, 100.0))
    assert request.pointing == pytest.approx((0.0, 0.0, -1.0))
    assert request.telescope_position_m == pytest.approx((0.0, 0.0, 0.0))
    assert request.telescope_pointing_deg == (0.0, 0.0)


def test_illuminator_request_configured_position_with_model_telescope(
    illuminator_model, telescope_model, site_model
):
    config = {
        "wavelength": 300.0 * u.nm,
        "light_source_position": [110.0 * u.m, 0.0 * u.m, 5.0 * u.m],
        "light_source_pointing": [-1.0, 0.0, 0.0],
    }
    request = light_sources.illuminator_request(
        illuminator_model, telescope_model, site_model, config
    )
    assert request.telescope_position_m == pytest.approx((10.0, 0.0, 5.0))
    assert request.pointing == pytest.approx((-1.0, 0.0, 0.0))
    assert request.telescope_pointing_deg == (0.0, 0.0)


def test_telescope_position_axes_offsets(telescope_model):
    telescope_model.parameters["array_element_position_ground"] = [0.0 * u.m] * 3
    telescope_model.parameters["axes_offsets"] = [50.0 * u.cm, 25.0 * u.cm]
    position = light_sources.telescope_position_m(telescope_model, [0.0, 100.0, 0.0], {})
    # Illuminator to the north: azimuth offset along the azimuth, altitude offset upwards.
    assert position == pytest.approx((0.0, 0.5, 0.25))
    telescope_model.parameters["axes_offsets"] = [50.0 * u.cm]
    position = light_sources.telescope_position_m(telescope_model, [0.0, 100.0, 0.0], {})
    assert position == pytest.approx((0.0, 0.5, 0.0))


def test_pointing_towards_telescope_zero_distance():
    with pytest.raises(ValueError, match="at the telescope position"):
        light_sources.pointing_towards_telescope([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])


def test_illuminator_request_events_per_level(illuminator_model, telescope_model, site_model):
    config = {
        "wavelength": 355.0 * u.nm,
        "flasher_photons": [1000, 2000],
        "number_of_events": [1, 2],
    }
    with pytest.raises(ValueError, match="one number of events"):
        light_sources.illuminator_request(illuminator_model, telescope_model, site_model, config)


@pytest.mark.parametrize(
    ("pointing", "expected"),
    [
        ([0.0, 0.0, -1.0], False),
        ([0.0, 0.0, 1.0], True),
        ([0.0, 0.0], True),
        (None, True),
        ("down", True),
        ([0.0, float("nan"), -1.0], True),
    ],
)
def test_uses_telescope_position(pointing, expected):
    assert light_sources.uses_telescope_position(pointing) is expected


def _flasher_request(**options):
    return FlasherRequest(
        **{
            "position_m": (0.02, 0.0, -0.5),
            "camera_radius_m": 1.2,
            "events": (2, 1),
            "photons": (1000, 2000),
            "wavelength_nm": 355.0,
            "angular": AngularDistribution("FilledCone", 2.5),
            "pulse": PulseShape("Gauss", 1.5, 0.0),
            "observation_level_m": 2158.0,
            "telescope": "LSTN-01",
            "site": "North",
            "model_version": "6.0.2",
            "run_number": 4,
            "seed": 9,
            "instrument_seed": 3,
        }
        | options
    )


def test_flasher_request_json_round_trip(tmp_path):
    request = _flasher_request()
    path = light_sources.request_to_json(request, tmp_path / "sub" / "request.json")
    assert light_sources.flasher_request_from_json(path) == request
    assert request.photons_per_pulse == (1000, 1000, 2000)


def test_flasher_command(tmp_path):
    tokens = FlasherSource.command(
        _flasher_request(), tmp_path, output=tmp_path / "f.pst", file_token="_x", executable="py"
    )
    assert tokens == [
        "py",
        "-m",
        "simtools.simulation_chain.flasher_stream",
        *("--request", str(tmp_path / "light_source_request_x.json")),
        *("--output-path", str(tmp_path)),
        *("--output", str(tmp_path / "f.pst")),
    ]
    assert light_sources.flasher_request_from_json(tokens[4]) == _flasher_request()
    assert "--output" not in FlasherSource.command(_flasher_request(), tmp_path)


def _illuminator_request(**options):
    return IlluminatorRequest(
        **{
            "position_m": (110.0, 0.0, 5.0),
            "pointing": (-1.0, 0.0, 0.0),
            "telescope_position_m": (10.0, 0.0, 5.0),
            "telescope_radius_m": 12.0,
            "telescope_pointing_deg": (90.0, 0.0),
            "events": 2,
            "photons": (1_000_000,),
            "bunch_size": 1.0,
            "wavelength_nm": 355.0,
            "angular": AngularDistribution("Lambertian", 20.0),
            "pulse": PulseShape("Gauss-Exponential", 0.4, 1.59),
            "observation_level_m": 2158.0,
            "run_number": 3,
            "seed": 8,
        }
        | options
    )


def test_illuminator_command_and_tables(tmp_path):
    tokens = IlluminatorSource.command(
        _illuminator_request(), tmp_path, output=tmp_path / "i.pst", file_token="_x"
    )
    positions = tmp_path / "light_source_telescope_position_x.dat"
    angular = tmp_path / "light_source_angular_distribution_x.dat"
    pulse = tmp_path / "light_source_pulse_shape_x.dat"
    assert tokens == [
        "obdeect-sources-point-source",
        *("-R", "3", "-N", "2"),
        *("-x", "11000", "-y", "0", "-z", "500", "-d", "-1,0,0"),
        *("-n", "1000000", "-b", "1", "-s", "355"),
        *("-p", str(pulse), "-a", str(angular), "-h", "2158"),
        *("--telpos-file", str(positions), "--seed", "8", "-o", str(tmp_path / "i.pst")),
    ]
    assert positions.read_text(encoding="utf-8").split() == ["1000", "0", "500", "1200"]
    angles, intensity = np.loadtxt(angular, unpack=True)
    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(20.0)
    assert intensity[0] == pytest.approx(1.0)
    assert np.all(np.diff(intensity) < 0.0)
    times, rates = np.loadtxt(pulse, unpack=True)
    assert np.all(np.diff(times) > 0.0)
    assert rates.max() > 0.0
    assert np.all(rates >= 0.0)


def test_illuminator_command_with_program_tokens(tmp_path):
    request = _illuminator_request(
        angular=AngularDistribution("FilledCone", 2.0), pulse=PulseShape("Exponential", 0.0, 3.0)
    )
    tokens = IlluminatorSource.command(request, tmp_path)
    assert tokens[tokens.index("-a") + 1] == "FilledCone:2"
    assert tokens[tokens.index("-p") + 1] == "Exp:3"
    assert "-o" not in tokens
    assert not (tmp_path / "light_source_angular_distribution.dat").exists()


def test_illuminator_tokens_without_tables(tmp_path):
    assert (
        light_sources.illuminator_angular_token(AngularDistribution("isotropic", 0.0), tmp_path)
        == "isotropic"
    )
    assert (
        light_sources.illuminator_angular_token(AngularDistribution("Gauss", 3.0), tmp_path)
        == "Gauss:3"
    )
    assert light_sources.illuminator_pulse_token(PulseShape("TopHat", 2.0, 0.0), tmp_path) == (
        "Simple:2"
    )
    assert light_sources.illuminator_pulse_token(PulseShape("Gauss", 0.0, 0.0), tmp_path) == (
        "Gauss:0"
    )
    # An instantaneous pulse needs no table; an unknown shape with a width has none.
    assert (
        light_sources.illuminator_pulse_token(PulseShape("Triangle", 0.0, 0.0), tmp_path)
        == "Simple:0"
    )
    with pytest.raises(ValueError, match="has no tabulated profile"):
        light_sources.illuminator_pulse_token(PulseShape("Triangle", 1.0, 0.0), tmp_path)


def test_registry_has_builtin_types():
    assert light_sources.get_source_type("flasher") is FlasherSource
    assert light_sources.get_source_type("illuminator") is IlluminatorSource
    assert light_sources.source_type_of_model("flat_fielding") is FlasherSource
    assert light_sources.source_type_of_model("illuminator") is IlluminatorSource


def test_registry_errors():
    with pytest.raises(ValueError, match="Unknown light source type 'laser'"):
        light_sources.get_source_type("laser")
    with pytest.raises(ValueError, match="No light source type handles"):
        light_sources.source_type_of_model("laser_beam")


def test_new_source_type_without_editing_existing_code(monkeypatch):
    class LaserSource:
        name = "laser"
        light_source_types = ("laser_beam",)

        @staticmethod
        def build_request(*_args, **_kwargs):
            return "request"

        @staticmethod
        def command(request, work_directory, output=None, file_token="", executable=None):
            return ["laser", request, str(work_directory)]

    monkeypatch.setattr(light_sources, "SOURCE_TYPES", dict(light_sources.SOURCE_TYPES))
    light_sources.register_source_type("laser", lambda: LaserSource)
    assert light_sources.get_source_type("laser") is LaserSource
    assert light_sources.source_type_of_model("laser_beam") is LaserSource
    assert light_sources.source_type_of_model("flat_fielding") is FlasherSource


def test_legacy_modules_do_not_import_obdeect_through_light_sources():
    code = (
        "import sys\n"
        "import simtools.simulation_chain.light_sources as m\n"
        "m.get_source_type('flasher'); m.get_source_type('illuminator')\n"
        "assert not any(name.startswith('obdeect') for name in sys.modules)\n"
    )
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
