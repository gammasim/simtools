import math
import sys
import types
from types import SimpleNamespace

import astropy.units as u
import numpy as np
import pytest

from simtools.atmosphere import sixs_backend
from simtools.atmosphere.optical_tables import get_backend
from simtools.atmosphere.sixs_backend import (
    SixSBackend,
    _map_runs,
    _phase_zenith_deg,
    scattering_angle_deg,
)

WAVELENGTH_UM = [0.5, 1.0]
ALTITUDE_KM = [1.0, 5.0]
HG_ASYMMETRY = 0.7
SCALE_HEIGHT_KM = 8.0


def _mock_scattering_angle(solar_zenith, view_zenith, azimuth_difference):
    """Independent implementation of the 6S scattering-angle convention for the mock."""
    sun, view, diff = np.radians([solar_zenith, view_zenith, azimuth_difference])
    cos_angle = -(np.cos(sun) * np.cos(view) + np.sin(sun) * np.sin(view) * np.cos(diff))
    return float(np.degrees(np.arccos(cos_angle)))


def _henyey_greenstein(cos_angle, asymmetry):
    return (1 - asymmetry**2) / (4 * np.pi * (1 + asymmetry**2 - 2 * asymmetry * cos_angle) ** 1.5)


def _fake_outputs(wavelength, altitude, geometry, angle_offset, nan_phase):
    """Mock 6S outputs: optical depths decrease with altitude, phase is HG(0.7)."""
    scale = np.exp(-altitude / SCALE_HEIGHT_KM)
    angle = _mock_scattering_angle(geometry.solar_z, geometry.view_z, geometry.view_a)
    phase = _henyey_greenstein(np.cos(np.radians(angle)), HG_ASYMMETRY)
    return SimpleNamespace(
        values={"scattering_angle": angle + angle_offset, "version": "1.1"},
        rat={
            "optical_depth_total": SimpleNamespace(
                rayleigh=(1.0 + 0.1 * wavelength) * scale,
                aerosol=(0.2 + 0.05 * wavelength) * scale,
                total=np.nan,
            ),
            "single_scattering_albedo": SimpleNamespace(aerosol=0.9),
            "phase_function_I": SimpleNamespace(
                rayleigh=np.nan, aerosol=np.nan if nan_phase else phase, total=np.nan
            ),
        },
        trans={
            "global_gas": SimpleNamespace(upward=np.exp(-0.3 * wavelength * scale)),
        },
    )


class _FakeGeometry:
    class User:
        solar_z = solar_a = view_z = view_a = 0.0


class _FakeAltitudes:
    def __init__(self):
        self.target_alt_pres = None

    def set_target_custom_altitude(self, altitude):
        self.target_alt_pres = -altitude

    def set_sensor_satellite_level(self):
        """Sensor at satellite level (no effect on the mock)."""


def _make_fake_py6s(calls, executable, angle_offset, nan_phase):
    atmos_profile = SimpleNamespace(
        MidlatitudeSummer=2,
        PredefinedType=lambda code: ("predefined", code),
        UserWaterAndOzone=lambda water, ozone: ("user", water, ozone),
    )
    aero_profile = SimpleNamespace(
        Continental=1,
        PredefinedType=lambda code: ("aerosol", code),
    )

    class SixS:
        __version__ = "1.9.2"

        def __init__(self, path=None):
            self.sixs_path = path or ("/opt/6s/sixsV1.1" if executable else None)
            self.outputs = None

        def run(self):
            calls.append(
                {
                    "wavelength": self.wavelength,
                    "altitude": -self.altitudes.target_alt_pres,
                    "atmos": self.atmos_profile,
                    "aero": self.aero_profile,
                    "aot550": self.aot550,
                    "solar_z": self.geometry.solar_z,
                    "view_z": self.geometry.view_z,
                    "view_a": self.geometry.view_a,
                }
            )
            self.outputs = _fake_outputs(
                self.wavelength,
                -self.altitudes.target_alt_pres,
                self.geometry,
                angle_offset,
                nan_phase,
            )

    module = types.ModuleType("Py6S")
    module.SixS = SixS
    module.AtmosProfile = atmos_profile
    module.AeroProfile = aero_profile
    module.Geometry = _FakeGeometry
    module.Altitudes = _FakeAltitudes
    module.Wavelength = lambda value: value
    return module


@pytest.fixture
def fake_py6s(monkeypatch):
    """Install a mock Py6S module; returns the list of recorded 6S runs."""

    def install(executable=True, angle_offset=0.0, nan_phase=False):
        calls = []
        module = _make_fake_py6s(calls, executable, angle_offset, nan_phase)
        monkeypatch.setitem(sys.modules, "Py6S", module)
        return calls

    return install


def test_missing_py6s_raises_actionable_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "Py6S", None)
    with pytest.raises(ImportError, match="pip install Py6S"):
        SixSBackend().compute(WAVELENGTH_UM * u.um, ALTITUDE_KM * u.km)


def test_missing_executable_raises_actionable_error(fake_py6s):
    fake_py6s(executable=False)
    with pytest.raises(RuntimeError, match="6S executable not found"):
        SixSBackend().compute(WAVELENGTH_UM * u.um, ALTITUDE_KM * u.km)


def test_one_run_per_phase_angle_and_optical_point(fake_py6s):
    calls = fake_py6s()
    SixSBackend().compute(WAVELENGTH_UM * u.um, ALTITUDE_KM * u.km)
    # 178 reachable angles (3 to 180 deg) per wavelength, plus one run per (wavelength, altitude)
    assert len(calls) == 2 * 178 + 2 * 2


def test_optical_outputs_are_mapped_to_arrays(fake_py6s):
    fake_py6s()
    props = SixSBackend().compute(WAVELENGTH_UM * u.um, ALTITUDE_KM * u.km)
    wavelength = np.array(WAVELENGTH_UM)[:, None]
    scale = np.exp(-np.array(ALTITUDE_KM)[None, :] / SCALE_HEIGHT_KM)
    np.testing.assert_allclose(props.tau_molecular, (1.0 + 0.1 * wavelength) * scale)
    np.testing.assert_allclose(props.tau_aerosol, (0.2 + 0.05 * wavelength) * scale)
    np.testing.assert_allclose(props.aerosol_ssa, np.full((2, 2), 0.9))
    np.testing.assert_allclose(props.altitude.to_value(u.km), ALTITUDE_KM)
    np.testing.assert_allclose(props.wavelength.to_value(u.um), WAVELENGTH_UM)


def test_tau_absorption_is_minus_log_of_upward_transmittance(fake_py6s):
    fake_py6s()
    props = SixSBackend().compute(WAVELENGTH_UM * u.um, ALTITUDE_KM * u.km)
    wavelength = np.array(WAVELENGTH_UM)[:, None]
    scale = np.exp(-np.array(ALTITUDE_KM)[None, :] / SCALE_HEIGHT_KM)
    upward = np.exp(-0.3 * wavelength * scale)
    np.testing.assert_allclose(props.tau_absorption, -np.log(upward))


def test_aerosol_and_atmosphere_settings_reach_6s(fake_py6s):
    calls = fake_py6s()
    SixSBackend(aot550=0.2).compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    assert calls[0]["atmos"] == ("predefined", 2)
    assert calls[0]["aero"] == ("aerosol", 1)
    assert calls[0]["aot550"] == pytest.approx(0.2)


def test_user_water_and_ozone_atmosphere(fake_py6s):
    calls = fake_py6s()
    backend = SixSBackend(ozone=0.3, water=2.0)
    backend.compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    assert calls[0]["atmos"] == ("user", 2.0, 0.3)


def test_geometry_of_phase_and_optical_runs(fake_py6s):
    calls = fake_py6s()
    SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    optical = [call for call in calls if call["altitude"] > 0]
    assert {(call["solar_z"], call["view_z"]) for call in optical} == {(0.0, 0.0)}
    phase = [call for call in calls if call["altitude"] == 0]
    for call in phase:
        assert call["view_a"] == pytest.approx(180.0)
        assert call["solar_z"] == pytest.approx(call["view_z"])
        assert call["solar_z"] < sixs_backend.MAX_ZENITH_DEG


def test_phase_grid_extends_to_zero_and_is_normalised(fake_py6s):
    fake_py6s()
    props = SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    angles = props.phase_angle.to_value(u.deg)
    assert angles[0] == pytest.approx(0.0)
    assert angles[-1] == pytest.approx(180.0)
    assert angles.size == 180  # 0 deg plus 2 to 180 deg in 1 deg steps
    assert np.all(np.diff(angles) > 0)
    mu = np.cos(np.radians(angles))
    for wavelength_index in range(2):
        phase = props.aerosol_phase[wavelength_index, 0]
        # mu decreases with angle, so the trapezoid integral over mu is negative
        integral = -np.trapezoid(phase, mu)
        assert 2.0 * np.pi * integral == pytest.approx(1.0, rel=1e-9)
    np.testing.assert_allclose(props.aerosol_phase[:, 0], props.aerosol_phase[:, 1])


def test_extension_below_reachable_angles_follows_henyey_greenstein(fake_py6s):
    fake_py6s()
    props = SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    angles = props.phase_angle.to_value(u.deg)
    phase = props.aerosol_phase[0, 0]
    asymmetry = props.aerosol_asymmetry[0, 0]

    def shape(angle):
        mu = math.cos(math.radians(angle))
        return 1.0 / (1 + asymmetry**2 - 2 * asymmetry * mu) ** 1.5

    # Values at 0 and 2 deg are extensions; their ratio to the value at 3 deg follows the HG shape
    assert phase[angles == 0][0] / phase[angles == 3][0] == pytest.approx(
        shape(0.0) / shape(3.0), rel=1e-3
    )
    assert phase[angles == 2][0] / phase[angles == 3][0] == pytest.approx(
        shape(2.0) / shape(3.0), rel=1e-3
    )


def test_asymmetry_of_henyey_greenstein_phase_function(fake_py6s):
    fake_py6s()
    props = SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    np.testing.assert_allclose(props.aerosol_asymmetry, HG_ASYMMETRY, atol=1e-3)


def test_non_finite_phase_function_raises(fake_py6s):
    fake_py6s(nan_phase=True)
    with pytest.raises(RuntimeError, match="non-finite aerosol phase function"):
        SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)


def test_scattering_angle_mismatch_with_6s_raises(fake_py6s):
    fake_py6s(angle_offset=1.0)
    with pytest.raises(RuntimeError, match="geometry convention does not match"):
        SixSBackend().compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)


def test_metadata_contains_versions_and_settings(fake_py6s):
    fake_py6s()
    props = SixSBackend(aot550=0.3, processes=1).compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    assert props.metadata["backend"] == "6s"
    assert props.metadata["py6s_version"] == "1.9.2"
    assert props.metadata["sixs_version"] == "1.1"
    assert props.metadata["aot550"] == pytest.approx(0.3)
    assert props.metadata["atmosphere_profile"] == "MidlatitudeSummer"
    assert props.metadata["aerosol_profile"] == "Continental"
    assert len(props.metadata["phase_angle_deg"]) == 180


def test_wavelength_validation(fake_py6s):
    fake_py6s()
    backend = SixSBackend()
    with pytest.raises(ValueError, match=r"wavelengths must be between 0\.25 and 4\.0"):
        backend.compute([0.2, 1.0] * u.um, [1.0, 5.0] * u.km)
    with pytest.raises(ValueError, match=r"wavelengths must be between 0\.25 and 4\.0"):
        backend.compute([0.5, 4.5] * u.um, [1.0, 5.0] * u.km)
    with pytest.raises(ValueError, match="At least one value"):
        backend.compute([] * u.um, [1.0, 5.0] * u.km)
    props = backend.compute([0.25, 4.0] * u.um, [1.0, 5.0] * u.km)
    np.testing.assert_allclose(props.wavelength.to_value(u.um), [0.25, 4.0])


def test_altitude_validation(fake_py6s):
    fake_py6s()
    backend = SixSBackend()
    with pytest.raises(ValueError, match=r"below 100\.0 km"):
        backend.compute([0.5, 1.0] * u.um, [1.0, 100.0] * u.km)
    with pytest.raises(ValueError, match=r"above 0\.0 km"):
        backend.compute([0.5, 1.0] * u.um, [-1.0, 5.0] * u.km)


def test_sea_level_altitude_zero_is_rejected(fake_py6s):
    calls = fake_py6s()
    backend = SixSBackend()
    with pytest.raises(ValueError, match="Give sea level as a small positive altitude") as info:
        backend.compute([0.5, 1.0] * u.um, [0.0, 5.0] * u.km)
    assert "0.001 km (1 m)" in str(info.value)
    assert calls == []  # rejected before any 6S run
    props = backend.compute([0.5, 1.0] * u.um, [0.001, 5.0] * u.km)
    np.testing.assert_allclose(props.altitude.to_value(u.m), [1.0, 5000.0])


def test_quantities_in_other_units_are_converted(fake_py6s):
    calls = fake_py6s()
    props = SixSBackend().compute([500.0, 1000.0] * u.nm, [1000.0, 5000.0] * u.m)
    np.testing.assert_allclose(props.wavelength.to_value(u.um), [0.5, 1.0])
    np.testing.assert_allclose(props.altitude.to_value(u.km), [1.0, 5.0])
    assert {call["altitude"] for call in calls if call["altitude"] > 0} == {1.0, 5.0}
    assert {call["wavelength"] for call in calls} == {0.5, 1.0}


def test_altitude_plain_numbers_are_km(fake_py6s):
    calls = fake_py6s()
    props = SixSBackend().compute([0.5, 1.0], [1.0, 5.0])
    assert {call["altitude"] for call in calls if call["altitude"] > 0} == {1.0, 5.0}
    np.testing.assert_allclose(props.altitude.to_value(u.km), [1.0, 5.0])


def test_unknown_profile_names_raise(fake_py6s):
    fake_py6s()
    with pytest.raises(ValueError, match="Unknown atmosphere profile 'Nope'"):
        SixSBackend(atmosphere_profile="Nope").compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)
    with pytest.raises(ValueError, match="Unknown aerosol profile 'PredefinedType'"):
        SixSBackend(aerosol_profile="PredefinedType").compute([0.5, 1.0] * u.um, [1.0, 5.0] * u.km)


def test_ozone_and_water_must_be_given_together():
    with pytest.raises(ValueError, match="together"):
        SixSBackend(ozone=0.3)
    with pytest.raises(ValueError, match="together"):
        SixSBackend(water=2.0)


def test_processes_must_be_positive():
    with pytest.raises(ValueError, match="at least 1"):
        SixSBackend(processes=0)


def test_phase_angle_validation():
    with pytest.raises(ValueError, match=r"within \[0, 180\] deg"):
        SixSBackend(phase_angle=[-1.0, 10.0])
    with pytest.raises(ValueError, match="at least two angles"):
        SixSBackend(phase_angle=[1.0, 2.0])
    backend = SixSBackend(phase_angle=[10.0, 20.0] * u.deg)
    np.testing.assert_allclose(backend.phase_angle_deg, [0.0, 10.0, 20.0, 180.0])


def test_scattering_angle_matches_6s_sample_output():
    # 6S output in py6s/wvlinux: solar zenith 32, view zenith 23, azimuth difference 74 deg
    assert math.isclose(scattering_angle_deg(32.0, 23.0, 74.0), 146.90, abs_tol=0.01)
    assert math.isclose(scattering_angle_deg(32.0, 23.0, 180.0), 125.0, abs_tol=1e-9)
    assert math.isclose(scattering_angle_deg(32.0, 23.0, 0.0), 171.0, abs_tol=1e-9)


@pytest.mark.parametrize("angle", [3.0, 90.0, 180.0])
def test_phase_geometry_gives_requested_scattering_angle(angle):
    zenith = float(_phase_zenith_deg(angle))
    assert math.isclose(scattering_angle_deg(zenith, zenith, 180.0), angle, abs_tol=1e-9)


def test_map_runs_keeps_task_order_for_any_process_count():
    tasks = [-3, 2, -1, 5]
    assert _map_runs(abs, tasks, 1) == [3, 2, 1, 5]
    assert _map_runs(abs, tasks, 2) == [3, 2, 1, 5]


def test_registration_as_backend_6s():
    backend = get_backend("6s", aot550=0.2)
    assert isinstance(backend, SixSBackend)
    assert backend.name == "6s"
    assert backend.aot550 == pytest.approx(0.2)
