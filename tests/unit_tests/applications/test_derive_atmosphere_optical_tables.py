from pathlib import Path
from unittest.mock import Mock, patch

import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

from simtools.applications import derive_atmosphere_optical_tables as app
from simtools.atmosphere.optical_tables import (
    AtmosphericOpticalProperties,
    read_optical_properties,
    write_optical_properties,
)

SIXS_NONE = {
    "atmosphere_profile": None,
    "aerosol_profile": None,
    "aot550": None,
    "ozone": None,
    "water": None,
    "sixs_path": None,
    "processes": None,
    "phase_angles": None,
}


def _args(**overrides):
    args = {
        **SIXS_NONE,
        "backend": "ecsv",
        "ecsv_file": None,
        "wavelength_min": 300.0,
        "wavelength_max": 400.0,
        "wavelength_step": 50.0,
        "altitudes": None,
        "altitude_min": None,
        "altitude_max": None,
        "altitude_steps": None,
        "observation_level": 2000.0,
        "depolarization": 0.0,
        "transmission_file": "obdeect_transmission.txt",
        "scattering_file": "obdeect_scattering.txt",
        "properties_file": None,
    }
    args.update(overrides)
    return args


def _write_synthetic_ecsv(file_name, wavelength_nm):
    """Exponential synthetic atmosphere on a log-spaced altitude grid, written as ECSV."""
    wl = np.asarray(wavelength_nm, dtype=float)[:, None]
    altitude_m = np.geomspace(1.0, 1.0e5, 20)
    h_km = altitude_m[None, :] / 1000.0
    shape = (wl.shape[0], altitude_m.size)
    properties = AtmosphericOpticalProperties(
        wavelength=wl[:, 0] * u.nm,
        altitude=altitude_m * u.m,
        tau_molecular=0.00865 * (wl / 1000.0) ** -4.05 * np.exp(-h_km / 8.0),
        tau_aerosol=0.1 * (wl / 550.0) ** -1.3 * np.exp(-h_km / 1.5),
        tau_absorption=np.zeros(shape),
        aerosol_ssa=np.full(shape, 0.9),
        aerosol_asymmetry=np.full(shape, 0.65),
        metadata={"backend": "synthetic"},
    )
    write_optical_properties(properties, file_name)


def test_backend_kwargs_ecsv(tmp_test_directory):
    kwargs = app._backend_kwargs(_args(ecsv_file=f"{tmp_test_directory}/x.ecsv"))
    assert kwargs == {"file_name": f"{tmp_test_directory}/x.ecsv"}


def test_backend_kwargs_ecsv_requires_file_and_rejects_6s_options():
    with pytest.raises(ValueError, match="--ecsv_file is required"):
        app._backend_kwargs(_args())
    with pytest.raises(ValueError, match="Options only apply to --backend 6s: aot550"):
        app._backend_kwargs(_args(ecsv_file="x.ecsv", aot550=0.1))


def test_backend_kwargs_6s_passes_phase_angles_as_quantity():
    args = _args(backend="6s", phase_angles=[0.0, 30.0, 90.0], aot550=0.2)
    kwargs = app._backend_kwargs(args)
    assert kwargs["aot550"] == pytest.approx(0.2)
    assert "file_name" not in kwargs
    assert isinstance(kwargs["phase_angle"], u.Quantity)
    assert kwargs["phase_angle"].unit == u.deg
    assert_quantity_allclose(kwargs["phase_angle"], [0.0, 30.0, 90.0] * u.deg)


def test_backend_kwargs_6s_without_phase_angles_uses_backend_default():
    assert app._backend_kwargs(_args(backend="6s")) == {}


def test_backend_kwargs_6s_rejects_ecsv_file():
    with pytest.raises(ValueError, match="--ecsv_file only applies to --backend ecsv"):
        app._backend_kwargs(_args(backend="6s", ecsv_file="x.ecsv"))


def test_altitudes_explicit_list_and_log_range():
    assert_quantity_allclose(app._altitudes(_args(altitudes=[1.0, 5000.0])), [1.0, 5000.0] * u.m)
    log_range = app._altitudes(_args(altitude_min=1.0, altitude_max=1000.0, altitude_steps=4))
    assert_quantity_allclose(log_range, [1.0, 10.0, 100.0, 1000.0] * u.m)


def test_altitudes_must_be_positive():
    with pytest.raises(ValueError, match="Altitudes must be positive"):
        app._altitudes(_args(altitudes=[0.0, 5000.0]))
    with pytest.raises(ValueError, match="Altitudes must be positive"):
        app._altitudes(_args(altitude_min=0.0, altitude_max=1000.0, altitude_steps=4))


def test_altitudes_option_errors():
    with pytest.raises(ValueError, match="Use either --altitudes"):
        app._altitudes(_args(altitudes=[1.0], altitude_min=1.0))
    with pytest.raises(ValueError, match="Give --altitudes or all of"):
        app._altitudes(_args(altitude_min=1.0))


def test_wavelengths_include_upper_limit_on_grid():
    grid = app._wavelengths(_args(wavelength_min=300.0, wavelength_max=350.0, wavelength_step=25.0))
    assert_quantity_allclose(grid, [300.0, 325.0, 350.0] * u.nm)


def test_wavelengths_reject_bad_range():
    with pytest.raises(ValueError, match="wavelength_step > 0"):
        app._wavelengths(_args(wavelength_step=0.0))
    with pytest.raises(ValueError, match="wavelength_max >= --wavelength_min"):
        app._wavelengths(_args(wavelength_min=400.0, wavelength_max=300.0))


def _properties_on_wavelengths(wavelength_nm):
    shape = (len(wavelength_nm), 2)
    return AtmosphericOpticalProperties(
        wavelength=wavelength_nm * u.nm,
        altitude=[1000.0, 2000.0] * u.m,
        tau_molecular=np.full(shape, 0.1),
        tau_aerosol=np.zeros(shape),
        tau_absorption=np.zeros(shape),
        aerosol_ssa=np.full(shape, 0.5),
        aerosol_asymmetry=np.zeros(shape),
    )


def test_summary_wavelength_is_350_nm_inside_grid_else_grid_minimum():
    inside = _properties_on_wavelengths([300.0, 400.0])
    assert_quantity_allclose(app._summary_wavelength(inside), 350.0 * u.nm)
    outside = _properties_on_wavelengths([400.0, 500.0])
    assert_quantity_allclose(app._summary_wavelength(outside), 400.0 * u.nm)


def test_log_summary_reports_shortest_and_longest(tmp_test_directory):
    file_name = f"{tmp_test_directory}/optical.ecsv"
    _write_synthetic_ecsv(file_name, np.arange(300.0, 401.0, 50.0))
    properties = read_optical_properties(file_name)
    logger = Mock()
    app._log_summary(logger, properties, 2000.0 * u.m)
    messages = [call.args[0] for call in logger.info.call_args_list]
    assert len(messages) == 2
    assert messages[0].startswith("Vertical transmission at 350.0 nm (shortest wavelength)")
    assert messages[1].startswith("Vertical transmission at 400.0 nm (longest wavelength)")
    assert "from 100 km to 2000.0 m a.s.l.:" in messages[0]


@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_without_properties_file_writes_only_obdeect_tables(
    mock_application_start, tmp_test_directory
):
    ecsv_file = f"{tmp_test_directory}/optical.ecsv"
    _write_synthetic_ecsv(ecsv_file, np.arange(300.0, 401.0, 10.0))
    context = Mock()
    context.args = _args(ecsv_file=ecsv_file, altitudes=[1.0, 5000.0], observation_level=2000.0)
    context.paths.get_output_directory.return_value = tmp_test_directory
    mock_application_start.return_value = context

    app.main()

    assert Path(f"{tmp_test_directory}/obdeect_transmission.txt").exists()
    assert not Path(f"{tmp_test_directory}/properties.ecsv").exists()


@patch("simtools.application.definition.ApplicationDefinition.start")
def test_main_writes_tables_and_depolarization(mock_application_start, tmp_test_directory):
    ecsv_file = f"{tmp_test_directory}/optical.ecsv"
    _write_synthetic_ecsv(ecsv_file, np.arange(300.0, 701.0, 10.0))
    context = Mock()
    context.args = _args(
        ecsv_file=ecsv_file,
        wavelength_min=300.0,
        wavelength_max=500.0,
        wavelength_step=100.0,
        altitudes=[1.0, 5000.0, 10000.0],
        observation_level=2147.0,
        depolarization=0.02,
        properties_file="properties.ecsv",
    )
    context.paths.get_output_directory.return_value = tmp_test_directory
    mock_application_start.return_value = context

    app.main()

    scattering = Path(f"{tmp_test_directory}/obdeect_scattering.txt").read_text(encoding="utf-8")
    assert "depolarization 0.02" in scattering.splitlines()
    transmission = Path(f"{tmp_test_directory}/obdeect_transmission.txt").read_text(
        encoding="utf-8"
    )
    assert any(line.startswith("# H2= 2.147, H1=") for line in transmission.splitlines())
    restored = read_optical_properties(f"{tmp_test_directory}/properties.ecsv")
    assert restored.depolarization == pytest.approx(0.02)
    assert any(
        "Vertical transmission at 350.0 nm" in call.args[0]
        for call in context.logger.info.call_args_list
    )
