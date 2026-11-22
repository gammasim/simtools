#!/usr/bin/python3

import sys
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from astropy.tests.helper import assert_quantity_allclose

from simtools.atmosphere import optical_tables
from simtools.atmosphere.optical_tables import (
    AtmosphericOpticalProperties,
    EcsvBackend,
    RadiativeTransferBackend,
    get_backend,
    read_optical_properties,
    register_backend,
    write_obdeect_scattering_table,
    write_obdeect_transmission_table,
    write_optical_properties,
)

HG_G = 0.65
N_ANGLES = 181


def _altitude_m():
    return np.geomspace(2000.0, 100000.0, 40)


def _exponential_properties(wavelength_nm=np.arange(300.0, 401.0), altitude_m=None):
    """Exponential synthetic atmosphere with a Henyey-Greenstein aerosol phase function."""
    altitude_m = _altitude_m() if altitude_m is None else altitude_m
    wl = np.asarray(wavelength_nm, dtype=float)
    h_km = altitude_m[np.newaxis, :] / 1000.0
    wl_um = wl[:, np.newaxis] / 1000.0
    tau_mol = 0.00865 * wl_um**-4.05 * np.exp(-h_km / 8.0)
    tau_aer = 0.1 * (wl[:, np.newaxis] / 550.0) ** -1.3 * np.exp(-h_km / 1.5)
    tau_abs = 0.01 * np.exp(-h_km / 3.0) * np.ones_like(wl[:, np.newaxis])
    ssa = np.full(tau_mol.shape, 0.9)
    asymmetry = np.full(tau_mol.shape, HG_G)
    angles = np.linspace(0.0, 180.0, N_ANGLES)
    phase = _henyey_greenstein(np.cos(np.radians(angles)), HG_G)
    phase = np.broadcast_to(phase, (wl.size, altitude_m.size, N_ANGLES)).copy()
    return AtmosphericOpticalProperties(
        wavelength=wl * u.nm,
        altitude=altitude_m * u.m,
        tau_molecular=tau_mol,
        tau_aerosol=tau_aer,
        tau_absorption=tau_abs,
        aerosol_ssa=ssa,
        aerosol_asymmetry=asymmetry,
        phase_angle=angles * u.deg,
        aerosol_phase=phase,
        depolarization=0.02,
        metadata={"backend": "synthetic", "version": "1", "settings": {"model": "exp"}},
    )


def _henyey_greenstein(cos_theta, g):
    return (1 - g**2) / (4 * np.pi * (1 + g**2 - 2 * g * cos_theta) ** 1.5)


def _analytic_total(wavelength_nm, altitude_m):
    """Analytic total optical depth of the synthetic atmosphere (same as _exponential_properties)."""
    h_km = altitude_m / 1000.0
    tau_mol = 0.00865 * (wavelength_nm / 1000.0) ** -4.05 * np.exp(-h_km / 8.0)
    tau_aer = 0.1 * (wavelength_nm / 550.0) ** -1.3 * np.exp(-h_km / 1.5)
    return tau_mol + tau_aer + 0.01 * np.exp(-h_km / 3.0)


def _read_data_rows(path):
    with open(path, encoding="utf-8") as file:
        return [line.split() for line in file if line.strip() and not line.startswith("#")]


def _minimal_properties(n_wl=2, n_h=3):
    shape = (n_wl, n_h)
    return AtmosphericOpticalProperties(
        wavelength=np.linspace(300.0, 310.0, n_wl) * u.nm,
        altitude=np.linspace(1000.0, 3000.0, n_h) * u.m,
        tau_molecular=np.tile(np.array([0.9, 0.5, 0.1])[:n_h], (n_wl, 1)),
        tau_aerosol=np.zeros(shape),
        tau_absorption=np.zeros(shape),
        aerosol_ssa=np.full(shape, 0.5),
        aerosol_asymmetry=np.full(shape, 0.0),
    )


def test_total_optical_depth():
    properties = _exponential_properties()
    total = properties.total_optical_depth()
    assert total.shape == (101, 40)
    assert_quantity_allclose(
        total, _analytic_total(np.arange(300.0, 401.0)[:, None], _altitude_m()[None, :]), rtol=1e-12
    )


def test_properties_store_metadata_and_depolarization():
    properties = _exponential_properties()
    assert properties.depolarization == pytest.approx(0.02)
    assert properties.metadata["backend"] == "synthetic"
    assert properties.phase_angle.unit == u.deg
    assert properties.wavelength.unit == u.nm
    assert properties.altitude.unit == u.m


def test_interpolate_wavelength_linear_and_exact_at_nodes():
    properties = _exponential_properties(np.array([300.0, 400.0]))
    middle = properties.interpolate_wavelength([350.0] * u.nm)
    expected = 0.5 * (properties.tau_molecular[0] + properties.tau_molecular[1])
    assert middle.tau_molecular.shape == (1, properties.altitude.size)
    assert middle.tau_molecular[0] == pytest.approx(expected, rel=1e-12)
    assert middle.aerosol_phase.shape == (1, properties.altitude.size, N_ANGLES)
    assert middle.depolarization == pytest.approx(0.02)
    assert middle.metadata == properties.metadata
    nodes = properties.interpolate_wavelength([300.0, 400.0] * u.nm)
    assert np.array_equal(nodes.tau_aerosol, properties.tau_aerosol[[0, 1]])


def test_interpolate_wavelength_without_phase():
    properties = _minimal_properties()
    resolved = properties.interpolate_wavelength([305.0] * u.nm)
    assert resolved.aerosol_phase is None
    assert resolved.phase_angle is None


def test_interpolate_single_wavelength_source_raises():
    properties = _minimal_properties(n_wl=1, n_h=3)
    with pytest.raises(ValueError, match="at least two tabulated values"):
        properties.interpolate_wavelength([300.0] * u.nm)


def test_interpolate_wavelength_outside_range_raises():
    properties = _minimal_properties()
    with pytest.raises(ValueError, match="outside the tabulated range"):
        properties.interpolate_wavelength([299.0] * u.nm)


def test_transmission_table_layout(tmp_test_directory):
    properties = _exponential_properties()
    file_name = f"{tmp_test_directory}/trans.dat"
    write_obdeect_transmission_table(properties, file_name, observation_level=2000.0 * u.m)
    lines = Path(file_name).read_text(encoding="utf-8").splitlines()
    assert "# backend = synthetic" in lines
    header = [line for line in lines if line.startswith("# H2=")]
    assert len(header) == 1
    assert header[0].startswith("# H2= 2.000, H1=")
    heights_km = [float(value) for value in header[0].split("H1=")[1].split()]
    assert len(heights_km) == 39  # All table altitudes above 2 km.
    assert heights_km[0] == pytest.approx(_altitude_m()[1] / 1000.0, abs=5e-4)
    rows = _read_data_rows(file_name)
    assert len(rows) == 101  # Integer nm 300 to 400.
    assert int(rows[0][0]) == 300
    assert int(rows[-1][0]) == 400
    assert all(len(row) == 1 + len(heights_km) for row in rows)


def test_transmission_table_depth_values(tmp_test_directory):
    """Depths at the table nodes equal total(H2) - total(h); depth is zero at H2 and increases."""
    properties = _exponential_properties()
    file_name = f"{tmp_test_directory}/trans.dat"
    write_obdeect_transmission_table(properties, file_name, observation_level=2000.0 * u.m)
    rows = _read_data_rows(file_name)
    altitudes = _altitude_m()[1:]
    for row in rows[::25]:
        wavelength = float(row[0])
        values = np.array([float(value) for value in row[1:]])
        expected = _analytic_total(wavelength, 2000.0) - _analytic_total(wavelength, altitudes)
        assert np.allclose(values, expected, rtol=0, atol=2e-6)
        assert np.all(np.diff(values) >= 0)
        assert values[0] >= 0


def test_transmission_table_zero_depth_at_observation_level(tmp_test_directory):
    """With altitude-independent optical depth, every depth at H2 is zero."""
    shape = (2, 3)
    properties = AtmosphericOpticalProperties(
        wavelength=[300.0, 301.0] * u.nm,
        altitude=[2000.0, 3000.0, 4000.0] * u.m,
        tau_molecular=np.full(shape, 0.3),
        tau_aerosol=np.zeros(shape),
        tau_absorption=np.zeros(shape),
        aerosol_ssa=np.full(shape, 0.5),
        aerosol_asymmetry=np.zeros(shape),
    )
    file_name = f"{tmp_test_directory}/flat.dat"
    write_obdeect_transmission_table(properties, file_name, observation_level=2000.0 * u.m)
    rows = _read_data_rows(file_name)
    assert all(float(value) == pytest.approx(0.0, abs=1e-9) for row in rows for value in row[1:])


def test_transmission_table_explicit_heights(tmp_test_directory):
    properties = _exponential_properties()
    file_name = f"{tmp_test_directory}/trans.dat"
    grid = _altitude_m()
    heights = grid[[5, 10]] * u.m
    write_obdeect_transmission_table(
        properties, file_name, observation_level=2000.0 * u.m, heights=heights
    )
    header = next(
        line
        for line in Path(file_name).read_text(encoding="utf-8").splitlines()
        if line.startswith("# H2=")
    )
    assert header.split("H1=")[1].split() == [f"{h / 1000:.3f}" for h in grid[[5, 10]]]
    row = _read_data_rows(file_name)[0]
    assert float(row[1]) == pytest.approx(
        _analytic_total(300.0, 2000.0) - _analytic_total(300.0, grid[5]), abs=2e-6
    )


def test_transmission_table_interpolates_observation_level(tmp_test_directory):
    """Observation level between grid nodes: total interpolated linearly in log(altitude)."""
    properties = _exponential_properties()
    file_name = f"{tmp_test_directory}/trans.dat"
    grid = _altitude_m()
    level = 2100.0
    write_obdeect_transmission_table(
        properties, file_name, observation_level=level * u.m, heights=[grid[5]] * u.m
    )
    row = _read_data_rows(file_name)[0]
    weight = (np.log(level) - np.log(grid[0])) / (np.log(grid[1]) - np.log(grid[0]))
    total_grid = _analytic_total(300.0, grid[:2])
    total_level = total_grid[0] + weight * (total_grid[1] - total_grid[0])
    assert float(row[1]) == pytest.approx(total_level - _analytic_total(300.0, grid[5]), abs=2e-6)


def test_transmission_table_integer_grid_from_fractional_wavelengths(tmp_test_directory):
    properties = _minimal_properties(n_wl=2, n_h=3)
    properties = AtmosphericOpticalProperties(
        wavelength=[300.4, 301.6] * u.nm,
        altitude=properties.altitude,
        tau_molecular=properties.tau_molecular,
        tau_aerosol=properties.tau_aerosol,
        tau_absorption=properties.tau_absorption,
        aerosol_ssa=properties.aerosol_ssa,
        aerosol_asymmetry=properties.aerosol_asymmetry,
        metadata={"backend": "x"},
    )
    file_name = f"{tmp_test_directory}/trans.dat"
    write_obdeect_transmission_table(properties, file_name, observation_level=1000.0 * u.m)
    rows = _read_data_rows(file_name)
    assert [int(row[0]) for row in rows] == [301]
    assert "# backend = x" in Path(file_name).read_text(encoding="utf-8")


def test_transmission_table_errors(tmp_test_directory):
    file_name = f"{tmp_test_directory}/trans.dat"
    properties = _exponential_properties(np.array([999.0, 1000.0]))
    with pytest.raises(ValueError, match="limited to wavelengths below 1000 nm"):
        write_obdeect_transmission_table(properties, file_name, observation_level=2000.0 * u.m)
    properties = _exponential_properties()
    with pytest.raises(ValueError, match="needs 1 to 99 heights"):
        write_obdeect_transmission_table(
            properties,
            file_name,
            observation_level=2000.0 * u.m,
            heights=np.arange(3000.0, 3000.0 + 100 * 10.0, 10.0) * u.m,
        )
    dense = _exponential_properties(altitude_m=np.geomspace(2000.0, 100000.0, 120))
    with pytest.raises(ValueError, match="needs 1 to 99 heights"):
        write_obdeect_transmission_table(dense, file_name, observation_level=2000.0 * u.m)
    with pytest.raises(ValueError, match="must be above the observation level"):
        write_obdeect_transmission_table(
            properties, file_name, observation_level=2000.0 * u.m, heights=[1500.0, 3000.0] * u.m
        )
    with pytest.raises(ValueError, match="strictly ascending"):
        write_obdeect_transmission_table(
            properties, file_name, observation_level=2000.0 * u.m, heights=[3000.0, 2500.0] * u.m
        )
    with pytest.raises(ValueError, match="outside the tabulated range"):
        write_obdeect_transmission_table(properties, file_name, observation_level=1000.0 * u.m)
    with pytest.raises(ValueError, match="No integer wavelength"):
        write_obdeect_transmission_table(
            _wavelength_only_properties([300.2, 300.8]), file_name, observation_level=1000.0 * u.m
        )


def _wavelength_only_properties(wavelength):
    base = _minimal_properties(n_wl=2, n_h=3)
    return AtmosphericOpticalProperties(
        wavelength=wavelength * u.nm,
        altitude=base.altitude,
        tau_molecular=base.tau_molecular,
        tau_aerosol=base.tau_aerosol,
        tau_absorption=base.tau_absorption,
        aerosol_ssa=base.aerosol_ssa,
        aerosol_asymmetry=base.aerosol_asymmetry,
    )


def test_scattering_table_rows(tmp_test_directory):
    properties = _exponential_properties(np.array([340.0, 341.0]), _altitude_m()[:5])
    file_name = f"{tmp_test_directory}/scat.dat"
    write_obdeect_scattering_table(properties, file_name)
    text = Path(file_name).read_text(encoding="utf-8").splitlines()
    assert "# wavelength_nm altitude_km tau_mol tau_aer ssa g" in text
    assert "depolarization 0.02" in text
    assert any(line.startswith("angles_deg 0 ") and line.endswith(" 180") for line in text)
    rows = [
        line.split()
        for line in text
        if line
        and not line.startswith("#")
        and not line.startswith(("phase", "angles_deg", "depolarization"))
    ]
    assert len(rows) == 2 * 5
    first = rows[0]
    assert first[0] == "340"
    assert float(first[1]) == pytest.approx(_altitude_m()[0] / 1000.0, rel=1e-7)
    assert float(first[2]) == pytest.approx(0.00865 * 0.340**-4.05 * np.exp(-2.0 / 8.0), rel=1e-7)
    assert float(first[3]) == pytest.approx(
        0.1 * (340.0 / 550.0) ** -1.3 * np.exp(-2.0 / 1.5), rel=1e-7
    )
    assert float(first[4]) == pytest.approx(0.9, rel=1e-12)
    assert float(first[5]) == pytest.approx(HG_G, rel=1e-12)
    assert [int(row[0]) for row in rows] == [340] * 5 + [341] * 5


def test_scattering_table_phase_block(tmp_test_directory):
    properties = _exponential_properties(np.array([340.0, 341.0]), _altitude_m()[:4])
    file_name = f"{tmp_test_directory}/scat.dat"
    write_obdeect_scattering_table(properties, file_name)
    phase_lines = [
        line.split()
        for line in Path(file_name).read_text(encoding="utf-8").splitlines()
        if line.startswith("phase ")
    ]
    assert len(phase_lines) == 2 * 4
    assert phase_lines[0][1] == "340"
    assert len(phase_lines[0]) == 3 + N_ANGLES
    values = np.array([float(value) for value in phase_lines[0][3:]])
    assert len(values) == N_ANGLES
    angles = np.radians(np.linspace(0.0, 180.0, N_ANGLES))
    cos_theta = np.cos(angles)
    integral = np.trapezoid(values[::-1], cos_theta[::-1])
    assert 2 * np.pi * integral == pytest.approx(1.0, rel=1e-3)


def test_scattering_writer_rejects_unsupported_depolarization(tmp_test_directory):
    properties = _minimal_properties().with_depolarization(0.5)
    with pytest.raises(ValueError, match=r"depolarization.*0\.5"):
        write_obdeect_scattering_table(properties, Path(tmp_test_directory) / "scat.dat")


def test_scattering_writer_rejects_invalid_phase(tmp_test_directory):
    properties = _exponential_properties(np.array([340.0, 341.0]), _altitude_m()[:4])
    properties.aerosol_phase[0, 0, 0] = 0.0
    with pytest.raises(ValueError, match=r"phase.*positive"):
        write_obdeect_scattering_table(properties, Path(tmp_test_directory) / "scat.dat")
    properties.aerosol_phase[:] = 1.0
    with pytest.raises(ValueError, match=r"phase.*normal"):
        write_obdeect_scattering_table(properties, Path(tmp_test_directory) / "scat.dat")


def test_scattering_table_without_phase(tmp_test_directory):
    file_name = f"{tmp_test_directory}/scat.dat"
    write_obdeect_scattering_table(_minimal_properties(), file_name)
    text = Path(file_name).read_text(encoding="utf-8")
    assert "phase " not in text
    assert "angles_deg" not in text
    assert "depolarization" not in text


def test_scattering_table_errors(tmp_test_directory):
    file_name = f"{tmp_test_directory}/scat.dat"
    base = _exponential_properties(np.array([340.0, 341.0]), _altitude_m()[:4])
    shifted = AtmosphericOpticalProperties(
        wavelength=base.wavelength,
        altitude=base.altitude,
        tau_molecular=base.tau_molecular,
        tau_aerosol=base.tau_aerosol,
        tau_absorption=base.tau_absorption,
        aerosol_ssa=base.aerosol_ssa,
        aerosol_asymmetry=base.aerosol_asymmetry,
        phase_angle=np.linspace(10.0, 170.0, 9) * u.deg,
        aerosol_phase=np.ones((2, 4, 9)),
    )
    with pytest.raises(ValueError, match=r"need angles from 0\.0 to 180\.0 deg"):
        write_obdeect_scattering_table(shifted, file_name)
    with pytest.raises(ValueError, match="No integer wavelength"):
        write_obdeect_scattering_table(_wavelength_only_properties([300.2, 300.8]), file_name)


def test_ecsv_round_trip_is_lossless(tmp_test_directory):
    properties = _exponential_properties(np.array([300.25, 320.5, 400.75]), _altitude_m()[:6])
    file_name = f"{tmp_test_directory}/optical.ecsv"
    write_optical_properties(properties, file_name)
    restored = read_optical_properties(file_name)
    assert np.array_equal(restored.wavelength.to_value(u.nm), properties.wavelength.value)
    assert np.array_equal(restored.altitude.to_value(u.m), properties.altitude.value)
    for name in (
        "tau_molecular",
        "tau_aerosol",
        "tau_absorption",
        "aerosol_ssa",
        "aerosol_asymmetry",
        "aerosol_phase",
    ):
        assert np.array_equal(getattr(restored, name), getattr(properties, name)), name
    assert np.array_equal(restored.phase_angle.to_value(u.deg), properties.phase_angle.value)
    assert restored.depolarization == properties.depolarization
    assert restored.metadata == properties.metadata


def test_ecsv_round_trip_without_phase(tmp_test_directory):
    properties = _minimal_properties()
    file_name = f"{tmp_test_directory}/optical.ecsv"
    write_optical_properties(properties, file_name)
    restored = read_optical_properties(file_name)
    assert restored.aerosol_phase is None
    assert restored.phase_angle is None
    assert restored.metadata == {}
    assert np.array_equal(restored.tau_molecular, properties.tau_molecular)


def test_read_optical_properties_rejects_incomplete_grid(tmp_test_directory):
    file_name = f"{tmp_test_directory}/optical.ecsv"
    write_optical_properties(_minimal_properties(n_wl=2, n_h=3), file_name)
    lines = Path(file_name).read_text(encoding="utf-8").splitlines()
    broken_file = f"{tmp_test_directory}/broken.ecsv"
    with open(broken_file, "w", encoding="utf-8") as file:
        file.write("\n".join(lines[:-1]) + "\n")  # Drop the last (wavelength, altitude) row.
    with pytest.raises(ValueError, match="not a complete"):
        read_optical_properties(broken_file)


def test_ecsv_rejects_misaligned_coordinates(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "optical.ecsv"
    write_optical_properties(_minimal_properties(), file_name)
    table = Table.read(file_name, format="ascii.ecsv")
    table["altitude"][4] = table["altitude"][3]
    table.write(file_name, format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match=r"complete.*grid"):
        read_optical_properties(file_name)


def test_ecsv_converts_coordinate_units(tmp_test_directory):
    file_name = Path(tmp_test_directory) / "optical.ecsv"
    properties = _minimal_properties()
    write_optical_properties(properties, file_name)
    table = Table.read(file_name, format="ascii.ecsv")
    table["wavelength"] = table["wavelength"].quantity.to(u.um)
    table["altitude"] = table["altitude"].quantity.to(u.km)
    table.write(file_name, format="ascii.ecsv", overwrite=True)
    restored = read_optical_properties(file_name)
    assert_quantity_allclose(restored.wavelength, properties.wavelength)
    assert_quantity_allclose(restored.altitude, properties.altitude)


def test_ecsv_backend_interpolates_wavelength_and_altitude(tmp_test_directory):
    properties = _exponential_properties(np.array([300.0, 400.0]), _altitude_m()[:5])
    file_name = f"{tmp_test_directory}/optical.ecsv"
    write_optical_properties(properties, file_name)
    backend = EcsvBackend(file_name)
    assert backend.name == "ecsv"
    altitudes = [np.sqrt(_altitude_m()[0] * _altitude_m()[1])] * u.m  # Geometric mean.
    result = backend.compute([350.0] * u.nm, altitudes)
    assert result.tau_molecular.shape == (1, 1)
    expected_mol = 0.5 * (
        0.5 * (properties.tau_molecular[0, 0] + properties.tau_molecular[0, 1])
        + 0.5 * (properties.tau_molecular[1, 0] + properties.tau_molecular[1, 1])
    )
    assert result.tau_molecular[0, 0] == pytest.approx(expected_mol, rel=1e-12)
    assert result.aerosol_ssa[0, 0] == pytest.approx(0.9, rel=1e-12)
    assert result.aerosol_phase.shape == (1, 1, N_ANGLES)
    assert result.depolarization == pytest.approx(0.02)


def test_ecsv_backend_errors_outside_grid(tmp_test_directory):
    properties = _exponential_properties(np.array([300.0, 400.0]), _altitude_m()[:5])
    file_name = f"{tmp_test_directory}/optical.ecsv"
    write_optical_properties(properties, file_name)
    backend = EcsvBackend(file_name)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        backend.compute([500.0] * u.nm, [3000.0] * u.m)
    with pytest.raises(ValueError, match="outside the tabulated range"):
        backend.compute([350.0] * u.nm, [1000.0] * u.m)


def test_validation_shapes_and_grids():
    base = _minimal_properties()
    kwargs = {
        "wavelength": base.wavelength,
        "altitude": base.altitude,
        "tau_molecular": base.tau_molecular,
        "tau_aerosol": base.tau_aerosol,
        "tau_absorption": base.tau_absorption,
        "aerosol_ssa": base.aerosol_ssa,
        "aerosol_asymmetry": base.aerosol_asymmetry,
    }
    with pytest.raises(ValueError, match="non-empty 1-D grid"):
        AtmosphericOpticalProperties(**{**kwargs, "wavelength": [] * u.nm})
    with pytest.raises(ValueError, match="strictly ascending"):
        AtmosphericOpticalProperties(**{**kwargs, "wavelength": [310.0, 300.0] * u.nm})
    with pytest.raises(ValueError, match="strictly ascending"):
        AtmosphericOpticalProperties(**{**kwargs, "altitude": [3000.0, 1000.0] * u.m})
    with pytest.raises(ValueError, match="must be positive"):
        AtmosphericOpticalProperties(**{**kwargs, "altitude": [0.0, 1000.0] * u.m})
    with pytest.raises(ValueError, match="must have shape"):
        AtmosphericOpticalProperties(**{**kwargs, "tau_aerosol": np.zeros((3, 3))})
    with pytest.raises(ValueError, match="must be finite"):
        AtmosphericOpticalProperties(**{**kwargs, "tau_aerosol": np.full((2, 3), np.nan)})


def test_validation_optical_depth_and_ranges():
    base = _minimal_properties()
    kwargs = {
        "wavelength": base.wavelength,
        "altitude": base.altitude,
        "tau_molecular": base.tau_molecular,
        "tau_aerosol": base.tau_aerosol,
        "tau_absorption": base.tau_absorption,
        "aerosol_ssa": base.aerosol_ssa,
        "aerosol_asymmetry": base.aerosol_asymmetry,
    }
    with pytest.raises(ValueError, match="non-negative and non-increasing"):
        AtmosphericOpticalProperties(**{**kwargs, "tau_molecular": -np.ones((2, 3))})
    increasing = np.tile([0.1, 0.5, 0.9], (2, 1))
    with pytest.raises(ValueError, match="non-negative and non-increasing"):
        AtmosphericOpticalProperties(**{**kwargs, "tau_aerosol": increasing})
    with pytest.raises(ValueError, match=r"aerosol_ssa must be in \[0, 1\]"):
        AtmosphericOpticalProperties(**{**kwargs, "aerosol_ssa": np.full((2, 3), 1.5)})
    with pytest.raises(ValueError, match=r"aerosol_asymmetry must be in \(-1, 1\)"):
        AtmosphericOpticalProperties(**{**kwargs, "aerosol_asymmetry": np.ones((2, 3))})
    with pytest.raises(ValueError, match="depolarization must be in"):
        AtmosphericOpticalProperties(**kwargs, depolarization=1.5)


def test_validation_phase():
    base = _minimal_properties()
    kwargs = {
        "wavelength": base.wavelength,
        "altitude": base.altitude,
        "tau_molecular": base.tau_molecular,
        "tau_aerosol": base.tau_aerosol,
        "tau_absorption": base.tau_absorption,
        "aerosol_ssa": base.aerosol_ssa,
        "aerosol_asymmetry": base.aerosol_asymmetry,
    }
    angles = np.array([0.0, 90.0, 180.0]) * u.deg
    with pytest.raises(ValueError, match="must be given together"):
        AtmosphericOpticalProperties(**kwargs, phase_angle=angles)
    with pytest.raises(ValueError, match="must be given together"):
        AtmosphericOpticalProperties(**kwargs, aerosol_phase=np.ones((2, 3, 3)))
    with pytest.raises(ValueError, match="strictly ascending"):
        AtmosphericOpticalProperties(
            **kwargs, phase_angle=[0.0, 180.0, 90.0] * u.deg, aerosol_phase=np.ones((2, 3, 3))
        )
    with pytest.raises(ValueError, match=r"phase_angle must lie within \[0.0, 180.0\]"):
        AtmosphericOpticalProperties(
            **kwargs, phase_angle=[0.0, 200.0] * u.deg, aerosol_phase=np.ones((2, 3, 2))
        )
    with pytest.raises(ValueError, match="must have shape"):
        AtmosphericOpticalProperties(**kwargs, phase_angle=angles, aerosol_phase=np.ones((2, 3, 2)))
    with pytest.raises(ValueError, match="must be non-negative"):
        AtmosphericOpticalProperties(
            **kwargs, phase_angle=angles, aerosol_phase=-np.ones((2, 3, 3))
        )


class _DummyBackend(RadiativeTransferBackend):
    name = "dummy"

    def __init__(self, scale=1.0):
        self.scale = scale

    def compute(self, wavelength, altitude):
        return _minimal_properties()


@pytest.fixture
def isolated_registry(monkeypatch):
    """Registry with the built-in ECSV backend only, as after a fresh import."""
    monkeypatch.setattr(optical_tables, "_BACKENDS", {EcsvBackend.name: EcsvBackend})
    monkeypatch.delitem(sys.modules, optical_tables.SIXS_BACKEND_MODULE, raising=False)
    return optical_tables._BACKENDS


def test_with_depolarization_returns_copy_and_validates():
    properties = _exponential_properties()
    updated = properties.with_depolarization(0.05)
    assert updated.depolarization == pytest.approx(0.05)
    assert properties.depolarization == pytest.approx(0.02)
    assert updated is not properties
    assert updated.tau_molecular is properties.tau_molecular
    assert updated.phase_angle is properties.phase_angle
    assert _minimal_properties().with_depolarization(0).depolarization == pytest.approx(0.0)
    with pytest.raises(ValueError, match="depolarization must be in"):
        properties.with_depolarization(-0.1)
    with pytest.raises(ValueError, match="depolarization must be in"):
        properties.with_depolarization(1.1)


def test_with_depolarization_writes_scattering_line(tmp_test_directory):
    file_name = f"{tmp_test_directory}/scat.dat"
    properties = _minimal_properties().with_depolarization(0.03)
    write_obdeect_scattering_table(properties, file_name)
    assert "depolarization 0.03" in Path(file_name).read_text(encoding="utf-8").splitlines()


def test_vertical_transmission_from_top_to_observation_level():
    properties = _exponential_properties(np.array([300.0, 400.0]))
    grid = _altitude_m()
    level = 2100.0
    transmission = properties.vertical_transmission(level * u.m)
    assert transmission.shape == (2,)
    weight = (np.log(level) - np.log(grid[0])) / (np.log(grid[1]) - np.log(grid[0]))
    for index, wavelength in enumerate([300.0, 400.0]):
        total = _analytic_total(wavelength, grid[:2])
        at_level = total[0] + weight * (total[1] - total[0])
        depth = at_level - _analytic_total(wavelength, grid[-1])
        assert transmission[index] == pytest.approx(np.exp(-depth), rel=1e-9)
    assert np.all((transmission > 0) & (transmission <= 1))


def test_vertical_transmission_is_one_for_flat_layer_without_depth():
    shape = (2, 3)
    properties = AtmosphericOpticalProperties(
        wavelength=[300.0, 301.0] * u.nm,
        altitude=[2000.0, 3000.0, 4000.0] * u.m,
        tau_molecular=np.zeros(shape),
        tau_aerosol=np.zeros(shape),
        tau_absorption=np.zeros(shape),
        aerosol_ssa=np.full(shape, 0.5),
        aerosol_asymmetry=np.zeros(shape),
    )
    assert properties.vertical_transmission(2000.0 * u.m) == pytest.approx([1.0, 1.0])


def test_vertical_transmission_level_outside_grid_raises():
    properties = _exponential_properties()
    with pytest.raises(ValueError, match="outside the tabulated range"):
        properties.vertical_transmission(1.0 * u.m)


def test_get_backend_ecsv(isolated_registry, tmp_test_directory):
    backend = get_backend("ecsv", file_name=f"{tmp_test_directory}/x.ecsv")
    assert isinstance(backend, EcsvBackend)
    assert backend.file_name.name == "x.ecsv"


def test_register_and_get_backend(isolated_registry):
    register_backend("dummy", _DummyBackend)
    backend = get_backend("dummy", scale=2.0)
    assert isinstance(backend, _DummyBackend)
    assert backend.scale == pytest.approx(2.0)
    assert isinstance(backend.compute([300.0] * u.nm, [1000.0] * u.m), AtmosphericOpticalProperties)


def test_get_backend_unknown_name(isolated_registry):
    with pytest.raises(ValueError, match=r"Unknown atmospheric backend 'modtran'.*'ecsv'"):
        get_backend("modtran")


def test_get_backend_imports_sixs_module_lazily(isolated_registry, monkeypatch):
    """The 6S module is imported on first request; its import registers the backend."""
    imported = []

    def fake_import(module_name):
        imported.append(module_name)
        register_backend(optical_tables.SIXS_BACKEND_NAME, _DummyBackend)

    monkeypatch.setattr(optical_tables.importlib, "import_module", fake_import)
    assert isinstance(get_backend("6s"), _DummyBackend)
    assert imported == [optical_tables.SIXS_BACKEND_MODULE]


def test_get_backend_6s_not_registered_by_module(isolated_registry, monkeypatch):
    """If the 6S module does not register a backend, the request fails with the known names."""
    monkeypatch.setattr(optical_tables.importlib, "import_module", lambda name: None)
    with pytest.raises(ValueError, match="Known backends"):
        get_backend("6s")


def test_get_backend_6s_skips_import_when_registered(isolated_registry, monkeypatch):
    register_backend(optical_tables.SIXS_BACKEND_NAME, _DummyBackend)

    def fail_import(module_name):
        raise AssertionError(f"unexpected import of {module_name}")

    monkeypatch.setattr(optical_tables.importlib, "import_module", fail_import)
    assert isinstance(get_backend("6s"), _DummyBackend)


def test_abstract_backend_cannot_be_instantiated():
    with pytest.raises(TypeError):
        RadiativeTransferBackend()
