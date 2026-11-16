"""Tests for the quantum-efficiency envelope of the obdeect chain."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from astropy.table import Column, Table

from simtools.simulation_software import efficiency
from simtools.simulation_software.efficiency import (
    ENVELOPE_FLOOR,
    NODE_WIDTH_NM,
    ArrayEnvelope,
    TelescopeEfficiency,
)


class _Telescope:
    """Telescope model with a QE table and a QE variation (the parts of a model the envelope reads)."""

    def __init__(self, name, table, qe_variation):
        self.name = name
        self.parameters = {
            "quantum_efficiency": {"value": f"qe_{name}.ecsv", "instrument": name},
        }
        self._table = table
        self._qe_variation = qe_variation

    def get_parameter_table(self, parameter_name):
        assert parameter_name == "quantum_efficiency"
        return self._table

    def get_parameter_value(self, parameter_name):
        assert parameter_name == "qe_variation"
        return self._qe_variation


def _qe_table(wavelength, efficiency_values):
    return Table(
        {
            "wavelength": Column(wavelength, unit="nm"),
            "efficiency": Column(efficiency_values, unit="dimensionless"),
        }
    )


def _array(*telescopes, model_version="6.0.0"):
    return SimpleNamespace(
        model_version=model_version,
        telescope_models={f"{t.name}-{index}": t for index, t in enumerate(telescopes)},
    )


def test_headroom_from_qe_variation():
    assert efficiency.headroom(0.0) == pytest.approx(1.0)
    assert efficiency.headroom(0.063) == pytest.approx(1.0 / 1.378)


def test_envelope_setting_defaults_to_off():
    assert efficiency.envelope_setting({}) is False
    assert efficiency.envelope_setting({"efficiency_envelope": "qe"}) is True
    assert efficiency.envelope_setting({"efficiency_envelope": "none"}) is False


def test_envelope_setting_rejects_unknown_value():
    with pytest.raises(ValueError, match=r"Invalid efficiency_envelope option: 'yes'"):
        efficiency.envelope_setting({"efficiency_envelope": "yes"})


def test_sim_telarray_qe_interpolates_linearly_on_grid():
    table = _qe_table([500.0, 300.0], [0.4, 0.2])

    qe = efficiency.sim_telarray_qe(table, grid=np.array([300.0, 400.0, 600.0]))

    assert qe == pytest.approx([0.2, 0.3, 0.4])


def test_sim_telarray_qe_converts_wavelength_units():
    table = Table({"wavelength": Column([0.3, 0.5], unit="micron"), "efficiency": [0.2, 0.4]})

    qe = efficiency.sim_telarray_qe(table, grid=np.array([400.0]))

    assert qe == pytest.approx([0.3])


def test_envelope_bound_is_largest_qe_over_headroom_with_floor():
    first = TelescopeEfficiency("A", "a.dat", np.array([0.2, 0.0, 0.4]), headroom=0.5)
    second = TelescopeEfficiency("B", "b.dat", np.array([0.1, 0.0, 0.2]), headroom=0.8)

    bound = efficiency.envelope_bound([first, second])

    assert bound == pytest.approx([0.4, ENVELOPE_FLOOR, 0.8])


def test_scaled_table_stays_below_headroom():
    telescopes = (
        TelescopeEfficiency("A", "a.dat", np.array([0.2, 0.3, 0.4]), headroom=0.5),
        TelescopeEfficiency("B", "b.dat", np.array([0.1, 0.3, 0.2]), headroom=0.8),
    )
    bound = efficiency.envelope_bound(telescopes)

    for telescope in telescopes:
        scaled = telescope.qe / bound
        assert np.all(scaled <= telescope.headroom + 1e-12)


def test_telescope_efficiencies_one_entry_per_telescope_type():
    lst = _Telescope("LSTN-01", _qe_table([300.0, 500.0], [0.2, 0.4]), qe_variation=0.063)
    mst = _Telescope("MSTN-04", _qe_table([300.0, 500.0], [0.1, 0.3]), qe_variation=0.05)

    types = efficiency.telescope_efficiencies(_array(lst, lst, mst))

    assert [t.name for t in types] == ["LSTN-01", "MSTN-04"]
    assert types[0].table_name == "qe_LSTN-01-LSTN-01.dat"
    assert types[0].headroom == pytest.approx(efficiency.headroom(0.063))
    assert types[1].headroom == pytest.approx(efficiency.headroom(0.05))


def test_telescope_efficiencies_reject_table_without_shared_name():
    telescope = _Telescope("LSTN-01", _qe_table([300.0], [0.2]), qe_variation=0.063)
    telescope.parameters["quantum_efficiency"] = {"value": "qe.dat"}

    with pytest.raises(ValueError, match=r"has no sim_telarray file name"):
        efficiency.telescope_efficiencies(_array(telescope))


def test_array_envelope_bounds_all_telescope_types():
    lst = _Telescope("LSTN-01", _qe_table([300.0, 500.0], [0.2, 0.4]), qe_variation=0.063)
    mst = _Telescope("MSTN-04", _qe_table([300.0, 500.0], [0.5, 0.1]), qe_variation=0.063)

    envelope = efficiency.array_envelope(_array(lst, mst))

    # grid index = wavelength - 100 nm; the bound is the larger QE / headroom of the two types
    assert envelope.bound.shape == efficiency.WAVELENGTH_GRID_NM.shape
    assert envelope.bound[200] == pytest.approx(0.5 / efficiency.headroom(0.063))
    assert envelope.bound[400] == pytest.approx(0.4 / efficiency.headroom(0.063))
    for telescope in envelope.telescopes:
        assert np.all(telescope.qe / envelope.bound <= telescope.headroom + 1e-12)


def test_write_envelope_has_two_nodes_per_bin(tmp_test_directory):
    path = efficiency.write_envelope(
        Path(tmp_test_directory) / "envelope.dat",
        np.array([0.5, 0.25]),
        grid=np.array([100, 101]),
    )

    lines = path.read_text(encoding="utf-8").splitlines()
    assert lines == [
        "100.000000 0.5",
        f"{100 + NODE_WIDTH_NM:.6f} 0.5",
        "101.000000 0.25",
        f"{101 + NODE_WIDTH_NM:.6f} 0.25",
    ]


def test_write_scaled_tables_writes_named_files(tmp_test_directory):
    first = TelescopeEfficiency("A", "qe_a.dat", np.array([0.2, 0.4]), headroom=0.5)
    envelope = ArrayEnvelope(bound=np.array([0.4, 0.8]), telescopes=(first,))

    directory = efficiency.write_scaled_tables(
        Path(tmp_test_directory) / "qe-table", envelope, grid=np.array([100, 101])
    )

    assert directory == Path(tmp_test_directory) / "qe-table"
    lines = (directory / "qe_a.dat").read_text(encoding="utf-8").splitlines()
    assert lines == ["100 0.5", "101 0.5"]


def test_write_scaled_tables_accepts_identical_shared_tables(tmp_test_directory):
    qe = np.array([0.2, 0.4])
    first = TelescopeEfficiency("A", "shared.dat", qe, headroom=0.5)
    second = TelescopeEfficiency("A2", "shared.dat", qe.copy(), headroom=0.5)
    envelope = ArrayEnvelope(bound=np.array([0.4, 0.8]), telescopes=(first, second))

    efficiency.write_scaled_tables(
        Path(tmp_test_directory) / "qe", envelope, grid=np.array([100, 101])
    )

    assert (Path(tmp_test_directory) / "qe" / "shared.dat").is_file()


def test_write_scaled_tables_rejects_shared_name_with_different_efficiency(tmp_test_directory):
    first = TelescopeEfficiency("A", "shared.dat", np.array([0.2, 0.4]), headroom=0.5)
    second = TelescopeEfficiency("B", "shared.dat", np.array([0.1, 0.4]), headroom=0.5)
    envelope = ArrayEnvelope(bound=np.array([0.4, 0.8]), telescopes=(first, second))

    with pytest.raises(ValueError, match=r"share the QE table shared.dat"):
        efficiency.write_scaled_tables(Path(tmp_test_directory) / "qe", envelope)


def test_envelope_directory_per_run_and_model_version(run_context):
    directory = efficiency.envelope_directory(run_context, 3, "6.0.0")

    assert directory == run_context.paths.get_output_directory(["efficiency", "run000003", "6.0.0"])
    assert directory.is_dir()
