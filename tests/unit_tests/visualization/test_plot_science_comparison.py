import math
from pathlib import Path

import astropy.units as u
import numpy as np
import pytest
from astropy.table import Table
from matplotlib.backends.backend_pdf import PdfPages

from simtools.visualization import plot_science_comparison as plots

EMPTY_ECSV = "empty.ecsv"

ASCII_ECSV = "ascii.ecsv"
B_ECSV = "b.ecsv"
BASELINE = "baseline"
C_ECSV = "c.ecsv"
CANDIDATE = "candidate"
D_ECSV = "d.ecsv"
MISSING_ECSV = "missing.ecsv"
RELATIVE = "relative"
REPORT_PDF = "report.pdf"
SHOWER = "shower"
TITLE = "title"

LABELS = ("v1", "v2")
COMPONENT_COLUMNS = ("ref", "masts", "filt", "pixel", "qe", "atm_trans", "eff", "eff_atm")


def write_table(path, scale=1.0, wavelength=(300.0, 400.0, 500.0), legacy=False, ecsv=True):
    table = Table({"wl": np.array(wavelength) * u.nm})
    for column in COMPONENT_COLUMNS:
        table[column] = np.full(len(wavelength), 0.5 * scale)
    table["C4" if legacy else "cherenkov_detected"] = np.array([1.0, 2.0, 0.0]) * scale
    table["nsb_detected"] = np.array([1.0, 1.0, 1.0]) * scale
    table.write(path, format=ASCII_ECSV if ecsv else "ascii.basic", overwrite=True)
    return path


def test_plot_metric_changes_writes_one_page(tmp_test_directory):
    output = Path(tmp_test_directory) / REPORT_PDF
    changes = {
        "cam_eff": {BASELINE: 0.5, CANDIDATE: 0.6, "unit": "", RELATIVE: 0.2},
        "nsb_rate": {BASELINE: 0.0, CANDIDATE: 0.1, "unit": "GHz", RELATIVE: None},
    }
    with PdfPages(output) as pdf:
        plots.plot_metric_changes(changes, TITLE, pdf)
    assert output.stat().st_size > 0


def test_plot_spectra_compares_common_channels(tmp_test_directory):
    root = Path(tmp_test_directory)
    baseline = {
        SHOWER: write_table(root / "b_shower.ecsv", legacy=True, ecsv=False),
        "nsb": write_table(root / "b_nsb.ecsv"),
        "muon": write_table(root / "b_muon.ecsv", wavelength=(1, 2, 3)),
    }
    candidate = {
        SHOWER: write_table(root / "c_shower.ecsv", scale=1.1),
        "nsb": write_table(root / "c_nsb.ecsv", scale=0.9),
        "muon": write_table(root / "c_muon.ecsv"),
        "extra": write_table(root / "c_extra.ecsv"),
    }
    with PdfPages(root / REPORT_PDF) as pdf:
        diagnostics = plots.plot_spectra(baseline, candidate, TITLE, LABELS, pdf)
    assert diagnostics == ["Cannot compare muon spectrum: wavelength grids differ"]
    assert (root / REPORT_PDF).stat().st_size > 0


def test_plot_spectra_without_comparable_tables_writes_nothing(tmp_test_directory):
    root = Path(tmp_test_directory)
    with PdfPages(root / REPORT_PDF) as pdf:
        assert plots.plot_spectra({}, {}, TITLE, LABELS, pdf) == []
        diagnostics = plots.plot_spectra(
            {"nsb": root / MISSING_ECSV}, {"nsb": root / MISSING_ECSV}, TITLE, LABELS, pdf
        )
    assert diagnostics[0].startswith("Cannot compare nsb spectrum")


def test_plot_components_writes_absolute_and_relative_pages(tmp_test_directory):
    root = Path(tmp_test_directory)
    baseline = write_table(root / B_ECSV)
    candidate = write_table(root / C_ECSV, scale=1.2)
    with PdfPages(root / REPORT_PDF) as pdf:
        assert plots.plot_components(baseline, candidate, TITLE, LABELS, pdf) is None
        assert pdf.get_pagecount() == 2


def test_component_page_limits_absolute_efficiency_to_one():
    wavelength = np.array([300.0, 400.0])
    groups = {"Mirrors": {"Reflectivity": (np.array([0.2, 0.5]), np.array([0.3, 0.6]))}}
    absolute = plots._component_page(wavelength, u.nm, groups, False, LABELS)
    relative = plots._component_page(wavelength, u.nm, groups, True, LABELS)
    assert math.isclose(absolute.axes[0].get_ylim()[1], 1.0)
    assert relative.axes[0].get_ylim()[1] > 1.0


def test_plot_components_reports_missing_or_invalid_columns(tmp_test_directory):
    root = Path(tmp_test_directory)
    baseline = write_table(root / B_ECSV)
    table = Table.read(baseline)
    table.remove_column("qe")
    table.write(root / C_ECSV, format=ASCII_ECSV)
    table["qe"] = [np.nan, 1.0, 1.0]
    table.write(root / D_ECSV, format=ASCII_ECSV)
    with PdfPages(root / REPORT_PDF) as pdf:
        missing = plots.plot_components(baseline, root / C_ECSV, TITLE, LABELS, pdf)
        invalid = plots.plot_components(baseline, root / D_ECSV, TITLE, LABELS, pdf)
    assert "qe" in missing
    assert "non-finite values" in invalid


def test_plot_spectra_falls_back_to_optical_throughput(tmp_test_directory):
    root = Path(tmp_test_directory) / "throughput-only"
    root.mkdir()
    tables = {}
    for name, scale in (("b", 1.0), ("c", 1.1)):
        table = Table({"wl": [300.0, 400.0] * u.nm, "eff": [0.2 * scale, 0.3 * scale]})
        path = root / f"{name}.ecsv"
        table.write(path, format=ASCII_ECSV)
        tables[name] = {SHOWER: path}
    with PdfPages(root / REPORT_PDF) as pdf:
        assert plots.plot_spectra(tables["b"], tables["c"], TITLE, LABELS, pdf) == []
        assert pdf.get_pagecount() == 1


def test_read_pair_rejects_invalid_grids_and_columns(tmp_test_directory):
    root = Path(tmp_test_directory) / "grids"
    root.mkdir()
    baseline = write_table(root / B_ECSV)
    descending = write_table(root / C_ECSV, wavelength=(500.0, 400.0, 300.0))
    shifted = write_table(root / D_ECSV, wavelength=(300.0, 400.0, 500.000001))
    with pytest.raises(ValueError, match="strictly increasing"):
        plots._read_pair(descending, baseline)
    with pytest.raises(ValueError, match="grids differ"):
        plots._read_pair(baseline, shifted)
    table = Table({"wl": [300.0, 400.0], "qe": [[0.1, 0.2], [0.3, 0.4]]})
    with pytest.raises(ValueError, match="one-dimensional"):
        plots._curve(table, "qe")
    empty = Table({"wl": np.array([], dtype=float)})
    empty.write(root / EMPTY_ECSV, format=ASCII_ECSV)
    with pytest.raises(ValueError, match="non-empty"):
        plots._read_pair(root / EMPTY_ECSV, baseline)
