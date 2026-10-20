from pathlib import Path

import astropy.units as u
import numpy as np
import pytest

from simtools.visualization import plot_science_comparison as plots

_CAMERA_EFFICIENCY_METRIC = "cam_eff"
_BASELINE_KEY = "baseline"
_CANDIDATE_KEY = "candidate"
_VALUE_KEY = "value"
_REPORT_PDF = "report.pdf"
_PNG_GLOB = "*.png"
_SHOWER_CHANNEL = "shower"

_BASELINE_TABLE_NAME = "base.ecsv"
_CANDIDATE_TABLE_NAME = "candidate.ecsv"
_ECSV_FORMAT = "ascii.ecsv"


def _capture_figures(monkeypatch):
    """Capture plot figures so axes can be inspected after PDF generation."""
    from matplotlib import pyplot as plt

    monkeypatch.setattr(plots, "plt", plt)
    original_subplots = plots.plt.subplots
    figures = []

    def capture_figure(*args, **kwargs):
        figure, axes = original_subplots(*args, **kwargs)
        figures.append(figure)
        return figure, axes

    monkeypatch.setattr(plots.plt, "subplots", capture_figure)
    return figures


def _assert_top_row_wavelength_axes(figure):
    for axis in figure.axes[:3]:
        assert axis.get_xlabel() == "Wavelength (nm)"
        assert axis.get_xticklabels()
        assert all(label.get_visible() for label in axis.get_xticklabels())


def test_plot_comparison_writes_scalar_png_and_vector_products(tmp_test_directory):
    metrics = {
        _CAMERA_EFFICIENCY_METRIC: {
            _BASELINE_KEY: {_VALUE_KEY: 0.5, "unit": ""},
            _CANDIDATE_KEY: {_VALUE_KEY: 0.6},
        },
        "nsb_rate": {
            _BASELINE_KEY: {_VALUE_KEY: 1.0, "unit": "GHz"},
            _CANDIDATE_KEY: {_VALUE_KEY: 1.2},
        },
    }
    output = Path(tmp_test_directory)

    products, diagnostics = plots.plot_comparison(
        metrics,
        {"label": _BASELINE_KEY, "software": {"simtools_version": "0.37"}},
        {"label": _CANDIDATE_KEY, "software": {"simtools_version": "0.38"}},
        output,
    )

    assert {Path(path).suffix for path in products.values()} == {".png", ".pdf"}
    assert all(Path(path).is_file() for path in products.values())
    assert diagnostics == []


def test_plot_comparison_requires_comparable_metrics(tmp_test_directory):
    with pytest.raises(ValueError, match="without comparable metrics"):
        plots.plot_comparison({}, {}, {}, tmp_test_directory)


def test_plot_comparison_titles_pdf_page_as_telescope_efficiency(tmp_test_directory, monkeypatch):
    from matplotlib.backends.backend_pdf import PdfPages

    output = Path(tmp_test_directory)
    report_path = output / _REPORT_PDF
    metrics = {
        _CAMERA_EFFICIENCY_METRIC: {
            _BASELINE_KEY: {_VALUE_KEY: 0.5},
            _CANDIDATE_KEY: {_VALUE_KEY: 0.6},
        }
    }
    figures = _capture_figures(monkeypatch)

    with PdfPages(report_path) as report:
        products, diagnostics = plots.plot_comparison(
            metrics,
            {"label": "v0.37.0", "case": {"telescope": "LSTN-02"}},
            {"label": "v0.39.0"},
            output,
            figure_format=(),
            pdf_pages=report,
        )

    assert products == {}
    assert diagnostics == []
    assert report_path.is_file()
    assert not list(output.glob(_PNG_GLOB))
    assert figures[0]._suptitle.get_text() == "Telescope efficiency: v0.37.0 vs v0.39.0 (LSTN-02)"


def test_wavelength_plot_reports_unequal_grids_without_resampling(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline = root / _BASELINE_TABLE_NAME
    candidate = root / _CANDIDATE_TABLE_NAME
    Table({"wl": [300, 400], "eff": [0.2, 0.3]}).write(baseline, format=_ECSV_FORMAT)
    Table({"wl": [300, 410], "eff": [0.3, 0.4]}).write(candidate, format=_ECSV_FORMAT)

    product, diagnostic = plots.plot_wavelength_comparison(baseline, candidate, root)

    assert product is None
    assert "grids differ" in diagnostic.lower()


def test_wavelength_plot_converts_units_and_writes_difference(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline = Table({"wl": [300, 400] * u.nm, "eff": [0.2, 0.3]})
    candidate = Table({"wl": [3000, 4000] * u.AA, "eff": [0.3, 0.4]})
    baseline_path = root / _BASELINE_TABLE_NAME
    candidate_path = root / _CANDIDATE_TABLE_NAME
    baseline.write(baseline_path, format=_ECSV_FORMAT)
    candidate.write(candidate_path, format=_ECSV_FORMAT)

    product, diagnostic = plots.plot_wavelength_comparison(baseline_path, candidate_path, root)

    assert Path(product).is_file()
    assert diagnostic is None


def test_wavelengths_without_units_use_the_camera_table_nm_contract():
    from astropy.table import Table

    baseline = Table({"wl": [300, 400]})
    candidate = Table({"wl": [300, 400]})

    aligned, diagnostic = plots._aligned_wavelengths(baseline, candidate)

    assert str(aligned[1]) == "nm"
    assert diagnostic is None


def test_wavelength_plot_reads_legacy_whitespace_baseline(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline_path = root / _BASELINE_TABLE_NAME
    candidate_path = root / _CANDIDATE_TABLE_NAME
    baseline_path.write_text("wl eff\n300 0.2\n400 0.3\n", encoding="utf-8")
    Table({"wl": [300, 400] * u.nm, "eff": [0.3, 0.4]}).write(candidate_path, format=_ECSV_FORMAT)

    product, diagnostic = plots.plot_wavelength_comparison(baseline_path, candidate_path, root)

    assert Path(product).is_file()
    assert diagnostic is None


def test_wavelength_comparison_plots_three_channels_and_writes_pdf_page(
    tmp_test_directory, monkeypatch
):
    from astropy.table import Table
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(tmp_test_directory)
    baseline_paths = {}
    candidate_paths = {}
    for channel in (_SHOWER_CHANNEL, "muon", "nsb"):
        baseline_path = root / f"baseline-{channel}.ecsv"
        candidate_path = root / f"candidate-{channel}.ecsv"
        Table({"wl": [300, 400] * u.nm, "eff": [0.2, 0.3]}).write(
            baseline_path, format=_ECSV_FORMAT
        )
        Table({"wl": [300, 400] * u.nm, "eff": [0.3, 0.4]}).write(
            candidate_path, format=_ECSV_FORMAT
        )
        baseline_paths[channel] = baseline_path
        candidate_paths[channel] = candidate_path
    report_path = root / _REPORT_PDF
    figures = _capture_figures(monkeypatch)

    with PdfPages(report_path) as report:
        product, diagnostic = plots.plot_wavelength_comparisons(
            baseline_paths,
            candidate_paths,
            root,
            "v0.37.0",
            "v0.39.0",
            report,
        )

    assert product is None
    assert report_path.stat().st_size > 0
    assert not list(root.glob(_PNG_GLOB))
    assert diagnostic is None
    _assert_top_row_wavelength_axes(figures[0])


def test_efficiency_component_comparison_adds_two_pdf_pages_without_pngs(
    tmp_test_directory, monkeypatch
):
    from astropy.table import Table
    from matplotlib.backends.backend_pdf import PdfPages

    root = Path(tmp_test_directory)
    baseline_path = root / _BASELINE_TABLE_NAME
    candidate_path = root / _CANDIDATE_TABLE_NAME
    columns = {
        "wl": [300, 400],
        "ref": [0.8, 0.9],
        "masts": [0.9, 0.9],
        "filt": [0.7, 0.8],
        "pixel": [0.6, 0.7],
        "qe": [0.5, 0.6],
        "atm_trans": [0.8, 0.9],
        "eff": [0.1, 0.2],
        "eff_atm": [0.08, 0.18],
    }
    Table(columns).write(baseline_path, format=_ECSV_FORMAT)
    candidate_columns = {
        key: np.asarray(value) * 1.01 if key != "wl" else value for key, value in columns.items()
    }
    Table(candidate_columns).write(candidate_path, format=_ECSV_FORMAT)
    report_path = root / _REPORT_PDF
    figures = _capture_figures(monkeypatch)

    with PdfPages(report_path) as report:
        product, diagnostic = plots.plot_efficiency_component_comparisons(
            baseline_path,
            candidate_path,
            root,
            "v0.37.0",
            "v0.39.0",
            report,
        )

    assert product is None
    assert diagnostic is None
    assert report_path.stat().st_size > 0
    assert not list(root.glob(_PNG_GLOB))
    _assert_top_row_wavelength_axes(figures[0])


@pytest.mark.parametrize(
    ("baseline_table", "candidate_table", "message"),
    [
        ({"wavelength": [300], "eff": [0.2]}, {"wl": [300], "eff": [0.3]}, "no wl"),
        ({"wl": [300], "response": [0.2]}, {"wl": [300], "response": [0.3]}, "no eff"),
    ],
)
def test_wavelength_plot_reports_missing_columns(
    tmp_test_directory, baseline_table, candidate_table, message
):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline_path = root / _BASELINE_TABLE_NAME
    candidate_path = root / _CANDIDATE_TABLE_NAME
    Table(baseline_table).write(baseline_path, format=_ECSV_FORMAT)
    Table(candidate_table).write(candidate_path, format=_ECSV_FORMAT)

    product, diagnostic = plots.plot_wavelength_comparison(baseline_path, candidate_path, root)

    assert product is None
    assert message in diagnostic


@pytest.mark.parametrize(
    ("channel", "legacy_column", "current_column"),
    [
        (_SHOWER_CHANNEL, "C4", "cherenkov_detected"),
        ("muon", "C4", "cherenkov_detected"),
        ("nsb", "N4", "nsb_detected"),
    ],
)
def test_spectra_use_detected_light_instead_of_optical_throughput(
    channel, legacy_column, current_column
):
    from astropy.table import Table

    baseline = Table({"wl": [300, 400], "eff": [0.1, 0.2], legacy_column: [2.0, 3.0]})
    candidate = Table({"wl": [300, 400], "eff": [0.2, 0.3], current_column: [3.0, 4.0]})

    arrays, diagnostic = plots._wavelength_arrays(baseline, candidate, channel)

    assert diagnostic is None
    np.testing.assert_allclose(arrays[1], [2.0, 3.0])
    np.testing.assert_allclose(arrays[2], [3.0, 4.0])
    assert "Detected" in arrays[4]


def test_wavelength_alignment_converts_units_for_legacy_unitless_grid():
    from astropy.table import Table

    baseline = Table({"wl": [300, 400]})
    candidate = Table({"wl": [0.3, 0.4] * u.um})

    aligned, diagnostic = plots._aligned_wavelengths(baseline, candidate)

    assert diagnostic is None
    np.testing.assert_allclose(aligned[0], [300, 400])


@pytest.mark.parametrize("wavelengths", [[], [300, 300], [400, 300], [300, np.nan]])
def test_wavelength_alignment_rejects_invalid_grids(wavelengths):
    from astropy.table import Table

    table = Table({"wl": wavelengths})
    aligned, diagnostic = plots._aligned_wavelengths(table, table)

    assert aligned is None
    assert "strictly increasing" in diagnostic


@pytest.mark.parametrize("values", [[0.1, np.nan], [0.1, np.inf], [[0.1], [0.2]]])
def test_spectrum_comparison_rejects_non_finite_or_multidimensional_values(values):
    from astropy.table import Table

    table = Table({"wl": [300, 400], "eff": values})
    arrays, diagnostic = plots._wavelength_arrays(table, table, _SHOWER_CHANNEL)

    assert arrays is None
    assert "finite scalar values" in diagnostic
