from pathlib import Path

import astropy.units as u
import pytest

from simtools.visualization import plot_science_comparison as plots


def test_plot_comparison_writes_scalar_png_and_vector_products(tmp_test_directory):
    metrics = {
        "cam_eff": {"baseline": {"value": 0.5, "unit": ""}, "candidate": {"value": 0.6}},
        "nsb_rate": {
            "baseline": {"value": 1.0, "unit": "GHz"},
            "candidate": {"value": 1.2},
        },
    }
    output = Path(tmp_test_directory)

    products, diagnostics = plots.plot_comparison(
        metrics,
        {"label": "baseline", "software": {"simtools_version": "0.37"}},
        {"label": "candidate", "software": {"simtools_version": "0.38"}},
        output,
    )

    assert {Path(path).suffix for path in products.values()} == {".png", ".pdf"}
    assert all(Path(path).is_file() for path in products.values())
    assert diagnostics == []


def test_plot_comparison_requires_comparable_metrics(tmp_test_directory):
    with pytest.raises(ValueError, match="without comparable metrics"):
        plots.plot_comparison({}, {}, {}, tmp_test_directory)


def test_wavelength_plot_reports_unequal_grids_without_resampling(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline = root / "base.ecsv"
    candidate = root / "candidate.ecsv"
    Table({"wl": [300, 400], "eff": [0.2, 0.3]}).write(baseline, format="ascii.ecsv")
    Table({"wl": [300, 410], "eff": [0.3, 0.4]}).write(candidate, format="ascii.ecsv")

    product, diagnostic = plots.plot_wavelength_comparison(baseline, candidate, root)

    assert product is None
    assert "grids differ" in diagnostic.lower()


def test_wavelength_plot_converts_units_and_writes_difference(tmp_test_directory):
    from astropy.table import Table

    root = Path(tmp_test_directory)
    baseline = Table({"wl": [300, 400] * u.nm, "eff": [0.2, 0.3]})
    candidate = Table({"wl": [3000, 4000] * u.AA, "eff": [0.3, 0.4]})
    baseline_path = root / "base.ecsv"
    candidate_path = root / "candidate.ecsv"
    baseline.write(baseline_path, format="ascii.ecsv")
    candidate.write(candidate_path, format="ascii.ecsv")

    product, diagnostic = plots.plot_wavelength_comparison(baseline_path, candidate_path, root)

    assert Path(product).is_file()
    assert diagnostic is None


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
    baseline_path = root / "base.ecsv"
    candidate_path = root / "candidate.ecsv"
    Table(baseline_table).write(baseline_path, format="ascii.ecsv")
    Table(candidate_table).write(candidate_path, format="ascii.ecsv")

    product, diagnostic = plots.plot_wavelength_comparison(baseline_path, candidate_path, root)

    assert product is None
    assert message in diagnostic
