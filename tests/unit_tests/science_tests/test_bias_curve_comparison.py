"""Tests for bias-curve comparison helpers."""

from pathlib import Path

import numpy as np
import pytest
from astropy.table import Table

from simtools.science_tests import bias_curve_comparison


def test_max_relative_deviation_interpolates_over_common_range():
    baseline = Table({"threshold": [10, 20, 30], "NSB rate (Hz)": [100.0, 50.0, 20.0]})
    candidate = Table({"threshold": [15, 25, 35], "NSB rate (Hz)": [67.5, 31.5, 4.5]})

    deviation = bias_curve_comparison._max_relative_deviation(baseline, candidate, "NSB rate (Hz)")

    assert deviation[0] == pytest.approx(0.1)
    assert deviation[1] == [15.0, 30.0]


def test_max_relative_deviation_rejects_no_common_range():
    baseline = Table({"threshold": [10, 20], "NSB rate (Hz)": [100.0, 50.0]})
    candidate = Table({"threshold": [30, 40], "NSB rate (Hz)": [30.0, 20.0]})

    with pytest.raises(ValueError, match="no common threshold range"):
        bias_curve_comparison._max_relative_deviation(baseline, candidate, "NSB rate (Hz)")


def test_max_relative_deviation_rejects_missing_curve_rows():
    baseline = Table({"threshold": [10, 20, 30], "NSB rate (Hz)": [100.0, np.nan, 20.0]})
    candidate = Table({"threshold": [10, 20, 30], "NSB rate (Hz)": [90.0, 54.0, 18.0]})

    with pytest.raises(ValueError, match="missing or non-finite"):
        bias_curve_comparison._max_relative_deviation(baseline, candidate, "NSB rate (Hz)")


def test_max_relative_deviation_rejects_zero_baseline_rate():
    baseline = Table({"threshold": [10, 20], "NSB rate (Hz)": [0.0, 50.0]})
    candidate = Table({"threshold": [10, 20], "NSB rate (Hz)": [1.0, 45.0]})

    with pytest.raises(ValueError, match="relative deviation is undefined"):
        bias_curve_comparison._max_relative_deviation(baseline, candidate, "NSB rate (Hz)")


def test_read_curve_requires_columns_and_unique_thresholds(tmp_test_directory):
    path = Path(tmp_test_directory) / "curve.ecsv"
    Table(
        {
            "threshold": [10, 10],
            "NSB rate (Hz)": [2.0, 1.0],
            "Proton rate (Hz)": [3.0, 2.0],
        }
    ).write(path, format="ascii.ecsv")

    with pytest.raises(ValueError, match="unique threshold values"):
        bias_curve_comparison._read_curve(path)


def test_read_curve_rejects_missing_columns_and_negative_rates(tmp_test_directory):
    path = Path(tmp_test_directory) / "curve.ecsv"
    Table({"threshold": [10, 20]}).write(path, format="ascii.ecsv")
    with pytest.raises(ValueError, match="missing columns"):
        bias_curve_comparison._read_curve(path)

    Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [2.0, -1.0],
            "Proton rate (Hz)": [3.0, 2.0],
        }
    ).write(path, format="ascii.ecsv", overwrite=True)
    with pytest.raises(ValueError, match="non-negative"):
        bias_curve_comparison._read_curve(path)


def test_load_results_keys_curves_by_telescope(tmp_test_directory):
    root = Path(tmp_test_directory) / "release"
    table = Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [100.0, 50.0],
            "Proton rate (Hz)": [10.0, 5.0],
        }
    )
    (root / "LSTN-01").mkdir(parents=True)
    table.write(root / "LSTN-01" / "bias_curve.ecsv", format="ascii.ecsv")

    curves = bias_curve_comparison.load_results(root)

    assert list(curves) == ["LSTN-01"]
    assert len(curves["LSTN-01"]) == 2


def test_load_results_requires_explicit_telescope_for_file(tmp_test_directory):
    path = Path(tmp_test_directory) / "bias_curve.ecsv"
    Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [100.0, 50.0],
            "Proton rate (Hz)": [10.0, 5.0],
        }
    ).write(path, format="ascii.ecsv")

    with pytest.raises(ValueError, match="exactly one telescope"):
        bias_curve_comparison.load_results(path)
    assert list(bias_curve_comparison.load_results(path, ["LSTN-01"])) == ["LSTN-01"]


def test_load_results_rejects_empty_duplicate_and_unselected_paths(tmp_test_directory):
    root = Path(tmp_test_directory) / "release"
    with pytest.raises(ValueError, match=r"No bias_curve.ecsv files"):
        bias_curve_comparison.load_results(root)
    for subdirectory in ("first", "second"):
        path = root / subdirectory / "LSTN-01" / "bias_curve.ecsv"
        path.parent.mkdir(parents=True)
        Table(
            {
                "threshold": [10, 20],
                "NSB rate (Hz)": [100.0, 50.0],
                "Proton rate (Hz)": [10.0, 5.0],
            }
        ).write(path, format="ascii.ecsv")
    with pytest.raises(ValueError, match="Multiple bias-curve files"):
        bias_curve_comparison.load_results(root)
    with pytest.raises(ValueError, match=r"No bias_curve.ecsv files"):
        bias_curve_comparison.load_results(root, ["MSTN-01"])


def test_load_results_rejects_wrong_filename(tmp_test_directory):
    path = Path(tmp_test_directory) / "other.ecsv"
    path.write_text("table", encoding="utf-8")
    with pytest.raises(ValueError, match=r"Expected bias_curve.ecsv"):
        bias_curve_comparison.load_results(path, ["LSTN-01"])


def test_compare_results_reports_deviation_and_tolerance():
    baseline = Table(
        {
            "threshold": [10, 20, 30],
            "NSB rate (Hz)": [100.0, 50.0, 20.0],
            "Proton rate (Hz)": [10.0, 5.0, 2.0],
        }
    )
    candidate = Table(
        {
            "threshold": [10, 20, 30],
            "NSB rate (Hz)": [90.0, 50.0, 20.0],
            "Proton rate (Hz)": [10.0, 4.0, 2.0],
        }
    )

    changes = bias_curve_comparison.compare_results(
        baseline, candidate, {"NSB rate (Hz)": 0.05, "Proton rate (Hz)": 0.3}
    )

    assert changes["NSB rate (Hz)"]["max_relative_deviation"] == pytest.approx(0.1)
    assert changes["NSB rate (Hz)"]["advisory"] is True
    assert changes["Proton rate (Hz)"]["advisory"] is False


def test_compare_results_rejects_unknown_tolerance():
    table = Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [100.0, 50.0],
            "Proton rate (Hz)": [10.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="unknown bias-curve metric"):
        bias_curve_comparison.compare_results(table, table, {"unknown": 0.1})


def test_compare_results_accepts_two_zero_rate_curves():
    table = Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [0.0, 0.0],
            "Proton rate (Hz)": [0.0, 0.0],
        }
    )

    changes = bias_curve_comparison.compare_results(table, table)

    assert changes["NSB rate (Hz)"]["max_relative_deviation"] == 0
    assert changes["Proton rate (Hz)"]["max_relative_deviation"] == 0


def test_max_relative_deviation_rejects_non_finite_deviation():
    baseline = Table({"threshold": [10, 20], "NSB rate (Hz)": [1e-308, 1e-308]})
    candidate = Table({"threshold": [10, 20], "NSB rate (Hz)": [1e308, 1e308]})
    with pytest.raises(ValueError, match="relative deviation is not finite"):
        bias_curve_comparison._max_relative_deviation(baseline, candidate, "NSB rate (Hz)")


def test_plot_comparison_adds_report_figure(mocker):
    table = Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [100.0, 50.0],
            "Proton rate (Hz)": [10.0, 5.0],
        }
    )
    pages = mocker.Mock()
    pages.add.side_effect = bias_curve_comparison.plt.close

    bias_curve_comparison.plot_comparison(table, table, "LSTN-01", ("v1", "v2"), pages)

    pages.add.assert_called_once()


def test_plot_comparison_handles_zero_rate_curves(mocker):
    table = Table(
        {
            "threshold": [10, 20],
            "NSB rate (Hz)": [0.0, 0.0],
            "Proton rate (Hz)": [0.0, 0.0],
        }
    )
    pages = mocker.Mock()
    pages.add.side_effect = bias_curve_comparison.plt.close

    bias_curve_comparison.plot_comparison(table, table, "LSTN-01", ("v1", "v2"), pages)

    pages.add.assert_called_once()


def test_max_relative_deviation_requires_two_finite_rates():
    table = Table({"threshold": [10, 20], "NSB rate (Hz)": [100.0, np.nan]})
    with pytest.raises(ValueError, match="missing or non-finite"):
        bias_curve_comparison._finite_curve(table, "NSB rate (Hz)")
