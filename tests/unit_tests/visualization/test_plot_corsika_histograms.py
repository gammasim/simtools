#!/usr/bin/python3

import matplotlib.pyplot as plt
import numpy as np
import pytest
from astropy import units as u

from simtools.visualization import plot_corsika_histograms
from simtools.visualization.plot_corsika_histograms import (
    _build_all_photon_figures,
    export_all_photon_figures_pdf,
)


@pytest.fixture
def hist_1d_factory():
    def factory(**overrides):
        base = {
            "is_1d": True,
            "hist_values": [np.array([1, 2, 3])],
            "x_bin_edges": [np.array([0, 1, 2, 3])],
            "x_axis_title": "X",
            "x_axis_unit": u.m,
            "y_axis_title": "Y",
            "y_axis_unit": u.dimensionless_unscaled,
            "x_bins": ["", "", "", ""],
            "log_y": False,
            "title": "1D Histogram",
            "input_file_name": "file1",
        }
        base.update(overrides)
        return base

    return factory


@pytest.fixture
def hist_2d_factory():
    def factory(**overrides):
        base = {
            "is_1d": False,
            "hist_values": [np.array([[1, 2], [3, 4]])],
            "x_bin_edges": [np.array([0, 1, 2])],
            "y_bin_edges": [np.array([0, 1, 2])],
            "x_axis_title": "X",
            "x_axis_unit": u.m,
            "y_axis_title": "Y",
            "y_axis_unit": u.m,
            "z_axis_title": "Z",
            "z_axis_unit": u.dimensionless_unscaled,
            "log_z": False,
            "title": "2D Histogram",
            "input_file_name": "file1",
        }
        base.update(overrides)
        return base

    return factory


def test_export_all_photon_figures_pdf(tmp_path, mocker, hist_1d_factory):
    mock_hist = mocker.Mock()
    mock_hist.hist = {
        "test_hist": hist_1d_factory(
            title="Test Histogram",
            input_file_name="test_file",
        )
    }

    # Patch save_figures_to_single_document to avoid actual file creation
    save_fig_patch = mocker.patch(
        "simtools.visualization.plot_corsika_histograms.save_figures_to_single_document"
    )

    pdf_path = tmp_path / "output.pdf"
    export_all_photon_figures_pdf(mock_hist, pdf_path)

    save_fig_patch.assert_called_once()
    args, _ = save_fig_patch.call_args
    assert args[1] == pdf_path


def test_build_all_photon_figures_1d_and_2d(mocker, hist_1d_factory, hist_2d_factory):
    mock_hist_1d = mocker.Mock()
    mock_hist_1d.hist = {"hist_1d": hist_1d_factory(title="1D Histogram")}
    mock_hist_2d = mocker.Mock()
    mock_hist_2d.hist = {"hist_2d": hist_2d_factory(title="2D Histogram", input_file_name="file2")}

    figs_1d = _build_all_photon_figures([mock_hist_1d])
    figs_2d = _build_all_photon_figures([mock_hist_2d])

    assert len(figs_1d) == 1
    assert hasattr(figs_1d[0], "savefig")
    assert len(figs_2d) == 1
    assert hasattr(figs_2d[0], "savefig")


def test_get_axis_label():
    label_with_unit = plot_corsika_histograms._get_axis_label("Energy", u.GeV)
    label_without_unit = plot_corsika_histograms._get_axis_label("Count", u.dimensionless_unscaled)

    assert label_with_unit == "Energy (GeV)"
    assert label_without_unit == "Count"


def test_plot_1d_single_histogram(mocker, hist_1d_factory):
    hist_dict = hist_1d_factory(title="Test 1D")
    figs = plot_corsika_histograms._plot_1d([hist_dict])
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_1d_multiple_histograms(mocker, hist_1d_factory):
    hist_dict1 = hist_1d_factory(title="Test 1D", input_file_name="file1")
    hist_dict2 = hist_1d_factory(hist_values=[np.array([2, 3, 4])], input_file_name="file2")
    figs = plot_corsika_histograms._plot_1d([hist_dict1, hist_dict2])
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_1d_with_labels(mocker, hist_1d_factory):
    hist_dict1 = hist_1d_factory(title="Test 1D", input_file_name="file1")
    hist_dict2 = hist_1d_factory(hist_values=[np.array([2, 3, 4])], input_file_name="file2")
    labels = ["First", "Second"]
    figs = plot_corsika_histograms._plot_1d([hist_dict1, hist_dict2], labels=labels)
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_1d_log_scale(hist_1d_factory):
    hist_dict = hist_1d_factory(
        hist_values=[np.array([1, 10, 100])],
        x_bin_edges=[np.array([1, 10, 100, 1000])],
        x_bins=["", "", "", "log"],
        log_y=True,
        title="Log 1D",
    )
    figs = plot_corsika_histograms._plot_1d([hist_dict])
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_1d_empty_list():
    figs = plot_corsika_histograms._plot_1d([])
    assert figs == []


def test_plot_2d_single_histogram(mocker, hist_2d_factory):
    hist_dict = hist_2d_factory()
    figs = plot_corsika_histograms._plot_2d([hist_dict])
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_2d_multiple_histograms(mocker, hist_2d_factory):
    hist_dict1 = hist_2d_factory(input_file_name="file1")
    hist_dict2 = hist_2d_factory(hist_values=[np.array([[2, 3], [4, 5]])], input_file_name="file2")
    figs = plot_corsika_histograms._plot_2d([hist_dict1, hist_dict2])
    assert len(figs) == 2
    for fig in figs:
        assert hasattr(fig, "savefig")


def test_plot_2d_with_labels(mocker, hist_2d_factory):
    hist_dict1 = hist_2d_factory(input_file_name="file1")
    hist_dict2 = hist_2d_factory(hist_values=[np.array([[2, 3], [4, 5]])], input_file_name="file2")
    labels = ["First", "Second"]
    figs = plot_corsika_histograms._plot_2d([hist_dict1, hist_dict2], labels=labels)
    assert len(figs) == 2
    for fig in figs:
        assert hasattr(fig, "savefig")


def test_plot_2d_log_scale(mocker, hist_2d_factory):
    hist_dict = hist_2d_factory(
        hist_values=[np.array([[1, 10], [100, 1000]])],
        x_bin_edges=[np.array([1, 10, 100])],
        y_bin_edges=[np.array([1, 10, 100])],
        log_z=True,
        title="Log 2D",
    )
    figs = plot_corsika_histograms._plot_2d([hist_dict])
    assert len(figs) == 1
    assert hasattr(figs[0], "savefig")


def test_plot_2d_empty_list():
    figs = plot_corsika_histograms._plot_2d([])
    assert figs == []


def test_extract_uncertainty_with_value():
    uncertainties = [[1, 2, 3], [4, 5, 6]]
    result = plot_corsika_histograms._extract_uncertainty(uncertainties, 1)
    assert result == [4, 5, 6]


def test_extract_uncertainty_with_none():
    uncertainties = [None, [4, 5, 6]]
    result = plot_corsika_histograms._extract_uncertainty(uncertainties, 0)
    assert result is None


def test_extract_uncertainty_uncertainties_none():
    result = plot_corsika_histograms._extract_uncertainty(None, 0)
    assert result is None


def test_plot_histogram_curve_with_uncertainties(mocker):
    fig, ax = plt.subplots()
    bin_centers = np.array([1, 2, 3])
    hist_values = np.array([10, 20, 30])
    uncertainties = np.array([1, 2, 3])
    color = "blue"
    label = "Test"
    errorbar_mock = mocker.patch.object(ax, "errorbar")
    plot_corsika_histograms._plot_histogram_curve(
        ax, bin_centers, hist_values, uncertainties, color, label
    )
    errorbar_mock.assert_called_once()
    plt.close(fig)


def test_plot_histogram_curve_without_uncertainties(mocker):
    fig, ax = plt.subplots()
    bin_centers = np.array([1, 2, 3])
    hist_values = np.array([10, 20, 30])
    uncertainties = None
    color = "red"
    label = "NoErr"
    plot_mock = mocker.patch.object(ax, "plot")
    plot_corsika_histograms._plot_histogram_curve(
        ax, bin_centers, hist_values, uncertainties, color, label
    )
    plot_mock.assert_called_once()
    plt.close(fig)
