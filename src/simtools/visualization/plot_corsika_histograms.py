"""Visualize Cherenkov photon distributions from CORSIKA."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib import colormaps, colors

from simtools.visualization.visualize import save_figures_to_single_document

_logger = logging.getLogger(__name__)


def _plot_2d(hist_list, labels=None):
    """
    Plot 2D Cherenkov photon distributions.

    Parameters
    ----------
    hist_list: list
        List of histogram dictionaries.
    labels: list or None
        Optional list of labels for the input files. If None, uses file names.

    Returns
    -------
    list
        List of figures.
    """
    if not hist_list:
        return []

    all_figs = []

    for i_file, hist_dict in enumerate(hist_list):
        hist_values = hist_dict["hist_values"]
        x_bin_edges = hist_dict["x_bin_edges"]
        y_bin_edges = hist_dict["y_bin_edges"]

        for i_hist, _ in enumerate(x_bin_edges):
            fig, ax = plt.subplots()
            if hist_dict.get("log_z", False):
                max_val = np.amax(hist_values[i_hist])
                norm = colors.LogNorm(vmin=1, vmax=np.amax([max_val, 2]))
            else:
                norm = None
            mesh = ax.pcolormesh(
                x_bin_edges[i_hist], y_bin_edges[i_hist], hist_values[i_hist], norm=norm
            )
            ax.set_xlabel(_get_axis_label(hist_dict["x_axis_title"], hist_dict["x_axis_unit"]))
            ax.set_ylabel(_get_axis_label(hist_dict["y_axis_title"], hist_dict["y_axis_unit"]))
            ax.set_xlim(np.amin(x_bin_edges[i_hist]), np.amax(x_bin_edges[i_hist]))
            ax.set_ylim(np.amin(y_bin_edges[i_hist]), np.amax(y_bin_edges[i_hist]))
            ax.set_facecolor("black")
            cbar = fig.colorbar(mesh)
            cbar.set_label(_get_axis_label(hist_dict["z_axis_title"], hist_dict["z_axis_unit"]))

            if labels is not None and i_file < len(labels):
                label = labels[i_file]
            else:
                label = Path(hist_dict.get("input_file_name", f"File {i_file}")).name
            ax.set_title(f"{hist_dict['title']} - {label}")

            all_figs.append(fig)
            plt.close()

    return all_figs


def _get_histogram_label(hist_dict, i_file, labels):
    """Get label for histogram curve."""
    if labels is not None and i_file < len(labels):
        return labels[i_file]
    return Path(hist_dict.get("input_file_name", f"File {i_file}")).name


def _extract_uncertainty(uncertainties, i_hist):
    """Extract uncertainty values if available."""
    if uncertainties is not None and uncertainties[i_hist] is not None:
        return uncertainties[i_hist]
    return None


def _plot_histogram_curve(ax, bin_centers, hist_values, uncertainties, color, label):
    """Plot a single histogram curve with or without error bars."""
    common_params = {
        "color": color,
        "label": label,
        "marker": "o",
        "markersize": 3,
        "linestyle": "-",
        "linewidth": 0.5,
    }

    if uncertainties is not None:
        ax.errorbar(
            bin_centers,
            hist_values,
            yerr=uncertainties,
            capsize=2,
            capthick=0.5,
            **common_params,
        )
    else:
        ax.plot(bin_centers, hist_values, **common_params)


def _configure_plot_scales(ax, hist):
    """Configure x and y axis scales."""
    if len(hist["x_bins"]) > 3 and hist["x_bins"][3] == "log":
        ax.set_xscale("log")
    if hist["log_y"] is True:
        ax.set_yscale("log")


def _plot_1d(hist_list, labels=None):
    """
    Plot 1D Cherenkov photon distributions.

    Parameters
    ----------
    hist_list: list
        List of histogram dictionaries from different files.
    labels: list or None
        Optional list of labels for the histogram curves. If None, uses file names.

    Returns
    -------
    list
        List of figures.
    """
    if not hist_list:
        return []

    hist = hist_list[0]
    plot_colors = colormaps["tab10"](np.linspace(0, 1, len(hist_list)))
    fig, ax = plt.subplots()

    for i_file, (hist_dict, color) in enumerate(zip(hist_list, plot_colors)):
        hist_values = hist_dict["hist_values"]
        x_bin_edges = hist_dict["x_bin_edges"]
        uncertainties = hist_dict.get("uncertainties")
        label = _get_histogram_label(hist_dict, i_file, labels)

        for i_hist, x_edges in enumerate(x_bin_edges):
            bin_centers = (x_edges[:-1] + x_edges[1:]) / 2
            unc = _extract_uncertainty(uncertainties, i_hist)
            _plot_histogram_curve(ax, bin_centers, hist_values[i_hist], unc, color, label)

    ax.set_xlabel(_get_axis_label(hist["x_axis_title"], hist["x_axis_unit"]))
    ax.set_ylabel(_get_axis_label(hist["y_axis_title"], hist["y_axis_unit"]))
    _configure_plot_scales(ax, hist)
    ax.set_title(f"{hist['title']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.close(fig)
    return [fig]


def _get_axis_label(title, unit):
    """Return axis label with unit if applicable."""
    if unit is not u.dimensionless_unscaled:
        return f"{title} ({unit})"
    return f"{title}"


def _build_all_photon_figures(histograms_list, labels=None):
    """Build list of all photon histogram figures.

    Parameters
    ----------
    histograms_list: list
        List of CorsikaHistograms instances from different input files.
    labels: list or None
        Optional list of labels for the input files. If None, uses file names.

    Returns
    -------
    list
        List of figures.
    """
    all_figs = []

    if not isinstance(histograms_list, list):
        histograms_list = [histograms_list]

    hist_keys = list(histograms_list[0].hist.keys())

    for hist_key in hist_keys:
        hist_from_all_files = [h.hist[hist_key] for h in histograms_list]

        if hist_from_all_files[0]["is_1d"]:
            all_figs.extend(_plot_1d(hist_from_all_files, labels=labels))
        else:
            all_figs.extend(_plot_2d(hist_from_all_files, labels=labels))

    return all_figs


def export_all_photon_figures_pdf(histograms_instance, pdf_file_name, labels=None):
    """
    Build and save all photon histogram figures into a single PDF.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms or list
        Single histogram instance or list of CorsikaHistograms instances from multiple files.
        When a list is provided, 1D histograms from all files are combined into single plots
        with different colors, while 2D histograms of the same type are plotted sequentially.
    pdf_file_name: str or Path
        Name of the output pdf file to save the histograms.
    labels: list or None
        Optional list of labels for the input files. If None, file names are used as labels.
        The order should match the order of histograms_instance if it's a list.
    """
    save_figures_to_single_document(
        _build_all_photon_figures(histograms_instance, labels=labels), Path(pdf_file_name)
    )
