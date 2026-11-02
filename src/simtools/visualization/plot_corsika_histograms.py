"""Visualize Cherenkov photon distributions from CORSIKA."""

from pathlib import Path

import numpy as np
from astropy import units as u

from simtools.visualization.matplotlib_backend import lazy_module
from simtools.visualization.matplotlib_backend import pyplot as plt
from simtools.visualization.visualize import pdf_report

matplotlib = lazy_module("matplotlib")
colors = lazy_module("matplotlib.colors")


def _axis_label(hist_dict, axis):
    """Return the label of axis 'x', 'y' or 'z', with its unit if applicable."""
    title, unit = hist_dict[f"{axis}_axis_title"], hist_dict[f"{axis}_axis_unit"]
    if unit is not u.dimensionless_unscaled:
        return f"{title} ({unit})"
    return f"{title}"


def _get_histogram_label(hist_dict, i_file, labels):
    """Return the user label for an input file, or its file name."""
    if labels is not None and i_file < len(labels):
        return labels[i_file]
    return Path(hist_dict.get("input_file_name", f"File {i_file}")).name


def _plot_2d(hist_list, labels=None):
    """Plot 2D Cherenkov photon distributions, one figure per histogram."""
    all_figs = []
    for i_file, hist_dict in enumerate(hist_list):
        hist_values = hist_dict["hist_values"]
        x_bin_edges = hist_dict["x_bin_edges"]
        y_bin_edges = hist_dict["y_bin_edges"]
        label = _get_histogram_label(hist_dict, i_file, labels)

        for i_hist, _ in enumerate(x_bin_edges):
            fig, ax = plt.subplots()
            norm = None
            if hist_dict.get("log_z", False):
                max_val = np.amax(hist_values[i_hist])
                norm = colors.LogNorm(vmin=1, vmax=np.amax([max_val, 2]))
            mesh = ax.pcolormesh(
                x_bin_edges[i_hist],
                y_bin_edges[i_hist],
                hist_values[i_hist],
                norm=norm,
                rasterized=True,
            )
            ax.set_xlabel(_axis_label(hist_dict, "x"))
            ax.set_ylabel(_axis_label(hist_dict, "y"))
            ax.set_xlim(np.amin(x_bin_edges[i_hist]), np.amax(x_bin_edges[i_hist]))
            ax.set_ylim(np.amin(y_bin_edges[i_hist]), np.amax(y_bin_edges[i_hist]))
            ax.set_facecolor("black")
            fig.colorbar(mesh).set_label(_axis_label(hist_dict, "z"))
            ax.set_title(f"{hist_dict['title']} - {label}")

            all_figs.append(fig)
            plt.close()

    return all_figs


def _plot_histogram_curve(ax, bin_centers, hist_values, uncertainties, color, label):
    """Plot one histogram curve, with error bars if uncertainties are given."""
    style = {
        "color": color,
        "label": label,
        "marker": "o",
        "markersize": 3,
        "linestyle": "-",
        "linewidth": 0.5,
    }
    if uncertainties is None:
        ax.plot(bin_centers, hist_values, **style)
    else:
        ax.errorbar(bin_centers, hist_values, yerr=uncertainties, capsize=2, capthick=0.5, **style)


def _plot_1d(hist_list, labels=None):
    """Plot 1D Cherenkov photon distributions from all files in one figure."""
    if not hist_list:
        return []

    hist = hist_list[0]
    plot_colors = matplotlib.colormaps["tab10"](np.linspace(0, 1, len(hist_list)))
    fig, ax = plt.subplots()

    for i_file, (hist_dict, color) in enumerate(zip(hist_list, plot_colors)):
        uncertainties = hist_dict.get("uncertainties")
        label = _get_histogram_label(hist_dict, i_file, labels)
        for i_hist, x_edges in enumerate(hist_dict["x_bin_edges"]):
            bin_centers = (x_edges[:-1] + x_edges[1:]) / 2
            unc = uncertainties[i_hist] if uncertainties is not None else None
            _plot_histogram_curve(
                ax, bin_centers, hist_dict["hist_values"][i_hist], unc, color, label
            )

    ax.set_xlabel(_axis_label(hist, "x"))
    ax.set_ylabel(_axis_label(hist, "y"))
    if len(hist["x_bins"]) > 3 and hist["x_bins"][3] == "log":
        ax.set_xscale("log")
    if hist["log_y"] is True:
        ax.set_yscale("log")
    if hist.get("y_axis_min") is not None:
        ax.set_ylim(bottom=float(hist["y_axis_min"]))
    ax.set_title(f"{hist['title']}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.close(fig)
    return [fig]


def _build_all_photon_figures(histograms_list, labels=None):
    """Build the figures of all photon histograms of one or more CorsikaHistograms."""
    all_figs = []

    if not isinstance(histograms_list, list):
        histograms_list = [histograms_list]
    for hist_key in histograms_list[0].hist:
        hist_from_all_files = [h.hist[hist_key] for h in histograms_list]
        plot_function = _plot_1d if hist_from_all_files[0]["is_1d"] else _plot_2d
        all_figs.extend(plot_function(hist_from_all_files, labels=labels))

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
    with pdf_report(Path(pdf_file_name)) as report:
        for figure in _build_all_photon_figures(histograms_instance, labels=labels):
            report.add(figure)
