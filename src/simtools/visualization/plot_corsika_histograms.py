"""Visualize Cherenkov photon distributions from CORSIKA."""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib import colors

from simtools.visualization.visualize import save_figures_to_single_document

_logger = logging.getLogger(__name__)


def _plot_2d(hist):
    """
    Plot 2D Cherenkov photon distributions.

    Parameters
    ----------
    hist: dict
        Histogram dictionary.

    Returns
    -------
    list
        List of figures.
    """
    hist_values = hist["hist_values"]
    x_bin_edges = hist["x_bin_edges"]
    y_bin_edges = hist["y_bin_edges"]

    all_figs = []
    for i_hist, _ in enumerate(x_bin_edges):
        fig, ax = plt.subplots()
        if hist.get("log_z", False):
            norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values[i_hist]), 2]))
        else:
            norm = None
        mesh = ax.pcolormesh(
            x_bin_edges[i_hist], y_bin_edges[i_hist], hist_values[i_hist], norm=norm
        )
        ax.set_xlabel(_get_axis_label(hist["x_axis_title"], hist["x_axis_unit"]))
        ax.set_ylabel(_get_axis_label(hist["y_axis_title"], hist["y_axis_unit"]))
        ax.set_xlim(np.amin(x_bin_edges[i_hist]), np.amax(x_bin_edges[i_hist]))
        ax.set_ylim(np.amin(y_bin_edges[i_hist]), np.amax(y_bin_edges[i_hist]))
        ax.set_facecolor("black")
        cbar = fig.colorbar(mesh)
        cbar.set_label(_get_axis_label(hist["z_axis_title"], hist["z_axis_unit"]))
        all_figs.append(fig)
        ax.set_title(f"{hist['file_name']}")
        plt.close()

    return all_figs


def _plot_1d(hist):
    """
    Plot 1D Cherenkov photon distributions.

    Parameters
    ----------
    hist: dict
        Histogram dictionary.

    Returns
    -------
    list
        List of figures.
    """
    hist_values = hist["hist_values"]
    x_bin_edges = hist["x_bin_edges"]

    all_figs = []
    for i_hist, _ in enumerate(x_bin_edges):
        fig, ax = plt.subplots()
        ax.bar(
            x_bin_edges[i_hist][:-1],
            hist_values[i_hist],
            align="edge",
            width=np.abs(np.diff(x_bin_edges[i_hist])),
        )
        ax.set_xlabel(_get_axis_label(hist["x_axis_title"], hist["x_axis_unit"]))
        ax.set_ylabel(_get_axis_label(hist["y_axis_title"], hist["y_axis_unit"]))
        if hist["log_y"] is True:
            ax.set_yscale("log")
        ax.set_title(f"{hist['file_name']}")
        all_figs.append(fig)
        plt.close(fig)
    return all_figs


def _get_axis_label(title, unit):
    """Return axis label with unit if applicable."""
    if unit is not u.dimensionless_unscaled:
        return f"{title} ({unit})"
    return f"{title}"


def _build_all_photon_figures(histograms):
    """Build list of all photon histogram figures."""
    all_figs = []
    for hist in histograms.hist.values():
        if hist["is_1d"]:
            all_figs.extend(_plot_1d(hist))
        else:
            all_figs.extend(_plot_2d(hist))

    return all_figs


def export_all_photon_figures_pdf(histograms_instance, pdf_file_name):
    """
    Build and save all photon histogram figures into a single PDF.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        Histograms to be plotted.
    pdf_file_name: str or Path
        Name of the output pdf file to save the histograms.
    """
    save_figures_to_single_document(
        _build_all_photon_figures(histograms_instance), Path(pdf_file_name)
    )
