"""Visualize Cherenkov photon distributions from CORSIKA."""

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u
from matplotlib import colors
from matplotlib.backends.backend_pdf import PdfPages

_logger = logging.getLogger(__name__)


def _kernel_plot_2d_photons(histograms_instance, property_name, log_z=False):
    """
    Provide helper functions for plotting Cherenkov photon distributions saved in CorsikaHistograms.

    Create the figure of a 2D plot. The parameter name indicate which plot.
    Choices are "counts", "density", "direction", "time_altitude", and "num_photons_per_telescope".

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    property_name: string
        Name of the quantity. Options are: "counts", "density", "direction", "time_altitude" and
        "num_photons_per_telescope".
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    Raises
    ------
    ValueError
        if property is not allowed.
    """
    if property_name not in histograms_instance.dict_2d_distributions:
        msg = (
            f"This property does not exist. The valid entries are "
            f"{histograms_instance.dict_2d_distributions}"
        )
        _logger.error(msg)
        raise ValueError(msg)
    _function = getattr(
        histograms_instance,
        histograms_instance.dict_2d_distributions[property_name]["function"],
    )
    hist_values, x_bin_edges, y_bin_edges = _function()

    all_figs = []
    for i_hist, _ in enumerate(x_bin_edges):
        fig, ax = plt.subplots()
        if log_z is True:
            norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values[i_hist]), 2]))
        else:
            norm = None
        mesh = ax.pcolormesh(
            x_bin_edges[i_hist], y_bin_edges[i_hist], hist_values[i_hist], norm=norm
        )
        if (
            histograms_instance.dict_2d_distributions[property_name]["x axis unit"]
            is not u.dimensionless_unscaled
        ):
            ax.set_xlabel(
                f"{histograms_instance.dict_2d_distributions[property_name]['x bin edges']} "
                f"({histograms_instance.dict_2d_distributions[property_name]['x axis unit']})"
            )
        else:
            ax.set_xlabel(
                f"{histograms_instance.dict_2d_distributions[property_name]['x bin edges']} "
            )
        if (
            histograms_instance.dict_2d_distributions[property_name]["y axis unit"]
            is not u.dimensionless_unscaled
        ):
            ax.set_ylabel(
                f"{histograms_instance.dict_2d_distributions[property_name]['y bin edges']} "
                f"({histograms_instance.dict_2d_distributions[property_name]['y axis unit']})"
            )
        else:
            ax.set_ylabel(
                f"{histograms_instance.dict_2d_distributions[property_name]['y bin edges']} "
            )
        ax.set_xlim(np.amin(x_bin_edges[i_hist]), np.amax(x_bin_edges[i_hist]))
        ax.set_ylim(np.amin(y_bin_edges[i_hist]), np.amax(y_bin_edges[i_hist]))
        ax.set_facecolor("black")
        fig.colorbar(mesh)
        all_figs.append(fig)
        if histograms_instance.individual_telescopes is False:
            ax.set_title(
                f"{histograms_instance.dict_2d_distributions[property_name]['file name']}_all_tels"
            )
        else:
            ax.text(
                0.99,
                0.99,
                "tel. " + str(i_hist),
                ha="right",
                va="top",
                transform=ax.transAxes,
                color="white",
            )
            ax.set_title(
                f"{histograms_instance.dict_2d_distributions[property_name]['file name']}"
                f"_tel_index_{histograms_instance.telescope_indices[i_hist]}",
            )
        plt.close()

    return all_figs


def plot_2d_counts(histograms_instance, log_z=True):
    """
    Plot the 2D histogram of the photon positions on the ground.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_2d_photons(histograms_instance, "counts", log_z=log_z)


def plot_2d_density(histograms_instance, log_z=True):
    """
    Plot the 2D histogram of the photon density distribution on the ground.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_2d_photons(histograms_instance, "density", log_z=log_z)


def plot_2d_direction(histograms_instance, log_z=True):
    """
    Plot the 2D histogram of the incoming direction of photons.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_2d_photons(histograms_instance, "direction", log_z=log_z)


def plot_2d_time_altitude(histograms_instance, log_z=True):
    """
    Plot the 2D histogram of the time and altitude where the photon was produced.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_2d_photons(histograms_instance, "time_altitude", log_z=log_z)


def plot_2d_num_photons_per_telescope(histograms_instance, log_z=True):
    """
    Plot the 2D histogram of the number of photons per event and per telescope.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_z: bool
        if True, the intensity of the color bar is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_2d_photons(histograms_instance, "num_photons_per_telescope", log_z=log_z)


def _kernel_plot_1d_photons(histograms_instance, property_name, log_y=True):
    """
    Create the figure of a 1D plot. The parameter property indicate which plot.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    property_name: string
        Name of the quantity. Choices are
        "counts", "density", "direction", "time", "altitude", "num_photons_per_event", and
        "num_photons_per_telescope".
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    Raises
    ------
    ValueError
        if property is not allowed.
    """
    if property_name not in histograms_instance.dict_1d_distributions:
        msg = (
            f"This property does not exist. The valid entries are "
            f"{histograms_instance.dict_1d_distributions}"
        )
        _logger.error(msg)
        raise ValueError(msg)

    _function = getattr(
        histograms_instance,
        histograms_instance.dict_1d_distributions[property_name]["function"],
    )
    hist_values, bin_edges = _function()
    all_figs = []
    for i_hist, _ in enumerate(bin_edges):
        fig, ax = plt.subplots()
        ax.bar(
            bin_edges[i_hist][:-1],
            hist_values[i_hist],
            align="edge",
            width=np.abs(np.diff(bin_edges[i_hist])),
        )
        if (
            histograms_instance.dict_1d_distributions[property_name]["axis unit"]
            is not u.dimensionless_unscaled
        ):
            ax.set_xlabel(
                f"{histograms_instance.dict_1d_distributions[property_name]['bin edges']} "
                f"({histograms_instance.dict_1d_distributions[property_name]['axis unit']})"
            )
        else:
            ax.set_xlabel(
                f"{histograms_instance.dict_1d_distributions[property_name]['bin edges']} "
            )
        if property_name == "density":
            ax.set_ylabel(
                f"Density ({histograms_instance.dict_1d_distributions[property_name]['axis unit']}"
                r"$^{-2}$)"
            )
        else:
            ax.set_ylabel("Counts")

        if log_y is True:
            ax.set_yscale("log")
        if histograms_instance.individual_telescopes is False:
            ax.set_title(
                f"{histograms_instance.dict_1d_distributions[property_name]['file name']}_all_tels"
            )
        else:
            ax.set_title(
                f"{histograms_instance.dict_1d_distributions[property_name]['file name']}"
                f"_tel_index_{histograms_instance.telescope_indices[i_hist]}",
            )
        all_figs.append(fig)
        plt.close(fig)
    return all_figs


def plot_wavelength_distr(histograms_instance, log_y=True):
    """
    Plot the 1D distribution of the photon wavelengths.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_1d_photons(histograms_instance, "wavelength", log_y=log_y)


def plot_counts_distr(histograms_instance, log_y=True):
    """
    Plot the 1D distribution, i.e. the radial distribution, of the photons on the ground.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_1d_photons(histograms_instance, "counts", log_y=log_y)


def plot_density_distr(histograms_instance, log_y=True):
    """
    Plot the photon density distribution on the ground.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_1d_photons(histograms_instance, "density", log_y=log_y)


def plot_time_distr(histograms_instance, log_y=True):
    """
    Plot the distribution times in which the photons were generated in ns.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_1d_photons(histograms_instance, "time", log_y=log_y)


def plot_altitude_distr(histograms_instance, log_y=True):
    """
    Plot the distribution of altitude in which the photons were generated in km.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.
    """
    return _kernel_plot_1d_photons(histograms_instance, "altitude", log_y=log_y)


def plot_photon_per_event_distr(histograms_instance, log_y=True):
    """
    Plot the distribution of the number of Cherenkov photons per event.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_1d_photons(histograms_instance, "num_photons_per_event", log_y=log_y)


def plot_photon_per_telescope_distr(histograms_instance, log_y=True):
    """
    Plot the distribution of the number of Cherenkov photons per telescope.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    return _kernel_plot_1d_photons(histograms_instance, "num_photons_per_telescope", log_y=log_y)


def plot_1d_event_header_distribution(
    histograms_instance, event_header_element, log_y=True, bins=50, hist_range=None
):
    """
    Plot the distribution of the quantity given by .

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    event_header_element: str
        The key to the CORSIKA event header element.
    log_y: bool
        if True, the intensity of the Y axis is given in logarithmic scale.
    bins: float
        Number of bins for the histogram.
    hist_range: 2-tuple
        Tuple to define the range of the histogram.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    hist_values, bin_edges = histograms_instance.event_1d_histogram(
        event_header_element, bins=bins, hist_range=hist_range
    )
    fig, ax = plt.subplots()
    ax.bar(
        bin_edges[:-1],
        hist_values,
        align="edge",
        width=np.abs(np.diff(bin_edges)),
    )
    if (
        histograms_instance.event_information[event_header_element].unit
        is not u.dimensionless_unscaled
    ):
        ax.set_xlabel(
            f"{event_header_element} ("
            f"{histograms_instance.event_information[event_header_element].unit})"
        )
    else:
        ax.set_xlabel(f"{event_header_element}")
    ax.set_ylabel("Counts")

    if log_y is True:
        ax.set_yscale("log")
    ax.set_title(f"hist_1d_{event_header_element}")
    return fig


def plot_2d_event_header_distribution(
    histograms_instance,
    event_header_element_1,
    event_header_element_2,
    log_z=True,
    bins=50,
    hist_range=None,
):
    """
    Plot the distribution of the quantity given by CorsikaHistograms.

    Parameters
    ----------
    histograms_instance: corsika.corsika_histograms.CorsikaHistograms
        instance of corsika.corsika_histograms.CorsikaHistograms.
    event_header_element_1: str
        The first key to the CORSIKA event header element
    event_header_element_2: str
        The second key to the CORSIKA event header element.
    log_z: bool
        if True, the intensity of the Y axis is given in logarithmic scale.
    bins: float
        Number of bins for the histogram.
    hist_range: 2-tuple
        Tuple to define the range of the histogram.

    Returns
    -------
    list
        List of figures for the given telescopes.

    """
    hist_values, x_bin_edges, y_bin_edges = histograms_instance.event_2d_histogram(
        event_header_element_1, event_header_element_2, bins=bins, hist_range=hist_range
    )
    fig, ax = plt.subplots()
    if log_z is True:
        norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values), 2]))
    else:
        norm = None
    mesh = ax.pcolormesh(x_bin_edges, y_bin_edges, hist_values, norm=norm)

    if (
        histograms_instance.event_information[event_header_element_1].unit
        is not u.dimensionless_unscaled
    ):
        ax.set_xlabel(
            f"{event_header_element_1} ("
            f"{histograms_instance.event_information[event_header_element_1].unit})"
        )
    else:
        ax.set_xlabel(f"{event_header_element_2}")
    if (
        histograms_instance.event_information[event_header_element_2].unit
        is not u.dimensionless_unscaled
    ):
        ax.set_ylabel(
            f"{event_header_element_2} "
            f"({histograms_instance.event_information[event_header_element_2].unit})"
        )
    else:
        ax.set_ylabel(f"{event_header_element_2}")

    ax.set_facecolor("black")
    ax.set_title(f"hist_2d_{event_header_element_1}_{event_header_element_2}")
    fig.colorbar(mesh)
    return fig


def save_figs_to_pdf(figs, pdf_file_name):
    """
    Save figures from corsika histograms to an output pdf file.

    Parameters
    ----------
    figs: list or numpy.array
        List with the figures output by corsika_output_visualize.py.
    pdf_file_name: str or Path
        Name of the pdf file.
    """
    pdf_pages = PdfPages(Path(pdf_file_name).absolute().as_posix())
    for fig in figs:
        plt.tight_layout()
        pdf_pages.savefig(fig)
    pdf_pages.close()


def build_all_photon_figures(histograms_instance, test: bool = False):
    """Return list of all photon histogram figures for the given instance.

    When test is True, only generate the first two figure groups to reduce runtime.
    """
    plot_function_names = sorted(
        [
            name
            for name, obj in globals().items()
            if name.startswith("plot_")
            and "event_header_distribution" not in name
            and callable(obj)
        ]
    )
    if test:
        plot_function_names = plot_function_names[:2]

    figure_list = []
    module_obj = globals()
    for fn_name in plot_function_names:
        plot_fn = module_obj[fn_name]
        figs = plot_fn(histograms_instance)
        for fig in figs:
            figure_list.append(fig)
    return np.array(figure_list).flatten()


def export_all_photon_figures_pdf(histograms_instance, test: bool = False):
    """Build and save all photon histogram figures into a single PDF.

    The PDF name is derived from the HDF5 file name core and written under output_path.
    """
    figs = build_all_photon_figures(histograms_instance, test=test)
    core_name = re.sub(r"\.hdf5$", "", Path(histograms_instance.hdf5_file_name).name)
    output_file_name = Path(histograms_instance.output_path).joinpath(f"{core_name}.pdf")
    save_figs_to_pdf(figs, output_file_name)
    return output_file_name


def derive_event_1d_histograms(
    histograms_instance,
    event_1d_header_keys,
    pdf: bool,
    hdf5: bool,
    overwrite: bool = False,
):
    """Create 1D event header histograms; optionally save to PDF and/or HDF5."""
    figure_list = []
    for key in event_1d_header_keys:
        if pdf:
            fig = plot_1d_event_header_distribution(histograms_instance, key)
            figure_list.append(fig)
        if hdf5:
            histograms_instance.export_event_header_1d_histogram(
                key, bins=50, hist_range=None, overwrite=overwrite
            )
    if pdf:
        figs_array = np.array(figure_list).flatten()
        pdf_name = Path(histograms_instance.output_path).joinpath(
            f"{Path(histograms_instance.hdf5_file_name).name}_event_1d_histograms.pdf"
        )
        save_figs_to_pdf(figs_array, pdf_name)
        return pdf_name
    return None


def derive_event_2d_histograms(
    histograms_instance,
    event_2d_header_keys,
    pdf: bool,
    hdf5: bool,
    overwrite: bool = False,
):
    """Create 2D event header histograms in pairs; optionally save PDF and/or HDF5.

    If an odd number of keys is provided, the last one is ignored (with a warning).
    """
    if len(event_2d_header_keys) % 2 == 1:
        _logger.warning(
            "An odd number of keys was passed to generate 2D histograms.\n"
            "The last key is being ignored."
        )

    figure_list = []
    for i, _ in enumerate(event_2d_header_keys[::2]):
        if pdf:
            fig = plot_2d_event_header_distribution(
                histograms_instance, event_2d_header_keys[i], event_2d_header_keys[i + 1]
            )
            figure_list.append(fig)
        if hdf5:
            histograms_instance.export_event_header_2d_histogram(
                event_2d_header_keys[i],
                event_2d_header_keys[i + 1],
                bins=50,
                hist_range=None,
                overwrite=overwrite,
            )
    if pdf:
        figs_array = np.array(figure_list).flatten()
        pdf_name = Path(histograms_instance.output_path).joinpath(
            f"{Path(histograms_instance.hdf5_file_name).name}_event_2d_histograms.pdf"
        )
        save_figs_to_pdf(figs_array, pdf_name)
        return pdf_name
    return None
