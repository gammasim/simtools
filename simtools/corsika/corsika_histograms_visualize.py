import logging

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
from astropy import units as u

_logger = logging.getLogger(__name__)


def _kernel_plot_2D_photons(histograms_instance, property_name, log_z=False):
    """
    The next functions below are used by the the CorsikaHistograms class to plot all sort of
    information from the Cherenkov photons saved.

    Create the figure of a 2D plot. The parameter `name` indicate which plot.
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
    list
        List of the figure names.

    Raises
    ------
    ValueError
        if `property` is not allowed.
    """
    if property_name not in histograms_instance._dict_2D_distributions:
        msg = (
            f"This property does not exist. The valid entries are "
            f"{histograms_instance._dict_2D_distributions}"
        )
        _logger.error(msg)
        raise ValueError
    function = getattr(
        histograms_instance,
        histograms_instance._dict_2D_distributions[property_name]["function"],
    )
    hist_values, x_edges, y_edges = function()

    all_figs = []
    fig_names = []
    for i_hist, _ in enumerate(x_edges):
        fig, ax = plt.subplots()
        if log_z is True:
            norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values[i_hist]), 2]))
        else:
            norm = None
        mesh = ax.pcolormesh(x_edges[i_hist], y_edges[i_hist], hist_values[i_hist], norm=norm)
        if (
            histograms_instance._dict_2D_distributions[property_name]["x edges unit"]
            is not u.dimensionless_unscaled
        ):
            ax.set_xlabel(
                f"{histograms_instance._dict_2D_distributions[property_name]['x edges']} "
                f"({histograms_instance._dict_2D_distributions[property_name]['x edges unit']})"
            )
        else:
            ax.set_xlabel(
                f"{histograms_instance._dict_2D_distributions[property_name]['x edges']} "
            )
        if (
            histograms_instance._dict_2D_distributions[property_name]["y edges"]
            is not u.dimensionless_unscaled
        ):
            ax.set_ylabel(
                f"{histograms_instance._dict_2D_distributions[property_name]['y edges']} "
                f"({histograms_instance._dict_2D_distributions[property_name]['y edges unit']})"
            )
        else:
            ax.set_ylabel(
                f"{histograms_instance._dict_2D_distributions[property_name]['y edges']} "
            )
        ax.set_xlim(np.amin(x_edges[i_hist]), np.amax(x_edges[i_hist]))
        ax.set_ylim(np.amin(y_edges[i_hist]), np.amax(y_edges[i_hist]))
        ax.set_facecolor("black")
        fig.colorbar(mesh)
        all_figs.append(fig)
        if histograms_instance.individual_telescopes is False:
            fig_names.append(
                f"{histograms_instance._dict_2D_distributions[property_name]['file name']}"
                f"_all_tels.png"
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
            fig_names.append(
                f"{histograms_instance._dict_2D_distributions[property_name]['file name']}"
                f"_tel_index_{histograms_instance.telescope_indices[i_hist]}.png",
            )
        plt.close()

    return all_figs, fig_names


def plot_2D_counts(histograms_instance, log_z=True):
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
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(histograms_instance, "counts", log_z=log_z)


def plot_2D_density(histograms_instance, log_z=True):
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
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(histograms_instance, "density", log_z=log_z)


def plot_2D_direction(histograms_instance, log_z=True):
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
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(histograms_instance, "direction", log_z=log_z)


def plot_2D_time_altitude(histograms_instance, log_z=True):
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
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(histograms_instance, "time_altitude", log_z=log_z)


def plot_2D_num_photons_per_telescope(histograms_instance, log_z=True):
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
    list
        List of the figure names.

    """
    return _kernel_plot_2D_photons(histograms_instance, "num_photons_per_telescope", log_z=log_z)


def _kernel_plot_1D_photons(histograms_instance, property_name, log_y=True):
    """
    Create the figure of a 1D plot. The parameter `property` indicate which plot.

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
    list
        List of the figure names.

    Raises
    ------
    ValueError
        if `property` is not allowed.
    """
    if property_name not in histograms_instance._dict_1D_distributions:
        msg = (
            f"This property does not exist. The valid entries are "
            f"{histograms_instance._dict_1D_distributions}"
        )
        _logger.error(msg)
        raise ValueError

    function = getattr(
        histograms_instance,
        histograms_instance._dict_1D_distributions[property_name]["function"],
    )
    hist_values, edges = function()
    all_figs = []
    fig_names = []
    for i_hist, _ in enumerate(edges):
        fig, ax = plt.subplots()
        ax.bar(
            edges[i_hist][:-1],
            hist_values[i_hist],
            align="edge",
            width=np.abs(np.diff(edges[i_hist])),
        )
        if (
            histograms_instance._dict_1D_distributions[property_name]["edges unit"]
            is not u.dimensionless_unscaled
        ):
            ax.set_xlabel(
                f"{histograms_instance._dict_1D_distributions[property_name]['edges']} "
                f"({histograms_instance._dict_1D_distributions[property_name]['edges unit']})"
            )
        else:
            ax.set_xlabel(f"{histograms_instance._dict_1D_distributions[property_name]['edges']} ")
        ax.set_ylabel("Counts")

        if log_y is True:
            ax.set_yscale("log")
        if histograms_instance.individual_telescopes is False:
            fig_names.append(
                f"{histograms_instance._dict_1D_distributions[property_name]['file name']}"
                f"_all_tels.png"
            )
        else:
            fig_names.append(
                f"{histograms_instance._dict_1D_distributions[property_name]['file name']}"
                f"_tel_index_{histograms_instance.telescope_indices[i_hist]}.png",
            )
        all_figs.append(fig)
        plt.close()
    return all_figs, fig_names


def plot_wavelength_distr(histograms_instance, log_y=True):
    """
    Plots the 1D distribution of the photon wavelengths

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
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(histograms_instance, "wavelength", log_y=log_y)


def plot_counts_distr(histograms_instance, log_y=True):
    """
    Plots the 1D distribution, i.e. the radial distribution, of the photons on the ground.

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
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(histograms_instance, "counts", log_y=log_y)


def plot_density_distr(histograms_instance, log_y=True):
    """
    Plots the photon density distribution on the ground.

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
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(histograms_instance, "density", log_y=log_y)


def plot_time_distr(histograms_instance, log_y=True):
    """
    Plots the distribution times in which the photons were generated in ns.

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
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(histograms_instance, "time", log_y=log_y)


def plot_altitude_distr(histograms_instance, log_y=True):
    """
    Plots the distribution of altitude in which the photons were generated in km.

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
    list
        List of the figure names.
    """
    return _kernel_plot_1D_photons(histograms_instance, "altitude", log_y=log_y)


def plot_photon_per_event_distr(histograms_instance, log_y=True):
    """
    Plots the distribution of the number of Cherenkov photons per event.

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
    list
        List of the figure names.

    """

    return _kernel_plot_1D_photons(histograms_instance, "num_photons_per_event", log_y=log_y)


def plot_photon_per_telescope_distr(histograms_instance, log_y=True):
    """
    Plots the distribution of the number of Cherenkov photons per telescope.

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
    list
        List of the figure names.

    """

    return _kernel_plot_1D_photons(histograms_instance, "num_photons_per_telescope", log_y=log_y)


def plot_1D_event_header_distribution(
    histograms_instance, event_header_element, log_y=True, bins=50, hist_range=None
):
    """
    Plots the distribution of the quantity given by .

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
    list
        List of the figure names.

    """
    hist_values, edges = histograms_instance.event_1D_histogram(
        event_header_element, bins=bins, hist_range=hist_range
    )
    fig, ax = plt.subplots()
    ax.bar(
        edges[:-1],
        hist_values,
        align="edge",
        width=np.abs(np.diff(edges)),
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
    fig_name = f"hist_1D_{event_header_element}"
    return fig, fig_name


def plot_2D_event_header_distribution(
    histograms_instance,
    event_header_element_1,
    event_header_element_2,
    log_z=True,
    bins=50,
    hist_range=None,
):
    """
    Plots the distribution of the quantity given by .

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
    list
        List of the figure names.

    """
    hist_values, x_edges, y_edges = histograms_instance.event_2D_histogram(
        event_header_element_1, event_header_element_2, bins=bins, hist_range=hist_range
    )
    fig, ax = plt.subplots()
    if log_z is True:
        norm = colors.LogNorm(vmin=1, vmax=np.amax([np.amax(hist_values), 2]))
    else:
        norm = None
    mesh = ax.pcolormesh(x_edges, y_edges, hist_values, norm=norm)

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
    fig.colorbar(mesh)
    fig_name = f"hist_2D_{event_header_element_1}_{event_header_element_2}"
    return fig, fig_name
