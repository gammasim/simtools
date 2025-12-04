"""Trigger rate calculation for telescopes and arrays of telescopes."""

import logging

import numpy as np
from astropy import units as u
from ctao_cr_spectra.definitions import IRFDOC_PROTON_SPECTRUM

from simtools.io import ascii_handler, io_handler
from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.sim_events.histograms import IOEventDataHistograms
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)


def telescope_trigger_rates(args_dict):
    """
    Calculate trigger rates for single telescopes or arrays of telescopes.

    Main function to read event data, fill histograms, and derive trigger rates.


    """
    if args_dict.get("array_layout_name"):
        telescope_configs = get_array_elements_from_db_for_layouts(
            args_dict["array_layout_name"],
            args_dict.get("site"),
            args_dict.get("model_version"),
        )
    else:
        telescope_configs = ascii_handler.collect_data_from_file(args_dict["telescope_ids"])[
            "telescope_configs"
        ]

    for array_name, telescope_ids in telescope_configs.items():
        _logger.info(
            f"Processing file: {args_dict['event_data_file']} with telescope config: {array_name}"
        )
        histograms = IOEventDataHistograms(
            args_dict["event_data_file"], array_name=array_name, telescope_list=telescope_ids
        )
        histograms.fill()

        _calculate_trigger_rates(histograms, array_name)

        if args_dict["plot_histograms"]:
            plot_simtel_event_histograms.plot(
                histograms.histograms,
                output_path=io_handler.IOHandler().get_output_directory(),
                array_name=array_name,
            )


def _calculate_trigger_rates(histograms, array_name):
    """
    Calculate trigger rates from the filled histograms.

    Missing

    - custom definition of energy spectra

    """
    efficiency = histograms.histograms.get("energy_eff", {}).get("histogram")
    energy_axis = histograms.histograms.get("energy_eff", {}).get("bin_edges")

    cr_spectrum = get_cosmic_ray_spectrum()
    _logger.info(f"Cosmic ray spectrum: {cr_spectrum}")
    e_min = energy_axis[:-1] * u.TeV
    e_max = energy_axis[1:] * u.TeV
    cr_rates = (
        np.array(
            [
                cr_spectrum.integrate_energy(e1, e2).decompose(bases=[u.s, u.cm, u.sr]).value
                for e1, e2 in zip(e_min, e_max)
            ]
        )
        * histograms.file_info["scatter_area"].to("cm2").value
        * histograms.file_info["solid_angle"].to("sr").value
        * u.Hz
    )
    trigger_rates = efficiency * cr_rates
    trigger_rate = np.sum(trigger_rates, axis=0)

    _logger.info(f"Scatter area from MC: {histograms.file_info['scatter_area'].to('m2')}")
    _logger.info(f"Solid angle from MC: {histograms.file_info['solid_angle']}")
    _logger.info(f"Trigger rate for {array_name} array: {trigger_rate.to('Hz')}")

    histograms.histograms["cr_rates_mc"] = histograms.get_histogram_definition(
        histogram=cr_rates.value,
        bin_edges=energy_axis,
        title="Cosmic Ray Rates (MC)",
        axis_titles=["Energy (TeV)", "Cosmic Ray Rate (Hz)"],
        plot_scales={"x": "log", "y": "log"},
    )
    histograms.histograms["trigger_rates"] = histograms.get_histogram_definition(
        histogram=trigger_rates.value,
        bin_edges=energy_axis,
        title="Trigger Rates (MC)",
        axis_titles=["Energy (TeV)", "Trigger Rate (Hz)"],
        plot_scales={"x": "log", "y": "log"},
    )

    return cr_rates, trigger_rates, trigger_rate


def get_cosmic_ray_spectrum():
    """
    Return the cosmic ray spectrum.

    To be extended in future to read a larger variety of spectra.

    Returns
    -------
    astropy.units.Quantity
        Cosmic ray spectrum.
    """
    return IRFDOC_PROTON_SPECTRUM
