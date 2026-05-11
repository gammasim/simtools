"""Trigger rate calculation for telescopes and arrays of telescopes."""

import logging
from pathlib import Path

import numpy as np
import yaml
from astropy import units as u
from ctao_cr_spectra.definitions import IRFDOC_ELECTRON_SPECTRUM, IRFDOC_PROTON_SPECTRUM
from ctao_cr_spectra.spectral import LogParabola, PowerLaw, PowerLawWithExponentialGaussian
from scipy import integrate

from simtools.io import ascii_handler, io_handler
from simtools.layout.array_layout_utils import get_array_elements_from_db_for_layouts
from simtools.sim_events.histograms import EventDataHistograms
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)

#: Mapping from primary particle common name to default cosmic-ray spectrum.
PARTICLE_SPECTRUM_MAP = {
    "proton": IRFDOC_PROTON_SPECTRUM,
    "electron": IRFDOC_ELECTRON_SPECTRUM,
}

#: Supported spectrum type names in YAML spectrum files.
_SPECTRUM_TYPES = {
    "PowerLaw": PowerLaw,
    "LogParabola": LogParabola,
    "PowerLawWithExponentialGaussian": PowerLawWithExponentialGaussian,
}


def telescope_trigger_rates(args_dict):
    """
    Calculate trigger rates for single telescopes or arrays of telescopes.

    Main function to read event data, fill histograms, and derive trigger rates.

    Parameters
    ----------
    args_dict : dict
        Dictionary of command line arguments.
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
        histograms = EventDataHistograms(
            args_dict["event_data_file"], array_name=array_name, telescope_list=telescope_ids
        )
        histograms.fill()

        _calculate_trigger_rates(histograms, array_name, args_dict.get("cr_spectrum"))

        if args_dict["plot_histograms"]:
            plot_simtel_event_histograms.plot(
                histograms.histograms,
                output_path=io_handler.IOHandler().get_output_directory(),
                array_name=array_name,
            )


def _calculate_trigger_rates(histograms, array_name, cr_spectrum_file=None):
    """
    Calculate trigger rates from the filled histograms.

    Parameters
    ----------
    histograms : EventDataHistograms
        Filled histogram object containing event data and file info.
    array_name : str
        Name of the telescope array configuration.
    cr_spectrum_file : str or Path, optional
        Path to a YAML file describing the cosmic-ray spectrum. If None, the spectrum
        is selected automatically based on the primary particle in the simulation data.
    """
    efficiency = histograms.histograms.get("energy_eff", {}).get("histogram")
    energy_axis = histograms.histograms.get("energy_eff", {}).get("bin_edges")

    primary_particle = histograms.file_info.get("primary_particle")
    cr_spectrum = get_cosmic_ray_spectrum(primary_particle, cr_spectrum_file)
    _logger.info(f"Cosmic ray spectrum: {cr_spectrum}")
    e_min = energy_axis[:-1] * u.TeV
    e_max = energy_axis[1:] * u.TeV
    cr_rates = (
        np.array(
            [
                _integrate_energy_spectrum(cr_spectrum, e1, e2)
                .decompose(bases=[u.s, u.cm, u.sr])
                .value
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


def get_cosmic_ray_spectrum(primary_particle=None, cr_spectrum_file=None):
    """
    Return the cosmic-ray spectrum to use for trigger rate calculations.

    If a YAML spectrum file is provided, the spectrum is loaded from that file.
    Otherwise, the spectrum is selected based on the primary particle name. If the
    particle is not found in the default map, a warning is logged and the proton
    spectrum is used as a fallback.

    Parameters
    ----------
    primary_particle : str, optional
        Common name of the simulated primary particle (e.g. 'proton', 'electron').
    cr_spectrum_file : str or Path, optional
        Path to a YAML file describing the cosmic-ray spectrum.

    Returns
    -------
    spectrum
        A callable spectrum object from ctao_cr_spectra.
    """
    if cr_spectrum_file is not None:
        return _load_spectrum_from_file(cr_spectrum_file)

    if primary_particle is not None:
        spectrum = PARTICLE_SPECTRUM_MAP.get(primary_particle)
        if spectrum is None:
            _logger.warning(
                f"No default spectrum for primary particle '{primary_particle}'. "
                "Falling back to IRFDOC_PROTON_SPECTRUM."
            )
            return IRFDOC_PROTON_SPECTRUM
        return spectrum

    return IRFDOC_PROTON_SPECTRUM


def _load_spectrum_from_file(yaml_path):
    """
    Load a cosmic-ray spectrum from a YAML configuration file.

    The YAML file must contain a 'type' key with one of the supported spectrum class
    names ('PowerLaw', 'LogParabola', 'PowerLawWithExponentialGaussian'), a
    'normalization' key with a numeric value, and a 'normalization_unit' key with
    an astropy-parseable unit string. Additional keys are passed as keyword arguments
    to the spectrum class constructor.

    Parameters
    ----------
    yaml_path : str or Path
        Path to the YAML spectrum definition file.

    Returns
    -------
    spectrum
        A spectrum object from ctao_cr_spectra.

    Examples
    --------
    Example YAML file content for a power-law spectrum:

    .. code-block:: yaml

        type: PowerLaw
        normalization: 9.8e-6
        normalization_unit: 1 / (cm2 s TeV sr)
        index: -2.62
        e_ref: 1.0
        e_ref_unit: TeV
    """
    with Path(yaml_path).open(encoding="utf-8") as f:
        config = yaml.safe_load(f)

    spectrum_type = config.get("type")
    if spectrum_type not in _SPECTRUM_TYPES:
        raise ValueError(
            f"Unsupported spectrum type '{spectrum_type}'. Supported types: {list(_SPECTRUM_TYPES)}"
        )

    normalization = config["normalization"] * u.Unit(config["normalization_unit"])

    kwargs = {"normalization": normalization}
    for key in ("index", "a", "b", "f", "mu", "sigma"):
        if key in config:
            kwargs[key] = config[key]
    if "e_ref" in config:
        kwargs["e_ref"] = config["e_ref"] * u.Unit(config.get("e_ref_unit", "TeV"))

    return _SPECTRUM_TYPES[spectrum_type](**kwargs)


def _integrate_energy_spectrum(spectrum, energy_min, energy_max):
    """
    Integrate a spectrum over an energy range.

    Uses the analytical integration method when the spectrum provides
    integrate_energy, and falls back to numerical integration
    (scipy.integrate.quad) otherwise.

    Parameters
    ----------
    spectrum : callable
        A spectrum object from ctao_cr_spectra.
    energy_min : astropy.units.Quantity
        Lower bound of the energy integration interval.
    energy_max : astropy.units.Quantity
        Upper bound of the energy integration interval.

    Returns
    -------
    astropy.units.Quantity
        Integrated spectrum value over the energy range.
    """
    if callable(getattr(spectrum, "integrate_energy", None)):
        return spectrum.integrate_energy(energy_min, energy_max)

    spectrum_unit = spectrum(energy_min).unit
    flux_unit = spectrum_unit * u.TeV

    def integrand(e_tev):
        return spectrum(e_tev * u.TeV).to_value(spectrum_unit)

    result, _ = integrate.quad(integrand, energy_min.to_value("TeV"), energy_max.to_value("TeV"))
    return result * flux_unit
