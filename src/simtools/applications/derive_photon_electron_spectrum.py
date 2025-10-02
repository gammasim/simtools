#!/usr/bin/python3

r"""
    Derive single photon electron spectrum from a given amplitude spectrum.

    Normalizes singe-p.e. amplitude distribution to mean amplitude of 1.0,
    as required by sim_telarray. Allows to fold in afterpulse distribution
    to prompt a spectrum. Uses the sim_telarray tool 'norm_spe' to normalize
    the spectra.

    Input files can be in ecsv format (preferred) or in the sim_telarray legacy format.

    Two output files with identical data are written to the output directory:

    - 'output_file'.ecsv: Single photon electron spectrum in ecsv format (data and metadata).
    - 'output_file'.dat: Single photon electron spectrum in sim_telarray format.

    Example
    -------

    .. code-block:: console

        simtools-derive-photon-electron-spectrum \\
            --input_spectrum spectrum_photon_electron.ecsv \\
            --afterpulse_spectrum spectrum_afterpulse.ecsv \\
            --step_size 0.02 \\
            --max_amplitude 42.0 \\
            --use_norm_spe \\
            --output_path ./tests/output \\
            --output_file spectrum_photon_electron_afterpulse.ecsv

    For an example of how to plot the single photon electron spectrum, see the
    integration test 'tests/integration_tests/config/plot_tabular_data_for_single_pe_data.yml'.

"""

from pathlib import Path

from simtools.application_control import get_application_label, startup_application
from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum
from simtools.configuration import configurator


def _parse():
    """Parse command line configuration."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description="Derive single photon electron spectrum from a given amplitude spectrum.",
    )
    config.parser.add_argument(
        "--input_spectrum",
        help="File with amplitude spectrum.",
        type=Path,
        required=True,
    )
    config.parser.add_argument(
        "--afterpulse_spectrum",
        help="File with afterpulse spectrum.",
        type=Path,
        required=False,
    )
    config.parser.add_argument(
        "--step_size",
        help="Step size in amplitude spectrum",
        type=float,
        default=0.02,
        required=False,
    )
    config.parser.add_argument(
        "--max_amplitude",
        help="Maximum amplitude for single p.e. for amplitude spectrum",
        type=float,
        default=42.0,
        required=False,
    )
    config.parser.add_argument(
        "--scale_afterpulse_spectrum",
        help="Scale afterpulse spectrum by the given factor",
        type=float,
        default=1.0,
        required=False,
    )
    config.parser.add_argument(
        "--afterpulse_amplitude_range",
        help="Amplitude range in pe for afterpulse calculation",
        type=float,
        nargs=2,
        default=[0.0, 42.0],
        required=False,
    )
    config.parser.add_argument(
        "--fit_afterpulse",
        help="Fit afterpulse spectrum with an exponential decay function.",
        action="store_true",
        required=False,
    )
    config.parser.add_argument(
        "--afterpulse_decay_factor_fixed_value",
        help="Fix decay factor in afterpulse fit (free fit parameter if not set set).",
        type=float,
        default=15.0,
        required=False,
    )
    config.parser.add_argument(
        "--use_norm_spe",
        help="Use sim_telarray tool 'norm_spe' to normalize the spectrum.",
        action="store_true",
        required=False,
    )

    return config.initialize(db_config=False, output=True, simulation_model=["telescope"])


def main():
    """Derive single photon electron spectrum from a given amplitude spectrum."""
    app_context = startup_application(_parse)

    single_pe = SinglePhotonElectronSpectrum(app_context.args)
    single_pe.derive_single_pe_spectrum()
    single_pe.write_single_pe_spectrum()


if __name__ == "__main__":
    main()
