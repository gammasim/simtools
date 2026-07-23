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

from simtools.application.definition import ApplicationDefinition
from simtools.camera.single_photon_electron_spectrum import SinglePhotonElectronSpectrum
from simtools.configuration import arguments as cli

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "input_spectrum", help="File with amplitude spectrum.", type=Path, required=True
    ),
    cli.ArgumentDefinition(
        "afterpulse_spectrum", help="File with afterpulse spectrum.", type=Path, required=False
    ),
    cli.ArgumentDefinition(
        "step_size",
        help="Step size in amplitude spectrum",
        type=float,
        default=0.02,
        required=False,
    ),
    cli.ArgumentDefinition(
        "max_amplitude",
        help="Maximum amplitude for single p.e. for amplitude spectrum",
        type=float,
        default=42.0,
        required=False,
    ),
    cli.ArgumentDefinition(
        "scale_afterpulse_spectrum",
        help="Scale afterpulse spectrum by the given factor",
        type=float,
        default=1.0,
        required=False,
    ),
    cli.ArgumentDefinition(
        "afterpulse_amplitude_range",
        help="Amplitude range in pe for afterpulse calculation",
        type=float,
        nargs=2,
        default=[0.0, 42.0],
        required=False,
    ),
    cli.ArgumentDefinition(
        "fit_afterpulse",
        help="Fit afterpulse spectrum with an exponential decay function.",
        action="store_true",
        required=False,
    ),
    cli.ArgumentDefinition(
        "afterpulse_decay_factor_fixed_value",
        help="Fix decay factor in afterpulse fit (free fit parameter if not set set).",
        type=float,
        default=15.0,
        required=False,
    ),
    cli.ArgumentDefinition(
        "use_norm_spe",
        help="Use sim_telarray tool 'norm_spe' to normalize the spectrum.",
        action="store_true",
        required=False,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.IGNORE_MISSING_DESIGN_MODEL,
        cli.SITE,
        cli.TELESCOPE,
        *cli.PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    single_pe = SinglePhotonElectronSpectrum(app_context.args)
    single_pe.derive_single_pe_spectrum()
    single_pe.write_single_pe_spectrum()


if __name__ == "__main__":
    main()
