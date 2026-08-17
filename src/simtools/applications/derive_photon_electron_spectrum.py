#!/usr/bin/python3

"""Derive single photon electron spectrum from a given amplitude spectrum."""

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
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
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
