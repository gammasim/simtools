#!/usr/bin/python3
"""Characterize and compare atmospheric model parameters."""

from simtools.application.definition import ApplicationDefinition
from simtools.application.model_reader import require_model_reader
from simtools.atmosphere.atmosphere_characterization import characterize_atmospheres
from simtools.configuration import arguments as cli
from simtools.configuration.argument_helpers import site as validate_site

_ATMOSPHERE = "atmosphere"
_PROFILE = "atmospheric_profile"
_TRANSMISSION = "atmospheric_transmission"
_RELATIVE_DIFFERENCE_LIMIT = "relative_difference_limit"
_DEFAULT_RELATIVE_DIFFERENCE_LIMIT = 35.0

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        cli.ArgumentDefinition(
            _ATMOSPHERE,
            action="append",
            nargs=3,
            metavar=("SITE", "PROFILE_VERSION", "TRANSMISSION_VERSION"),
            required=True,
            help="Atmosphere input; repeat to compare. The first input is the reference.",
        ),
        cli.ArgumentDefinition(
            _RELATIVE_DIFFERENCE_LIMIT,
            type=float,
            default=_DEFAULT_RELATIVE_DIFFERENCE_LIMIT,
            help="Symmetric relative-difference axis limit in percent (default: 35).",
        ),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
)


def main():
    """Execute atmospheric characterization using the configured model repository."""
    context = APPLICATION.start()
    reader = require_model_reader(context.model_reader)
    output_path = context.io_handler.get_output_directory()
    atmospheres = []
    for site, profile_version, transmission_version in context.args[_ATMOSPHERE]:
        site = validate_site(site)
        tables = {}
        for parameter, version in (
            (_PROFILE, profile_version),
            (_TRANSMISSION, transmission_version),
        ):
            tables[parameter] = reader.export_model_file(
                parameter=parameter,
                site=site,
                array_element_name=None,
                parameter_version=version,
                export_file_as_table=True,
                dest=output_path,
            )
        atmospheres.append(
            {
                "label": f"{site} profile {profile_version} transmission {transmission_version}",
                "site": site,
                "profile_version": profile_version,
                "transmission_version": transmission_version,
                "profile": tables[_PROFILE],
                "transmission": tables[_TRANSMISSION],
            }
        )
    characterize_atmospheres(
        atmospheres,
        output_path,
        relative_difference_limit=context.args[_RELATIVE_DIFFERENCE_LIMIT],
    )


if __name__ == "__main__":
    main()
