#!/usr/bin/python3
"""Generate sim_telarray configuration files for a given array."""

import astropy.units as u

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.array_model import ArrayModel
from simtools.simulation.configuration import get_model_writer
from simtools.simulation.nsb import NsbSettings

_ARGUMENTS = (
    cli.ZENITH_ANGLE(
        action="store",
        nargs=None,
        default=None,
        help=(
            "Zenith angle in degrees for the zenith-dependent NSB scaling of the pixel rates; "
            "without it, the NSB rates are not scaled."
        ),
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    array_model = ArrayModel(
        label=app_context.args["label"],
        model_version=app_context.args["model_version"],
        site=app_context.args.get("site"),
        layout_name=app_context.args.get("array_layout_name"),
        array_elements=app_context.args.get("array_element_list"),
        model_reader=app_context.model_reader,
        context=app_context,
    )
    array_model.print_telescope_list()
    writer = get_model_writer(array_model, context=app_context)
    zenith_angle = app_context.args.get("zenith_angle")
    if zenith_angle is not None:
        writer.nsb_settings = NsbSettings(
            zenith_angle=zenith_angle.to_value(u.deg), manual_factor=None, autoscale=True
        )
    writer.export_all_simtel_config_files()


if __name__ == "__main__":
    main()
