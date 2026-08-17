#!/usr/bin/python3
"""Generate sim_telarray configuration files for a given array."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.model.array_model import ArrayModel

_ARGUMENTS = ()


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.MODEL_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        *cli.layout_selection_arguments(),
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    array_model = ArrayModel(
        label=app_context.args["label"],
        model_version=app_context.args["model_version"],
        site=app_context.args.get("site"),
        layout_name=app_context.args.get("array_layout_name"),
        array_elements=app_context.args.get("array_elements"),
    )
    array_model.print_telescope_list()
    array_model.export_all_simtel_config_files()


if __name__ == "__main__":
    main()
