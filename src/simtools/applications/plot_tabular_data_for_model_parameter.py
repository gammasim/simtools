#!/usr/bin/python3
"""Plot tabular data for a single model parameter using default plotting configurations."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.visualization import plot_tables

_ARGUMENTS = (
    cli.ArgumentDefinition("parameter", type=str, required=True, help="Parameter name."),
    cli.ArgumentDefinition(
        "plot_type",
        help="Plot type as defined in the schema file.",
        type=str,
        required=True,
        default=None,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        cli.PARAMETER_VERSION,
        cli.OVERWRITE_MODEL_PARAMETERS,
        cli.SITE,
        cli.TELESCOPE,
        *cli.OUTPUT_PATH_ARGUMENTS,
    ),
    database=True,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()

    plot_configs, output_files = plot_tables.generate_plot_configurations(
        parameter=app_context.args["parameter"],
        parameter_version=app_context.args["parameter_version"],
        site=app_context.args["site"],
        telescope=app_context.args.get("telescope"),
        output_path=app_context.io_handler.get_output_directory(),
        plot_type=app_context.args["plot_type"],
    )

    for plot_config, output_file in zip(plot_configs, output_files):
        plot_tables.plot(config=plot_config, output_file=output_file)
        MetadataCollector.dump(app_context.args, output_file=output_file, add_activity_name=True)


if __name__ == "__main__":
    main()
