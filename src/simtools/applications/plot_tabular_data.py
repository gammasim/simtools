#!/usr/bin/python3
"""Plot tabular data read from a file or from the model parameter repository.

Data file mode (``--plot_config``) applies a plotting configuration to a data file.
Model parameter mode (``--parameter``) plots a model parameter using the plotting
configurations defined in its schema file.
"""

import simtools.utils.general as gen
from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.constants import PLOT_CONFIG_SCHEMA
from simtools.data_model import schema
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import ascii_handler
from simtools.utils import value_conversion
from simtools.visualization import plot_tables

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "plot_config",
        help="Plotting configuration file name (data file mode).",
        type=str,
        exclusive_group="plot source",
        exclusive_group_required=True,
    ),
    cli.ArgumentDefinition(
        "parameter",
        help="Model parameter name to plot (model parameter mode, e.g., 'atmospheric_profile').",
        type=str,
        exclusive_group="plot source",
        exclusive_group_required=True,
    ),
    cli.DATA_SEARCH_PATH(default=None, aliases=("table_data_path",)),
    cli.ArgumentDefinition(
        "output_file",
        help="Output file name (without suffix; required in data file mode).",
        type=str,
        default=None,
    ),
    cli.ArgumentDefinition(
        "plot_type",
        help=(
            "Plot type as defined in the schema file (required in model parameter mode; "
            "use 'all' to plot all types)."
        ),
        type=str,
        default=None,
    ),
    cli.PARAMETER_VERSION,
    cli.OVERWRITE_MODEL_PARAMETERS,
    cli.SITE,
    cli.TELESCOPE,
)


def _lower_dict_keys(value):
    """Recursively convert dictionary keys to lowercase."""
    if isinstance(value, dict):
        return {key.lower(): _lower_dict_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_lower_dict_keys(item) for item in value]
    return value


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    model_repository=True,
    arguments=(*_ARGUMENTS, *cli.OUTPUT_PATH_ARGUMENTS),
    usage=(
        "simtools-plot-tabular-data (--plot_config config_file_name --output_file output_file_name"
        " | --parameter parameter_name --parameter_version version --site site --plot_type type)"
    ),
    needs=("model", "corsika"),
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    if app_context.args.get("parameter"):
        _plot_model_parameter(app_context)
    else:
        _plot_data_file(app_context)


def _plot_data_file(app_context):
    """Plot a data file using a plotting configuration file."""
    output_file = app_context.args.get("output_file")
    if not output_file:
        raise ValueError("Argument output_file is required with plot_config.")

    plot_config = schema.validate_dict_using_schema(
        ascii_handler.collect_data_from_file(app_context.args["plot_config"]),
        PLOT_CONFIG_SCHEMA,
    )
    plot_config = _lower_dict_keys(plot_config)
    if "__SETTING_WORKFLOW__" in str(plot_config):
        setting_workflow = gen.extract_subdirectories_from_path(
            app_context.args["plot_config"],
            anchor="input",
        )
        plot_config = value_conversion.replace_placeholders_recursively(
            plot_config,
            {"__SETTING_WORKFLOW__": setting_workflow},
        )

    plot_tables.plot(
        config=plot_config["plot"],
        output_file=app_context.paths.get_output_file(output_file),
        data_path=app_context.args.get("data_search_path"),
        model_reader=app_context.model_reader,
        context=app_context,
    )

    MetadataCollector.dump(
        app_context.args,
        app_context.paths.get_output_file(output_file),
        add_activity_name=True,
        context=app_context,
    )


def _plot_model_parameter(app_context):
    """Plot a model parameter using the plot configurations from its schema file."""
    if not app_context.args.get("plot_type"):
        raise ValueError("Argument plot_type is required with parameter.")

    plot_configs, output_files = plot_tables.generate_plot_configurations(
        parameter=app_context.args["parameter"],
        parameter_version=app_context.args["parameter_version"],
        site=app_context.args["site"],
        telescope=app_context.args.get("telescope"),
        output_path=app_context.paths.get_output_directory(),
        plot_type=app_context.args["plot_type"],
        model_reader=app_context.model_reader,
        context=app_context,
    )

    for plot_config, output_file in zip(plot_configs, output_files):
        plot_tables.plot(
            config=plot_config,
            output_file=output_file,
            model_reader=app_context.model_reader,
            context=app_context,
        )
        MetadataCollector.dump(
            app_context.args, output_file=output_file, add_activity_name=True, context=app_context
        )


if __name__ == "__main__":
    main()
