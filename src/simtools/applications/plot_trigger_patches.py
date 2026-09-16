#!/usr/bin/python3
"""Plot camera trigger groups and pre-summed pixel patches."""

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.visualization import plot_pixels, plot_trigger_patches, visualize
from simtools.visualization.matplotlib_backend import pyplot as plt

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        cli.MODEL_VERSION,
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
    parameters = app_context.model_reader.get_model_parameters(
        site=app_context.args["site"],
        array_element_name=app_context.args["telescope"],
        collection="telescopes",
        model_version=app_context.args.get("model_version"),
    )
    configuration = plot_pixels.resolve_camera_components(
        app_context.model_reader,
        parameters,
    )
    figure = plot_trigger_patches.plot_trigger_patches(
        configuration,
        app_context.args["telescope"],
    )
    output_file = app_context.io_handler.get_output_directory() / (
        f"trigger-patches-{app_context.args['telescope']}.png"
    )
    visualize.save_figure(figure, output_file)
    plt.close(figure)


if __name__ == "__main__":
    main()
