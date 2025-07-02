#!/usr/bin/python3
r"""
Plot tabular data for a single model parameter using default plotting configurations.

Uses plotting configurations as defined in the model parameters schema files.

Command line arguments
----------------------
parameter (str, required)
    Model parameter to plot (e.g., 'atmospheric_profile').
parameter_version (str, required)
    Version of the model parameter to plot (e.g., '1.0.0').
site (str, required)
    Site for which the model parameter is defined (e.g., 'North').
telescope (str, optional)
    Telescope for which the model parameter is defined (e.g., 'LSTN-01').
output_file (str, required)
    Output file name (without suffix).
plot_type (str, optional)
    Type of plot as defined in the schema file.
    Use '--plot_type all' to plot all types defined in the schema.

Example
-------

Plot tabular data for a specific type defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data-for-model-parameter \\
        --parameter atmospheric_profile \\
        --parameter_version 1.0.0 \\
        --site North \\
        --plot_type refractive_index_vs_altitude

Plot tabular data for all types defined in the schema file:

.. code-block:: console

    simtools-plot-tabular-data-for-model-parameter \\
        --parameter fadc_pulse_shape
        --parameter_version 1.0.0 \\
        --site North \\
        --telescope LSTN-01 \\
        --plot_type all

"""

import logging
from pathlib import Path

import simtools.utils.general as gen
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io_operations import io_handler
from simtools.visualization import plot_tables


def _parse(label, description):
    """Parse command line configuration."""
    config = configurator.Configurator(label=label, description=description)

    config.parser.add_argument("--parameter", type=str, required=True, help="Parameter name.")
    config.parser.add_argument(
        "--plot_type",
        help="Plot type as defined in the schema file.",
        type=str,
        required=True,
        default=None,
    )
    return config.initialize(db_config=True, simulation_model=["telescope", "parameter_version"])


def main():
    """Plot tabular data."""
    args_dict, db_config = _parse(
        label=Path(__file__).stem,
        description="Plots tabular data for a model parameter.",
    )
    logger = logging.getLogger()
    logger.setLevel(gen.get_log_level_from_user(args_dict.get("log_level", "INFO")))
    io_handler_instance = io_handler.IOHandler()

    plot_configs, output_files = plot_tables.generate_plot_configurations(
        parameter=args_dict["parameter"],
        parameter_version=args_dict["parameter_version"],
        site=args_dict["site"],
        telescope=args_dict.get("telescope"),
        output_path=io_handler_instance.get_output_directory(),
        plot_type=args_dict["plot_type"],
    )

    for plot_config, output_file in zip(plot_configs, output_files):
        plot_tables.plot(
            config=plot_config,
            output_file=output_file,
            db_config=db_config,
        )
        MetadataCollector.dump(args_dict, output_file=output_file, add_activity_name=True)


if __name__ == "__main__":
    main()
