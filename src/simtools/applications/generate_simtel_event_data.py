#!/usr/bin/python3

r"""
Generate a reduced dataset of event data from output of telescope simulations.

Processes sim_telarray output files (typically of type '.simtel.zst') and creates
reduced datasets containing shower information, array-level parameters, and data about
triggered telescopes.

The output consists of an HDF5 or FITS file containing the following tables:

**FILE_INFO**

+-------------------+---------+-----------------------------------------------+
| Field             | Type    | Description                                   |
+===================+=========+===============================================+
| file_name         | string  | Name of the file                              |
+-------------------+---------+-----------------------------------------------+
| file_id           | int64   | Internal unique identifier for the file       |
+-------------------+---------+-----------------------------------------------+
| particle_id       | int64   | PDG particle ID (e.g., 14 for proton)         |
+-------------------+---------+-----------------------------------------------+
| energy_min        | float32 | Minimum simulated energy (TeV)                |
+-------------------+---------+-----------------------------------------------+
| energy_max        | float32 | Maximum simulated energy (TeV)                |
+-------------------+---------+-----------------------------------------------+
| viewcone_min      | float32 | Min viewcone angle (deg)                      |
+-------------------+---------+-----------------------------------------------+
| viewcone_max      | float32 | Max viewcone angle (deg)                      |
+-------------------+---------+-----------------------------------------------+
| core_scatter_min  | float32 | Min core scatter radius (m)                   |
+-------------------+---------+-----------------------------------------------+
| core_scatter_max  | float32 | Max core scatter radius (m)                   |
+-------------------+---------+-----------------------------------------------+
| zenith            | float32 | Zenith angle (deg)                            |
+-------------------+---------+-----------------------------------------------+
| azimuth           | float32 | Azimuth angle (deg)                           |
+-------------------+---------+-----------------------------------------------+
| nsb_level         | float64 | Night sky background level (factor to dark)   |
+-------------------+---------+-----------------------------------------------+

**SHOWERS**

+------------------+---------+-----------------------------------------------+
| Field            | Type    | Description                                   |
+==================+=========+===============================================+
| shower_id        | int64   | Shower identifier                             |
+------------------+---------+-----------------------------------------------+
| event_id         | int64   | Event identifier (depends on reuse of showers)|
+------------------+---------+-----------------------------------------------+
| file_id          | int64   | Internal unique identifier for the file       |
+------------------+---------+-----------------------------------------------+
| simulated_energy | float64 | Simulated primary energy (TeV)                |
+------------------+---------+-----------------------------------------------+
| x_core           | float64 | Shower core X position on ground (m)          |
+------------------+---------+-----------------------------------------------+
| y_core           | float64 | Shower core Y position on ground (m)          |
+------------------+---------+-----------------------------------------------+
| shower_azimuth   | float64 | Direction of shower azimuth (deg)             |
+------------------+---------+-----------------------------------------------+
| shower_altitude  | float64 | Direction of shower altitude (deg)            |
+------------------+---------+-----------------------------------------------+
| area_weight      | float64 | Weighting factor for sampling area            |
+------------------+---------+-----------------------------------------------+

**TRIGGERS**

+-----------------+---------+-----------------------------------------------+
| Field           | Type    | Description                                   |
+=================+=========+===============================================+
| shower_id       | int64   | Shower identifier                             |
+-----------------+---------+-----------------------------------------------+
| event_id        | int64   | Event identifier (depends on reuse of showers)|
+-----------------+---------+-----------------------------------------------+
| file_id         | int64   | Internal unique identifier for the file       |
+-----------------+---------+-----------------------------------------------+
| array_altitude  | float64 | Altitude of array pointing direction (deg)    |
+-----------------+---------+-----------------------------------------------+
| array_azimuth   | float64 | Azimuth of array pointing direction (deg)     |
+-----------------+---------+-----------------------------------------------+
| telescope_list  | string  | Comma-separated list of triggered telescopes  |
+-----------------+---------+-----------------------------------------------+

Several files generated with this application can be combined into a single
dataset using the 'simtools-merge-tables' command.

Command line arguments
----------------------
prefix (str, required)
    Path prefix for the input files.
output_file (str, required)
    Output file path.
max_files (int, optional, default=100)
    Maximum number of input files to process.
print_dataset_information (int, optional, default=0)
    Print information about the datasets in the generated reduced event dataset.

Example
-------
Generate a reduced dataset from input files and save the result.

.. code-block:: console

    simtools-production-extract-mc-event-data \\
    simtools-generate-simtel-event-data \\
        --input 'path/to/input_files/gamma_*dark*.simtel.zst' \\
        --output_file output_file.hdf5 \\
        --max_files 50 \\
        --print_dataset_information 10


To read a reduced event data file, use the following command reading on of the test files:

.. code-block:: console

    import h5py

    test_file = "tests/resources/proton_za20deg_azm000deg_North_alpha_6.0.0_reduced_event_data.hdf5"

    with h5py.File(test_file, "r") as f:
        triggers = f["/TRIGGERS"]
        for row in triggers:
            print({name: row[name] for name in row.dtype.names})

"""

from pathlib import Path

from simtools.application_control import get_application_label, startup_application
from simtools.configuration import configurator
from simtools.data_model.metadata_collector import MetadataCollector
from simtools.io import io_handler, table_handler
from simtools.sim_events.writer import EventDataWriter


def _parse():
    """Parse command line arguments."""
    config = configurator.Configurator(
        label=get_application_label(__file__),
        description=(
            "Process files and store reduced dataset with event information, "
            "array information and triggered telescopes."
        ),
    )

    config.parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input file path (wildcards allowed; e.g., '/data_path/gamma_*dark*.simtel.zst')",
    )
    config.parser.add_argument(
        "--max_files", type=int, default=100, help="Maximum number of input files to process."
    )
    config.parser.add_argument(
        "--print_dataset_information",
        type=int,
        help="Print data set information for the given number of events.",
        default=0,
    )
    return config.initialize(db_config=False, output=True)


def main():
    """Generate a reduced dataset of event data from output of telescope simulations."""
    app_context = startup_application(_parse, setup_io_handler=False)
    app_context.logger.info(f"Loading input files from: {app_context.args['input']}")

    input_pattern = Path(app_context.args["input"])
    files = list(input_pattern.parent.glob(input_pattern.name))
    if not files:
        app_context.logger.warning("No matching input files found.")
        return

    output_filepath = io_handler.IOHandler().get_output_file(app_context.args["output_file"])
    generator = EventDataWriter(files, app_context.args["max_files"])
    tables = generator.process_files()
    table_handler.write_tables(tables, output_filepath, overwrite_existing=True)
    MetadataCollector.dump(
        args_dict=app_context.args, output_file=output_filepath.with_suffix(".yml")
    )

    if app_context.args["print_dataset_information"] > 0:
        for table in tables:
            table.pprint(max_lines=app_context.args["print_dataset_information"], max_width=-1)


if __name__ == "__main__":
    main()
