#!/usr/bin/python3

"""Generate a reduced dataset of event data from output of telescope simulations."""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.configuration import arguments as cli
from simtools.io import io_handler, table_handler
from simtools.sim_events.metadata import build_simulation_metadata, build_standard_metadata
from simtools.sim_events.writer import EventDataWriter

_ARGUMENTS = (
    cli.ArgumentDefinition(
        "simtel_file",
        type=str,
        required=True,
        help="Input file path (wildcards allowed; e.g., '/path/to/gamma_*dark*.simtel.zst')",
    ),
    cli.ArgumentDefinition(
        "max_files",
        type=int,
        default=None,
        help="Maximum number of input files to process (default: all).",
    ),
    cli.ArgumentDefinition(
        "print_dataset_information",
        type=int,
        help="Print data set information for the given number of events.",
        default=0,
    ),
)


APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(
        *_ARGUMENTS,
        *cli.OUTPUT_PATH_ARGUMENTS,
        *cli.OUTPUT_ARGUMENTS,
    ),
    initialize_output=True,
    setup_io_handler=False,
)


def main():
    """See CLI description."""
    app_context = APPLICATION.start()
    app_context.logger.info(f"Loading input files from: {app_context.args['simtel_file']}")

    input_pattern = Path(app_context.args["simtel_file"])
    files = list(input_pattern.parent.glob(input_pattern.name))
    if not files:
        app_context.logger.warning("No matching input files found.")
        return

    output_filepath = io_handler.IOHandler().get_output_file(app_context.args["output_file"])
    if output_filepath.suffix.lower() not in (".hdf5", ".h5"):
        raise ValueError(
            f"Unsupported reduced event data format for '{output_filepath}'. "
            "Only HDF5 files with suffix '.hdf5' or '.h5' are supported."
        )
    generator = EventDataWriter(files, app_context.args["max_files"])
    tables = generator.process_files()
    table_handler.write_tables(
        tables,
        output_filepath,
        overwrite_existing=True,
        file_type="HDF5",
        metadata_documents={
            "METADATA": build_standard_metadata(app_context.args, output_filepath),
            "SIMULATION_METADATA": build_simulation_metadata(
                generator.get_simulation_input_metadata()
            ),
        },
    )

    if app_context.args["print_dataset_information"] > 0:
        for table in tables:
            table.pprint(max_lines=app_context.args["print_dataset_information"], max_width=-1)


if __name__ == "__main__":
    main()
