#!/usr/bin/python3
r"""
Derive the relative per-pixel rate map of the FlashCam DC flasher.

The map is built from the true photo-electrons per pixel in sim_telarray files of a
low-intensity full flasher simulation (sim_telarray parameter ``store_photoelectrons``).
The relative rate of each pixel is its photo-electron sum divided by the camera mean.

Example
-------
.. code-block:: console

    simtools-derive-flasher-dc-pixel-map \
        --simtel_files flasher_run_*.simtel.zst \
        --telescope MSTN-design \
        --output_file flasher_dc_relative_pixel_map.ecsv \
        --plot
"""

from pathlib import Path

from simtools.application.definition import ApplicationDefinition
from simtools.camera.flasher_dc_pixel_map import (
    plot_pixel_map,
    read_camera,
    relative_pixel_map,
    sum_true_photoelectrons,
    write_pixel_map,
)
from simtools.configuration import arguments as cli
from simtools.utils.general import resolve_file_patterns

SIMTEL_FILES = "simtel_files"
OUTPUT_FILE = "output_file"
PLOT = "plot"
DEFAULT_OUTPUT_FILE = "flasher_dc_relative_pixel_map.ecsv"

_ARGUMENTS = (
    cli.ArgumentDefinition(
        SIMTEL_FILES,
        nargs="+",
        required=True,
        help=(
            "sim_telarray files with stored true photo-electrons (store_photoelectrons); "
            "wildcards allowed"
        ),
    ),
    cli.ArgumentDefinition(
        "telescope",
        type=str,
        required=True,
        help="Telescope name (e.g., MSTN-design)",
    ),
    cli.ArgumentDefinition(
        OUTPUT_FILE,
        type=str,
        default=DEFAULT_OUTPUT_FILE,
        help="Output ECSV file with the relative pixel map",
    ),
    cli.ArgumentDefinition(
        PLOT,
        action="store_true",
        default=False,
        help="Plot the relative pixel map on the camera",
    ),
)

APPLICATION = ApplicationDefinition.for_module(
    __name__,
    arguments=(*_ARGUMENTS, *cli.OUTPUT_PATH_ARGUMENTS),
)


def main():
    """See CLI description."""
    context = APPLICATION.start()
    args = context.args
    output_dir = context.paths.get_output_directory()
    simtel_files = sorted(resolve_file_patterns(args[SIMTEL_FILES]))
    telescope = args["telescope"]

    pe_sum, number_of_events = sum_true_photoelectrons(simtel_files, telescope)
    table = relative_pixel_map(pe_sum)
    write_pixel_map(
        table,
        Path(output_dir) / args[OUTPUT_FILE],
        number_of_events=number_of_events,
        telescope=telescope,
        input_files=simtel_files,
    )
    if args[PLOT]:
        camera = read_camera(simtel_files[0], telescope)
        plot_pixel_map(table, camera, Path(output_dir) / Path(args[OUTPUT_FILE]).stem)


if __name__ == "__main__":
    main()
