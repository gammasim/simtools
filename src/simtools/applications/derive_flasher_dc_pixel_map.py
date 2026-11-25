#!/usr/bin/python3
r"""
Derive the relative per-pixel rate map of the FlashCam DC flasher.

The map is built from the true photo-electrons per pixel in sim_telarray files of a
low-intensity full flasher simulation (sim_telarray parameter ``store_photoelectrons``).
The relative rate is the photo-electron sum divided by the mean over enabled pixels.
Slots listed as HV-disabled in the simulation metadata are excluded and set to zero.

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

import numpy as np
from astropy.table import Table

from simtools.application.definition import ApplicationDefinition
from simtools.camera.flasher_dc_pixel_map import (
    active_pixel_mask,
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
    table = _make_pixel_map(pe_sum, active_pixel_mask(simtel_files, telescope))
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


def _make_pixel_map(pe_sum, active_mask):
    """Normalize active pixels and preserve disabled camera slots as zeros."""
    pe = np.asarray(pe_sum, dtype=np.float64)
    active = np.asarray(active_mask, dtype=bool)
    if active.shape != pe.shape:
        raise ValueError("The active pixel mask must match the photo-electron array shape.")
    if not np.any(active):
        raise ValueError("The active pixel mask contains no active pixels.")

    active_map = relative_pixel_map(pe[active])
    table = Table(
        {
            "pixel_id": np.arange(pe.size, dtype=np.int64),
            "relative_rate": np.zeros(pe.size, dtype=np.float64),
            "relative_rate_error": np.zeros(pe.size, dtype=np.float64),
        }
    )
    table["relative_rate"][active] = active_map["relative_rate"]
    table["relative_rate_error"][active] = active_map["relative_rate_error"]
    return table


if __name__ == "__main__":
    main()
