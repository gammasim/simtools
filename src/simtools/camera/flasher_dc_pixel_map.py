"""Relative per-pixel rate map of the FlashCam DC flasher from true photo-electrons.

The map is derived from sim_telarray files written with the ``store_photoelectrons``
parameter. The relative rate of pixel i is ``pe_sum_i / mean(pe_sum)``, so the map has
mean 1 and can scale a flat DC rate per pixel.
"""

import logging
from pathlib import Path

import numpy as np
from astropy.table import Table
from eventio import SimTelFile

from simtools.model.camera import Camera
from simtools.sim_events.simtel_io_metadata import (
    get_sim_telarray_telescope_id_to_telescope_name_mapping,
)
from simtools.visualization.plot_camera import plot_pixel_layout_with_image
from simtools.visualization.visualize import save_figure

_logger = logging.getLogger(__name__)

PIXEL_ID = "pixel_id"
RELATIVE_RATE = "relative_rate"
RELATIVE_RATE_ERROR = "relative_rate_error"


def sum_true_photoelectrons(simtel_files, telescope):
    """
    Sum the true photo-electrons per pixel over all events of the given telescope.

    Parameters
    ----------
    simtel_files : list of str or Path
        sim_telarray files with stored true photo-electrons.
    telescope : str
        Telescope name (e.g., MSTN-design).

    Returns
    -------
    tuple
        Photo-electron sum per pixel and number of events with a telescope PE record.

    Raises
    ------
    ValueError
        If the telescope is not in a file, or no true photo-electrons are stored.
    """
    pe_sum = None
    number_of_events = 0
    for file_name in simtel_files:
        tel_id = _telescope_id(file_name, telescope)
        with SimTelFile(file_name, skip_calibration=False) as simtel:
            pe_tel_id = _photoelectron_telescope_id(simtel.header["tel_id"], tel_id, file_name)
            for event in simtel:
                record = (event.get("photoelectrons") or {}).get(pe_tel_id)
                if record is None:
                    continue
                number_of_events += 1
                counts = np.asarray(record["photoelectrons"], dtype=np.float64)
                pe_sum = counts.copy() if pe_sum is None else _add_counts(pe_sum, counts, file_name)
        _logger.info(f"Read events from '{file_name}' for telescope '{telescope}'.")

    if number_of_events == 0:
        raise ValueError(
            f"No true photo-electrons found for telescope '{telescope}' in the input files. "
            "Set the sim_telarray parameter 'store_photoelectrons' in the simulation."
        )
    return pe_sum, number_of_events


def _telescope_id(file_name, telescope):
    """Return the sim_telarray telescope ID of a telescope name in a file."""
    tel_id_map = get_sim_telarray_telescope_id_to_telescope_name_mapping(file_name)
    tel_id = next((k for k, v in tel_id_map.items() if v == telescope), None)
    if tel_id is None:
        raise ValueError(f"Telescope '{telescope}' not found in file '{file_name}'.")
    return tel_id


def _photoelectron_telescope_id(telescope_ids, telescope_id, file_name):
    """Return the array index used by eventio photoelectron records."""
    try:
        return list(telescope_ids).index(telescope_id)
    except ValueError as exc:
        raise ValueError(
            f"Telescope ID '{telescope_id}' not found in simtel run header from '{file_name}'."
        ) from exc


def _add_counts(pe_sum, counts, file_name):
    """Add per-pixel counts to the running sum, checking the camera size."""
    if pe_sum.shape != counts.shape:
        raise ValueError(
            f"Inconsistent number of pixels in file '{file_name}': "
            f"{counts.shape[0]} instead of {pe_sum.shape[0]}."
        )
    return pe_sum + counts


def relative_pixel_map(pe_sum):
    """
    Compute the relative pixel rate map from the photo-electron sum per pixel.

    Parameters
    ----------
    pe_sum : numpy.ndarray
        True photo-electrons per pixel.

    Returns
    -------
    astropy.table.Table
        Table with columns ``pixel_id``, ``relative_rate`` (mean 1) and
        ``relative_rate_error`` (Poisson error, sqrt(N)/mean).

    Raises
    ------
    ValueError
        If the mean photo-electron count per pixel is zero.
    """
    pe = np.asarray(pe_sum, dtype=np.float64)
    mean = pe.mean() if pe.size else 0.0
    if mean <= 0:
        raise ValueError("Mean photo-electron count per pixel is zero; cannot form a relative map.")
    return Table(
        {
            PIXEL_ID: np.arange(pe.size, dtype=np.int64),
            RELATIVE_RATE: pe / mean,
            RELATIVE_RATE_ERROR: np.sqrt(pe) / mean,
        }
    )


def active_pixel_mask(simtel_files, telescope):
    """Return a mask using the HV-disabled pixel list in each simulation file."""
    reference_mask = None
    for file_name in simtel_files:
        tel_id = _telescope_id(file_name, telescope)
        with SimTelFile(file_name, skip_calibration=False) as simtel:
            description = simtel.telescope_descriptions[tel_id]
            camera_size = len(description["camera_settings"]["pixel_x"])
            try:
                disabled = description["disabled_pixels"]["HV_disabled"]
            except KeyError as exc:
                raise ValueError(
                    f"Disabled-pixel metadata is missing for telescope '{telescope}' "
                    f"in '{file_name}'."
                ) from exc
            mask = np.ones(camera_size, dtype=bool)
            disabled_ids = np.asarray(disabled, dtype=np.int64)
            if np.any((disabled_ids < 0) | (disabled_ids >= camera_size)):
                raise ValueError(f"Disabled pixel ID is outside the camera range in '{file_name}'.")
            mask[disabled_ids] = False
            if reference_mask is None:
                reference_mask = mask
            elif not np.array_equal(reference_mask, mask):
                raise ValueError(
                    f"Disabled-pixel metadata differs between input files for '{telescope}'."
                )
    if reference_mask is None:
        raise ValueError(f"No sim_telarray files supplied for telescope '{telescope}'.")
    return reference_mask


def write_pixel_map(table, output_file, *, number_of_events=None, telescope=None, input_files=None):
    """
    Write the relative pixel map as ECSV.

    Parameters
    ----------
    table : astropy.table.Table
        Relative pixel map from :func:`relative_pixel_map`.
    output_file : str or Path
        Output ECSV file.
    number_of_events : int, optional
        Number of events with a telescope PE record, stored in the table metadata.
    telescope : str, optional
        Telescope name, stored in the table metadata.
    input_files : list of str or Path, optional
        Input sim_telarray files, stored in the table metadata as ``input_files``.
    """
    metadata = {
        "number_of_events": number_of_events,
        "telescope": telescope,
        "input_files": None if input_files is None else [str(f) for f in input_files],
    }
    table.meta.update({key: value for key, value in metadata.items() if value is not None})
    table.write(Path(output_file), format="ascii.ecsv", overwrite=True)


def read_camera(simtel_file, telescope):
    """Return the camera geometry of a telescope from a sim_telarray file."""
    tel_id = _telescope_id(simtel_file, telescope)
    with SimTelFile(simtel_file, skip_calibration=False) as simtel:
        tel_desc = simtel.telescope_descriptions[tel_id]
    return Camera(
        telescope_name=telescope,
        camera_config_file=None,
        focal_length=tel_desc["camera_settings"]["focal_length"],
        camera_config_dict=tel_desc["camera_settings"],
    )


def plot_pixel_map(table, telescope_camera, output_file):
    """
    Plot the relative pixel rates on the camera.

    Parameters
    ----------
    table : astropy.table.Table
        Relative pixel map from :func:`relative_pixel_map`.
    telescope_camera : simtools.model.camera.Camera
        Camera geometry of the telescope.
    output_file : str or Path
        Output figure file.
    """
    fig = plot_pixel_layout_with_image(
        telescope_camera,
        image=np.asarray(table[RELATIVE_RATE]),
        color_bar_label="Relative DC rate",
    )
    save_figure(fig, output_file, log_title="flasher DC pixel map")
