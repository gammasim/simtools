"""Convert CORSIKA IACT photons to streams and camera streams to sim_telarray input."""

import astropy.units as u

from simtools.corsika.corsika_config import corsika_telescope_ids
from simtools.simulation_software.programs import program_command

SOFTWARE = "obdeect"
FROM_CORSIKA_PROGRAM = "obdeect-stream-from-corsika"
TO_SIMTEL_PROGRAM = "obdeect-stream-to-simtel"


def from_corsika_command(input_file):
    """
    Return the command tokens converting a CORSIKA output file to a photon stream.

    Parameters
    ----------
    input_file : str or Path
        CORSIKA output file, or ``STDIN`` when reading a stream.

    Returns
    -------
    list of str
        Command tokens; the stream is written to stdout.
    """
    return [*program_command(SOFTWARE, FROM_CORSIKA_PROGRAM), str(input_file)]


def to_simtel_command(camera_map, input_file, allow_pending=(), scaled_factors=()):
    """
    Return the command tokens converting a camera-frame photon stream to sim_telarray input.

    Parameters
    ----------
    camera_map : str or Path
        Camera map file (see :func:`write_camera_map`).
    input_file : str or Path
        Photon stream file, or ``STDIN`` when reading a stream.
    allow_pending : iterable of str
        Efficiency factors pending upstream that sim_telarray does not apply.
    scaled_factors : iterable of str
        Efficiency factors whose sim_telarray table holds f/f_env.

    Returns
    -------
    list of str
        Command tokens; the converted stream is written to stdout.
    """
    command = [*program_command(SOFTWARE, TO_SIMTEL_PROGRAM), "--camera-map", str(camera_map)]
    for factor in allow_pending:
        command.extend(["--allow-pending", factor])
    for factor in scaled_factors:
        command.extend(["--scaled-factor", factor])
    command.append(str(input_file))
    return command


def write_camera_map(telescope_models, path):
    """
    Write the camera map of the telescopes of an array and return its file.

    Each line is ``telescope_id mirror_class camera_rotation_deg``, with the CORSIKA telescope
    id of the telescope.

    Parameters
    ----------
    telescope_models : dict
        Telescope models keyed by array element name, in the order of the array.
    path : Path
        Camera map file.

    Returns
    -------
    Path
        The file written.
    """
    ids = corsika_telescope_ids(telescope_models)
    lines = []
    for name, telescope in telescope_models.items():
        mirror_class = telescope.get_parameter_value("mirror_class")
        rotation_deg = telescope.get_parameter_value_with_unit("camera_rotate").to_value(u.deg)
        lines.append(f"{ids[name]} {mirror_class} {rotation_deg}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
