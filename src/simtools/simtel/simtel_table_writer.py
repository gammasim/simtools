"""Writer for sim_telarray table data files."""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def write_simtel_table(parameter_name, value, dest_dir, telescope_name):
    """
    Write a table parameter to a space-separated ASCII file for sim_telarray.

    Parameters
    ----------
    parameter_name : str
        Parameter name (used as filename prefix).
    value : dict
        Table data in row-oriented format with keys ``columns`` (list of str)
        and ``rows`` (list of lists of float).
    dest_dir : str or Path
        Directory to write the file into.
    telescope_name : str
        Telescope name (used as filename suffix).

    Returns
    -------
    str
        Basename of the written file (``{parameter_name}-{telescope_name}.dat``).

    Raises
    ------
    ValueError
        If ``value`` is not a dict containing ``columns`` and ``rows``.
    """
    if not isinstance(value, dict) or "columns" not in value or "rows" not in value:
        raise ValueError(
            f"Table value for '{parameter_name}' must be a dict with 'columns' and 'rows' keys, "
            f"got {type(value).__name__}."
        )

    file_name = f"{parameter_name}-{telescope_name}.dat"
    file_path = Path(dest_dir) / file_name
    logger.debug(f"Writing sim_telarray table file {file_path}")

    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write(f"# {' '.join(value['columns'])}\n")
        for row in value["rows"]:
            fh.write(" ".join(str(v) for v in row) + "\n")

    return file_name
