"""Writer for sim_telarray table data files."""

import logging
from pathlib import Path

import numpy as np

from simtools.simtel.pulse_shapes import generate_pulse_from_rise_fall_times

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


def write_light_pulse_table_gauss_exp_conv(
    file_path,
    width_ns,
    exp_decay_ns,
    fadc_sum_bins,
    dt_ns=0.1,
    rise_range=(0.1, 0.9),
    fall_range=(0.9, 0.1),
    time_margin_ns=10.0,
):
    """Write a pulse table for a Gaussian convolved with a causal exponential.

    Parameters
    ----------
    file_path : str or pathlib.Path
        Destination path of the ASCII pulse table to write. Parent directory must exist.
    width_ns : float
        Target rise time in ns between the fractional levels defined by ``rise_range``.
    exp_decay_ns : float
        Target fall time in ns between the fractional levels defined by ``fall_range``.
    fadc_sum_bins : int
        Length of the FADC integration window (treated as ns here) used to derive
        the internal time sampling window of the solver as [-(margin), bins + margin].
    dt_ns : float, optional
        Time sampling step in ns for the generated pulse table.
    rise_range : tuple[float, float], optional
        Fractional amplitude bounds (low, high) for rise-time definition.
    fall_range : tuple[float, float], optional
        Fractional amplitude bounds (high, low) for fall-time definition.
    time_margin_ns : float, optional
        Margin in ns to add to both ends of the FADC window when ``fadc_sum_bins`` is given.

    Returns
    -------
    pathlib.Path
        The path to the created pulse table file.

    Notes
    -----
    The underlying model is a Gaussian convolved with a causal exponential. The model
    parameters (sigma, tau) are solved such that the normalized pulse matches the requested
    rise and fall times. The pulse is normalized to a peak amplitude of 1.
    """
    if width_ns is None or exp_decay_ns is None:
        raise ValueError("width_ns (rise 10-90) and exp_decay_ns (fall 90-10) are required")
    logger.info(
        "Generating pulse-shape table with "
        f"rise{int(rise_range[0] * 100)}-{int(rise_range[1] * 100)}={width_ns} ns, "
        f"fall{int(fall_range[0] * 100)}-{int(fall_range[1] * 100)}={exp_decay_ns} ns, "
        f"dt={dt_ns} ns"
    )
    width = float(fadc_sum_bins)
    t_start_ns = -abs(time_margin_ns + width)
    t_stop_ns = +abs(time_margin_ns + width)
    t, y = generate_pulse_from_rise_fall_times(
        width_ns,
        exp_decay_ns,
        dt_ns=dt_ns,
        rise_range=rise_range,
        fall_range=fall_range,
        t_start_ns=t_start_ns,
        t_stop_ns=t_stop_ns,
        center_on_peak=True,
    )

    return write_ascii_pulse_table(file_path, t, y)


def write_angular_distribution_table_lambertian(
    file_path,
    max_angle_deg,
    n_samples=100,
):
    """Write a Lambertian angular distribution table (I(t) ~ cos(t)).

    Parameters
    ----------
    file_path : str or pathlib.Path
        Destination path of the ASCII table to write. Parent directory must exist.
    max_angle_deg : float
        Maximum angle (deg) for the distribution sampling range [0, max_angle_deg].
    n_samples : int, optional
        Number of samples (including end point) from 0 to max_angle_deg. Default 100.

    Returns
    -------
    pathlib.Path
        Path to created angular distribution table.
    """
    logger.info(
        f"Generating Lambertian angular distribution table up to {max_angle_deg} deg "
        f"with {n_samples} samples"
    )
    angles = np.linspace(0.0, float(max_angle_deg), int(n_samples), dtype=float)
    intensities = np.cos(np.deg2rad(angles))
    intensities[intensities < 0] = 0.0
    if intensities.max() > 0:
        intensities /= intensities.max()

    return write_ascii_angle_distribution_table(file_path, angles, intensities)


def write_ascii_pulse_table(file_path, t, y):
    """Write two-column ASCII pulse table."""
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("# time[ns] amplitude\n")
        for ti, yi in zip(t, y):
            fh.write(f"{ti:.6f} {yi:.8f}\n")
    return Path(file_path)


def write_ascii_angle_distribution_table(file_path, angles, intensities):
    """Write two-column ASCII angular distribution table."""
    with open(file_path, "w", encoding="utf-8") as fh:
        fh.write("# angle[deg] relative_intensity\n")
        for a, i in zip(angles, intensities):
            fh.write(f"{a:.6f} {i:.8f}\n")
    return Path(file_path)
