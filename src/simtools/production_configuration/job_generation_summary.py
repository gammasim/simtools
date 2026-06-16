"""Bookkeeping helpers for streaming production job generation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class GeneratedRowSummary:
    """
    Track generated-row ranges without retaining all rows.

    Attributes
    ----------
    count : int
        Number of rows added to the summary.
    energy_min, energy_max, core_scatter_max, view_cone_min, view_cone_max : tuple or None
        Minimum and maximum bounds for the corresponding generated-row quantities.
    """

    count: int = 0
    energy_min: object = None
    energy_max: object = None
    core_scatter_max: object = None
    view_cone_min: object = None
    view_cone_max: object = None

    def add(self, row):
        """
        Add one generated row to the summary.

        Parameters
        ----------
        row : dict
            Generated job row containing energy, core-scatter, and view-cone quantities.
        """
        self.count += 1
        self.energy_min = update_quantity_bounds(self.energy_min, row["energy_min"])
        self.energy_max = update_quantity_bounds(self.energy_max, row["energy_max"])
        self.core_scatter_max = update_quantity_bounds(
            self.core_scatter_max, row["core_scatter_max"]
        )
        self.view_cone_min = update_quantity_bounds(self.view_cone_min, row["view_cone_min"])
        self.view_cone_max = update_quantity_bounds(self.view_cone_max, row["view_cone_max"])


@dataclass
class ShowerRoundingSummary:
    """
    Aggregate total-showers rounding warnings for large grids.

    Attributes
    ----------
    count : int
        Number of grid points requiring a total-showers adjustment.
    first_effective_total_showers : int or None
        Effective total showers for the first adjusted grid point.
    first_showers_per_run : int or None
        Showers per run for the first adjusted grid point.
    first_adjusted_total_showers : int or None
        Rounded total showers for the first adjusted grid point.
    """

    count: int = 0
    first_effective_total_showers: int | None = None
    first_showers_per_run: int | None = None
    first_adjusted_total_showers: int | None = None

    def add(self, effective_total_showers, selected_showers_per_run, adjusted_total_showers):
        """
        Record one total-showers rounding adjustment.

        Parameters
        ----------
        effective_total_showers : int
            Total showers requested for the grid point after any configured scaling.
        selected_showers_per_run : int
            Showers per run selected for the grid point.
        adjusted_total_showers : int
            Total showers after rounding up to full runs.
        """
        self.count += 1
        if self.first_effective_total_showers is None:
            self.first_effective_total_showers = effective_total_showers
            self.first_showers_per_run = selected_showers_per_run
            self.first_adjusted_total_showers = adjusted_total_showers

    def log(self, logger):
        """
        Log one aggregate warning for all recorded adjustments.

        Parameters
        ----------
        logger : logging.Logger
            Logger used for the warning message.
        """
        if self.count == 0:
            return
        logger.warning(
            "total_showers was not divisible by showers_per_run for %d grid point(s); "
            "adjusted each affected point to keep equal showers per run. "
            "First adjustment: total_showers=%s, showers_per_run=%s, adjusted_total_showers=%s.",
            self.count,
            self.first_effective_total_showers,
            self.first_showers_per_run,
            self.first_adjusted_total_showers,
        )


@dataclass
class SimulationJobContext:
    """
    Resolved inputs shared by simulation-job generation passes.

    Attributes
    ----------
    grid_axes : dict
        Normalized cartesian axes for primary, model, and interaction expansion.
    energy_ranges : list
        Energy range pairs used when building job rows.
    showers_per_run : int
        Baseline showers per run.
    showers_per_run_power_law : tuple or None
        Optional energy-dependent showers-per-run scaling definition.
    showers_per_run_scaling : str
        Zenith-dependent showers-per-run scaling mode.
    total_showers : int or None
        Requested total showers per grid point.
    total_showers_scaling : str
        Total-showers scaling mode.
    zenith_angle_scaling_factor : float
        Scaling factor for zenith-dependent total showers.
    energy_max_scaling : tuple or None
        Optional zenith-dependent maximum-energy scaling definition.
    number_of_runs : int
        Fixed number of runs when total showers is not configured.
    run_number : int
        First run number to assign at each grid point.
    core_scatter : list
        Configured core-scatter definition.
    view_cone_min : astropy.units.Quantity
        Configured minimum view-cone angle.
    configured_view_cone_max : astropy.units.Quantity
        Configured maximum view-cone angle.
    core_scatter_number : int
        Number of core-scatter samples.
    nsb_rates_per_model_version : dict
        NSB interpolation rates keyed by model version.
    observation_grids_per_model_version : dict
        Generated observation grids keyed by model version.
    resolved_layout_names : dict
        Resolved array layout names keyed by model version.
    """

    grid_axes: dict
    energy_ranges: list
    showers_per_run: int
    showers_per_run_power_law: tuple | None
    showers_per_run_scaling: str
    total_showers: int | None
    total_showers_scaling: str
    zenith_angle_scaling_factor: float
    energy_max_scaling: tuple | None
    number_of_runs: int
    run_number: int
    core_scatter: list
    view_cone_min: object
    configured_view_cone_max: object
    core_scatter_number: int
    nsb_rates_per_model_version: dict
    observation_grids_per_model_version: dict
    resolved_layout_names: dict


def update_quantity_bounds(bounds, value):
    """
    Update quantity min/max bounds with one value.

    Parameters
    ----------
    bounds : tuple or None
        Existing ``(minimum, maximum)`` quantity bounds, or None for the first value.
    value : astropy.units.Quantity
        Quantity to include in the bounds.

    Returns
    -------
    tuple
        Updated ``(minimum, maximum)`` quantity bounds.
    """
    if bounds is None:
        return value, value
    value_for_min = value.to(bounds[0].unit)
    value_for_max = value.to(bounds[1].unit)
    return min(bounds[0], value_for_min), max(bounds[1], value_for_max)


def format_quantity_bounds(bounds):
    """
    Format precomputed quantity min/max bounds.

    Parameters
    ----------
    bounds : tuple
        ``(minimum, maximum)`` quantity bounds to format.

    Returns
    -------
    str
        Compact value or range string with unit.
    """
    quantity_min, quantity_max = bounds
    summary_unit = quantity_max.unit
    min_value = quantity_min.to_value(summary_unit)
    max_value = quantity_max.to_value(summary_unit)
    if np.isclose(min_value, max_value):
        return f"{max_value:.6g} {summary_unit}"
    return f"[{min_value:.6g}, {max_value:.6g}] {summary_unit}"


def log_streamed_row_summary(summary, logger):
    """
    Log a compact summary collected during streaming generation.

    Parameters
    ----------
    summary : GeneratedRowSummary
        Summary object populated while job rows were streamed.
    logger : logging.Logger
        Logger used for summary messages.
    """
    if summary.count == 0:
        logger.info("Generated 0 simulation rows after applying all clipping and scaling rules.")
        return
    logger.info("Generated %d simulation rows.", summary.count)
    logger.info(
        "Energy range after clipping/scaling: Emin %s, Emax %s.",
        format_quantity_bounds(summary.energy_min),
        format_quantity_bounds(summary.energy_max),
    )
    logger.info("Core scatter max range: %s.", format_quantity_bounds(summary.core_scatter_max))
    logger.info(
        "View cone range: min %s, max %s.",
        format_quantity_bounds(summary.view_cone_min),
        format_quantity_bounds(summary.view_cone_max),
    )
