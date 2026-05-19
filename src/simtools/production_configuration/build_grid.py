"""Build simulation execution grid based on the production configuration and lookup tables."""

import itertools
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import EarthLocation
from astropy.time import Time

from simtools.configuration import defaults
from simtools.io.ascii_handler import collect_data_from_file
from simtools.layout.array_layout_utils import resolve_array_layout_name
from simtools.model.site_model import SiteModel
from simtools.production_configuration.corsika_limits_lookup import CorsikaLimitsLookup
from simtools.production_configuration.grid_engine import ProductionGridEngine
from simtools.utils.general import ensure_list

_GRID_AXES = [
    "primary",
    "azimuth_angle",
    "zenith_angle",
    "model_version",
    "corsika_le_interaction",
    "corsika_he_interaction",
]

_GRID_AXIS_DEFAULTS = {
    "corsika_le_interaction": defaults.CORSIKA_LE_INTERACTION,
    "corsika_he_interaction": defaults.CORSIKA_HE_INTERACTION,
}


def load_axes(file_path):
    """Load axes definitions from a YAML or JSON file."""
    if not Path(file_path).exists():
        raise FileNotFoundError(f"Axes file {file_path} not found.")

    return collect_data_from_file(file_path)


def resolve_observing_time(observing_time, coordinate_system):
    """Resolve observing time from CLI arguments."""
    if observing_time:
        return Time(observing_time, scale="utc")
    if coordinate_system == "ra_dec":
        return Time.now()
    return None


def resolve_single_model_version(model_version):
    """Resolve one model version for helpers that require a scalar version."""
    if isinstance(model_version, list):
        return model_version[0]
    return model_version


def build_observing_location(site, model_version):
    """Build observing location from the site model."""
    site_model = SiteModel(model_version=resolve_single_model_version(model_version), site=site)
    return EarthLocation(
        lat=site_model.get_parameter_value_with_unit("reference_point_latitude"),
        lon=site_model.get_parameter_value_with_unit("reference_point_longitude"),
        height=site_model.get_parameter_value_with_unit("reference_point_altitude"),
    )


def build_production_grid_engine(args_dict, array_layout_name=None):
    """Build a production-grid engine from application arguments."""
    coordinate_system = args_dict.get("coordinate_system", "horizontal")
    observing_location = None
    if coordinate_system == "ra_dec":
        observing_location = build_observing_location(
            site=args_dict["site"],
            model_version=args_dict["model_version"],
        )
    resolved_layout_name = array_layout_name or resolve_array_layout_name(
        args_dict.get("array_layout_name"),
        resolve_single_model_version(args_dict.get("model_version")),
    )

    return ProductionGridEngine(
        axes=load_axes(args_dict["axes"]),
        coordinate_system=coordinate_system,
        observing_location=observing_location,
        observing_time=resolve_observing_time(
            args_dict.get("observing_time"),
            coordinate_system,
        ),
        lookup_table=args_dict.get("lookup_table"),
        array_layout_name=resolved_layout_name,
    )


def build_job_grid_metadata(args_dict):
    """Build metadata stored alongside serialized executable job grids."""
    observing_time = resolve_observing_time(
        args_dict.get("observing_time"),
        args_dict.get("coordinate_system", "horizontal"),
    )
    return {
        "site": args_dict.get("site"),
        "simulation_software": args_dict.get("simulation_software"),
        "coordinate_system": args_dict.get("coordinate_system", "horizontal"),
        "observing_time_utc": observing_time.isot if observing_time else None,
        "observing_time_scale": observing_time.scale if observing_time else None,
        "lookup_table": str(args_dict["lookup_table"]) if args_dict.get("lookup_table") else None,
    }


def normalize_grid_axes(args_dict):
    """Return normalized grid axes for cartesian product expansion."""
    return {
        axis: (
            ensure_list(args_dict[axis])
            if axis in args_dict and args_dict[axis] is not None
            else [_GRID_AXIS_DEFAULTS[axis]]
            if axis in _GRID_AXIS_DEFAULTS
            else [None]
        )
        for axis in _GRID_AXES
    }


def normalize_energy_ranges(energy_range):
    """Normalize energy range argument to a list of ``(e_min, e_max)`` pairs."""
    if isinstance(energy_range, tuple) and len(energy_range) == 2:
        return [energy_range]

    if isinstance(energy_range, list):
        if len(energy_range) == 2 and all(hasattr(item, "to") for item in energy_range):
            return [(energy_range[0], energy_range[1])]
        if all(isinstance(item, (list, tuple)) and len(item) == 2 for item in energy_range):
            return [tuple(item) for item in energy_range]

    raise ValueError(
        "energy_range must be one pair (e_min, e_max) or a list of (e_min, e_max) pairs."
    )


def get_energy_range_for_zenith_angle(
    zenith_angle, energy_range_pair, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """
    Return a zenith-dependent energy range pair or ``None`` to skip the step.

    The lower energy bound is clipped to the lookup-table threshold for the
    requested direction. If the threshold exceeds the configured upper bound,
    the simulation step is skipped.
    """
    if corsika_limits is None:
        return energy_range_pair

    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)

    azimuth_angle = 0.0 * u.deg if azimuth_angle is None else azimuth_angle
    interpolated_limits = corsika_limits.interpolate_point(zenith_angle, azimuth_angle, nsb_level)
    lower_energy_threshold = interpolated_limits["lower_energy_threshold"] * u.TeV

    energy_min, energy_max = energy_range_pair
    if lower_energy_threshold > energy_max.to(lower_energy_threshold.unit):
        return None
    if lower_energy_threshold <= energy_min.to(lower_energy_threshold.unit):
        return energy_range_pair
    return lower_energy_threshold.to(energy_min.unit), energy_max


def get_core_scatter_max_for_zenith_angle(
    zenith_angle, core_scatter, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """
    Return zenith-dependent max core-scatter value.

    The lookup-table scatter radius is treated as an upper limit and therefore
    clipped against the user-provided maximum value.
    """
    if corsika_limits is None:
        return core_scatter[1]

    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)

    azimuth_angle = 0.0 * u.deg if azimuth_angle is None else azimuth_angle
    interpolated_limits = corsika_limits.interpolate_point(zenith_angle, azimuth_angle, nsb_level)
    lookup_scatter_max = interpolated_limits["upper_scatter_radius"] * u.m
    configured_scatter_max = core_scatter[1]
    return min(configured_scatter_max, lookup_scatter_max.to(configured_scatter_max.unit))


def get_viewcone_max_for_zenith_angle(
    zenith_angle, view_cone, corsika_limits, azimuth_angle=None, nsb_level=1.0
):
    """Return zenith-dependent max viewcone value."""
    if corsika_limits is None:
        return view_cone[1]

    if not isinstance(corsika_limits, CorsikaLimitsLookup):
        corsika_limits = CorsikaLimitsLookup(corsika_limits)

    azimuth_angle = 0.0 * u.deg if azimuth_angle is None else azimuth_angle
    interpolated_limits = corsika_limits.interpolate_point(zenith_angle, azimuth_angle, nsb_level)
    lookup_viewcone_max = interpolated_limits["viewcone_radius"] * u.deg
    configured_viewcone_max = view_cone[1]
    return min(configured_viewcone_max, lookup_viewcone_max.to(configured_viewcone_max.unit))


def calculate_log_energy_midpoint(energy_range_pair):
    """Return the geometric-mean energy for an energy range pair."""
    energy_min, energy_max = energy_range_pair

    if not isinstance(energy_min, u.Quantity) or not isinstance(energy_max, u.Quantity):
        raise TypeError("energy_range_pair must contain astropy Quantity values.")

    energy_min_tev = energy_min.to(u.TeV)
    energy_max_tev = energy_max.to(u.TeV)

    if energy_min_tev <= 0 * u.TeV or energy_max_tev <= 0 * u.TeV:
        raise ValueError("Energy range values must be strictly positive.")

    mean_log_energy = np.mean(
        [
            np.log10(energy_min_tev.value),
            np.log10(energy_max_tev.value),
        ]
    )
    return 10**mean_log_energy * u.TeV


def calculate_scaled_nshow(
    energy_range_pair,
    baseline_nshow,
    nshow_power_index=None,
    reference_energy=None,
):
    """Return an energy-dependent nshow value."""
    if baseline_nshow < 1:
        raise ValueError("baseline_nshow must be a positive integer.")

    if nshow_power_index is None:
        return baseline_nshow

    if reference_energy is None:
        raise ValueError("reference_energy is required when nshow_power_index is configured.")

    midpoint_energy = calculate_log_energy_midpoint(energy_range_pair)
    scaling_factor = (midpoint_energy / reference_energy.to(midpoint_energy.unit)).to_value(
        u.dimensionless_unscaled
    ) ** nshow_power_index
    scaled_nshow = int(np.ceil(baseline_nshow * scaling_factor))

    if scaled_nshow < 1:
        raise ValueError("Scaled nshow must be at least 1.")

    return scaled_nshow


def _clip_energy_range_from_threshold(energy_range_pair, lower_energy_threshold):
    """Clip the lower energy bound of a configured energy range."""
    if lower_energy_threshold is None:
        return energy_range_pair

    energy_min, energy_max = energy_range_pair
    lower_energy_threshold = lower_energy_threshold.to(energy_min.unit)
    if lower_energy_threshold > energy_max.to(lower_energy_threshold.unit):
        return None
    if lower_energy_threshold <= energy_min:
        return energy_range_pair
    return lower_energy_threshold, energy_max


def _clip_max_quantity(configured_max, lookup_max):
    """Clip a configured maximum value against an interpolated lookup value."""
    if lookup_max is None:
        return configured_max
    return min(configured_max, lookup_max.to(configured_max.unit))


def _build_simulation_jobs_from_production_grid(
    args_dict,
    energy_ranges,
    grid_axes,
    number_of_runs,
    run_number,
    nshow,
    nshow_power_index,
    reference_energy,
):
    """Build simulation jobs from a shared production-grid definition."""
    core_scatter_number = int(args_dict["core_scatter"][0])
    view_cone_min = args_dict["view_cone"][0]
    rows = []
    grid_points_by_layout = {}

    for (
        primary,
        model_version,
        corsika_le,
        corsika_he,
    ) in itertools.product(
        grid_axes["primary"],
        grid_axes["model_version"],
        grid_axes["corsika_le_interaction"],
        grid_axes["corsika_he_interaction"],
    ):
        resolved_layout_name = resolve_array_layout_name(
            args_dict.get("array_layout_name"),
            model_version,
        )
        if resolved_layout_name not in grid_points_by_layout:
            grid_points_by_layout[resolved_layout_name] = build_production_grid_engine(
                args_dict,
                array_layout_name=resolved_layout_name,
            ).generate_simulation_grid()

        grid_points = grid_points_by_layout[resolved_layout_name]
        for point in grid_points:
            selected_core_scatter_max = _clip_max_quantity(
                args_dict["core_scatter"][1],
                point.get("scatter_radius"),
            )
            selected_viewcone_max = _clip_max_quantity(
                args_dict["view_cone"][1],
                point.get("viewcone_radius"),
            )

            for energy_range_pair in energy_ranges:
                selected_energy_range_pair = _clip_energy_range_from_threshold(
                    energy_range_pair,
                    point.get("lower_energy_threshold"),
                )
                if selected_energy_range_pair is None:
                    continue

                selected_nshow = calculate_scaled_nshow(
                    selected_energy_range_pair, nshow, nshow_power_index, reference_energy
                )

                for row_index in range(number_of_runs):
                    rows.append(
                        {
                            "primary": primary,
                            "azimuth_angle": point["azimuth"],
                            "zenith_angle": point["zenith_angle"],
                            "model_version": model_version,
                            "array_layout_name": resolved_layout_name,
                            "corsika_le_interaction": corsika_le,
                            "corsika_he_interaction": corsika_he,
                            "energy_min": selected_energy_range_pair[0],
                            "energy_max": selected_energy_range_pair[1],
                            "core_scatter_number": core_scatter_number,
                            "core_scatter_max": selected_core_scatter_max,
                            "view_cone_min": view_cone_min,
                            "view_cone_max": selected_viewcone_max,
                            "nshow": selected_nshow,
                            "run_number": run_number + row_index,
                        }
                    )
    return rows


def build_simulation_jobs(args_dict):
    """
    Build simulation job parameters based on the production configuration and lookup tables.

    Each entry corresponds to a single simulation job/run and contains all parameters needed to
    launch that simulation (e.g., primary, zenith, azimuth, energy range, model version, etc.).
    The energy range and core-scatter radius can be adjusted based on the direction-dependent
    lookup-table limits. The number of events (``nshow``) can be scaled with energy according
    to a configurable power law.

    Parameters
    ----------
    args_dict : dict
        Production-job configuration.

    Returns
    -------
    list[dict]
        Simulation job parameter dictionaries for all combinations of configured axes.
    """
    grid_axes = normalize_grid_axes(args_dict)
    energy_ranges = normalize_energy_ranges(args_dict["energy_range"])
    if args_dict.get("axes"):
        nshow_power_index = args_dict.get("nshow_power_index")
        reference_energy = args_dict.get("nshow_reference_energy")
        if nshow_power_index is not None and reference_energy is not None:
            reference_energy = u.Quantity(reference_energy)

        return _build_simulation_jobs_from_production_grid(
            args_dict=args_dict,
            energy_ranges=energy_ranges,
            grid_axes=grid_axes,
            number_of_runs=int(args_dict.get("number_of_runs", 1)),
            run_number=int(args_dict.get("run_number") or 1),
            nshow=args_dict["nshow"],
            nshow_power_index=nshow_power_index,
            reference_energy=reference_energy,
        )

    corsika_limits_path = args_dict.get("corsika_limits")
    corsika_limits_by_layout = {}

    core_scatter = args_dict["core_scatter"]
    view_cone = args_dict["view_cone"]
    core_scatter_number = int(core_scatter[0])
    view_cone_min = view_cone[0]
    nshow = args_dict["nshow"]
    nshow_power_index = args_dict.get("nshow_power_index")
    reference_energy = args_dict.get("nshow_reference_energy")
    if nshow_power_index is not None and reference_energy is not None:
        reference_energy = u.Quantity(reference_energy)

    combinations = list(
        itertools.product(
            grid_axes["primary"],
            grid_axes["azimuth_angle"],
            grid_axes["zenith_angle"],
            grid_axes["model_version"],
            grid_axes["corsika_le_interaction"],
            grid_axes["corsika_he_interaction"],
            energy_ranges,
        )
    )

    number_of_runs = args_dict.get("number_of_runs", 1)
    run_number = int(args_dict.get("run_number") or 1)

    rows = []
    row_index = 0
    for (
        primary,
        azimuth,
        zenith,
        model_version,
        corsika_le,
        corsika_he,
        energy_range_pair,
    ) in combinations:
        resolved_layout_name = resolve_array_layout_name(
            args_dict.get("array_layout_name"),
            model_version,
        )
        corsika_limits = None
        if corsika_limits_path is not None:
            if resolved_layout_name not in corsika_limits_by_layout:
                corsika_limits_by_layout[resolved_layout_name] = CorsikaLimitsLookup(
                    corsika_limits_path,
                    array_layout_name=resolved_layout_name,
                )
            corsika_limits = corsika_limits_by_layout[resolved_layout_name]

        selected_energy_range_pair = get_energy_range_for_zenith_angle(
            zenith,
            energy_range_pair,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        if selected_energy_range_pair is None:
            continue

        selected_core_scatter_max = get_core_scatter_max_for_zenith_angle(
            zenith,
            core_scatter,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        selected_viewcone_max = get_viewcone_max_for_zenith_angle(
            zenith,
            view_cone,
            corsika_limits,
            azimuth_angle=azimuth,
        )
        selected_nshow = calculate_scaled_nshow(
            selected_energy_range_pair, nshow, nshow_power_index, reference_energy
        )

        for row_index in range(number_of_runs):
            rows.append(
                {
                    "primary": primary,
                    "azimuth_angle": azimuth,
                    "zenith_angle": zenith,
                    "model_version": model_version,
                    "array_layout_name": resolved_layout_name,
                    "corsika_le_interaction": corsika_le,
                    "corsika_he_interaction": corsika_he,
                    "energy_min": selected_energy_range_pair[0],
                    "energy_max": selected_energy_range_pair[1],
                    "core_scatter_number": core_scatter_number,
                    "core_scatter_max": selected_core_scatter_max,
                    "view_cone_min": view_cone_min,
                    "view_cone_max": selected_viewcone_max,
                    "nshow": selected_nshow,
                    "run_number": run_number + row_index,
                }
            )
    return rows
