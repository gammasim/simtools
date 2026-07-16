"""Estimate required Monte Carlo statistics from trigger histogram products."""

import logging

import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools import settings
from simtools.io import io_handler
from simtools.io.file_type import validate_file_type
from simtools.production_configuration.derive_corsika_limits import FILE_INFO_COLUMNS
from simtools.production_configuration.histogram_output_metadata import (
    extract_histogram_output_metadata,
)
from simtools.production_configuration.trigger_histograms import (
    load_trigger_histograms,
)
from simtools.visualization.plot_simtel_event_histograms import (
    plot_monte_carlo_statistics_diagnostics,
)

_logger = logging.getLogger(__name__)


def _get_metadata_quantity(metadata_row, column_name, default_unit):
    """Return a quantity-valued metadata field from an astropy table row.

    Parameters
    ----------
    metadata_row : astropy.table.row.Row
        Metadata row selected from the trigger-histogram metadata table.
    column_name : str
        Column name to read from the row.
    default_unit : astropy.units.UnitBase or str
        Unit to apply when the source column has no unit metadata.

    Returns
    -------
    astropy.units.Quantity
        Quantity reconstructed from the row scalar and the table-column unit.
    """
    value = metadata_row[column_name]
    column = metadata_row.table[column_name]
    unit = getattr(column, "unit", None) or default_unit
    return u.Quantity(value, unit)


def _resolve_effective_upper_bound(original_value, override_value, unit, label):
    """Return an effective upper bound, validating an optional reduced override."""
    original_value = u.Quantity(original_value).to(unit)
    if original_value.value <= 0.0:
        raise ValueError(f"Original {label} must be positive.")
    if override_value is None:
        return original_value

    override_value = u.Quantity(override_value).to(unit)
    if override_value.value <= 0.0:
        raise ValueError(f"Reduced {label} must be positive.")
    if override_value > original_value:
        raise ValueError(f"Reduced {label} cannot exceed the original simulated {label}.")
    return override_value


def _resolve_effective_throw_radius(original_radius, radius_override=None):
    """Return the effective throw radius, validating optional overrides."""
    return _resolve_effective_upper_bound(
        original_radius,
        radius_override,
        u.m,
        "core scatter radius",
    )


def _resolve_effective_view_cone_radius(original_radius, radius_override=None):
    """Return the effective maximum view-cone radius, validating optional overrides."""
    return _resolve_effective_upper_bound(
        original_radius,
        radius_override,
        u.deg,
        "view cone radius",
    )


def _power_law_bin_integrals(bin_lows, bin_highs, spectral_index):
    """Integrate a power law across energy bins."""
    exponent = spectral_index + 1.0
    bin_lows = np.asarray(bin_lows, dtype=float)
    bin_highs = np.asarray(bin_highs, dtype=float)
    if np.isclose(exponent, 0.0):
        return np.log(bin_highs / bin_lows)
    return (np.power(bin_highs, exponent) - np.power(bin_lows, exponent)) / exponent


def _compute_spectral_bin_weights(energy_edges, spectral_index, br_energy):
    """Return unnormalized spectral weights per energy bin."""
    energy_edges = np.asarray(energy_edges, dtype=float)
    br_energy_min = u.Quantity(br_energy[0]).to_value(u.TeV)
    br_energy_max = u.Quantity(br_energy[1]).to_value(u.TeV)
    if br_energy_min <= 0.0 or br_energy_max <= 0.0:
        raise ValueError("Thrown energy bounds must be positive.")
    if br_energy_min >= br_energy_max:
        raise ValueError("Thrown energy minimum must be smaller than the maximum.")

    lows = energy_edges[:-1]
    highs = energy_edges[1:]
    clipped_lows = np.maximum(lows, br_energy_min)
    clipped_highs = np.minimum(highs, br_energy_max)
    overlaps = clipped_highs > clipped_lows

    weights = np.zeros_like(lows, dtype=float)
    if np.any(overlaps):
        weights[overlaps] = _power_law_bin_integrals(
            clipped_lows[overlaps],
            clipped_highs[overlaps],
            spectral_index,
        )
    return weights


def _compute_energy_bin_probabilities(
    energy_edges,
    energy_totals,
    source_spectral_index,
    target_spectral_index,
    br_energy,
):
    """Return empirical per-bin probabilities reweighted from source to target spectrum."""
    energy_totals = np.asarray(energy_totals, dtype=float)
    total_simulated = np.sum(energy_totals)
    if total_simulated <= 0.0:
        raise ValueError("No simulated events available in reference energy bins.")

    if np.isclose(source_spectral_index, target_spectral_index):
        return energy_totals / total_simulated

    source_weights = _compute_spectral_bin_weights(energy_edges, source_spectral_index, br_energy)
    target_weights = _compute_spectral_bin_weights(energy_edges, target_spectral_index, br_energy)
    reweighting = np.divide(
        target_weights,
        source_weights,
        out=np.zeros_like(target_weights, dtype=float),
        where=source_weights > 0.0,
    )
    weighted_totals = energy_totals * reweighting
    total = np.sum(weighted_totals)
    if total <= 0.0:
        raise ValueError("Thrown energy range does not overlap any reference energy bins.")
    return weighted_totals / total


def _compute_expected_trigger_matrix(simulated_counts, triggered_counts, energy_probabilities):
    """Return the expected triggered counts per one thrown event."""
    simulated_counts = np.asarray(simulated_counts, dtype=float)
    triggered_counts = np.asarray(triggered_counts, dtype=float)
    energy_probabilities = np.asarray(energy_probabilities, dtype=float)

    energy_totals = simulated_counts.sum(axis=0)
    trigger_probabilities = np.divide(
        triggered_counts,
        energy_totals[np.newaxis, :],
        out=np.zeros_like(simulated_counts, dtype=float),
        where=energy_totals[np.newaxis, :] > 0,
    )
    return trigger_probabilities * energy_probabilities[np.newaxis, :]


def _compute_expected_counts(expected_triggers_per_event, required_total_events):
    """Return expected triggered counts for the solved total thrown events."""
    if not np.isfinite(required_total_events):
        return np.full_like(expected_triggers_per_event, np.nan, dtype=float)
    return np.asarray(expected_triggers_per_event, dtype=float) * float(required_total_events)


def _ceil_required_total_events(required_total_events):
    """Round the required total number of events up to the next integer."""
    if not np.isfinite(required_total_events):
        return required_total_events
    return int(np.ceil(required_total_events))


def _compute_relative_uncertainty(expected_counts):
    """Return Poisson relative uncertainty for expected counts."""
    expected_counts = np.asarray(expected_counts, dtype=float)
    return np.divide(
        1.0,
        np.sqrt(expected_counts),
        out=np.full_like(expected_counts, np.inf, dtype=float),
        where=expected_counts > 0.0,
    )


def _get_reference_matrix(reference_rows, column_name):
    """Return one per-bin matrix reconstructed from the flattened table."""
    rows = reference_rows.copy()
    if "core_distance_bin_index" in rows.colnames:
        rows.sort(["angular_distance_bin_index", "energy_bin_index", "core_distance_bin_index"])
        angular_indices = np.unique(rows["angular_distance_bin_index"])
        energy_indices = np.unique(rows["energy_bin_index"])
        core_indices = np.unique(rows["core_distance_bin_index"])
        return np.asarray(rows[column_name]).reshape(
            len(angular_indices), len(energy_indices), len(core_indices)
        )
    rows.sort(["angular_distance_bin_index", "energy_bin_index"])
    angular_indices = np.unique(rows["angular_distance_bin_index"])
    energy_indices = np.unique(rows["energy_bin_index"])
    return np.asarray(rows[column_name]).reshape(len(angular_indices), len(energy_indices))


def _get_reference_bin_edges(reference_rows):
    """Return the bin edges for energy and angular distance histograms."""
    axis_units = {
        "energy": u.TeV,
        "angular_distance": u.deg,
    }
    edges = []
    for axis, unit in axis_units.items():
        rows = reference_rows.copy()
        rows.sort(f"{axis}_bin_index")
        lows = np.unique(rows[f"{axis}_low"].quantity.to_value(unit))
        highs = np.unique(rows[f"{axis}_high"].quantity.to_value(unit))
        edges.append(np.concatenate([lows[:1], highs]))
    return edges[0], edges[1]


def _get_reference_core_edges(reference_rows):
    """Return core-distance bin edges for core-distance-binned trigger histograms."""
    if "core_distance_bin_index" not in reference_rows.colnames:
        raise ValueError(
            "Core-distance-binned trigger histograms are required to use reduced_core_radius. "
            "Rebuild trigger histograms with simtools-write-trigger-histograms."
        )
    rows = reference_rows.copy()
    rows.sort("core_distance_bin_index")
    lows = np.unique(rows["core_distance_low"].quantity.to_value(u.m))
    highs = np.unique(rows["core_distance_high"].quantity.to_value(u.m))
    return np.concatenate([lows[:1], highs])


def _compute_core_distance_weights(core_edges, effective_radius):
    """Return area-fraction weights for integrating core-distance bins."""
    effective_radius = u.Quantity(effective_radius).to_value(u.m)
    if effective_radius <= 0.0:
        raise ValueError("Effective core scatter radius must be positive.")

    core_edges = np.asarray(core_edges, dtype=float)
    lows = core_edges[:-1]
    highs = core_edges[1:]
    denominator = highs**2 - lows**2
    clipped_highs = np.minimum(highs, effective_radius)
    numerator = np.maximum(clipped_highs**2 - lows**2, 0.0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    ).clip(0.0, 1.0)


def _compute_view_cone_weights(angular_edges, effective_radius):
    """Return solid-angle-fraction weights for integrating angular-distance bins.

    Weights are computed from the spherical solid angle covered by each bin, not from
    the linear angular width. Bins intersected by the reduced view-cone radius receive
    a partial weight corresponding to the clipped solid-angle fraction.
    """
    effective_radius = u.Quantity(effective_radius).to_value(u.deg)
    if effective_radius <= 0.0:
        raise ValueError("Effective view cone radius must be positive.")

    angular_edges = np.asarray(angular_edges, dtype=float)
    lows = np.deg2rad(angular_edges[:-1])
    highs = np.deg2rad(angular_edges[1:])
    clipped_highs = np.deg2rad(np.minimum(angular_edges[1:], effective_radius))
    denominator = np.cos(lows) - np.cos(highs)
    numerator = np.maximum(np.cos(lows) - np.cos(clipped_highs), 0.0)
    return np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator, dtype=float),
        where=denominator > 0.0,
    ).clip(0.0, 1.0)


def _collapse_core_distance_counts(
    simulated_counts, triggered_counts, core_edges, effective_radius
):
    """Collapse 3D count matrices over core distance using area-fraction weights."""
    weights = _compute_core_distance_weights(core_edges, effective_radius)
    simulated_counts = np.asarray(simulated_counts, dtype=float)
    triggered_counts = np.asarray(triggered_counts, dtype=float)
    return (
        np.sum(simulated_counts * weights[np.newaxis, np.newaxis, :], axis=2),
        np.sum(triggered_counts * weights[np.newaxis, np.newaxis, :], axis=2),
    )


def _restrict_view_cone_counts(simulated_counts, triggered_counts, angular_edges, effective_radius):
    """Scale angular-distance bins to a reduced maximum view-cone radius."""
    weights = _compute_view_cone_weights(angular_edges, effective_radius)
    simulated_counts = np.asarray(simulated_counts, dtype=float)
    triggered_counts = np.asarray(triggered_counts, dtype=float)
    return (
        simulated_counts * weights[:, np.newaxis],
        triggered_counts * weights[:, np.newaxis],
    )


def _compute_trigger_efficiency(triggered_counts, simulated_counts):
    """Return trigger efficiency from triggered and simulated counts."""
    triggered_counts = np.asarray(triggered_counts, dtype=float)
    simulated_counts = np.asarray(simulated_counts, dtype=float)
    return np.divide(
        triggered_counts,
        simulated_counts,
        out=np.zeros_like(triggered_counts, dtype=float),
        where=simulated_counts > 0.0,
    )


def _compute_overall_trigger_probability(expected_triggers_per_event, energy_mask):
    """Return the weighted trigger probability in the selected optimization range."""
    return float(np.sum(np.asarray(expected_triggers_per_event, dtype=float)[:, energy_mask]))


def _log_reference_validation_summary(metadata_row, reference_rows):
    """Log a compact validation summary for one trigger-histogram reference."""
    total_simulated = int(np.sum(np.asarray(reference_rows["simulated_count"], dtype=float)))
    total_triggered = int(np.sum(np.asarray(reference_rows["triggered_count"], dtype=float)))
    overall_efficiency = (
        float(total_triggered / total_simulated) if total_simulated > 0 else float("nan")
    )
    zenith = _get_metadata_quantity(metadata_row, "zenith", u.deg).to_value(u.deg)
    azimuth = _get_metadata_quantity(metadata_row, "azimuth", u.deg).to_value(u.deg)
    nsb_level = metadata_row["nsb_level"]
    _logger.info(
        "Using trigger histogram for array_layout=%s "
        "(zenith=%.3f deg, azimuth=%.3f deg, nsb_level=%s): "
        "simulated_events=%d triggered_events=%d overall_trigger_efficiency=%.6g",
        metadata_row["array_name"],
        zenith,
        azimuth,
        nsb_level,
        total_simulated,
        total_triggered,
        overall_efficiency,
    )


def _log_overall_trigger_probability(metadata_row, overall_trigger_probability):
    """Log the overall trigger probability used for overall-target estimation."""
    _logger.info(
        "Overall trigger probability in selected optimization range for array_layout=%s "
        "(zenith=%.3f deg, azimuth=%.3f deg, nsb_level=%s): %.6g",
        metadata_row["array_name"],
        _get_metadata_quantity(metadata_row, "zenith", u.deg).to_value(u.deg),
        _get_metadata_quantity(metadata_row, "azimuth", u.deg).to_value(u.deg),
        metadata_row["nsb_level"],
        overall_trigger_probability,
    )


def _select_reference_rows(metadata_table, array_names):
    """Filter reference metadata rows according to optional user-facing selectors."""
    selected = metadata_table
    if array_names:
        selected = selected[np.isin(selected["array_name"].astype(str), array_names)]
    if len(selected) == 0:
        raise ValueError("No trigger histograms matched the requested selection.")
    return selected


def _optimization_energy_mask(energy_edges, optimization_energy):
    """Return the energy-bin mask used for the optimization criterion."""
    centers = 0.5 * (np.asarray(energy_edges[:-1]) + np.asarray(energy_edges[1:]))
    optimization_energy_min = u.Quantity(optimization_energy[0]).to_value(u.TeV)
    optimization_energy_max = u.Quantity(optimization_energy[1]).to_value(u.TeV)
    if optimization_energy_min >= optimization_energy_max:
        raise ValueError("Optimization energy minimum must be smaller than the maximum.")
    mask = (centers >= optimization_energy_min) & (centers <= optimization_energy_max)
    if not np.any(mask):
        raise ValueError("Optimization energy range does not overlap any reference energy bins.")
    return mask


def _estimate_required_events(
    expected_triggers_per_event,
    energy_mask,
    target_relative_uncertainty=None,
    target_triggered_events=None,
    overall_trigger_probability=None,
):
    """Solve for required total thrown events from estimable trigger bins."""
    if target_relative_uncertainty is not None and target_triggered_events is not None:
        raise ValueError(
            "Target relative uncertainty and target triggered events are mutually exclusive."
        )
    candidate_matrix, positive_mask = _prepare_estimation_candidates(
        expected_triggers_per_event, energy_mask
    )

    if target_relative_uncertainty is not None:
        return _estimate_required_events_from_uncertainty(
            candidate_matrix,
            positive_mask,
            target_relative_uncertainty,
        )

    if target_triggered_events is not None:
        return _estimate_required_events_from_total_trigger_target(
            target_triggered_events,
            overall_trigger_probability,
        )

    raise ValueError(
        "Either target relative uncertainty or target triggered events must be provided."
    )


def _prepare_estimation_candidates(expected_triggers_per_event, energy_mask):
    """Return candidate bins and masks restricted to the optimization energy range."""
    candidate_matrix = np.asarray(expected_triggers_per_event, dtype=float)[:, energy_mask]
    positive_mask = np.isfinite(candidate_matrix) & (candidate_matrix > 0.0)
    return candidate_matrix, positive_mask


def _estimate_required_events_from_uncertainty(
    candidate_matrix,
    positive_mask,
    target_relative_uncertainty,
):
    """Solve the per-bin uncertainty target using the worst relevant bin."""
    if target_relative_uncertainty <= 0.0:
        raise ValueError("Target relative uncertainty must be positive.")
    if not np.any(positive_mask):
        return np.inf

    required_trigger_count = 1.0 / float(target_relative_uncertainty) ** 2
    required_totals = np.full_like(candidate_matrix, -np.inf, dtype=float)
    required_totals[positive_mask] = required_trigger_count / candidate_matrix[positive_mask]
    return float(np.max(required_totals))


def _estimate_required_events_from_total_trigger_target(
    target_triggered_events,
    overall_trigger_probability,
):
    """Solve the overall triggered-event target using the selected-range trigger probability."""
    if target_triggered_events <= 0:
        raise ValueError("Target triggered events must be positive.")
    if overall_trigger_probability is None:
        raise ValueError("Overall trigger probability must be provided.")
    if overall_trigger_probability <= 0.0:
        return np.inf

    return float(target_triggered_events) / float(overall_trigger_probability)


def _resolve_energy_ranges(metadata_row, optimization_energy):
    """
    Resolve thrown and optimization energy ranges for one histogram row.

    Parameters
    ----------
    metadata_row : astropy.table.row.Row
        Metadata row selected from the trigger-histogram metadata table.
    optimization_energy : tuple of astropy.units.Quantity or None
        Optional user-specified optimization energy range.

    """
    br_energy = (
        _get_metadata_quantity(metadata_row, "energy_min", u.TeV),
        _get_metadata_quantity(metadata_row, "energy_max", u.TeV),
    )
    optimization_energy_min = None
    optimization_energy_max = None
    if optimization_energy is not None:
        optimization_energy_min, optimization_energy_max = optimization_energy
    resolved_optimization_energy = (
        br_energy[0] if optimization_energy_min is None else optimization_energy_min,
        br_energy[1] if optimization_energy_max is None else optimization_energy_max,
    )
    return br_energy, resolved_optimization_energy


def _extract_diagnostic_file_info(metadata_row):
    """Return observational metadata used to disambiguate diagnostic plot filenames."""
    return {
        "zenith": _get_metadata_quantity(metadata_row, "zenith", u.deg),
        "azimuth": _get_metadata_quantity(metadata_row, "azimuth", u.deg),
        "nsb_level": metadata_row["nsb_level"],
    }


def _build_result_metadata(
    metadata_row,
):
    """Build the metadata fields shared by all estimator result rows."""
    return extract_histogram_output_metadata(
        metadata_row,
        FILE_INFO_COLUMNS,
        include_array_name=True,
    )


def _build_table_metadata(args_dict):
    """Build table-level metadata for configuration values shared by all rows."""
    metadata = {
        "spectral_index": args_dict.get("spectral_index"),
        "target_relative_uncertainty": args_dict.get("target_relative_uncertainty"),
        "target_triggered_events": args_dict.get("target_triggered_events"),
        "optimization_energy_min": args_dict.get("optimization_energy_min"),
        "optimization_energy_max": args_dict.get("optimization_energy_max"),
        "reduced_core_radius": args_dict.get("reduced_core_radius"),
        "reduced_view_cone_radius": args_dict.get("reduced_view_cone_radius"),
    }
    return {key: value for key, value in metadata.items() if value is not None}


def _resolve_source_spectral_index(metadata_row):
    """Return the original simulated spectral index stored in histogram metadata."""
    if "spectral_index" in metadata_row.colnames:
        spectral_index = float(metadata_row["spectral_index"])
        if np.isfinite(spectral_index):
            return spectral_index

    _logger.warning(
        "No spectral index found in trigger histogram metadata for array_layout=%s; assuming -2.0.",
        metadata_row["array_name"],
    )
    return -2.0


def _resolve_reference_spectral_index(
    metadata_row, spectral_index_override, source_spectral_index=None
):
    """Return the target spectral index to use for one reference row."""
    if source_spectral_index is None:
        source_spectral_index = _resolve_source_spectral_index(metadata_row)
    if spectral_index_override is not None:
        spectral_index = float(spectral_index_override)
        _logger.info(
            "Using user-provided target spectral index %.6g for array_layout=%s "
            "(source spectral index %.6g from trigger histogram metadata).",
            spectral_index,
            metadata_row["array_name"],
            source_spectral_index,
        )
        return spectral_index

    _logger.info(
        "Using spectral index %.6g from trigger histogram metadata for array_layout=%s.",
        source_spectral_index,
        metadata_row["array_name"],
    )
    return source_spectral_index


def _resolved_table_spectral_index(selected_references, spectral_index_override):
    """Return the table-level spectral index metadata value."""
    if spectral_index_override is not None:
        return float(spectral_index_override)

    if "spectral_index" in selected_references.colnames:
        spectral_indices = np.asarray(selected_references["spectral_index"], dtype=float)
        spectral_indices = spectral_indices[np.isfinite(spectral_indices)]
        if spectral_indices.size == 0:
            return -2.0
        unique_indices = np.unique(np.round(spectral_indices, decimals=12))
        if unique_indices.size == 1:
            return float(unique_indices[0])
    return -2.0


def _prepare_reference_estimation_inputs(
    metadata_row,
    reference_rows,
    reduced_core_radius,
    reduced_view_cone_radius,
    optimization_energy,
    source_spectral_index,
    target_spectral_index,
):
    """Prepare per-reference matrices, ranges, and derived probabilities for estimation."""
    energy_edges, angular_edges = _get_reference_bin_edges(reference_rows)
    simulated_counts = _get_reference_matrix(reference_rows, "simulated_count")
    triggered_counts = _get_reference_matrix(reference_rows, "triggered_count")

    effective_radius = _resolve_effective_throw_radius(
        _get_metadata_quantity(metadata_row, "core_scatter_max", u.m),
        reduced_core_radius,
    )
    if simulated_counts.ndim == 3:
        core_edges = _get_reference_core_edges(reference_rows)
        simulated_counts, triggered_counts = _collapse_core_distance_counts(
            simulated_counts,
            triggered_counts,
            core_edges,
            effective_radius,
        )
    elif reduced_core_radius is not None:
        raise ValueError(
            "Core-distance-binned trigger histograms are required to use reduced_core_radius. "
            "Rebuild trigger histograms with simtools-write-trigger-histograms."
        )

    effective_view_cone_radius = _resolve_effective_view_cone_radius(
        _get_metadata_quantity(metadata_row, "viewcone_max", u.deg),
        reduced_view_cone_radius,
    )
    simulated_counts, triggered_counts = _restrict_view_cone_counts(
        simulated_counts,
        triggered_counts,
        angular_edges,
        effective_view_cone_radius,
    )
    trigger_efficiency = _compute_trigger_efficiency(triggered_counts, simulated_counts)
    br_energy, optimization_energy = _resolve_energy_ranges(metadata_row, optimization_energy)
    energy_probabilities = _compute_energy_bin_probabilities(
        energy_edges,
        simulated_counts.sum(axis=0),
        source_spectral_index,
        target_spectral_index,
        br_energy,
    )
    expected_triggers_per_event = _compute_expected_trigger_matrix(
        simulated_counts,
        triggered_counts,
        energy_probabilities,
    )
    energy_mask = _optimization_energy_mask(energy_edges, optimization_energy)
    overall_trigger_probability = _compute_overall_trigger_probability(
        expected_triggers_per_event,
        energy_mask,
    )
    return {
        "energy_edges": energy_edges,
        "angular_edges": angular_edges,
        "simulated_counts": simulated_counts,
        "triggered_counts": triggered_counts,
        "effective_radius": effective_radius,
        "effective_view_cone_radius": effective_view_cone_radius,
        "trigger_efficiency": trigger_efficiency,
        "br_energy": br_energy,
        "optimization_energy": optimization_energy,
        "expected_triggers_per_event": expected_triggers_per_event,
        "energy_mask": energy_mask,
        "overall_trigger_probability": overall_trigger_probability,
    }


def _build_result_row(
    metadata_row,
    bin_table,
    target_relative_uncertainty,
    target_triggered_events,
    reduced_core_radius,
    reduced_view_cone_radius,
    optimization_energy,
    spectral_index_override,
    plot_diagnostics,
):
    """Build one result row for a selected trigger histogram."""
    reference_id = metadata_row["reference_id"]
    reference_rows = bin_table[bin_table["reference_id"] == reference_id]
    source_spectral_index = _resolve_source_spectral_index(metadata_row)
    spectral_index = _resolve_reference_spectral_index(
        metadata_row,
        spectral_index_override,
        source_spectral_index=source_spectral_index,
    )
    _log_reference_validation_summary(metadata_row, reference_rows)
    prepared = _prepare_reference_estimation_inputs(
        metadata_row,
        reference_rows,
        reduced_core_radius,
        reduced_view_cone_radius,
        optimization_energy,
        source_spectral_index,
        spectral_index,
    )
    if target_triggered_events is not None:
        _log_overall_trigger_probability(metadata_row, prepared["overall_trigger_probability"])
    required_total_events = _estimate_required_events(
        prepared["expected_triggers_per_event"],
        prepared["energy_mask"],
        target_relative_uncertainty=target_relative_uncertainty,
        target_triggered_events=target_triggered_events,
        overall_trigger_probability=prepared["overall_trigger_probability"],
    )
    required_total_events = _ceil_required_total_events(required_total_events)
    expected_counts = _compute_expected_counts(
        prepared["expected_triggers_per_event"], required_total_events
    )
    relative_uncertainty = _compute_relative_uncertainty(expected_counts)

    if plot_diagnostics:
        plot_monte_carlo_statistics_diagnostics(
            io_handler.IOHandler().get_output_directory(),
            metadata_row["array_name"],
            _extract_diagnostic_file_info(metadata_row),
            prepared["energy_edges"],
            prepared["angular_edges"],
            expected_counts,
            relative_uncertainty,
        )

    return _build_result_metadata(
        metadata_row,
    ) | {
        "estimated_total_events": required_total_events,
        "br_energy_min": u.Quantity(prepared["br_energy"][0]).to(u.TeV),
        "br_energy_max": u.Quantity(prepared["br_energy"][1]).to(u.TeV),
    }


def estimate_monte_carlo_statistics(args_dict=None):
    """
    Estimate required total thrown events for one or more trigger histograms.

    Returns
    -------
    astropy.table.Table
        Results table containing required thrown-event estimates and limiting-bin
        diagnostics for each selected trigger histogram.
    """
    args_dict = args_dict or settings.config.args
    metadata_table, bin_table = load_trigger_histograms(args_dict.get("trigger_histogram_file"))

    selected_references = _select_reference_rows(
        metadata_table,
        args_dict.get("array_layout_name") or args_dict.get("array_names"),
    )
    if args_dict.get("plot_diagnostics") and len(selected_references) > 0:
        _logger.info(
            "Writing Monte Carlo statistics diagnostic plots to %s",
            io_handler.IOHandler().get_output_directory(),
        )

    output_rows = [
        _build_result_row(
            metadata_row,
            bin_table,
            args_dict.get("target_relative_uncertainty"),
            args_dict.get("target_triggered_events"),
            args_dict.get("reduced_core_radius"),
            args_dict.get("reduced_view_cone_radius"),
            (
                args_dict.get("optimization_energy_min"),
                args_dict.get("optimization_energy_max"),
            ),
            args_dict.get("spectral_index"),
            args_dict.get("plot_diagnostics"),
        )
        for metadata_row in selected_references
    ]
    results = Table(rows=output_rows)
    results.meta.update(_build_table_metadata(args_dict))
    resolved_spectral_index = _resolved_table_spectral_index(
        selected_references,
        args_dict.get("spectral_index"),
    )
    if resolved_spectral_index is not None:
        results.meta["spectral_index"] = resolved_spectral_index
    output_file = validate_file_type(
        io_handler.IOHandler().get_output_file(args_dict.get("output_file")), file_type="table"
    )
    results.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Writing Monte Carlo statistics estimates to {output_file}")
    return results
