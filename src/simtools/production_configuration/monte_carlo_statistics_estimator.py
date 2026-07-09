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


def _resolve_effective_throw_radius(original_radius, radius_override=None):
    """Return the effective throw radius, validating optional overrides."""
    original_radius = u.Quantity(original_radius).to(u.m)
    if original_radius.value <= 0.0:
        raise ValueError("Original core scatter radius must be positive.")
    if radius_override is None:
        return original_radius

    radius_override = u.Quantity(radius_override).to(u.m)
    if radius_override.value <= 0.0:
        raise ValueError("Reduced core scatter radius must be positive.")
    if radius_override > original_radius:
        raise ValueError(
            "Reduced core scatter radius cannot exceed the original simulated core scatter radius."
        )
    return radius_override


def _power_law_bin_integrals(bin_lows, bin_highs, spectral_index):
    """Integrate a power law across energy bins."""
    exponent = spectral_index + 1.0
    bin_lows = np.asarray(bin_lows, dtype=float)
    bin_highs = np.asarray(bin_highs, dtype=float)
    if np.isclose(exponent, 0.0):
        return np.log(bin_highs / bin_lows)
    return (np.power(bin_highs, exponent) - np.power(bin_lows, exponent)) / exponent


def _compute_energy_bin_probabilities(energy_edges, spectral_index, br_energy):
    """Return the normalized thrown-event probability per energy bin."""
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
    total = np.sum(weights)
    if total <= 0.0:
        raise ValueError("Thrown energy range does not overlap any reference energy bins.")
    return weights / total


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
    expected_triggers_per_event, energy_mask, target_relative_uncertainty
):
    """Solve for required total thrown events from estimable trigger bins."""
    if target_relative_uncertainty <= 0.0:
        raise ValueError("Target relative uncertainty must be positive.")

    required_trigger_count = 1.0 / float(target_relative_uncertainty) ** 2
    candidate_matrix = np.asarray(expected_triggers_per_event, dtype=float)[:, energy_mask]
    positive_mask = np.isfinite(candidate_matrix) & (candidate_matrix > 0.0)
    skipped_bins = int(candidate_matrix.size - np.count_nonzero(positive_mask))
    if not np.any(positive_mask):
        return np.inf, 0.0, (0, 0), 0, skipped_bins

    required_totals = np.full_like(candidate_matrix, -np.inf, dtype=float)
    required_totals[positive_mask] = required_trigger_count / candidate_matrix[positive_mask]
    limiting_flat_index = int(np.argmax(required_totals))
    limiting_index = np.unravel_index(limiting_flat_index, required_totals.shape)
    return (
        required_totals[limiting_index],
        candidate_matrix[limiting_index],
        limiting_index,
        int(np.count_nonzero(positive_mask)),
        skipped_bins,
    )


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


def _resolve_limiting_indices(energy_mask, limiting_index):
    """Map limiting indices from the masked matrix back to the original energy bins."""
    masked_energy_indices = np.flatnonzero(energy_mask)
    return limiting_index[0], masked_energy_indices[limiting_index[1]]


def _build_result_metadata(
    metadata_row,
    spectral_index,
    target_relative_uncertainty,
):
    """Build the metadata fields shared by all estimator result rows."""
    return extract_histogram_output_metadata(
        metadata_row,
        FILE_INFO_COLUMNS,
        include_array_name=True,
    ) | {
        "spectral_index": spectral_index,
        "target_relative_uncertainty": target_relative_uncertainty,
    }


def _build_result_row(
    metadata_row,
    bin_table,
    target_relative_uncertainty,
    spectral_index,
    reduced_core_radius,
    optimization_energy,
    plot_diagnostics,
):
    """Build one result row for a selected trigger histogram."""
    reference_id = metadata_row["reference_id"]
    reference_rows = bin_table[bin_table["reference_id"] == reference_id]
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
    trigger_efficiency = _compute_trigger_efficiency(triggered_counts, simulated_counts)
    br_energy, optimization_energy = _resolve_energy_ranges(metadata_row, optimization_energy)

    energy_probabilities = _compute_energy_bin_probabilities(
        energy_edges, spectral_index, br_energy
    )
    expected_triggers_per_event = _compute_expected_trigger_matrix(
        simulated_counts,
        triggered_counts,
        energy_probabilities,
    )
    energy_mask = _optimization_energy_mask(energy_edges, optimization_energy)
    (
        required_total_events,
        limiting_expected_per_event,
        limiting_index,
        optimization_bins_used,
        optimization_bins_skipped,
    ) = _estimate_required_events(
        expected_triggers_per_event,
        energy_mask,
        target_relative_uncertainty,
    )
    required_total_events = _ceil_required_total_events(required_total_events)
    expected_counts = _compute_expected_counts(expected_triggers_per_event, required_total_events)
    relative_uncertainty = _compute_relative_uncertainty(expected_counts)

    limiting_angular_index, limiting_energy_index = _resolve_limiting_indices(
        energy_mask, limiting_index
    )
    original_radius = _get_metadata_quantity(metadata_row, "core_scatter_max", u.m).to(u.m)

    if plot_diagnostics:
        plot_monte_carlo_statistics_diagnostics(
            io_handler.IOHandler().get_output_directory(),
            metadata_row["array_name"],
            energy_edges,
            angular_edges,
            expected_counts,
            relative_uncertainty,
        )

    return _build_result_metadata(metadata_row, spectral_index, target_relative_uncertainty) | {
        "estimated_total_events": required_total_events,
        "limiting_energy_low": energy_edges[limiting_energy_index] * u.TeV,
        "limiting_energy_high": energy_edges[limiting_energy_index + 1] * u.TeV,
        "limiting_angular_distance_low": angular_edges[limiting_angular_index] * u.deg,
        "limiting_angular_distance_high": angular_edges[limiting_angular_index + 1] * u.deg,
        "limiting_expected_trigger_count": (
            limiting_expected_per_event * required_total_events
            if np.isfinite(required_total_events)
            else 0.0
        ),
        "limiting_trigger_efficiency": trigger_efficiency[
            limiting_angular_index, limiting_energy_index
        ],
        "optimization_bins_used": optimization_bins_used,
        "optimization_bins_skipped": optimization_bins_skipped,
        "original_core_scatter_radius": original_radius,
        "effective_core_scatter_radius": effective_radius,
        "br_energy_min": u.Quantity(br_energy[0]).to(u.TeV),
        "br_energy_max": u.Quantity(br_energy[1]).to(u.TeV),
        "optimization_energy_min": u.Quantity(optimization_energy[0]).to(u.TeV),
        "optimization_energy_max": u.Quantity(optimization_energy[1]).to(u.TeV),
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
    trigger_histogram_file = args_dict.get("trigger_histogram_file") or args_dict.get("input")

    if trigger_histogram_file is None:
        raise ValueError("Use --trigger_histogram_file to provide a trigger-histogram file.")

    metadata_table, bin_table = load_trigger_histograms(trigger_histogram_file)

    selected_references = _select_reference_rows(
        metadata_table,
        args_dict.get("array_layout_name") or args_dict.get("array_names"),
    )

    output_rows = [
        _build_result_row(
            metadata_row,
            bin_table,
            args_dict.get("target_relative_uncertainty"),
            args_dict.get("spectral_index"),
            args_dict.get("reduced_core_radius"),
            (
                args_dict.get("optimization_energy_min"),
                args_dict.get("optimization_energy_max"),
            ),
            args_dict.get("plot_diagnostics"),
        )
        for metadata_row in selected_references
    ]
    results = Table(rows=output_rows)
    output_file = validate_file_type(
        io_handler.IOHandler().get_output_file(args_dict.get("output_file")), ".ecsv"
    )
    results.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Writing Monte Carlo statistics estimates to {output_file}")
    return results
