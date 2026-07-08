"""Estimate required Monte Carlo statistics from trigger histogram products."""

import logging

import astropy.units as u
import numpy as np
from astropy.table import Table

import simtools.utils.general as gen
from simtools.production_configuration.trigger_histograms import (
    load_trigger_histograms,
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


# TODO - not used!!
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


def _compute_energy_bin_probabilities(
    energy_edges, spectral_index, thrown_energy_min, thrown_energy_max
):
    """Return the normalized thrown-event probability per energy bin."""
    energy_edges = np.asarray(energy_edges, dtype=float)
    thrown_energy_min = u.Quantity(thrown_energy_min).to_value(u.TeV)
    thrown_energy_max = u.Quantity(thrown_energy_max).to_value(u.TeV)
    if thrown_energy_min <= 0.0 or thrown_energy_max <= 0.0:
        raise ValueError("Thrown energy bounds must be positive.")
    if thrown_energy_min >= thrown_energy_max:
        raise ValueError("Thrown energy minimum must be smaller than the maximum.")

    lows = energy_edges[:-1]
    highs = energy_edges[1:]
    clipped_lows = np.maximum(lows, thrown_energy_min)
    clipped_highs = np.minimum(highs, thrown_energy_max)
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


def _compute_expected_trigger_matrix(simulated_counts, trigger_efficiency, energy_probabilities):
    """Return the expected triggered counts per one thrown event."""
    simulated_counts = np.asarray(simulated_counts, dtype=float)
    trigger_efficiency = np.asarray(trigger_efficiency, dtype=float)
    energy_probabilities = np.asarray(energy_probabilities, dtype=float)

    energy_totals = simulated_counts.sum(axis=0)
    angular_conditionals = np.divide(
        simulated_counts,
        energy_totals[np.newaxis, :],
        out=np.zeros_like(simulated_counts, dtype=float),
        where=energy_totals[np.newaxis, :] > 0,
    )
    return angular_conditionals * energy_probabilities[np.newaxis, :] * trigger_efficiency


def _get_reference_matrix(bin_table, reference_id, column_name):
    """Return one per-bin matrix reconstructed from the flattened table."""
    rows = bin_table[bin_table["reference_id"] == reference_id]
    rows.sort(["angular_distance_bin_index", "energy_bin_index"])
    angular_indices = np.unique(rows["angular_distance_bin_index"])
    energy_indices = np.unique(rows["energy_bin_index"])
    return np.asarray(rows[column_name]).reshape(len(angular_indices), len(energy_indices))


def _get_reference_bin_edges(bin_table, reference_id):
    """Return the bin edges for energy and angular distance histograms."""
    axis_units = {
        "energy": u.TeV,
        "angular_distance": u.deg,
    }
    reference_rows = bin_table[bin_table["reference_id"] == reference_id]
    edges = []
    for axis, unit in axis_units.items():
        rows = reference_rows.copy()
        rows.sort(f"{axis}_bin_index")
        lows = np.unique(rows[f"{axis}_low"].quantity.to_value(unit))
        highs = np.unique(rows[f"{axis}_high"].quantity.to_value(unit))
        edges.append(np.concatenate([lows[:1], highs]))
    return edges[0], edges[1]


def _select_reference_rows(metadata_table, array_names):
    """Filter reference metadata rows according to optional user-facing selectors."""
    selected = metadata_table
    if array_names:
        selected = selected[np.isin(selected["array_name"], array_names)]
    if len(selected) == 0:
        raise ValueError("No trigger histograms matched the requested selection.")
    return selected


def _optimization_energy_mask(energy_edges, optimization_energy_min, optimization_energy_max):
    """Return the energy-bin mask used for the optimization criterion."""
    centers = 0.5 * (np.asarray(energy_edges[:-1]) + np.asarray(energy_edges[1:]))
    optimization_energy_min = u.Quantity(optimization_energy_min).to_value(u.TeV)
    optimization_energy_max = u.Quantity(optimization_energy_max).to_value(u.TeV)
    if optimization_energy_min >= optimization_energy_max:
        raise ValueError("Optimization energy minimum must be smaller than the maximum.")
    mask = (centers >= optimization_energy_min) & (centers <= optimization_energy_max)
    if not np.any(mask):
        raise ValueError("Optimization energy range does not overlap any reference energy bins.")
    return mask


def _estimate_required_events(
    expected_triggers_per_event, energy_mask, target_relative_uncertainty
):
    """Solve for the required total thrown events from the limiting predicted trigger bin."""
    if target_relative_uncertainty <= 0.0:
        raise ValueError("Target relative uncertainty must be positive.")

    required_trigger_count = 1.0 / float(target_relative_uncertainty) ** 2
    candidate_matrix = np.asarray(expected_triggers_per_event, dtype=float)[:, energy_mask]
    required_totals = np.divide(
        required_trigger_count,
        candidate_matrix,
        out=np.full_like(candidate_matrix, np.inf, dtype=float),
        where=candidate_matrix > 0.0,
    )
    limiting_flat_index = int(np.argmax(required_totals))
    limiting_index = np.unravel_index(limiting_flat_index, required_totals.shape)
    return required_totals[limiting_index], candidate_matrix[limiting_index], limiting_index


def _resolve_energy_ranges(metadata_row, args_dict):
    """Resolve thrown and optimization energy ranges for one histogram row."""
    thrown_energy_min = args_dict.get("thrown_energy_min")
    if thrown_energy_min is None:
        thrown_energy_min = _get_metadata_quantity(metadata_row, "energy_min", u.TeV)

    thrown_energy_max = args_dict.get("thrown_energy_max")
    if thrown_energy_max is None:
        thrown_energy_max = _get_metadata_quantity(metadata_row, "energy_max", u.TeV)

    optimization_energy_min = args_dict.get("optimization_energy_min")
    if optimization_energy_min is None:
        optimization_energy_min = thrown_energy_min

    optimization_energy_max = args_dict.get("optimization_energy_max")
    if optimization_energy_max is None:
        optimization_energy_max = thrown_energy_max
    return (
        thrown_energy_min,
        thrown_energy_max,
        optimization_energy_min,
        optimization_energy_max,
    )


def _build_result_row(metadata_row, bin_table, args_dict):
    """Build one result row for a selected trigger histogram."""
    reference_id = metadata_row["reference_id"]
    trigger_efficiency = _get_reference_matrix(bin_table, reference_id, "trigger_efficiency")
    energy_edges, angular_edges = _get_reference_bin_edges(bin_table, reference_id)
    simulated_counts = _get_reference_matrix(bin_table, reference_id, "simulated_count")

    effective_radius = _resolve_effective_throw_radius(
        _get_metadata_quantity(metadata_row, "core_scatter_max", u.m),
        args_dict.get("reduced_core_radius"),
    )
    (
        thrown_energy_min,
        thrown_energy_max,
        optimization_energy_min,
        optimization_energy_max,
    ) = _resolve_energy_ranges(metadata_row, args_dict)

    energy_probabilities = _compute_energy_bin_probabilities(
        energy_edges,
        args_dict["spectral_index"],
        thrown_energy_min,
        thrown_energy_max,
    )
    expected_triggers_per_event = _compute_expected_trigger_matrix(
        simulated_counts,
        trigger_efficiency,
        energy_probabilities,
    )
    energy_mask = _optimization_energy_mask(
        energy_edges,
        optimization_energy_min,
        optimization_energy_max,
    )
    required_total_events, limiting_expected_per_event, limiting_index = _estimate_required_events(
        expected_triggers_per_event,
        energy_mask,
        args_dict["target_relative_uncertainty"],
    )

    masked_energy_indices = np.flatnonzero(energy_mask)
    limiting_angular_index = limiting_index[0]
    limiting_energy_index = masked_energy_indices[limiting_index[1]]
    original_radius = _get_metadata_quantity(metadata_row, "core_scatter_max", u.m).to(u.m)

    return {
        "array_name": metadata_row["array_name"],
        "spectral_index": args_dict["spectral_index"],
        "target_relative_uncertainty": args_dict["target_relative_uncertainty"],
        "required_total_thrown_events": required_total_events,
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
        "original_core_scatter_radius": original_radius,
        "effective_core_scatter_radius": effective_radius,
        "br_energy_min": u.Quantity(thrown_energy_min).to(u.TeV),
        "br_energy_max": u.Quantity(thrown_energy_max).to(u.TeV),
        "optimization_energy_min": u.Quantity(optimization_energy_min).to(u.TeV),
        "optimization_energy_max": u.Quantity(optimization_energy_max).to(u.TeV),
    }


def estimate_monte_carlo_statistics(args_dict):
    """
    Estimate required total thrown events for one or more trigger histograms.

    Parameters
    ----------
    args_dict : dict
        Application arguments describing the histogram input file, optional array-name
        selector, spectral assumptions, optimization range, optional reduced core radius,
        and output file.

    Returns
    -------
    astropy.table.Table
        Results table containing required thrown-event estimates and limiting-bin
        diagnostics for each selected trigger histogram.

    Raises
    ------
    ValueError
        If the selected references are empty, the requested ranges are invalid,
        the reduced core radius is inconsistent with the histogram metadata, or
        the output file does not use the ECSV suffix.
    """
    metadata_table, bin_table = load_trigger_histograms(args_dict["input"])
    print("AAAAA", metadata_table)
    print("BBBB", bin_table)
    selected_references = _select_reference_rows(metadata_table, args_dict.get("array_names"))
    output_rows = [
        _build_result_row(metadata_row, bin_table, args_dict)
        for metadata_row in selected_references
    ]
    results = Table(rows=output_rows)
    output_file = gen.validate_file_type(args_dict["output_file"], ".ecsv")
    results.write(output_file, format="ascii.ecsv", overwrite=True)
    _logger.info(f"Wrote Monte Carlo statistics estimates to {output_file}")
    return results
