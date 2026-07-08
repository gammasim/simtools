"""Build and load trigger-histogram products from reduced event data."""

import copy

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack

import simtools.utils.general as gen
from simtools.io import io_handler, table_handler
from simtools.production_configuration.production_event_data_helpers import (
    accumulate_histograms_by_telescope_config,
    normalize_event_data_file,
    normalize_telescope_configs,
    resolve_telescope_configs,
)
from simtools.sim_events.histograms import EventDataHistograms

TRIGGER_HISTOGRAM_METADATA_TABLE = "TRIGGER_REFERENCE_METADATA"
TRIGGER_HISTOGRAM_BINS_TABLE = "TRIGGER_REFERENCE_BINS"
TRIGGER_HISTOGRAM_VALUES_TABLE = "TRIGGER_HISTOGRAM_VALUES"
TRIGGER_HISTOGRAM_EDGES_TABLE = "TRIGGER_HISTOGRAM_EDGES"
TRIGGER_TOPOLOGY_COUNTS_TABLE = "TRIGGER_TOPOLOGY_COUNTS"
TRIGGER_SUBSET_HISTOGRAMS_TABLE = "TRIGGER_SUBSET_HISTOGRAMS"

_DERIVED_HISTOGRAM_SUFFIXES = ("_eff", "_cumulative")


def _get_plot_directory_name(array_name, telescope_ids):
    """Return a readable directory name for plot output."""
    if array_name != "array_element_list":
        return array_name
    if not telescope_ids:
        return array_name
    return "_".join(str(telescope_id) for telescope_id in telescope_ids)


def _use_readable_inline_array_names(telescope_configs):
    """Return telescope configs with readable names for inline array-element lists."""
    readable_configs = []
    for config in telescope_configs:
        array_name = _get_plot_directory_name(config["array_name"], config["telescope_ids"])
        readable_configs.append(config | {"array_name": array_name})
    return readable_configs


def _get_angular_distance_bin_width(histograms):
    """Return the configured angular-distance bin width for metadata."""
    if histograms.angular_distance_bin_width is not None:
        return histograms.angular_distance_bin_width
    return np.diff(histograms.view_cone_bins)[0] * u.deg


def _quantity_value(value, unit):
    """Return a scalar value converted to unit, preserving missing values."""
    if hasattr(value, "to"):
        return value.to(unit)
    return value


def _data_range_value(histograms, name, index):
    """Return a stored data-range value, or NaN when unavailable."""
    data_range = getattr(histograms, "data_ranges", {}).get(name)
    if data_range is None:
        return np.nan
    return float(data_range[index])


def _create_histogram_tables(reference_specs):
    """Convert accumulated histogram products into metadata and bin tables."""
    metadata_tables = []
    bin_tables = []

    for spec in reference_specs:
        histograms = spec["histograms"]
        reference_id = spec["reference_id"]
        metadata_tables.append(
            _create_metadata_table(
                reference_id=reference_id,
                production_index=spec["production_index"],
                event_data_file=spec["event_data_file"],
                site=spec["site"],
                array_name=spec["array_name"],
                telescope_ids=spec["telescope_ids"],
                histograms=histograms,
            )
        )
        bin_tables.append(
            _create_bin_table(
                reference_id=reference_id,
                production_index=spec["production_index"],
                array_name=spec["array_name"],
                histograms=histograms,
            )
        )

    return (
        vstack(metadata_tables, metadata_conflicts="silent"),
        vstack(bin_tables, metadata_conflicts="silent"),
    )


def _create_trigger_topology_count_table(reference_specs):
    """Create count table for trigger multiplicities, combinations, and telescope participation."""
    rows = []
    for spec in reference_specs:
        topology = spec.get("trigger_topology") or {}
        for count_type in (
            "trigger_multiplicity",
            "trigger_combinations",
            "telescope_participation",
        ):
            for key, count in topology.get(count_type, {}).items():
                rows.append(
                    {
                        "reference_id": spec["reference_id"],
                        "count_type": count_type,
                        "subset": "",
                        "key": str(key),
                        "count": int(count),
                    }
                )
        for subset_name, multiplicity_counts in topology.get("subset_multiplicity", {}).items():
            for multiplicity, count in multiplicity_counts.items():
                rows.append(
                    {
                        "reference_id": spec["reference_id"],
                        "count_type": "subset_multiplicity",
                        "subset": str(subset_name),
                        "key": str(multiplicity),
                        "count": int(count),
                    }
                )

    table = Table(rows=rows, names=["reference_id", "count_type", "subset", "key", "count"])
    table.meta["EXTNAME"] = TRIGGER_TOPOLOGY_COUNTS_TABLE
    return table


def _quantity_bin_edges(histograms, quantity_name):
    """Return bin edges for a named event quantity."""
    if quantity_name == "energy":
        return histograms.energy_bins
    if quantity_name == "core_distance":
        return histograms.core_distance_bins
    if quantity_name == "angular_distance":
        return histograms.view_cone_bins
    raise ValueError(f"Unsupported quantity: {quantity_name}")


def _create_trigger_subset_histogram_table(reference_specs):
    """Create per-trigger-subset histograms for event-production comparisons."""
    rows = []
    for spec in reference_specs:
        topology = spec.get("trigger_topology") or {}
        subset_values = topology.get("subset_values", {})
        for quantity_name, values_by_subset in subset_values.items():
            bin_edges = _quantity_bin_edges(spec["histograms"], quantity_name)
            for subset_name, values in values_by_subset.items():
                counts, _ = np.histogram(values, bins=bin_edges)
                for bin_index, count in enumerate(counts):
                    rows.append(
                        {
                            "reference_id": spec["reference_id"],
                            "subset": str(subset_name),
                            "quantity": quantity_name,
                            "bin_index": bin_index,
                            "bin_low": float(bin_edges[bin_index]),
                            "bin_high": float(bin_edges[bin_index + 1]),
                            "count": int(count),
                        }
                    )

    table = Table(
        rows=rows,
        names=["reference_id", "subset", "quantity", "bin_index", "bin_low", "bin_high", "count"],
    )
    table.meta["EXTNAME"] = TRIGGER_SUBSET_HISTOGRAMS_TABLE
    return table


def _create_metadata_table(
    reference_id,
    production_index,
    event_data_file,
    site,
    array_name,
    telescope_ids,
    histograms,
):
    """Create the one-row metadata table for trigger histograms."""
    file_info = histograms.file_info
    metadata = Table(
        rows=[
            {
                "reference_id": reference_id,
                "production_index": production_index,
                "event_data_file": event_data_file,
                "site": site or "",
                "array_name": array_name,
                "telescope_ids": ",".join(telescope_ids) if telescope_ids else "",
                "primary_particle": file_info.get("primary_particle", ""),
                "zenith": file_info.get("zenith", 0.0 * u.deg),
                "azimuth": file_info.get("azimuth", 0.0 * u.deg),
                "nsb_level": file_info.get("nsb_level", 0.0),
                "energy_min": file_info.get("energy_min", 0.0 * u.TeV),
                "energy_max": file_info.get("energy_max", 0.0 * u.TeV),
                "viewcone_min": file_info.get("viewcone_min", 0.0 * u.deg),
                "viewcone_max": file_info.get("viewcone_max", 0.0 * u.deg),
                "core_scatter_min": file_info.get("core_scatter_min", 0.0 * u.m),
                "core_scatter_max": file_info.get("core_scatter_max", 0.0 * u.m),
                "scatter_area": file_info.get("scatter_area", 0.0 * u.cm**2),
                "solid_angle": file_info.get("solid_angle", 0.0 * u.sr),
                "angular_distance_min": _data_range_value(histograms, "angular_distance", 0)
                * u.deg,
                "angular_distance_max": _data_range_value(histograms, "angular_distance", 1)
                * u.deg,
                "energy_bins_per_decade": histograms.energy_bins_per_decade,
                "angular_distance_bin_width": _get_angular_distance_bin_width(histograms),
                "angular_distance_bin_count": len(histograms.view_cone_bins) - 1,
                "core_distance_bin_count": len(histograms.core_distance_bins) - 1,
                "total_simulated_events": int(
                    np.sum(
                        histograms.histograms["angular_distance_vs_energy_vs_core_distance_mc"][
                            "histogram"
                        ]
                    )
                ),
                "total_triggered_events": int(
                    np.sum(
                        histograms.histograms["angular_distance_vs_energy_vs_core_distance"][
                            "histogram"
                        ]
                    )
                ),
            }
        ]
    )
    metadata.meta["EXTNAME"] = TRIGGER_HISTOGRAM_METADATA_TABLE
    return metadata


def _is_persisted_histogram(name, histogram):
    """Return True for base histograms that should be serialized."""
    if name.endswith(_DERIVED_HISTOGRAM_SUFFIXES):
        return False
    return isinstance(histogram, dict) and histogram.get("histogram") is not None


def _create_histogram_value_table(reference_specs):
    """Create a flattened table containing all persisted histogram counts."""
    rows = []
    for spec in reference_specs:
        for histogram_name, histogram in spec["histograms"].histograms.items():
            if not _is_persisted_histogram(histogram_name, histogram):
                continue
            values = np.asarray(histogram["histogram"])
            for flat_index, value in enumerate(values.ravel()):
                indices = np.unravel_index(flat_index, values.shape)
                padded_indices = list(indices) + [-1] * (3 - len(indices))
                rows.append(
                    {
                        "reference_id": spec["reference_id"],
                        "histogram_name": histogram_name,
                        "dimension": values.ndim,
                        "index_0": padded_indices[0],
                        "index_1": padded_indices[1],
                        "index_2": padded_indices[2],
                        "value": float(value),
                    }
                )

    table = Table(rows=rows)
    table.meta["EXTNAME"] = TRIGGER_HISTOGRAM_VALUES_TABLE
    return table


def _iter_histogram_edges(histogram):
    """Yield one bin-edge array per histogram axis."""
    bin_edges = histogram["bin_edges"]
    if histogram["1d"]:
        yield 0, bin_edges
        return
    yield from enumerate(bin_edges)


def _create_histogram_edge_table(reference_specs):
    """Create a flattened table containing bin edges for persisted histograms."""
    rows = []
    for spec in reference_specs:
        for histogram_name, histogram in spec["histograms"].histograms.items():
            if not _is_persisted_histogram(histogram_name, histogram):
                continue
            for axis_index, edges in _iter_histogram_edges(histogram):
                for bin_index, edge in enumerate(np.asarray(edges, dtype=float)):
                    rows.append(
                        {
                            "reference_id": spec["reference_id"],
                            "histogram_name": histogram_name,
                            "axis_index": axis_index,
                            "bin_index": bin_index,
                            "edge": float(edge),
                        }
                    )

    table = Table(rows=rows)
    table.meta["EXTNAME"] = TRIGGER_HISTOGRAM_EDGES_TABLE
    return table


def _create_bin_table(reference_id, production_index, array_name, histograms):
    """Create the flattened per-bin histogram table."""
    triggered_hist = histograms.histograms["angular_distance_vs_energy_vs_core_distance"][
        "histogram"
    ]
    simulated_hist = histograms.histograms["angular_distance_vs_energy_vs_core_distance_mc"][
        "histogram"
    ]
    efficiency_hist = histograms.histograms["angular_distance_vs_energy_vs_core_distance_eff"][
        "histogram"
    ]
    energy_edges = histograms.energy_bins
    angular_edges = histograms.view_cone_bins
    core_edges = histograms.core_distance_bins

    rows = []
    for angular_index in range(len(angular_edges) - 1):
        for energy_index in range(len(energy_edges) - 1):
            for core_index in range(len(core_edges) - 1):
                rows.append(
                    {
                        "reference_id": reference_id,
                        "production_index": production_index,
                        "array_name": array_name,
                        "angular_distance_bin_index": angular_index,
                        "energy_bin_index": energy_index,
                        "core_distance_bin_index": core_index,
                        "angular_distance_low": angular_edges[angular_index] * u.deg,
                        "angular_distance_high": angular_edges[angular_index + 1] * u.deg,
                        "energy_low": energy_edges[energy_index] * u.TeV,
                        "energy_high": energy_edges[energy_index + 1] * u.TeV,
                        "core_distance_low": core_edges[core_index] * u.m,
                        "core_distance_high": core_edges[core_index + 1] * u.m,
                        "simulated_count": int(
                            simulated_hist[angular_index, energy_index, core_index]
                        ),
                        "triggered_count": int(
                            triggered_hist[angular_index, energy_index, core_index]
                        ),
                        "trigger_efficiency": float(
                            efficiency_hist[angular_index, energy_index, core_index]
                        ),
                    }
                )

    bin_table = Table(rows=rows)
    bin_table.meta["EXTNAME"] = TRIGGER_HISTOGRAM_BINS_TABLE
    return bin_table


def _process_production(
    file_path,
    telescope_configs,
    energy_bins_per_decade,
    angular_distance_bin_width,
    skip_invalid_event_data_files=False,
):
    """Read one production once and build trigger histograms for all telescope configurations."""
    finalized_histograms = accumulate_histograms_by_telescope_config(
        file_path,
        telescope_configs,
        energy_bins_per_decade=energy_bins_per_decade,
        angular_distance_bin_width=angular_distance_bin_width,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        fill_efficiency_histogram=True,
        collect_trigger_topology=True,
    )
    return [(histograms, topology) for _, histograms, topology in finalized_histograms]


def write_trigger_histograms(args_dict):
    """
    Build trigger histograms and write them to file.

    Parameters
    ----------
    args_dict : dict
        Application arguments.

    Returns
    -------
    tuple[astropy.table.Table, astropy.table.Table]
        Metadata and per-bin histogram tables written to the HDF5 output file.

    Raises
    ------
    ValueError
        If the output file does not use an HDF5 suffix or no supported telescope
        selection is provided.
    """
    production_patterns = normalize_event_data_file(args_dict["event_data_file"])
    telescope_configs = _use_readable_inline_array_names(
        normalize_telescope_configs(resolve_telescope_configs(args_dict))
    )
    output_file = gen.validate_file_type(
        io_handler.IOHandler().get_output_file(args_dict["output_file"]),
        expected_suffixes=[".hdf5", ".h5"],
    )

    reference_specs = []

    for production_index, pattern in enumerate(production_patterns):
        histogram_topology_pairs = _process_production(
            pattern,
            telescope_configs,
            energy_bins_per_decade=args_dict["energy_bins_per_decade"],
            angular_distance_bin_width=args_dict["angular_distance_bin_width"],
            skip_invalid_event_data_files=args_dict.get("skip_invalid_event_data_files", False),
        )

        for config, (histograms, trigger_topology) in zip(
            telescope_configs, histogram_topology_pairs
        ):
            # reference_id connects metadata and bin tables
            reference_id = gen.get_uuid()
            reference_specs.append(
                {
                    "reference_id": reference_id,
                    "production_index": production_index,
                    "event_data_file": pattern,
                    "site": args_dict.get("site"),
                    "array_name": config["array_name"],
                    "telescope_ids": config["telescope_ids"],
                    "histograms": histograms,
                    "trigger_topology": trigger_topology,
                }
            )

    metadata_table, bin_table = _create_histogram_tables(reference_specs)
    histogram_value_table = _create_histogram_value_table(reference_specs)
    histogram_edge_table = _create_histogram_edge_table(reference_specs)
    topology_count_table = _create_trigger_topology_count_table(reference_specs)
    subset_histogram_table = _create_trigger_subset_histogram_table(reference_specs)
    table_handler.write_tables(
        [
            metadata_table,
            bin_table,
            histogram_value_table,
            histogram_edge_table,
            topology_count_table,
            subset_histogram_table,
        ],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )
    return metadata_table, bin_table


def load_trigger_histograms(reference_file):
    """
    Load trigger-histogram metadata and bin tables from HDF5.

    Parameters
    ----------
    reference_file : str or pathlib.Path
        Path to the trigger-histogram HDF5 file.

    Returns
    -------
    tuple[astropy.table.Table, astropy.table.Table]
        Metadata table and per-bin histogram table read from the input file.
    """
    tables = table_handler.read_tables(
        reference_file,
        [TRIGGER_HISTOGRAM_METADATA_TABLE, TRIGGER_HISTOGRAM_BINS_TABLE],
        file_type="HDF5",
    )
    return tables[TRIGGER_HISTOGRAM_METADATA_TABLE], tables[TRIGGER_HISTOGRAM_BINS_TABLE]


def _row_value(row, name):
    """Return a scalar row value, preserving astropy quantities."""
    value = row[name]
    unit = getattr(value, "unit", None)
    table = getattr(row, "_table", None)
    if unit is None and table is not None and name in table.colnames:
        unit = getattr(table[name], "unit", None)
    if unit is not None:
        return value * unit
    return value


def _metadata_file_info(row):
    """Build EventDataHistograms.file_info from a metadata row."""
    return {
        "primary_particle": row["primary_particle"],
        "zenith": _row_value(row, "zenith"),
        "azimuth": _row_value(row, "azimuth"),
        "nsb_level": row["nsb_level"],
        "energy_min": _row_value(row, "energy_min"),
        "energy_max": _row_value(row, "energy_max"),
        "viewcone_min": _row_value(row, "viewcone_min"),
        "viewcone_max": _row_value(row, "viewcone_max"),
        "core_scatter_min": _row_value(row, "core_scatter_min"),
        "core_scatter_max": _row_value(row, "core_scatter_max"),
        "scatter_area": _row_value(row, "scatter_area"),
        "solid_angle": _row_value(row, "solid_angle"),
    }


def _metadata_data_ranges(row):
    """Build data_ranges from metadata values if present."""
    if "angular_distance_min" not in row.colnames or "angular_distance_max" not in row.colnames:
        return {}
    angular_min = _quantity_value(_row_value(row, "angular_distance_min"), u.deg)
    angular_max = _quantity_value(_row_value(row, "angular_distance_max"), u.deg)
    angular_min = angular_min.value if hasattr(angular_min, "value") else float(angular_min)
    angular_max = angular_max.value if hasattr(angular_max, "value") else float(angular_max)
    if np.isnan(angular_min) or np.isnan(angular_max):
        return {}
    return {"angular_distance": (float(angular_min), float(angular_max))}


def _rows_for_reference(table, reference_id):
    """Return table rows matching a reference id."""
    return table[table["reference_id"] == reference_id]


def _edges_by_histogram(edge_rows):
    """Return nested histogram edge arrays by histogram name and axis."""
    edges = {}
    for histogram_name in sorted(set(edge_rows["histogram_name"])):
        histogram_rows = edge_rows[edge_rows["histogram_name"] == histogram_name]
        edges[histogram_name] = {}
        for axis_index in sorted(set(histogram_rows["axis_index"])):
            axis_rows = histogram_rows[histogram_rows["axis_index"] == axis_index]
            axis_rows.sort("bin_index")
            edges[histogram_name][int(axis_index)] = np.asarray(axis_rows["edge"], dtype=float)
    return edges


def _histogram_values_by_name(value_rows):
    """Return histogram arrays reconstructed from flattened value rows."""
    histograms = {}
    for histogram_name in sorted(set(value_rows["histogram_name"])):
        histogram_rows = value_rows[value_rows["histogram_name"] == histogram_name]
        dimension = int(histogram_rows["dimension"][0])
        shape = tuple(
            int(np.max(histogram_rows[f"index_{axis_index}"])) + 1
            for axis_index in range(dimension)
        )
        values = np.zeros(shape, dtype=float)
        for row in histogram_rows:
            index = tuple(int(row[f"index_{axis_index}"]) for axis_index in range(dimension))
            values[index] = float(row["value"])
        histograms[histogram_name] = values
    return histograms


def _template_histograms(histograms):
    """Create metadata-rich empty histogram definitions for a loaded histogram object."""
    return histograms.get_empty_histogram_definitions()


def _apply_loaded_histograms(histograms, values_by_name, edges_by_name):
    """Attach loaded histogram arrays and exact bin edges to a histogram object."""
    templates = _template_histograms(histograms)
    loaded = {}
    loaded_bin_edges = {}
    for histogram_name, values in values_by_name.items():
        definition = copy.copy(templates.get(histogram_name, {}))
        definition.setdefault("1d", values.ndim == 1)
        definition["histogram"] = values
        definition["event_data"] = (
            None if definition["1d"] else tuple(None for _ in range(values.ndim))
        )
        axis_edges = edges_by_name[histogram_name]
        definition["bin_edges"] = (
            axis_edges[0]
            if values.ndim == 1
            else tuple(axis_edges[index] for index in range(values.ndim))
        )
        loaded[histogram_name] = definition

    if "energy" in edges_by_name:
        loaded_bin_edges["energy"] = edges_by_name["energy"][0]
    if "core_distance" in edges_by_name:
        loaded_bin_edges["core_distance"] = edges_by_name["core_distance"][0]
    if "angular_distance" in edges_by_name:
        loaded_bin_edges["viewcone"] = edges_by_name["angular_distance"][0]

    return loaded, loaded_bin_edges


def _histograms_from_reference_row(row, value_rows, edge_rows):
    """Reconstruct finalized EventDataHistograms for one reference row."""
    histograms = EventDataHistograms.create_accumulator(
        array_name=row["array_name"],
        telescope_list=list(filter(None, str(row["telescope_ids"]).split(","))),
        energy_bins_per_decade=int(row["energy_bins_per_decade"]),
        angular_distance_bin_count=int(row["angular_distance_bin_count"]) + 1,
        angular_distance_bin_width=_row_value(row, "angular_distance_bin_width"),
        core_distance_bin_count=int(row["core_distance_bin_count"]) + 1,
    )
    loaded_histograms, loaded_bin_edges = _apply_loaded_histograms(
        histograms,
        _histogram_values_by_name(value_rows),
        _edges_by_histogram(edge_rows),
    )
    histograms.set_loaded_histograms(
        loaded_histograms,
        file_info=_metadata_file_info(row),
        data_ranges=_metadata_data_ranges(row),
        loaded_bin_edges=loaded_bin_edges,
        contains_triggered_data=True,
    )
    histograms.calculate_efficiency_data()
    histograms.calculate_cumulative_data()
    return histograms


def load_event_data_histograms(reference_file, array_names=None, production_indices=None):
    """
    Load finalized EventDataHistograms objects from a trigger-histogram HDF5 file.

    Parameters
    ----------
    reference_file : str or pathlib.Path
        Path to the trigger-histogram HDF5 file.
    array_names : list[str], optional
        Restrict loaded histograms to these array names.
    production_indices : list[int], optional
        Restrict loaded histograms to these production indices.

    Returns
    -------
    list[tuple[astropy.table.Row, EventDataHistograms]]
        Metadata row and reconstructed histogram object for each reference.
    """
    tables = table_handler.read_tables(
        reference_file,
        [
            TRIGGER_HISTOGRAM_METADATA_TABLE,
            TRIGGER_HISTOGRAM_VALUES_TABLE,
            TRIGGER_HISTOGRAM_EDGES_TABLE,
        ],
        file_type="HDF5",
    )
    metadata = tables[TRIGGER_HISTOGRAM_METADATA_TABLE]
    values = tables[TRIGGER_HISTOGRAM_VALUES_TABLE]
    edges = tables[TRIGGER_HISTOGRAM_EDGES_TABLE]

    if array_names:
        metadata = metadata[np.isin(metadata["array_name"].astype(str), array_names)]
    if production_indices is not None:
        metadata = metadata[np.isin(metadata["production_index"], production_indices)]

    loaded = []
    for row in metadata:
        reference_id = row["reference_id"]
        loaded.append(
            (
                row,
                _histograms_from_reference_row(
                    row,
                    _rows_for_reference(values, reference_id),
                    _rows_for_reference(edges, reference_id),
                ),
            )
        )
    return loaded
