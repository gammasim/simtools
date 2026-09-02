"""Build and load trigger-histogram products from reduced event data."""

import copy
import logging
import os
import re
from pathlib import Path

import astropy.units as u
import h5py
import numpy as np
from astropy.table import Table, vstack

import simtools.utils.general as gen
from simtools.io import io_handler, table_handler
from simtools.io.ascii_handler import write_data_to_file
from simtools.io.file_type import validate_file_type
from simtools.job_execution.execution import map_ordered, options_from_args, submit_jobs
from simtools.job_execution.job import JobSpec
from simtools.production_configuration.production_event_data_helpers import (
    accumulate_histograms_by_telescope_config,
    normalize_telescope_configs,
    resolve_telescope_configs,
)
from simtools.production_configuration.production_file_selection import (
    ProductionManifest,
    check_manifest,
    get_manifest_schema_metadata,
    select_file_groups,
    selection_summary,
    stable_configuration_hash,
)
from simtools.settings import config as settings_config
from simtools.sim_events.histograms import EventDataHistograms
from simtools.sim_events.metadata import build_standard_metadata

_logger = logging.getLogger(__name__)

TRIGGER_HISTOGRAM_METADATA_TABLE = "TRIGGER_REFERENCE_METADATA"
TRIGGER_HISTOGRAM_BINS_TABLE = "TRIGGER_REFERENCE_BINS"
TRIGGER_TOPOLOGY_COUNTS_TABLE = "TRIGGER_TOPOLOGY_COUNTS"
TRIGGER_SUBSET_HISTOGRAMS_TABLE = "TRIGGER_SUBSET_HISTOGRAMS"
TRIGGER_HISTOGRAM_DENSE_GROUP = "TRIGGER_HISTOGRAM_DENSE"

_DERIVED_HISTOGRAM_SUFFIXES = ("_eff", "_cumulative")
_REDUCED_EVENT_DATA_SUFFIX = ".reduced_event_data.hdf5"
_PART_SUFFIX_PATTERN = re.compile(r"\.part\d+$")


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
    metadata_tables = [
        _create_metadata_table(
            reference_id=spec["reference_id"],
            production_index=spec["production_index"],
            site=spec["site"],
            array_name=spec["array_name"],
            telescope_ids=spec["telescope_ids"],
            histograms=spec["histograms"],
        )
        for spec in reference_specs
    ]
    bin_tables = [
        _create_bin_table(
            reference_id=spec["reference_id"],
            production_index=spec["production_index"],
            array_name=spec["array_name"],
            histograms=spec["histograms"],
        )
        for spec in reference_specs
    ]

    return (
        vstack(metadata_tables, metadata_conflicts="silent"),
        vstack(bin_tables, metadata_conflicts="silent"),
    )


def _iter_persisted_histograms(histograms):
    """Yield persisted histogram names and definitions."""
    for histogram_name, histogram in histograms.histograms.items():
        if _is_persisted_histogram(histogram_name, histogram):
            yield histogram_name, histogram


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
                "site": site or "",
                "array_name": array_name,
                "telescope_ids": ",".join(telescope_ids) if telescope_ids else "",
                "primary_particle": file_info.get("primary_particle", ""),
                "zenith": file_info.get("zenith", 0.0 * u.deg),
                "azimuth": file_info.get("azimuth", 0.0 * u.deg),
                "nsb_level": file_info.get("nsb_level", 0.0),
                "spectral_index": file_info.get("spectral_index", np.nan),
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


def _iter_histogram_edges(histogram):
    """Yield one bin-edge array per histogram axis."""
    bin_edges = histogram["bin_edges"]
    if histogram["1d"]:
        yield 0, bin_edges
        return
    yield from enumerate(bin_edges)


def _write_dense_histogram_payload(reference_specs, output_file):
    """Write persisted histogram arrays and bin edges as dense HDF5 datasets."""
    with h5py.File(output_file, "a") as hdf5_file:
        dense_group = hdf5_file.require_group(TRIGGER_HISTOGRAM_DENSE_GROUP)
        dense_group.attrs["format_version"] = 1
        for spec in reference_specs:
            reference_group = dense_group.create_group(str(spec["reference_id"]))
            for histogram_name, histogram in _iter_persisted_histograms(spec["histograms"]):
                histogram_group = reference_group.create_group(str(histogram_name))
                _create_dense_histogram_dataset(
                    histogram_group,
                    "values",
                    data=np.asarray(histogram["histogram"]),
                )
                for axis_index, edges in _iter_histogram_edges(histogram):
                    _create_dense_histogram_dataset(
                        histogram_group,
                        f"edges_{axis_index}",
                        data=np.asarray(edges, dtype=float),
                    )


def _create_dense_histogram_dataset(hdf5_group, dataset_name, data):
    """Create one compressed dataset for dense trigger-histogram payloads."""
    return hdf5_group.create_dataset(
        dataset_name,
        data=data,
        chunks=True,
        compression="gzip",
        compression_opts=6,
        shuffle=True,
    )


def _load_dense_histogram_payloads(reference_file):
    """Load dense histogram arrays and edges grouped by reference id."""
    with h5py.File(reference_file, "r") as hdf5_file:
        if TRIGGER_HISTOGRAM_DENSE_GROUP not in hdf5_file:
            return {}

        dense_group = hdf5_file[TRIGGER_HISTOGRAM_DENSE_GROUP]
        payloads = {}
        for reference_id, reference_group in dense_group.items():
            values_by_name = {}
            edges_by_name = {}
            for histogram_name, histogram_group in reference_group.items():
                values_by_name[histogram_name] = np.asarray(histogram_group["values"])
                axis_edges = {}
                for dataset_name, dataset in histogram_group.items():
                    if not dataset_name.startswith("edges_"):
                        continue
                    axis_index = int(dataset_name.split("_", maxsplit=1)[1])
                    axis_edges[axis_index] = np.asarray(dataset, dtype=float)
                edges_by_name[histogram_name] = axis_edges
            payloads[reference_id] = (values_by_name, edges_by_name)
        return payloads


def inspect_trigger_histogram_file(file_path, format_report=True):
    """
    Return trigger-histogram-specific consistency checks when applicable.

    Parameters
    ----------
    file_path : str or Path
        Path to the file to inspect.
    format_report : bool, optional
        Whether to format the report as human-readable text. Default is True.

    Returns
    -------
    str or dict or None
        Formatted report string, structured report dictionary, or None if the file
        does not contain trigger-histogram products.
    """
    with h5py.File(file_path, "r") as hdf5_file:
        has_metadata = TRIGGER_HISTOGRAM_METADATA_TABLE in hdf5_file
        has_dense_group = TRIGGER_HISTOGRAM_DENSE_GROUP in hdf5_file
        if not has_metadata and not has_dense_group:
            return None

        dense_reference_ids = (
            sorted(
                str(reference_id)
                for reference_id in hdf5_file[TRIGGER_HISTOGRAM_DENSE_GROUP].keys()
            )
            if has_dense_group
            else []
        )

    metadata_reference_ids = []
    if has_metadata:
        metadata = table_handler.read_tables(
            file_path,
            [TRIGGER_HISTOGRAM_METADATA_TABLE],
            file_type="HDF5",
        )[TRIGGER_HISTOGRAM_METADATA_TABLE]
        metadata_reference_ids = sorted(str(row["reference_id"]) for row in metadata)

    metadata_reference_id_set = set(metadata_reference_ids)
    dense_reference_id_set = set(dense_reference_ids)
    report = {
        "inspector": "trigger_histogram",
        "metadata_reference_count": len(metadata_reference_ids),
        "dense_reference_count": len(dense_reference_ids),
        "missing_dense_reference_ids": sorted(metadata_reference_id_set - dense_reference_id_set),
        "orphan_dense_reference_ids": sorted(dense_reference_id_set - metadata_reference_id_set),
    }
    return _format_trigger_histogram_inspection(report) if format_report else report


def _format_trigger_histogram_inspection(report):
    """Format trigger-histogram inspection results for console output."""
    lines = ["Trigger histogram consistency:"]
    lines.append(f"- metadata reference ids: {report['metadata_reference_count']}")
    lines.append(f"- dense payload reference ids: {report['dense_reference_count']}")
    if report["missing_dense_reference_ids"]:
        lines.append(
            "- missing dense payloads for metadata ids: "
            + ", ".join(report["missing_dense_reference_ids"])
        )
    else:
        lines.append("- missing dense payloads for metadata ids: none")
    if report["orphan_dense_reference_ids"]:
        lines.append(
            "- orphan dense payload ids without metadata rows: "
            + ", ".join(report["orphan_dense_reference_ids"])
        )
    else:
        lines.append("- orphan dense payload ids without metadata rows: none")
    return "\n".join(lines)


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
    core_distance_bin_width=None,
    skip_invalid_event_data_files=False,
    minimum_triggered_telescopes=2,
):
    """Read one production once and build trigger histograms for all telescope configurations."""
    finalized_histograms = accumulate_histograms_by_telescope_config(
        file_path,
        telescope_configs,
        energy_bins_per_decade=energy_bins_per_decade,
        angular_distance_bin_width=angular_distance_bin_width,
        core_distance_bin_width=core_distance_bin_width,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        fill_efficiency_histogram=True,
        collect_trigger_topology=True,
        minimum_triggered_telescopes=minimum_triggered_telescopes,
    )
    return [(histograms, topology) for _, histograms, topology in finalized_histograms]


def _execute_production_job(job_spec):
    """Execute one production-pattern histogram job in a worker process."""
    histogram_topology_pairs = _process_production(
        job_spec["production_pattern"],
        job_spec["telescope_configs"],
        energy_bins_per_decade=job_spec["energy_bins_per_decade"],
        angular_distance_bin_width=job_spec["angular_distance_bin_width"],
        core_distance_bin_width=job_spec.get("core_distance_bin_width"),
        skip_invalid_event_data_files=job_spec["skip_invalid_event_data_files"],
        minimum_triggered_telescopes=job_spec.get("minimum_triggered_telescopes", 2),
    )
    return [
        {
            "production_index": job_spec["production_index"],
            "site": job_spec["site"],
            "array_name": config["array_name"],
            "telescope_ids": config["telescope_ids"],
            "histograms": histograms,
            "trigger_topology": trigger_topology,
        }
        for config, (histograms, trigger_topology) in zip(
            job_spec["telescope_configs"], histogram_topology_pairs
        )
    ]


def discover_event_data_groups(event_data_directory):
    """
    Discover reduced event-data files and group partitioned products by filename.

    Parameters
    ----------
    event_data_directory : str or pathlib.Path
        Directory containing direct-child reduced event-data HDF5 files.

    Returns
    -------
    list[tuple[str, list[pathlib.Path]]]
        Sorted ``(group_name, files)`` pairs. A group name is the file stem with a
        trailing ``.part<digits>`` suffix removed.

    Raises
    ------
    FileNotFoundError
        If the input directory does not exist or is not a directory.
    ValueError
        If the directory does not contain matching reduced event-data files.
    """
    directory = Path(event_data_directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Event-data directory not found: {directory}")

    groups = {}
    for file_path in sorted(directory.glob(f"*{_REDUCED_EVENT_DATA_SUFFIX}")):
        if not file_path.is_file():
            continue
        file_stem = file_path.name.removesuffix(_REDUCED_EVENT_DATA_SUFFIX)
        group_name = _PART_SUFFIX_PATTERN.sub("", file_stem)
        groups.setdefault(group_name, []).append(file_path)

    if not groups:
        raise ValueError(
            "No reduced event-data files matching "
            f"'*{_REDUCED_EVENT_DATA_SUFFIX}' found in {directory}."
        )
    return [(group_name, groups[group_name]) for group_name in sorted(groups)]


def _write_trigger_histogram_product(
    args_dict,
    production_patterns,
    output_file,
    product_metadata=None,
    telescope_configs=None,
):
    """Build and write one trigger-histogram product from production sources."""
    telescope_configs = telescope_configs or _resolve_telescope_configs(args_dict)
    output_file = validate_file_type(output_file, file_type="hdf5")

    reference_specs = []
    job_specs = [
        {
            "production_index": production_index,
            "production_pattern": pattern,
            "site": args_dict.get("site"),
            "telescope_configs": telescope_configs,
            "energy_bins_per_decade": args_dict["energy_bins_per_decade"],
            "angular_distance_bin_width": args_dict["angular_distance_bin_width"],
            "core_distance_bin_width": args_dict.get("core_distance_bin_width"),
            "skip_invalid_event_data_files": args_dict.get("skip_invalid_event_data_files", False),
            "minimum_triggered_telescopes": args_dict.get("minimum_triggered_telescopes", 2),
        }
        for production_index, pattern in enumerate(production_patterns)
    ]
    _logger.info(
        "Processing %d trigger-histogram production pattern(s) with max_workers=%s",
        len(job_specs),
        args_dict.get("max_workers", 1),
    )
    for production_result in map_ordered(
        _execute_production_job,
        job_specs,
        backend=args_dict.get("backend", "local"),
        max_workers=args_dict.get("max_workers", 1),
        backend_config=args_dict.get("backend_config"),
    ):
        for spec in production_result:
            reference_specs.append(spec | {"reference_id": f"reference_{len(reference_specs)}"})

    metadata_table, bin_table = _create_histogram_tables(reference_specs)
    topology_count_table = _create_trigger_topology_count_table(reference_specs)
    subset_histogram_table = _create_trigger_subset_histogram_table(reference_specs)
    table_handler.write_tables(
        [
            metadata_table,
            bin_table,
            topology_count_table,
            subset_histogram_table,
        ],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
        metadata_documents={
            "METADATA": build_standard_metadata(
                args_dict,
                output_file,
                product_data_name="trigger_histograms",
            ),
        },
    )
    _write_dense_histogram_payload(reference_specs, output_file)
    if product_metadata is not None:
        _write_trigger_histogram_metadata(args_dict, output_file, product_metadata)
    return metadata_table, bin_table


def _write_trigger_histogram_metadata(args_dict, output_file, product_metadata):
    """Write a neighboring YAML manifest for a trigger-histogram product."""
    output_file = Path(output_file)
    metadata_file = output_file.with_suffix(".yml")
    manifest_schema = get_manifest_schema_metadata("trigger_histograms")
    manifest = {
        **manifest_schema,
        "product_type": "trigger_histograms",
        "status": "complete",
        "configuration": product_metadata["configuration"],
        "histogram_settings": _histogram_settings(args_dict),
        "array_selection": product_metadata["array_selection"],
        "input_run_numbers": product_metadata["run_numbers"],
        "input_files": {
            "reduced_event_data": [
                _relative_to_directory(path, metadata_file.parent)
                for path in product_metadata["input_files"]
            ],
        },
        "files": {
            "trigger_histograms": [_relative_to_directory(output_file, metadata_file.parent)],
        },
    }
    check_manifest(ProductionManifest(path=metadata_file, data=manifest))
    write_data_to_file(manifest, metadata_file)


def _write_directory_group_job(job_spec):
    """Build one directory-discovered trigger-histogram product."""
    args_dict = job_spec["args_dict"] | {
        "event_data_files": job_spec["event_data_files"],
        "backend": "local",
        "backend_config": None,
        "max_workers": 1,
    }
    _write_trigger_histogram_product(
        args_dict,
        [job_spec["event_data_files"]],
        job_spec["output_file"],
        product_metadata=job_spec.get("product_metadata"),
        telescope_configs=job_spec.get("telescope_configs"),
    )
    return str(job_spec["output_file"])


def _write_directory_products(args_dict):
    """Submit or execute one independent trigger-histogram product per directory group."""
    output_directory = io_handler.IOHandler().get_output_directory()
    event_data_directory = Path(args_dict["event_data_directory"]).resolve()
    groups = discover_event_data_groups(event_data_directory)
    runtime_args = (
        dict(settings_config.args) if args_dict.get("backend", "local") != "local" else None
    )
    runtime_db_config = dict(settings_config.db_config) if runtime_args is not None else None
    jobs = []
    for index, (group_name, event_data_files) in enumerate(groups):
        output_file = validate_file_type(
            output_directory / f"{group_name}.trigger_histograms.hdf5",
            file_type="hdf5",
        )
        jobs.append(
            JobSpec(
                job_id=f"trigger-histograms-{index:06d}",
                index=index,
                function=_write_directory_group_job,
                item={
                    "args_dict": args_dict,
                    "event_data_files": [str(file_path) for file_path in event_data_files],
                    "output_file": output_file,
                },
                runtime_args=runtime_args,
                runtime_db_config=runtime_db_config,
                mount_paths=(event_data_directory,),
                output_paths=(output_file,),
            )
        )
    _logger.info("Submitting %d trigger-histogram directory group(s)", len(jobs))
    return submit_jobs(jobs, options_from_args(args_dict, max_workers=args_dict.get("max_workers")))


def _write_production_selection_products(args_dict):
    """Submit one trigger-histogram product per selected production configuration group."""
    if args_dict.get("file_type", "reduced_event_data") != "reduced_event_data":
        raise ValueError("Production metadata input requires file_type='reduced_event_data'.")
    output_directory = io_handler.IOHandler().get_output_directory()
    selection_result = select_file_groups(
        args_dict["production_path"],
        selections=args_dict.get("select"),
        file_type="reduced_event_data",
        require_complete_runs=args_dict.get("require_complete_runs", False),
    )
    _logger.info("\n%s", selection_summary(selection_result))
    if not selection_result["groups"]:
        raise ValueError("Production metadata selection did not match any files.")

    production_path = Path(args_dict["production_path"]).resolve()
    runtime_args = (
        dict(settings_config.args) if args_dict.get("backend", "local") != "local" else None
    )
    runtime_db_config = dict(settings_config.db_config) if runtime_args is not None else None
    jobs = []
    for index, group in enumerate(selection_result["groups"]):
        telescope_configs = _resolve_group_telescope_configs(args_dict, group.configuration)
        product_identity = {
            "histogram_settings": _histogram_settings(args_dict),
            "array_selection": telescope_configs,
        }
        output_file = validate_file_type(
            output_directory
            / f"{_group_output_stem(group, product_identity)}.trigger_histograms.hdf5",
            file_type="hdf5",
        )
        metadata_file = output_file.with_suffix(".yml")
        if output_file.exists() or metadata_file.exists():
            raise FileExistsError(f"Trigger-histogram output already exists: {output_file}")
        jobs.append(
            JobSpec(
                job_id=f"trigger-histograms-{index:06d}",
                index=index,
                function=_write_directory_group_job,
                item={
                    "args_dict": args_dict,
                    "event_data_files": [str(file_path) for file_path in group.file_paths],
                    "output_file": output_file,
                    "product_metadata": {
                        "configuration": group.configuration,
                        "run_numbers": group.run_numbers,
                        "input_files": group.file_paths,
                        "array_selection": telescope_configs,
                    },
                    "telescope_configs": telescope_configs,
                },
                runtime_args=runtime_args,
                runtime_db_config=runtime_db_config,
                mount_paths=(production_path,),
                output_paths=(output_file, metadata_file),
            )
        )
    _logger.info("Submitting %d trigger-histogram production metadata group(s)", len(jobs))
    return submit_jobs(jobs, options_from_args(args_dict, max_workers=args_dict.get("max_workers")))


def write_trigger_histograms(args_dict):
    """
    Build trigger histograms and write them to file.

    Parameters
    ----------
    args_dict : dict
        Application arguments.

    Returns
    -------
    tuple[astropy.table.Table, astropy.table.Table] or simtools.job_execution.job.SubmissionHandle
        Metadata and per-bin histogram tables for explicit input patterns, or the submission
        handle for directory-mode jobs.

    Raises
    ------
    ValueError
        If the output file does not use an HDF5 suffix or no supported telescope
        selection is provided.
    """
    if args_dict.get("event_data_directory"):
        return _write_directory_products(args_dict)
    if args_dict.get("production_path"):
        return _write_production_selection_products(args_dict)

    production_patterns = gen.ensure_string_lists(args_dict["event_data_files"])
    output_file = io_handler.IOHandler().get_output_file(args_dict["output_file"])
    return _write_trigger_histogram_product(args_dict, production_patterns, output_file)


def _group_output_stem(group, product_identity=None):
    """Return a readable, stable output stem for a selected production group."""
    configuration = group.configuration
    parts = [
        _safe_stem_part(configuration.get("primary", "production")),
        _angle_stem_part("za", configuration.get("zenith_angle")),
        _safe_stem_part(configuration.get("corsika_he_interaction")),
    ]
    parts = [part for part in parts if part]
    parts.append(
        stable_configuration_hash(
            {
                "configuration": configuration,
                **(product_identity or {}),
            }
        )
    )
    return "_".join(parts)


def _angle_stem_part(prefix, value):
    """Return an angle value formatted for an output file stem."""
    if isinstance(value, dict) and set(value) == {"value", "unit"}:
        quantity = float(value["value"]) * u.Unit(value["unit"])
        return f"{prefix}{quantity.to_value(u.deg):g}"
    return None


def _safe_stem_part(value):
    """Return a filesystem-safe output stem component."""
    if value is None:
        return None
    return re.sub(r"[^A-Za-z0-9]+", "-", str(value)).strip("-").lower()


def _relative_to_directory(path, directory):
    """Return a portable path relative to a metadata directory."""
    path = Path(path).resolve()
    directory = Path(directory).resolve()
    return Path(os.path.relpath(path, directory)).as_posix()


def _resolve_telescope_configs(args_dict):
    """Return normalized resolved telescope configurations."""
    return _use_readable_inline_array_names(
        normalize_telescope_configs(resolve_telescope_configs(args_dict))
    )


def _resolve_group_telescope_configs(args_dict, configuration):
    """Resolve telescope configurations from one selected production configuration."""
    group_args = dict(args_dict)
    _set_group_argument(group_args, configuration, "site")
    _set_group_argument(group_args, configuration, "model_version", list_value=True)
    if group_args.get("array_element_list"):
        return _resolve_telescope_configs(group_args)
    _set_group_argument(group_args, configuration, "array_layout_name", list_value=True)
    return _resolve_telescope_configs(group_args)


def _set_group_argument(args_dict, configuration, key, list_value=False):
    """Set one telescope-resolution argument from metadata or reject an incompatible override."""
    if key not in configuration:
        raise ValueError(f"Selected production metadata is missing configuration.{key}.")
    metadata_value = configuration[key]
    expected = _as_argument_value(metadata_value, list_value)
    provided = args_dict.get(key)
    if provided is not None and provided != expected:
        raise ValueError(
            f"Explicit --{key}={provided} does not match selected production metadata {expected}."
        )
    args_dict[key] = expected


def _as_argument_value(value, list_value):
    """Return manifest values in the form accepted by the command-line resolver."""
    if not list_value:
        return value
    return value if isinstance(value, list) else [value]


def _histogram_settings(args_dict):
    """Return settings that define a trigger-histogram product."""
    return {
        "energy_bins_per_decade": args_dict["energy_bins_per_decade"],
        "angular_distance_bin_width": args_dict["angular_distance_bin_width"],
        "core_distance_bin_width": args_dict.get("core_distance_bin_width"),
        "minimum_triggered_telescopes": args_dict.get("minimum_triggered_telescopes", 2),
    }


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
        "spectral_index": row["spectral_index"] if "spectral_index" in row.colnames else np.nan,
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


def _histograms_from_reference_row_dense(row, values_by_name, edges_by_name):
    """Reconstruct finalized EventDataHistograms from dense HDF5 payloads."""
    histograms = EventDataHistograms.create_accumulator(
        array_name=row["array_name"],
        telescope_list=list(filter(None, str(row["telescope_ids"]).split(","))),
        energy_bins_per_decade=int(row["energy_bins_per_decade"]),
        angular_distance_bin_count=int(row["angular_distance_bin_count"]) + 1,
        angular_distance_bin_width=_row_value(row, "angular_distance_bin_width"),
        core_distance_bin_count=int(row["core_distance_bin_count"]) + 1,
    )
    loaded_histograms, loaded_bin_edges = _apply_loaded_histograms(
        histograms, values_by_name, edges_by_name
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
    dense_payloads = _load_dense_histogram_payloads(reference_file)
    metadata = table_handler.read_tables(
        reference_file,
        [TRIGGER_HISTOGRAM_METADATA_TABLE],
        file_type="HDF5",
    )[TRIGGER_HISTOGRAM_METADATA_TABLE]

    if array_names:
        metadata = metadata[np.isin(metadata["array_name"].astype(str), array_names)]
    if production_indices is not None:
        metadata = metadata[np.isin(metadata["production_index"], production_indices)]

    loaded = []
    for row in metadata:
        reference_id = str(row["reference_id"])
        values_by_name, edges_by_name = dense_payloads[reference_id]
        loaded.append(
            (
                row,
                _histograms_from_reference_row_dense(row, values_by_name, edges_by_name),
            )
        )
    return loaded
