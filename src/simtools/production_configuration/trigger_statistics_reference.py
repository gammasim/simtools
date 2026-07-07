"""Build and load trigger-statistics reference products from reduced event data."""

import logging

import astropy.units as u
import numpy as np
from astropy.table import Table, vstack

from simtools.io import io_handler, table_handler
from simtools.production_configuration.production_event_data_helpers import (
    accumulate_histograms_by_telescope_config,
    build_production_subdirectories,
    normalize_event_data_file,
    normalize_telescope_configs,
    resolve_telescope_configs,
)
from simtools.utils.general import get_uuid
from simtools.visualization import plot_simtel_event_histograms

_logger = logging.getLogger(__name__)

TRIGGER_REFERENCE_METADATA_TABLE = "TRIGGER_REFERENCE_METADATA"
TRIGGER_REFERENCE_BINS_TABLE = "TRIGGER_REFERENCE_BINS"


def _create_reference_tables(reference_specs):
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


def _create_metadata_table(
    reference_id,
    production_index,
    event_data_file,
    array_name,
    telescope_ids,
    histograms,
):
    """Create the one-row metadata table for a trigger-statistics reference."""
    file_info = histograms.file_info
    metadata = Table(
        rows=[
            {
                "reference_id": reference_id,
                "production_index": production_index,
                "event_data_file": event_data_file,
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
                "energy_bins_per_decade": histograms.energy_bins_per_decade,
                "angular_distance_bin_count": histograms.angular_distance_bin_count,
                "total_simulated_events": int(
                    np.sum(histograms.histograms["angular_distance_vs_energy_mc"]["histogram"])
                ),
                "total_triggered_events": int(
                    np.sum(histograms.histograms["angular_distance_vs_energy"]["histogram"])
                ),
            }
        ]
    )
    metadata.meta["EXTNAME"] = TRIGGER_REFERENCE_METADATA_TABLE
    return metadata


def _create_bin_table(reference_id, production_index, array_name, histograms):
    """Create the flattened per-bin reference table."""
    triggered_hist = histograms.histograms["angular_distance_vs_energy"]["histogram"]
    simulated_hist = histograms.histograms["angular_distance_vs_energy_mc"]["histogram"]
    efficiency_hist = histograms.histograms["angular_distance_vs_energy_eff"]["histogram"]
    energy_edges = histograms.energy_bins
    angular_edges = histograms.view_cone_bins
    scatter_area = histograms.file_info["scatter_area"].to(u.m**2).value
    effective_area = efficiency_hist * scatter_area

    rows = []
    for angular_index in range(len(angular_edges) - 1):
        for energy_index in range(len(energy_edges) - 1):
            rows.append(
                {
                    "reference_id": reference_id,
                    "production_index": production_index,
                    "array_name": array_name,
                    "angular_bin_index": angular_index,
                    "energy_bin_index": energy_index,
                    "angular_distance_low": angular_edges[angular_index] * u.deg,
                    "angular_distance_high": angular_edges[angular_index + 1] * u.deg,
                    "energy_low": energy_edges[energy_index] * u.TeV,
                    "energy_high": energy_edges[energy_index + 1] * u.TeV,
                    "simulated_count": int(simulated_hist[angular_index, energy_index]),
                    "triggered_count": int(triggered_hist[angular_index, energy_index]),
                    "trigger_efficiency": float(efficiency_hist[angular_index, energy_index]),
                    "effective_area": effective_area[angular_index, energy_index] * u.m**2,
                }
            )

    bin_table = Table(rows=rows)
    bin_table.meta["EXTNAME"] = TRIGGER_REFERENCE_BINS_TABLE
    return bin_table


def _process_production(
    file_path,
    telescope_configs,
    energy_bins_per_decade,
    angular_distance_bin_count,
    skip_invalid_event_data_files=False,
):
    """Read one production once and build trigger references for all telescope configurations."""
    finalized_histograms = accumulate_histograms_by_telescope_config(
        file_path,
        telescope_configs,
        energy_bins_per_decade=energy_bins_per_decade,
        angular_distance_bin_count=angular_distance_bin_count,
        skip_invalid_event_data_files=skip_invalid_event_data_files,
        fill_efficiency_histogram=True,
    )
    return [histograms for _, histograms in finalized_histograms]


def _plot_reference_histograms(histograms, output_dir, array_name):
    """Write diagnostic histograms for one trigger reference."""
    output_dir.mkdir(parents=True, exist_ok=True)
    histograms_to_plot = {
        name: histogram
        for name, histogram in histograms.histograms.items()
        if name.startswith("angular_distance")
    }
    plot_simtel_event_histograms.plot(
        histograms_to_plot,
        output_path=output_dir,
        array_name=array_name,
    )


def build_trigger_statistics_reference(args_dict):
    """
    Build and write a trigger-statistics reference HDF5 product.

    Parameters
    ----------
    args_dict : dict
        Application arguments describing the reduced event-data inputs, telescope
        selection, histogram binning, plotting options, and output file.

    Returns
    -------
    tuple[astropy.table.Table, astropy.table.Table]
        Metadata and per-bin reference tables written to the HDF5 output file.

    Raises
    ------
    ValueError
        If the output file does not use an HDF5 suffix or no supported telescope
        selection is provided.
    """
    production_patterns = normalize_event_data_file(args_dict["event_data_file"])
    telescope_configs = normalize_telescope_configs(resolve_telescope_configs(args_dict))
    output_dir = io_handler.IOHandler().get_output_directory()
    output_file = io_handler.IOHandler().get_output_file(args_dict["output_file"])

    if output_file.suffix.lower() not in (".hdf5", ".h5"):
        raise ValueError("output_file must be an HDF5 file with suffix '.hdf5' or '.h5'.")

    reference_specs = []
    is_multi_production = len(production_patterns) > 1
    production_subdirs = {}
    if is_multi_production and args_dict.get("plot_histograms"):
        production_subdirs = build_production_subdirectories(
            production_patterns,
            output_dir,
            is_multi_production,
        )

    for production_index, pattern in enumerate(production_patterns):
        accumulators = _process_production(
            pattern,
            telescope_configs,
            energy_bins_per_decade=args_dict["energy_bins_per_decade"],
            angular_distance_bin_count=args_dict["angular_distance_bin_count"],
            skip_invalid_event_data_files=args_dict.get("skip_invalid_event_data_files", False),
        )
        production_subdir = None
        if args_dict.get("plot_histograms"):
            production_subdir = production_subdirs.get(pattern, output_dir / "trigger_reference")

        for config, histograms in zip(telescope_configs, accumulators):
            reference_id = get_uuid()
            reference_specs.append(
                {
                    "reference_id": reference_id,
                    "production_index": production_index,
                    "event_data_file": pattern,
                    "array_name": config["array_name"],
                    "telescope_ids": config["telescope_ids"],
                    "histograms": histograms,
                }
            )
            if production_subdir is not None:
                _plot_reference_histograms(
                    histograms,
                    production_subdir / config["array_name"],
                    config["array_name"],
                )

    metadata_table, bin_table = _create_reference_tables(reference_specs)
    table_handler.write_tables(
        [metadata_table, bin_table],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )
    _logger.info(f"Wrote trigger-statistics reference to {output_file}")
    return metadata_table, bin_table


def load_trigger_statistics_reference(reference_file):
    """
    Load trigger-statistics reference metadata and bin tables from HDF5.

    Parameters
    ----------
    reference_file : str or pathlib.Path
        Path to the trigger-statistics reference HDF5 file.

    Returns
    -------
    tuple[astropy.table.Table, astropy.table.Table]
        Metadata table and per-bin reference table read from the input file.
    """
    tables = table_handler.read_tables(
        reference_file,
        [TRIGGER_REFERENCE_METADATA_TABLE, TRIGGER_REFERENCE_BINS_TABLE],
        file_type="HDF5",
    )
    return tables[TRIGGER_REFERENCE_METADATA_TABLE], tables[TRIGGER_REFERENCE_BINS_TABLE]
