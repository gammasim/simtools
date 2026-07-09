import astropy.units as u
import numpy as np
from astropy.table import Table

from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    _write_dense_histogram_payload,
)
from simtools.sim_events.production_comparison import (
    ProductionDescriptor,
    collect_production_metrics,
)


def _write_minimal_dense_trigger_histogram_file(output_file):
    """Write a compact trigger-histogram file using dense histogram datasets."""
    metadata = Table(
        rows=[
            {
                "reference_id": "ref-1",
                "production_index": 0,
                "event_data_file": "events.h5",
                "site": "North",
                "array_name": "alpha",
                "telescope_ids": "LSTN-01,MSTN-01",
                "primary_particle": "gamma",
                "zenith": 20.0 * u.deg,
                "azimuth": 0.0 * u.deg,
                "nsb_level": 1.0,
                "energy_min": 0.1 * u.TeV,
                "energy_max": 10.0 * u.TeV,
                "viewcone_min": 0.0 * u.deg,
                "viewcone_max": 2.0 * u.deg,
                "core_scatter_min": 0.0 * u.m,
                "core_scatter_max": 100.0 * u.m,
                "scatter_area": 1.0 * u.m**2,
                "solid_angle": 1.0 * u.sr,
                "angular_distance_min": 0.1 * u.deg,
                "angular_distance_max": 1.0 * u.deg,
                "energy_bins_per_decade": 1,
                "angular_distance_bin_width": 1.0 * u.deg,
                "angular_distance_bin_count": 2,
                "core_distance_bin_count": 2,
                "total_simulated_events": 4,
                "total_triggered_events": 2,
            }
        ]
    )
    metadata.meta["EXTNAME"] = TRIGGER_HISTOGRAM_METADATA_TABLE

    topology = Table(
        rows=[
            ("ref-1", "trigger_multiplicity", "", "1", 1),
            ("ref-1", "trigger_multiplicity", "", "2", 1),
            ("ref-1", "trigger_combinations", "", "MSTN-01", 1),
            ("ref-1", "trigger_combinations", "", "LSTN-01,MSTN-01", 1),
            ("ref-1", "telescope_participation", "", "MSTN-01", 2),
            ("ref-1", "telescope_participation", "", "LSTN-01", 1),
            ("ref-1", "subset_multiplicity", "mixed_type", "2", 1),
        ],
        names=["reference_id", "count_type", "subset", "key", "count"],
    )
    topology.meta["EXTNAME"] = TRIGGER_TOPOLOGY_COUNTS_TABLE

    subset_histograms = Table(
        rows=[
            ("ref-1", "mixed_type", "energy", 0, 0.1, 1.0, 1),
            ("ref-1", "mixed_type", "energy", 1, 1.0, 10.0, 0),
        ],
        names=["reference_id", "subset", "quantity", "bin_index", "bin_low", "bin_high", "count"],
    )
    subset_histograms.meta["EXTNAME"] = TRIGGER_SUBSET_HISTOGRAMS_TABLE
    table_handler.write_tables(
        [metadata, topology, subset_histograms], output_file, file_type="HDF5"
    )

    class _DenseHistograms:
        def __init__(self):
            self.histograms = {
                "energy_mc": {
                    "histogram": np.array([3.0, 1.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
                "energy": {
                    "histogram": np.array([1.0, 1.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
                "core_distance_mc": {
                    "histogram": np.array([2.0, 2.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
                "core_distance": {
                    "histogram": np.array([0.0, 2.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
                "angular_distance_mc": {
                    "histogram": np.array([2.0, 2.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
                "angular_distance": {
                    "histogram": np.array([1.0, 1.0]),
                    "bin_edges": np.array([0.1, 1.0, 10.0]),
                    "1d": True,
                },
            }

    _write_dense_histogram_payload(
        [{"reference_id": "ref-1", "histograms": _DenseHistograms()}],
        output_file,
    )


def test_collect_production_metrics_from_dense_trigger_histogram_file(tmp_path):
    """Collect comparison metrics from trigger histogram files with dense payloads."""
    trigger_histogram_file = tmp_path / "trigger_histograms_dense.hdf5"
    _write_minimal_dense_trigger_histogram_file(trigger_histogram_file)

    metrics = collect_production_metrics(
        [ProductionDescriptor("baseline", [str(trigger_histogram_file)])]
    )

    assert metrics[0].simulated_event_count == 4
    assert metrics[0].triggered_event_count == 2
    np.testing.assert_allclose(metrics[0].quantity_histograms["energy"]["simulated"][0], [3, 1])
