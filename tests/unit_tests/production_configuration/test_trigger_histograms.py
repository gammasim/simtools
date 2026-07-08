import astropy.units as u
import numpy as np
import pytest

from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_EDGES_TABLE,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_HISTOGRAM_VALUES_TABLE,
    _create_histogram_edge_table,
    _create_histogram_tables,
    _create_histogram_value_table,
    _get_plot_directory_name,
    _use_readable_inline_array_names,
    load_event_data_histograms,
    load_trigger_histograms,
)
from simtools.sim_events.histograms import EventDataHistograms


class _FakeHistograms:
    def __init__(self):
        self.energy_bins_per_decade = 4
        self.angular_distance_bin_count = 3
        self.angular_distance_bin_width = 1.0 * u.deg
        self.file_info = {
            "primary_particle": "gamma",
            "zenith": 20.0 * u.deg,
            "azimuth": 0.0 * u.deg,
            "nsb_level": 1.0,
            "energy_min": 0.1 * u.TeV,
            "energy_max": 10.0 * u.TeV,
            "viewcone_min": 0.0 * u.deg,
            "viewcone_max": 2.0 * u.deg,
            "core_scatter_min": 0.0 * u.m,
            "core_scatter_max": 120.0 * u.m,
            "scatter_area": (np.pi * (120.0**2)) * u.m**2,
            "solid_angle": 0.1 * u.sr,
        }
        self.histograms = {
            "angular_distance_vs_energy_vs_core_distance": {
                "histogram": np.array([[[1, 1], [1, 0]], [[0, 1], [0, 0]]])
            },
            "angular_distance_vs_energy_vs_core_distance_mc": {
                "histogram": np.array([[[2, 2], [1, 1]], [[1, 1], [1, 1]]])
            },
            "angular_distance_vs_energy_vs_core_distance_eff": {
                "histogram": np.array([[[0.5, 0.5], [1.0, 0.0]], [[0.0, 1.0], [0.0, 0.0]]])
            },
        }
        self.energy_bins = np.array([0.1, 1.0, 10.0])
        self.view_cone_bins = np.array([0.0, 1.0, 2.0])
        self.core_distance_bins = np.array([0.0, 60.0, 120.0])


def _full_fake_histograms():
    histograms = EventDataHistograms.create_accumulator(
        array_name="alpha",
        telescope_list=["LSTN-01"],
        energy_bins_per_decade=4,
        angular_distance_bin_width=1.0 * u.deg,
        core_distance_bin_count=3,
    )
    histograms.file_info = {
        "primary_particle": "gamma",
        "zenith": 20.0 * u.deg,
        "azimuth": 0.0 * u.deg,
        "nsb_level": 1.0,
        "energy_min": 0.1 * u.TeV,
        "energy_max": 10.0 * u.TeV,
        "viewcone_min": 0.0 * u.deg,
        "viewcone_max": 2.0 * u.deg,
        "core_scatter_min": 0.0 * u.m,
        "core_scatter_max": 120.0 * u.m,
        "scatter_area": (np.pi * (120.0**2)) * u.m**2,
        "solid_angle": 0.1 * u.sr,
    }
    histograms.data_ranges = {"angular_distance": (0.5, 1.5)}
    histograms.histograms = histograms._define_histograms(None, None, None)
    for name, histogram in histograms.histograms.items():
        if histogram["1d"]:
            shape = (len(histogram["bin_edges"]) - 1,)
        else:
            shape = tuple(len(edges) - 1 for edges in histogram["bin_edges"])
        histogram["histogram"] = np.full(shape, 2.0 if name.endswith("_mc") else 1.0)
        histogram["event_data"] = (
            None if histogram["1d"] else tuple(None for _ in histogram["event_data_column"])
        )
    histograms._filled_data_sets = 1
    histograms.calculate_efficiency_data()
    histograms.calculate_cumulative_data()
    return histograms


def test_create_histogram_tables_contains_expected_metadata_and_bins():
    metadata_table, bin_table = _create_histogram_tables(
        [
            {
                "reference_id": "ref-1",
                "production_index": 0,
                "event_data_file": "pattern*.hdf5",
                "site": "North",
                "array_name": "alpha",
                "telescope_ids": ["LSTN-01"],
                "histograms": _FakeHistograms(),
            }
        ]
    )

    assert metadata_table.meta["EXTNAME"] == TRIGGER_HISTOGRAM_METADATA_TABLE
    assert bin_table.meta["EXTNAME"] == TRIGGER_HISTOGRAM_BINS_TABLE
    assert metadata_table["reference_id"][0] == "ref-1"
    assert metadata_table["site"][0] == "North"
    assert metadata_table["angular_distance_bin_width"].quantity[0].to_value(
        u.deg
    ) == pytest.approx(1.0)
    assert metadata_table["angular_distance_bin_count"][0] == 2
    assert metadata_table["total_simulated_events"][0] == 10
    assert metadata_table["total_triggered_events"][0] == 4
    assert metadata_table["core_distance_bin_count"][0] == 2
    assert len(bin_table) == 8
    assert np.all(bin_table["reference_id"] == "ref-1")
    assert np.all(bin_table["core_distance_low"].quantity.to_value(u.m) >= 0.0)
    assert np.all(bin_table["trigger_efficiency"] >= 0.0)


def test_histogram_tables_round_trip_via_hdf5(tmp_path):
    metadata_table, bin_table = _create_histogram_tables(
        [
            {
                "reference_id": "ref-1",
                "production_index": 0,
                "event_data_file": "pattern*.hdf5",
                "site": "North",
                "array_name": "alpha",
                "telescope_ids": ["LSTN-01"],
                "histograms": _FakeHistograms(),
            }
        ]
    )

    output_file = tmp_path / "trigger_histograms.hdf5"
    table_handler.write_tables(
        [metadata_table, bin_table],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )

    loaded_metadata, loaded_bins = load_trigger_histograms(output_file)
    assert len(loaded_metadata) == 1
    assert len(loaded_bins) == 8
    assert loaded_metadata["array_name"][0] == "alpha"
    assert loaded_metadata["site"][0] == "North"
    assert loaded_bins["triggered_count"][0] == 1


def test_event_data_histograms_round_trip_via_hdf5(tmp_path):
    histograms = _full_fake_histograms()
    reference_specs = [
        {
            "reference_id": "ref-1",
            "production_index": 0,
            "event_data_file": "pattern*.hdf5",
            "site": "North",
            "array_name": "alpha",
            "telescope_ids": ["LSTN-01"],
            "histograms": histograms,
        }
    ]
    metadata_table, bin_table = _create_histogram_tables(reference_specs)
    value_table = _create_histogram_value_table(reference_specs)
    edge_table = _create_histogram_edge_table(reference_specs)
    output_file = tmp_path / "trigger_histograms.hdf5"
    table_handler.write_tables(
        [metadata_table, bin_table, value_table, edge_table],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )

    loaded = load_event_data_histograms(output_file)

    assert value_table.meta["EXTNAME"] == TRIGGER_HISTOGRAM_VALUES_TABLE
    assert edge_table.meta["EXTNAME"] == TRIGGER_HISTOGRAM_EDGES_TABLE
    assert len(loaded) == 1
    row, loaded_histograms = loaded[0]
    assert row["array_name"] == "alpha"
    assert loaded_histograms.array_name == "alpha"
    assert loaded_histograms.file_info["primary_particle"] == "gamma"
    assert loaded_histograms.data_ranges["angular_distance"] == pytest.approx((0.5, 1.5))
    np.testing.assert_allclose(
        loaded_histograms.histograms["angular_distance_vs_energy_vs_core_distance"]["histogram"],
        histograms.histograms["angular_distance_vs_energy_vs_core_distance"]["histogram"],
    )
    assert "angular_distance_vs_energy_vs_core_distance_eff" in loaded_histograms.histograms
    assert "energy_cumulative" in loaded_histograms.histograms


def test_plot_directory_name_uses_telescope_ids_for_inline_lists():
    assert _get_plot_directory_name("array_element_list", ["MSTS-01"]) == "MSTS-01"
    assert _get_plot_directory_name("alpha", ["MSTS-01"]) == "alpha"


def test_readable_inline_array_names_use_telescope_ids():
    configs = _use_readable_inline_array_names(
        [{"array_name": "array_element_list", "telescope_ids": ["MSTS-01"]}]
    )

    assert configs[0]["array_name"] == "MSTS-01"
