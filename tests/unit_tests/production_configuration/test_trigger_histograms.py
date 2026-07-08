import astropy.units as u
import numpy as np
import pytest

from simtools.io import table_handler
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    _create_histogram_tables,
    _get_plot_directory_name,
    _use_readable_inline_array_names,
    load_trigger_histograms,
)


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
            "angular_distance_vs_energy": {"histogram": np.array([[2, 1], [1, 0]])},
            "angular_distance_vs_energy_mc": {"histogram": np.array([[4, 2], [2, 2]])},
            "angular_distance_vs_energy_eff": {"histogram": np.array([[0.5, 0.5], [0.5, 0.0]])},
        }
        self.energy_bins = np.array([0.1, 1.0, 10.0])
        self.view_cone_bins = np.array([0.0, 1.0, 2.0])


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
    assert len(bin_table) == 4
    assert np.all(bin_table["reference_id"] == "ref-1")
    assert np.all(bin_table["effective_area"].quantity.to_value(u.m**2) >= 0.0)


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
    assert len(loaded_bins) == 4
    assert loaded_metadata["array_name"][0] == "alpha"
    assert loaded_metadata["site"][0] == "North"
    assert loaded_bins["triggered_count"][0] == 2


def test_plot_directory_name_uses_telescope_ids_for_inline_lists():
    assert _get_plot_directory_name("array_element_list", ["MSTS-01"]) == "MSTS-01"
    assert _get_plot_directory_name("alpha", ["MSTS-01"]) == "alpha"


def test_readable_inline_array_names_use_telescope_ids():
    configs = _use_readable_inline_array_names(
        [{"array_name": "array_element_list", "telescope_ids": ["MSTS-01"]}]
    )

    assert configs[0]["array_name"] == "MSTS-01"
