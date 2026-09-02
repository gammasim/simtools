from pathlib import Path
from types import SimpleNamespace

import astropy.units as u
import h5py
import numpy as np
import pytest
from astropy.table import Table

from simtools.io import table_handler
from simtools.io.ascii_handler import collect_data_from_file
from simtools.production_configuration.trigger_histograms import (
    TRIGGER_HISTOGRAM_BINS_TABLE,
    TRIGGER_HISTOGRAM_DENSE_GROUP,
    TRIGGER_HISTOGRAM_METADATA_TABLE,
    TRIGGER_SUBSET_HISTOGRAMS_TABLE,
    TRIGGER_TOPOLOGY_COUNTS_TABLE,
    _create_histogram_tables,
    _create_trigger_subset_histogram_table,
    _create_trigger_topology_count_table,
    _execute_production_job,
    _format_trigger_histogram_inspection,
    _get_plot_directory_name,
    _group_output_stem,
    _relative_to_directory,
    _resolve_group_telescope_configs,
    _use_readable_inline_array_names,
    _write_dense_histogram_payload,
    _write_directory_group_job,
    _write_directory_products,
    _write_trigger_histogram_metadata,
    discover_event_data_groups,
    inspect_trigger_histogram_file,
    load_event_data_histograms,
    write_trigger_histograms,
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
            "spectral_index": -2.0,
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
        "spectral_index": -2.0,
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
    assert metadata_table["spectral_index"][0] == pytest.approx(-2.0)
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


def test_event_data_histograms_round_trip_via_hdf5(tmp_path):
    histograms = _full_fake_histograms()
    reference_specs = [
        {
            "reference_id": "ref-1",
            "production_index": 0,
            "site": "North",
            "array_name": "alpha",
            "telescope_ids": ["LSTN-01"],
            "histograms": histograms,
        }
    ]
    metadata_table, bin_table = _create_histogram_tables(reference_specs)
    output_file = tmp_path / "trigger_histograms.hdf5"
    table_handler.write_tables(
        [metadata_table, bin_table],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )
    _write_dense_histogram_payload(reference_specs, output_file)

    loaded = load_event_data_histograms(output_file)

    assert len(loaded) == 1
    row, loaded_histograms = loaded[0]
    assert row["array_name"] == "alpha"
    assert loaded_histograms.array_name == "alpha"
    assert loaded_histograms.file_info["primary_particle"] == "gamma"
    assert loaded_histograms.file_info["spectral_index"] == pytest.approx(-2.0)
    assert loaded_histograms.data_ranges["angular_distance"] == pytest.approx((0.5, 1.5))
    assert all(isinstance(histogram, dict) for histogram in loaded_histograms.histograms.values())
    np.testing.assert_allclose(loaded_histograms.energy_bins, histograms.energy_bins)
    np.testing.assert_allclose(
        loaded_histograms.histograms["angular_distance_vs_energy_vs_core_distance"]["histogram"],
        histograms.histograms["angular_distance_vs_energy_vs_core_distance"]["histogram"],
    )
    assert "angular_distance_vs_energy_vs_core_distance_eff" in loaded_histograms.histograms
    assert "energy_cumulative" in loaded_histograms.histograms
    assert "reuse_mean_vs_energy" in loaded_histograms.histograms
    np.testing.assert_allclose(
        loaded_histograms.histograms["reuse_mean_vs_energy"]["histogram"],
        histograms.histograms["reuse_mean_vs_energy"]["histogram"],
    )

    with h5py.File(output_file, "r") as hdf5_file:
        assert TRIGGER_HISTOGRAM_DENSE_GROUP in hdf5_file


def test_trigger_topology_tables_are_created_from_reference_specs():
    histograms = _full_fake_histograms()
    reference_specs = [
        {
            "reference_id": "ref-1",
            "production_index": 0,
            "site": "North",
            "array_name": "alpha",
            "telescope_ids": ["LSTN-01"],
            "histograms": histograms,
            "trigger_topology": {
                "trigger_multiplicity": {2: 3},
                "trigger_combinations": {"LSTN-01,MSTN-01": 2},
                "telescope_participation": {"LSTN-01": 2, "MSTN-01": 2},
                "subset_multiplicity": {"mixed_type": {2: 2}},
                "subset_values": {
                    "energy": {"mixed_type": [0.2, 2.0]},
                    "core_distance": {"mixed_type": [10.0, 80.0]},
                    "angular_distance": {"mixed_type": [0.2, 1.2]},
                },
            },
        }
    ]

    topology_table = _create_trigger_topology_count_table(reference_specs)
    subset_histogram_table = _create_trigger_subset_histogram_table(reference_specs)

    assert topology_table.meta["EXTNAME"] == TRIGGER_TOPOLOGY_COUNTS_TABLE
    assert subset_histogram_table.meta["EXTNAME"] == TRIGGER_SUBSET_HISTOGRAMS_TABLE
    assert "trigger_combinations" in set(topology_table["count_type"])
    assert "mixed_type" in set(subset_histogram_table["subset"])
    assert np.sum(subset_histogram_table["count"]) == 6


def test_event_data_histograms_hdf5_filter_by_array_name(tmp_path):
    histograms = _full_fake_histograms()
    reference_specs = [
        {
            "reference_id": "ref-1",
            "production_index": 0,
            "site": "North",
            "array_name": "MSTS-01",
            "telescope_ids": ["MSTS-01"],
            "histograms": histograms,
        }
    ]
    metadata_table, bin_table = _create_histogram_tables(reference_specs)
    output_file = tmp_path / "trigger_histograms.hdf5"
    table_handler.write_tables(
        [metadata_table, bin_table],
        output_file,
        overwrite_existing=True,
        file_type="HDF5",
    )
    _write_dense_histogram_payload(reference_specs, output_file)

    loaded = load_event_data_histograms(output_file, array_names=["MSTS-01"])

    assert len(loaded) == 1
    row, loaded_histograms = loaded[0]
    assert row["array_name"] == "MSTS-01"
    assert loaded_histograms.array_name == "MSTS-01"


def test_plot_directory_name_uses_telescope_ids_for_inline_lists():
    assert _get_plot_directory_name("array_element_list", ["MSTS-01"]) == "MSTS-01"
    assert _get_plot_directory_name("alpha", ["MSTS-01"]) == "alpha"


def test_readable_inline_array_names_use_telescope_ids():
    configs = _use_readable_inline_array_names(
        [{"array_name": "array_element_list", "telescope_ids": ["MSTS-01"]}]
    )

    assert configs[0]["array_name"] == "MSTS-01"


def test_execute_production_job_returns_one_result_per_telescope_config(mocker):
    histograms = mocker.Mock()
    topology = {"trigger_multiplicity": {1: 2}}
    mocker.patch(
        "simtools.production_configuration.trigger_histograms._process_production",
        return_value=[(histograms, topology)],
    )

    result = _execute_production_job(
        {
            "production_index": 3,
            "production_pattern": "prod_a/*.hdf5",
            "site": "North",
            "telescope_configs": [{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}],
            "energy_bins_per_decade": 4,
            "angular_distance_bin_width": 1.0 * u.deg,
            "core_distance_bin_width": 20.0 * u.m,
            "skip_invalid_event_data_files": False,
        }
    )

    assert result == [
        {
            "production_index": 3,
            "site": "North",
            "array_name": "alpha",
            "telescope_ids": ["LSTN-01"],
            "histograms": histograms,
            "trigger_topology": topology,
        }
    ]


def test_discover_event_data_groups_collects_parts(tmp_test_directory):
    input_directory = Path(tmp_test_directory) / "reduced_event_data"
    input_directory.mkdir()
    for file_name in (
        "gamma.part0002.reduced_event_data.hdf5",
        "gamma.part0001.reduced_event_data.hdf5",
        "proton.reduced_event_data.hdf5",
        "ignored.hdf5",
    ):
        (input_directory / file_name).touch()
    (input_directory / "not-a-file.reduced_event_data.hdf5").mkdir()

    groups = discover_event_data_groups(input_directory)

    assert [(name, [path.name for path in files]) for name, files in groups] == [
        (
            "gamma",
            [
                "gamma.part0001.reduced_event_data.hdf5",
                "gamma.part0002.reduced_event_data.hdf5",
            ],
        ),
        ("proton", ["proton.reduced_event_data.hdf5"]),
    ]


def test_discover_event_data_groups_rejects_empty_directory(tmp_test_directory):
    input_directory = Path(tmp_test_directory) / "reduced_event_data"
    input_directory.mkdir()

    with pytest.raises(ValueError, match="No reduced event-data files"):
        discover_event_data_groups(input_directory)


def test_write_directory_products_submits_one_job_per_group(mocker, tmp_test_directory):
    tmp_test_directory = Path(tmp_test_directory)
    input_directory = tmp_test_directory / "reduced_event_data"
    input_directory.mkdir()
    for file_name in (
        "gamma.part0001.reduced_event_data.hdf5",
        "gamma.part0002.reduced_event_data.hdf5",
        "proton.reduced_event_data.hdf5",
    ):
        (input_directory / file_name).touch()
    mocker.patch(
        "simtools.production_configuration.trigger_histograms.io_handler.IOHandler"
    ).return_value.get_output_directory.return_value = tmp_test_directory / "output"
    submit_jobs = mocker.patch("simtools.production_configuration.trigger_histograms.submit_jobs")

    _write_directory_products(
        {
            "event_data_directory": input_directory,
            "output_path": tmp_test_directory / "output",
            "backend": "htcondor",
            "backend_config": {},
            "max_workers": 4,
        }
    )

    jobs = submit_jobs.call_args.args[0]
    assert [job.job_id for job in jobs] == [
        "trigger-histograms-000000",
        "trigger-histograms-000001",
    ]
    assert [job.output_paths[0].name for job in jobs] == [
        "gamma.trigger_histograms.hdf5",
        "proton.trigger_histograms.hdf5",
    ]
    assert jobs[0].item["event_data_files"] == [
        str(input_directory / "gamma.part0001.reduced_event_data.hdf5"),
        str(input_directory / "gamma.part0002.reduced_event_data.hdf5"),
    ]
    assert jobs[0].mount_paths == (input_directory.resolve(),)
    assert jobs[1].mount_paths == (input_directory.resolve(),)
    assert jobs[0].output_paths != jobs[1].output_paths


def test_write_directory_group_job_uses_local_inner_execution(mocker):
    write_product = mocker.patch(
        "simtools.production_configuration.trigger_histograms._write_trigger_histogram_product"
    )

    _write_directory_group_job(
        {
            "args_dict": {"backend": "htcondor", "max_workers": 8, "backend_config": {}},
            "event_data_files": ["gamma.part0001.reduced_event_data.hdf5"],
            "output_file": "gamma.trigger_histograms.hdf5",
        }
    )

    inner_args = write_product.call_args.args[0]
    assert inner_args["backend"] == "local"
    assert inner_args["backend_config"] is None
    assert inner_args["max_workers"] == 1


def test_trigger_histogram_metadata_uses_portable_paths_and_resolved_array_selection(
    tmp_test_directory,
):
    base_directory = Path(tmp_test_directory)
    input_file = base_directory / "grid-output" / "job-000001" / "gamma.hdf5"
    output_file = base_directory / "trigger_histograms" / "gamma.trigger_histograms.hdf5"
    input_file.parent.mkdir(parents=True)
    output_file.parent.mkdir()
    input_file.touch()
    output_file.touch()

    _write_trigger_histogram_metadata(
        {
            "energy_bins_per_decade": 10,
            "angular_distance_bin_width": 0.5 * u.deg,
            "core_distance_bin_width": 20 * u.m,
            "minimum_triggered_telescopes": 2,
        },
        output_file,
        {
            "configuration": {"primary": "gamma"},
            "run_numbers": [1],
            "input_files": [input_file],
            "array_selection": [{"array_name": "CTAO-North-Alpha", "telescope_ids": ["LSTN-01"]}],
        },
    )

    metadata = collect_data_from_file(output_file.with_suffix(".yml"))
    assert metadata["input_files"]["reduced_event_data"] == ["../grid-output/job-000001/gamma.hdf5"]
    assert metadata["array_selection"][0]["array_name"] == "CTAO-North-Alpha"


def test_group_output_stem_changes_with_histogram_settings_and_array_selection():
    group = SimpleNamespace(
        configuration={
            "primary": "gamma",
            "zenith_angle": {"value": 20.0, "unit": "deg"},
            "corsika_he_interaction": "qgs3",
        }
    )
    first = _group_output_stem(
        group,
        {
            "histogram_settings": {"minimum_triggered_telescopes": 2},
            "array_selection": [{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}],
        },
    )
    second = _group_output_stem(
        group,
        {
            "histogram_settings": {"minimum_triggered_telescopes": 3},
            "array_selection": [{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}],
        },
    )

    assert first != second


def test_relative_to_directory_handles_sibling_directories(tmp_test_directory):
    base_directory = Path(tmp_test_directory)
    path = base_directory / "grid-output" / "job-000001" / "file.hdf5"
    directory = base_directory / "trigger_histograms"

    assert _relative_to_directory(path, directory) == "../grid-output/job-000001/file.hdf5"


def test_production_group_telescope_configs_come_from_metadata(mocker):
    resolve = mocker.patch(
        "simtools.production_configuration.trigger_histograms._resolve_telescope_configs",
        return_value=[{"array_name": "CTAO-North-Alpha", "telescope_ids": ["LSTN-01"]}],
    )

    _resolve_group_telescope_configs(
        {},
        {
            "site": "North",
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
        },
    )

    resolved_args = resolve.call_args.args[0]
    assert resolved_args["site"] == "North"
    assert resolved_args["model_version"] == ["7.0.0"]
    assert resolved_args["array_layout_name"] == ["CTAO-North-Alpha"]


def test_production_group_telescope_configs_reject_mismatched_layout():
    with pytest.raises(ValueError, match="does not match selected production metadata"):
        _resolve_group_telescope_configs(
            {"array_layout_name": ["CTAO-North-Beta"]},
            {
                "site": "North",
                "model_version": "7.0.0",
                "array_layout_name": "CTAO-North-Alpha",
            },
        )


def test_production_metadata_histograms_reject_non_event_data_file_types():
    with pytest.raises(ValueError, match="requires file_type='reduced_event_data'"):
        write_trigger_histograms({"production_path": "production", "file_type": "sim_telarray"})


def test_write_trigger_histograms_dispatches_one_job_per_pattern(mocker, tmp_path):
    mocker.patch(
        "simtools.production_configuration.trigger_histograms.resolve_telescope_configs",
        return_value={"alpha": ["LSTN-01"]},
    )
    mocker.patch(
        "simtools.production_configuration.trigger_histograms.normalize_telescope_configs",
        return_value=[{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}],
    )
    mocker.patch(
        "simtools.production_configuration.trigger_histograms.io_handler.IOHandler"
    ).return_value.get_output_file.return_value = tmp_path / "trigger_histograms.hdf5"
    mock_process_pool = mocker.patch(
        "simtools.production_configuration.trigger_histograms.map_ordered",
        return_value=[
            [
                {
                    "production_index": 0,
                    "site": "North",
                    "array_name": "alpha",
                    "telescope_ids": ["LSTN-01"],
                    "histograms": _full_fake_histograms(),
                    "trigger_topology": {},
                }
            ],
            [
                {
                    "production_index": 1,
                    "site": "North",
                    "array_name": "alpha",
                    "telescope_ids": ["LSTN-01"],
                    "histograms": _full_fake_histograms(),
                    "trigger_topology": {},
                }
            ],
        ],
    )
    metadata_document = {"cta": {"product": {"id": "metadata-id"}}}
    mock_metadata = mocker.patch(
        "simtools.production_configuration.trigger_histograms.build_standard_metadata",
        return_value=metadata_document,
    )

    metadata_table, _ = write_trigger_histograms(
        {
            "event_data_files": ["prod_a/*.hdf5", "prod_b/*.hdf5"],
            "array_element_list": ["LSTN-01"],
            "energy_bins_per_decade": 4,
            "angular_distance_bin_width": 1.0 * u.deg,
            "core_distance_bin_width": 20.0 * u.m,
            "skip_invalid_event_data_files": False,
            "max_workers": 24,
            "site": "North",
            "output_file": "trigger_histograms.hdf5",
        }
    )

    mock_process_pool.assert_called_once()
    assert mock_process_pool.call_args.kwargs["max_workers"] == 24
    job_specs = mock_process_pool.call_args.args[1]
    assert [job_spec["production_index"] for job_spec in job_specs] == [0, 1]
    assert [job_spec["production_pattern"] for job_spec in job_specs] == [
        "prod_a/*.hdf5",
        "prod_b/*.hdf5",
    ]
    assert all(
        job_spec["core_distance_bin_width"].to_value(u.m) == pytest.approx(20.0)
        for job_spec in job_specs
    )
    assert list(metadata_table["reference_id"]) == ["reference_0", "reference_1"]
    assert list(metadata_table["production_index"]) == [0, 1]
    assert "event_data_file" not in metadata_table.colnames
    mock_metadata.assert_called_once_with(
        {
            "event_data_files": ["prod_a/*.hdf5", "prod_b/*.hdf5"],
            "array_element_list": ["LSTN-01"],
            "energy_bins_per_decade": 4,
            "angular_distance_bin_width": 1.0 * u.deg,
            "core_distance_bin_width": 20.0 * u.m,
            "skip_invalid_event_data_files": False,
            "max_workers": 24,
            "site": "North",
            "output_file": "trigger_histograms.hdf5",
        },
        tmp_path / "trigger_histograms.hdf5",
        product_data_name="trigger_histograms",
    )
    assert (
        table_handler.read_metadata_document(tmp_path / "trigger_histograms.hdf5", "METADATA")
        == metadata_document
    )


def test_inspect_trigger_histogram_file_reports_reference_mismatches(tmp_path):
    file_path = tmp_path / "trigger_histograms.hdf5"
    metadata = Table(rows=[{"reference_id": "ref-1"}, {"reference_id": "ref-2"}])
    metadata.meta["EXTNAME"] = TRIGGER_HISTOGRAM_METADATA_TABLE
    table_handler.write_tables([metadata], file_path, file_type="HDF5")
    with h5py.File(file_path, "a") as hdf5_file:
        dense_group = hdf5_file.create_group(TRIGGER_HISTOGRAM_DENSE_GROUP)
        dense_group.create_group("ref-1")
        dense_group.create_group("ref-3")

    report = inspect_trigger_histogram_file(file_path, format_report=False)
    formatted = _format_trigger_histogram_inspection(report)

    assert report["missing_dense_reference_ids"] == ["ref-2"]
    assert report["orphan_dense_reference_ids"] == ["ref-3"]
    assert "missing dense payloads for metadata ids: ref-2" in formatted
    assert "orphan dense payload ids without metadata rows: ref-3" in formatted
