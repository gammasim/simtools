"""Tests for production comparison workflows."""

import json
import logging
from pathlib import Path

import pytest

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.production_configuration import production_comparison
from simtools.production_configuration.production_file_selection import (
    ProductionManifest,
)
from simtools.sim_events.production_comparison import ProductionDescriptor

BASELINE_KEY = "baseline"
COMPARISON_SETS_KEY = "comparison_sets"
PLOT_STATISTICS_KEY = "plot_statistics"
COMPARISONS_KEY = "comparisons"
TEXT_ENCODING = "utf-8"
SIMULATED_EVENT_COUNT_KEY = "simulated_event_count"

ARRAY_LAYOUT_KEY = "array_layout_name"
CONFIGURATION_KEY = "configuration"

CONFIGURATIONS_KEY = "configurations"
LAYOUT_NAME = "layout-a"


def test_compare_array_layout_passes_metadata_to_plotting(mocker, tmp_test_directory):
    mocker.patch.object(production_comparison, "collect_production_metrics", return_value=[])
    mocker.patch.object(production_comparison, "_comparison_title", return_value=None)
    plot = mocker.patch.object(production_comparison.plot_event_level_production_comparison, "plot")
    context = {CONFIGURATIONS_KEY: {}}
    production_comparison._compare_array_layout(
        [], {}, None, Path(tmp_test_directory), comparison_context=context
    )
    assert plot.call_args.kwargs["comparison_context"] is context


def test_write_production_comparison_writes_each_selected_array_layout(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
    first_statistics_file = output_directory / "first" / "comparison_statistics.json"
    second_statistics_file = output_directory / "second" / "comparison_statistics.json"
    descriptors = mocker.sentinel.production_descriptors
    mocker.patch(
        "simtools.production_configuration.production_comparison.parse_production_arguments",
        return_value=descriptors,
    )
    mock_collect = mocker.patch(
        "simtools.production_configuration.production_comparison.collect_production_metrics",
        side_effect=[mocker.sentinel.first_metrics, mocker.sentinel.second_metrics],
    )
    mock_plot = mocker.patch(
        "simtools.production_configuration.production_comparison."
        "plot_event_level_production_comparison.plot",
        side_effect=[first_statistics_file, second_statistics_file],
    )
    mock_summary = mocker.patch(
        "simtools.production_configuration.production_comparison._write_comparison_summary"
    )

    production_comparison.write_production_comparison(
        {
            "production": [BASELINE_KEY, "baseline.hdf5", "candidate", "candidate.hdf5"],
            ARRAY_LAYOUT_KEY: ["first", "second"],
            "figure_format": ["png"],
        },
        output_directory,
    )

    assert mock_collect.call_args_list == [
        mocker.call(descriptors, array_names="first"),
        mocker.call(descriptors, array_names="second"),
    ]
    assert mock_plot.call_args_list == [
        mocker.call(
            mocker.sentinel.first_metrics,
            output_path=output_directory,
            array_layout_name="first",
            figure_format=["png"],
            report_mode="diagnostic",
            comparison_title="Layout: first",
        ),
        mocker.call(
            mocker.sentinel.second_metrics,
            output_path=output_directory,
            array_layout_name="second",
            figure_format=["png"],
            report_mode="diagnostic",
            comparison_title="Layout: second",
        ),
    ]
    mock_summary.assert_called_once_with(
        [first_statistics_file, second_statistics_file], output_directory, "diagnostic"
    )


def test_production_descriptor_pairs_from_metadata_matches_configurations(
    mocker, tmp_test_directory
):
    base_directory = Path(tmp_test_directory)

    def manifest(directory, interaction, zenith):
        return ProductionManifest(
            path=base_directory / directory / f"{zenith}.yml",
            data={
                CONFIGURATION_KEY: {
                    "primary": "gamma",
                    "zenith_angle": {"value": zenith, "unit": "deg"},
                    "corsika_he_interaction": interaction,
                },
                "histogram_settings": {"minimum_triggered_telescopes": 2},
                "array_selection": [{"array_name": "alpha", "telescope_ids": ["LSTN-01"]}],
                "files": {
                    "trigger_histograms": [f"{interaction}_{zenith}.trigger_histograms.hdf5"]
                },
            },
        )

    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_selected_trigger_histogram_manifests",
        side_effect=[
            [manifest(BASELINE_KEY, "qgs3", 20), manifest(BASELINE_KEY, "qgs3", 40)],
            [manifest("candidate", "epos", 20), manifest("candidate", "epos", 40)],
        ],
    )

    pairs = production_comparison._production_descriptor_pairs_from_metadata(
        {
            "baseline_path": base_directory / BASELINE_KEY,
            "candidate_path": base_directory / "candidate",
            "select": [],
            "compare_by": ["corsika_he_interaction"],
        }
    )

    assert len(pairs) == 2
    assert all(len(descriptors) == 2 for _, descriptors in pairs)
    assert all(
        len(descriptor.input_files) == 1 for _, descriptors in pairs for descriptor in descriptors
    )


def test_write_production_comparison_uses_metadata_pair_output_directory(
    mocker, tmp_test_directory
):
    output_directory = Path(tmp_test_directory) / "comparison"
    pairing_key = {CONFIGURATION_KEY: {"primary": "gamma"}}
    input_stem = "gamma-za67.5-qgs3-energy-min-30-gev-ctao-north-alpha.trigger_histograms"
    descriptors = [
        ProductionDescriptor(BASELINE_KEY, [f"{input_stem}.hdf5"]),
        ProductionDescriptor("candidate", [f"{input_stem}.hdf5"]),
    ]
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_production_descriptor_pairs_from_metadata",
        return_value=[(pairing_key, descriptors)],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_comparison._write_array_layout_comparisons"
    )

    production_comparison.write_production_comparison(
        {"baseline_path": BASELINE_KEY, ARRAY_LAYOUT_KEY: ["alpha"]},
        output_directory,
    )

    assert mock_write.call_args.args[:2] == (
        descriptors,
        {"baseline_path": BASELINE_KEY, ARRAY_LAYOUT_KEY: ["alpha"]},
    )
    assert mock_write.call_args.args[2] == output_directory / input_stem
    assert mock_write.call_args.args[3] == ["alpha"]
    inventory = json.loads((output_directory / "comparison_inventory.json").read_text())
    assert inventory["matched_pair_count"] == 1
    assert inventory["pairing_error_count"] == 0
    assert "pairs" not in inventory


def test_compact_metadata_comparison_writes_one_site_report(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
    descriptors = [
        ProductionDescriptor(BASELINE_KEY, ["baseline.hdf5"]),
        ProductionDescriptor("candidate", ["candidate.hdf5"]),
    ]
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_production_descriptor_pairs_from_metadata",
        return_value=[({CONFIGURATION_KEY: {"zenith_angle": 20}}, descriptors)],
    )
    figure = mocker.sentinel.figure

    def write_pair(*_args, compact_figures=None):
        compact_figures.append(figure)

    mocker.patch.object(
        production_comparison, "_write_array_layout_comparisons", side_effect=write_pair
    )
    save = mocker.patch.object(production_comparison, "save_figures_to_single_document")

    production_comparison.write_production_comparison(
        {"baseline_path": BASELINE_KEY, "report_mode": "compact"}, output_directory
    )

    save.assert_called_once_with([figure], output_directory / "trigger-report.pdf", close=True)


def test_write_production_comparison_uses_hash_only_for_duplicate_input_stems(
    mocker, tmp_test_directory
):
    output_directory = Path(tmp_test_directory) / "comparison"
    input_stem = "gamma-za67.5-qgs3-energy-min-30-gev-ctao-north-alpha.trigger_histograms"
    descriptors = [
        ProductionDescriptor(BASELINE_KEY, [f"{input_stem}.hdf5"]),
        ProductionDescriptor("candidate", [f"{input_stem}.hdf5"]),
    ]
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_production_descriptor_pairs_from_metadata",
        return_value=[
            ({CONFIGURATION_KEY: {"zenith_angle": 20}}, descriptors),
            ({CONFIGURATION_KEY: {"zenith_angle": 40}}, descriptors),
        ],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_comparison._write_array_layout_comparisons"
    )

    production_comparison.write_production_comparison(
        {"baseline_path": BASELINE_KEY, ARRAY_LAYOUT_KEY: ["alpha"]},
        output_directory,
    )

    output_paths = [call.args[2] for call in mock_write.call_args_list]
    assert len(output_paths) == 2
    assert output_paths[0].name != input_stem
    assert output_paths[1].name != input_stem
    assert output_paths[0].name != output_paths[1].name


def test_production_descriptor_pairs_rejects_unmatched_metadata(mocker, tmp_test_directory):
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "baseline.yml",
        data={CONFIGURATION_KEY: {}, "files": {"trigger_histograms": ["baseline.hdf5"]}},
    )
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_selected_trigger_histogram_manifests",
        side_effect=[[manifest], []],
    )

    with pytest.raises(ValueError, match="pairing failed"):
        production_comparison._production_descriptor_pairs_from_metadata(
            {"baseline_path": BASELINE_KEY, "candidate_path": "candidate"}
        )


def test_write_production_comparison_writes_matched_pairs_before_reporting_unmatched_metadata(
    mocker, caplog, tmp_test_directory
):
    base_directory = Path(tmp_test_directory)

    def manifest(directory, zenith):
        return ProductionManifest(
            path=base_directory / directory / f"{zenith}.yml",
            data={
                CONFIGURATION_KEY: {"zenith_angle": {"value": zenith, "unit": "deg"}},
                "files": {"trigger_histograms": [f"{directory}_{zenith}.hdf5"]},
            },
        )

    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_selected_trigger_histogram_manifests",
        side_effect=[
            [manifest(BASELINE_KEY, 20), manifest(BASELINE_KEY, 40)],
            [manifest("candidate", 20)],
        ],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_comparison._write_array_layout_comparisons"
    )

    with caplog.at_level(logging.WARNING, logger=production_comparison.__name__):
        production_comparison.write_production_comparison(
            {"baseline_path": BASELINE_KEY, "candidate_path": "candidate"},
            base_directory / "comparison",
        )

    mock_write.assert_called_once()
    assert "missing candidates=1, missing baselines=0" in caplog.text
    inventory = json.loads((base_directory / "comparison/comparison_inventory.json").read_text())
    assert inventory["matched_pair_count"] == 1
    assert inventory["pairing_error_count"] == 1
    assert "pairs" not in inventory
    descriptors = mock_write.call_args.args[0]
    assert [descriptor.label for descriptor in descriptors] == [BASELINE_KEY, "candidate"]
    assert descriptors[0].input_files == [str(base_directory / BASELINE_KEY / "baseline_20.hdf5")]
    assert descriptors[1].input_files == [str(base_directory / "candidate" / "candidate_20.hdf5")]


def test_write_production_comparison_rejects_completely_unmatched_metadata(
    mocker, tmp_test_directory
):
    pairing_error = production_comparison._ProductionPairingError(
        "Trigger-histogram metadata pairing failed: missing candidates=1, missing baselines=1.",
        [],
    )
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_production_descriptor_pairs_from_metadata",
        side_effect=pairing_error,
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_comparison._write_array_layout_comparisons"
    )

    with pytest.raises(ValueError, match="pairing failed"):
        production_comparison.write_production_comparison(
            {"baseline_path": BASELINE_KEY, "candidate_path": "candidate"},
            Path(tmp_test_directory) / "comparison",
        )

    mock_write.assert_not_called()


def test_selected_trigger_histogram_manifests_checks_matches(mocker):
    manifest = mocker.sentinel.manifest
    mocker.patch(
        "simtools.production_configuration.production_comparison.discover_product_manifests",
        return_value=[manifest],
    )
    mocker.patch(
        "simtools.production_configuration.production_comparison.filter_manifests",
        return_value=[manifest],
    )
    mock_check = mocker.patch(
        "simtools.production_configuration.production_comparison.check_manifest"
    )

    selected = production_comparison._selected_trigger_histogram_manifests("production", [])

    assert selected == [manifest]
    mock_check.assert_called_once_with(manifest)


def test_selected_trigger_histogram_manifests_rejects_empty_selection(mocker):
    mocker.patch(
        "simtools.production_configuration.production_comparison.discover_product_manifests",
        return_value=[],
    )
    mocker.patch(
        "simtools.production_configuration.production_comparison.filter_manifests",
        return_value=[],
    )

    with pytest.raises(ValueError, match="No trigger-histogram metadata"):
        production_comparison._selected_trigger_histogram_manifests("production", [])


def test_unique_manifests_by_pairing_key_rejects_duplicates(tmp_test_directory):
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "first.yml",
        data={CONFIGURATION_KEY: {}},
    )

    with pytest.raises(ValueError, match="More than one baseline"):
        production_comparison._unique_manifests_by_pairing_key(
            [manifest, manifest], set(), BASELINE_KEY
        )


@pytest.mark.parametrize(
    ("layout", "matches"),
    [
        ([LAYOUT_NAME], True),
        ((LAYOUT_NAME,), True),
        ([], False),
        ([LAYOUT_NAME, "other-layout"], False),
        (["different-layout"], False),
    ],
)
def test_pairing_normalizes_only_single_equivalent_layouts(layout, matches):
    field = ARRAY_LAYOUT_KEY
    baseline = {CONFIGURATION_KEY: {field: LAYOUT_NAME}}
    candidate = {CONFIGURATION_KEY: {field: layout}}
    actual = production_comparison._pairing_key(baseline, set()) == (
        production_comparison._pairing_key(candidate, set())
    )
    assert actual is matches


def test_single_trigger_histogram_file_handles_absolute_and_invalid_manifests(tmp_test_directory):
    absolute_file = Path(tmp_test_directory) / "trigger_histograms.hdf5"
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "manifest.yml",
        data={"files": {"trigger_histograms": [str(absolute_file)]}},
    )
    assert production_comparison._single_trigger_histogram_file(manifest) == absolute_file

    empty_manifest = ProductionManifest(path=manifest.path, data={"files": {}})
    with pytest.raises(ValueError, match="Expected exactly one"):
        production_comparison._single_trigger_histogram_file(empty_manifest)


def test_comparison_title_formats_shared_and_different_configuration_values():
    configuration = {
        BASELINE_KEY: {
            "primary": "gamma",
            "site": "North",
            "energy_min": {"value": 30, "unit": "GeV"},
            "energy_max": {"value": 100, "unit": "GeV"},
            "zenith_angle": {"value": 20, "unit": "deg"},
            "azimuth_angle": {"value": 0, "unit": "deg"},
            "corsika_he_interaction": "qgs3",
        },
        "candidate": {
            "primary": "gamma",
            "site": "North",
            "energy_min": {"value": 30, "unit": "GeV"},
            "energy_max": {"value": 200, "unit": "GeV"},
            "zenith_angle": {"value": 20, "unit": "deg"},
            "azimuth_angle": {"value": 10, "unit": "deg"},
            "corsika_he_interaction": "qgs3",
        },
    }

    title = production_comparison._comparison_title(
        {CONFIGURATIONS_KEY: configuration}, "CTAO-North-Alpha"
    )

    assert "Layout: CTAO-North-Alpha | Primary: gamma | Site: North" in title
    assert "baseline 30 GeV to 100 GeV; candidate 30 GeV to 200 GeV" in title
    assert "Zenith: 20 deg" in title
    assert "Azimuth: baseline 0 deg -> candidate 10 deg" in title
    assert "Interaction model: qgs3" in title
    assert production_comparison._comparison_title(None, None) is None


@pytest.mark.parametrize("pair_count", [1, 100])
def test_compact_comparison_summary_stays_small(tmp_test_directory, pair_count):
    output = Path(tmp_test_directory)
    statistics_file = output / "pair" / "alpha" / "comparison_statistics.json"
    statistics_file.parent.mkdir(parents=True)
    statistics_file.write_text(
        json.dumps(
            {
                "format_version": 1,
                BASELINE_KEY: {
                    "label": "base",
                    SIMULATED_EVENT_COUNT_KEY: 10,
                    "triggered_event_count": 5,
                },
                COMPARISON_SETS_KEY: [
                    {
                        "label": "candidate",
                        SIMULATED_EVENT_COUNT_KEY: 12,
                        "triggered_event_count": 6,
                    }
                ],
                PLOT_STATISTICS_KEY: {
                    "energy": {
                        "plot_name": "distribution_energy",
                        "baseline_label": "base",
                        "metric_type": "quantity_distribution",
                        "baseline_counts": [10] * 1000,
                        COMPARISONS_KEY: [
                            {
                                "candidate_label": "candidate",
                                "simulated": {
                                    "metric": "ks",
                                    "valid": True,
                                    "value": 0.2,
                                    "reason": "",
                                    "sample_size": 10,
                                    "bin_counts": [2] * 1000,
                                },
                                "triggered": {
                                    "metric": "ks",
                                    "value": None,
                                    "valid": False,
                                    "reason": "empty",
                                },
                            }
                        ],
                    }
                },
            }
        ),
        encoding=TEXT_ENCODING,
    )

    statistics_files = [statistics_file]
    for index in range(1, pair_count):
        path = output / f"pair-{index}" / "alpha" / statistics_file.name
        path.parent.mkdir(parents=True)
        path.write_text(statistics_file.read_text(encoding=TEXT_ENCODING), encoding=TEXT_ENCODING)
        statistics_files.append(path)
    production_comparison._write_comparison_summary(statistics_files, output, "compact")

    text = (output / statistics_file.name).read_text(encoding=TEXT_ENCODING)
    summary = json.loads(text)
    schema.validate_dict_using_schema(
        summary,
        schema_file=SCHEMA_PATH / "production_comparison_statistics.schema.yml",
        offline=True,
    )
    assert summary["format_version"] == 2
    assert summary[BASELINE_KEY][SIMULATED_EVENT_COUNT_KEY] == 10 * pair_count
    assert summary[COMPARISON_SETS_KEY][0][SIMULATED_EVENT_COUNT_KEY] == 12 * pair_count
    assert summary[PLOT_STATISTICS_KEY] == {}
    assert summary[COMPARISONS_KEY] == []
    assert len(text.splitlines()) <= 20
    assert len(text.encode(TEXT_ENCODING)) < 1024
    assert all(not path.exists() for path in statistics_files)
    assert all(not path.parent.exists() for path in statistics_files)
    assert output.is_dir()


def test_empty_comparison_cleanup_preserves_plots_and_external_directories(tmp_test_directory):
    root = Path(tmp_test_directory)
    output = root / "report"
    layout = output / "pair/layout"
    layout.mkdir(parents=True)
    plot_file = layout / "plot.png"
    plot_file.write_bytes(b"plot")
    production_comparison._remove_empty_comparison_directories(layout, output)
    assert plot_file.read_bytes() == b"plot"
    outside = root / "outside"
    outside.mkdir()
    production_comparison._remove_empty_comparison_directories(outside, output)
    assert outside.is_dir()


def test_version_one_report_remains_valid_under_current_schema():
    current_schema = schema.load_schema(SCHEMA_PATH / "production_comparison_statistics.schema.yml")
    schema.validate_dict_using_schema(
        {
            "format_version": 1,
            BASELINE_KEY: {
                "label": BASELINE_KEY,
                SIMULATED_EVENT_COUNT_KEY: 10,
                "triggered_event_count": 5,
            },
            COMPARISON_SETS_KEY: [],
            PLOT_STATISTICS_KEY: {},
        },
        json_schema=current_schema,
    )
