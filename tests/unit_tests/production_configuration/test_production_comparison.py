"""Tests for production comparison workflows."""

from pathlib import Path

import pytest

from simtools.constants import SCHEMA_PATH
from simtools.production_configuration import production_comparison
from simtools.production_configuration.production_file_selection import ProductionManifest


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
    mock_dump = mocker.patch(
        "simtools.production_configuration.production_comparison.MetadataCollector.dump"
    )

    production_comparison.write_production_comparison(
        {
            "production": ["baseline", "baseline.hdf5", "candidate", "candidate.hdf5"],
            "array_layout_name": ["first", "second"],
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
        ),
        mocker.call(
            mocker.sentinel.second_metrics,
            output_path=output_directory,
            array_layout_name="second",
            figure_format=["png"],
        ),
    ]
    assert [call.args[0]["array_layout_name"] for call in mock_dump.call_args_list] == [
        "first",
        "second",
    ]
    assert mock_dump.call_args.args[0]["schema_file"] == str(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )


def test_production_descriptor_pairs_from_metadata_matches_configurations(
    mocker, tmp_test_directory
):
    base_directory = Path(tmp_test_directory)

    def manifest(directory, interaction, zenith):
        return ProductionManifest(
            path=base_directory / directory / f"{zenith}.yml",
            data={
                "configuration": {
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
            [manifest("baseline", "qgs3", 20), manifest("baseline", "qgs3", 40)],
            [manifest("candidate", "epos", 20), manifest("candidate", "epos", 40)],
        ],
    )

    pairs = production_comparison._production_descriptor_pairs_from_metadata(
        {
            "baseline_path": base_directory / "baseline",
            "candidate_path": base_directory / "candidate",
            "select": [],
            "compare_by": ["corsika_he_interaction"],
        }
    )

    assert len(pairs) == 2
    assert all(len(descriptors) == 2 for _, descriptors in pairs)
    assert all(
        len(descriptor.trigger_histogram_files) == 1
        for _, descriptors in pairs
        for descriptor in descriptors
    )


def test_write_production_comparison_uses_metadata_pair_output_directory(
    mocker, tmp_test_directory
):
    output_directory = Path(tmp_test_directory) / "comparison"
    pairing_key = {"configuration": {"primary": "gamma"}}
    descriptors = [mocker.sentinel.baseline, mocker.sentinel.candidate]
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_production_descriptor_pairs_from_metadata",
        return_value=[(pairing_key, descriptors)],
    )
    mock_write = mocker.patch(
        "simtools.production_configuration.production_comparison._write_array_layout_comparisons"
    )

    production_comparison.write_production_comparison(
        {"baseline_path": "baseline", "array_layout_name": ["alpha"]},
        output_directory,
    )

    assert mock_write.call_args.args[:2] == (
        descriptors,
        {"baseline_path": "baseline", "array_layout_name": ["alpha"]},
    )
    assert mock_write.call_args.args[2].parent == output_directory
    assert mock_write.call_args.args[3] == ["alpha"]


def test_production_descriptor_pairs_rejects_unmatched_metadata(mocker, tmp_test_directory):
    manifest = ProductionManifest(
        path=Path(tmp_test_directory) / "baseline.yml",
        data={"configuration": {}, "files": {"trigger_histograms": ["baseline.hdf5"]}},
    )
    mocker.patch(
        "simtools.production_configuration.production_comparison."
        "_selected_trigger_histogram_manifests",
        side_effect=[[manifest], []],
    )

    with pytest.raises(ValueError, match="pairing failed"):
        production_comparison._production_descriptor_pairs_from_metadata(
            {"baseline_path": "baseline", "candidate_path": "candidate"}
        )


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
        data={"configuration": {}},
    )

    with pytest.raises(ValueError, match="More than one baseline"):
        production_comparison._unique_manifests_by_pairing_key(
            [manifest, manifest], set(), "baseline"
        )


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
