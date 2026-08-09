from pathlib import Path

import pytest

import simtools.sim_events.production_comparison as production_comparison
from simtools.applications import compare_productions


def test_parse_production_arguments_accepts_single_production(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments([["baseline", "base.h5"]])

    assert len(descriptors) == 1
    assert descriptors[0].label == "baseline"
    assert descriptors[0].input_files == ["base.h5"]


def test_parse_production_arguments_resolves_flattened_pairs(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        ["baseline", "base_*.h5", "candidate", "cand_*.h5"]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]

    assert descriptors[0].input_files == ["base_*.h5"]
    assert descriptors[1].input_files == ["cand_*.h5"]


def test_parse_production_arguments_rejects_duplicate_labels(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match="labels must be unique"):
        production_comparison.parse_production_arguments([["same", "a.h5"], ["same", "b.h5"]])


@pytest.mark.parametrize(
    ("arguments", "error_match"),
    [
        ([], "At least one production is required"),
        (["baseline", "base.h5", "dangling"], "label/file pairs"),
        ([["baseline", "  ,   "]], "has no input_file pattern"),
        ([["baseline", "a.h5"], ["candidate", 1]], "label/file pairs"),
    ],
)
def test_parse_production_arguments_error_paths(mocker, arguments, error_match):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match=error_match):
        production_comparison.parse_production_arguments(arguments)


def test_parse_production_arguments_rejects_unresolved_patterns(mocker):
    mocker.patch("simtools.sim_events.production_comparison.resolve_file_patterns", return_value=[])

    with pytest.raises(ValueError, match="does not resolve to any files"):
        production_comparison.parse_production_arguments([["baseline", "missing_*.h5"]])


def test_parse_production_arguments_accepts_nested_flattened_strings(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        [["baseline", "a.h5", "candidate", "b.h5"]]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]


def test_main_writes_comparison_statistics_metadata(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
    statistics_file = output_directory / "comparison_statistics.json"
    app_context = mocker.MagicMock()
    app_context.args = {
        "comparison_level": "events",
        "production": ["baseline", "baseline.hdf5"],
        "array_layout_name": ["CTAO-North-Alpha"],
    }
    app_context.io_handler.get_output_directory.return_value = output_directory
    mock_application = mocker.patch("simtools.applications.compare_productions.APPLICATION")
    mock_application.start.return_value = app_context
    mocker.patch("simtools.applications.compare_productions.parse_production_arguments")
    mocker.patch("simtools.applications.compare_productions.collect_production_metrics")
    mock_plot = mocker.patch(
        "simtools.applications.compare_productions.plot_event_level_production_comparison.plot",
        return_value=statistics_file,
    )
    mock_dump = mocker.patch("simtools.applications.compare_productions.MetadataCollector.dump")

    compare_productions.main()

    mock_plot.assert_called_once()
    metadata_args, metadata_file = mock_dump.call_args.args
    assert metadata_file == statistics_file
    assert metadata_args["output_file"] == str(statistics_file)
    assert metadata_args["output_file_format"] == "JSON"
    assert metadata_args["metadata_product_data_name"] == "production_comparison_statistics"


def test_main_runs_signal_comparison_for_layout(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
    statistics_file = output_directory / "LSTN-01" / "comparison_statistics.json"
    app_context = mocker.MagicMock()
    app_context.args = {
        "comparison_level": "signal",
        "production": ["baseline", "baseline.simtel"],
        "array_layout_name": ["CTAO-North-Alpha"],
    }
    app_context.io_handler.get_output_directory.return_value = output_directory
    mock_application = mocker.patch("simtools.applications.compare_productions.APPLICATION")
    mock_application.start.return_value = app_context
    mocker.patch("simtools.applications.compare_productions.parse_production_arguments")
    mocker.patch("simtools.applications.compare_productions.collect_signal_metrics")
    mocker.patch(
        "simtools.applications.compare_productions.plot_signal_level_production_comparison.plot",
        return_value=[statistics_file],
    )
    mock_dump = mocker.patch("simtools.applications.compare_productions.MetadataCollector.dump")

    compare_productions.main()

    mock_dump.assert_called_once()
    assert mock_dump.call_args.args[1] == statistics_file
