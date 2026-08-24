from pathlib import Path

import pytest

import simtools.sim_events.production_comparison as production_comparison
from simtools.applications import compare_productions
from simtools.configuration.commandline_parser import CommandLineParser
from simtools.constants import SCHEMA_PATH


def test_parse_production_arguments_requires_baseline_and_candidate(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match="At least two productions"):
        production_comparison.parse_production_arguments([["baseline", "base.h5"]])


def test_parse_production_arguments_resolves_flattened_pairs(mocker):
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        ["baseline", "base_*.h5", "candidate", "cand_*.h5"]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]

    assert descriptors[0].trigger_histogram_files == ["base_*.h5"]
    assert descriptors[1].trigger_histogram_files == ["cand_*.h5"]


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
        ([], "At least two productions are required"),
        (["baseline", "base.h5", "dangling"], "label/file pairs"),
        (
            [["baseline", "  ,   "], ["candidate", "candidate.h5"]],
            "has no trigger_histogram_file pattern",
        ),
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
        production_comparison.parse_production_arguments(
            [["baseline", "missing_*.h5"], ["candidate", "candidate.h5"]]
        )


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
        "production": [
            "baseline",
            "baseline.hdf5",
            "candidate",
            "candidate.hdf5",
        ],
        "array_layout_name": ["CTAO-North-Alpha"],
        "figure_format": ["pdf"],
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

    mock_plot.assert_called_once_with(
        mocker.ANY,
        output_path=output_directory,
        array_layout_name="CTAO-North-Alpha",
        figure_format=["pdf"],
    )
    metadata_args, metadata_file = mock_dump.call_args.args
    assert metadata_file == statistics_file
    assert metadata_args["output_file"] == str(statistics_file)
    assert metadata_args["output_file_format"] == "JSON"
    assert metadata_args["metadata_product_data_name"] == "production_comparison_statistics"
    assert metadata_args["schema_file"] == str(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )


def test_main_compares_each_selected_array_layout_separately(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
    first_statistics_file = output_directory / "first" / "comparison_statistics.json"
    second_statistics_file = output_directory / "second" / "comparison_statistics.json"
    app_context = mocker.MagicMock()
    app_context.args = {
        "comparison_level": "events",
        "production": ["baseline", "baseline.hdf5", "candidate", "candidate.hdf5"],
        "array_layout_name": ["first", "second"],
        "figure_format": ["png"],
    }
    app_context.io_handler.get_output_directory.return_value = output_directory
    mock_application = mocker.patch("simtools.applications.compare_productions.APPLICATION")
    mock_application.start.return_value = app_context
    descriptors = mocker.sentinel.production_descriptors
    mocker.patch(
        "simtools.applications.compare_productions.parse_production_arguments",
        return_value=descriptors,
    )
    mock_collect = mocker.patch(
        "simtools.applications.compare_productions.collect_production_metrics",
        side_effect=[mocker.sentinel.first_metrics, mocker.sentinel.second_metrics],
    )
    mock_plot = mocker.patch(
        "simtools.applications.compare_productions.plot_event_level_production_comparison.plot",
        side_effect=[first_statistics_file, second_statistics_file],
    )
    mock_dump = mocker.patch("simtools.applications.compare_productions.MetadataCollector.dump")

    compare_productions.main()

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


def test_application_exposes_events_comparison_level_without_unused_output_arguments():
    argument_names = {argument.name for argument in compare_productions.APPLICATION.all_arguments}

    assert {
        "output_file",
        "output_file_format",
        "skip_output_validation",
    }.isdisjoint(argument_names)
    assert "comparison_level" in argument_names
    assert {"test", "ignore_existing_parameter_version"}.isdisjoint(argument_names)
    assert "figure_format" in argument_names


def test_comparison_level_argument_accepts_events():
    parser = CommandLineParser()
    parser.add_argument_definitions(compare_productions.APPLICATION.all_arguments)

    args = parser.parse_args(
        [
            "--production",
            "baseline",
            "baseline.hdf5",
            "--production",
            "candidate",
            "candidate.hdf5",
            "--comparison_level",
            "events",
        ]
    )

    assert args.comparison_level == "events"


def test_main_rejects_unimplemented_comparison_level(mocker):
    app_context = mocker.MagicMock()
    app_context.args = {"comparison_level": "signals"}
    mock_application = mocker.patch("simtools.applications.compare_productions.APPLICATION")
    mock_application.start.return_value = app_context

    with pytest.raises(NotImplementedError, match="Comparison level 'signals' is not implemented"):
        compare_productions.main()
