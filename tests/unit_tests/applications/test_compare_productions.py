import sys
from pathlib import Path

import pytest

import simtools.sim_events.production_comparison as production_comparison
from simtools.applications import compare_productions
from simtools.configuration.commandline_parser import CommandLineParser


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


def test_main_delegates_to_production_comparison_workflow(mocker, tmp_test_directory):
    output_directory = Path(tmp_test_directory) / "comparison"
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
    mock_write = mocker.patch(
        "simtools.applications.compare_productions.write_production_comparison"
    )

    compare_productions.main()

    mock_write.assert_called_once_with(app_context.args, output_directory)


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


def test_application_parses_productions_without_select(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "compare_productions.py",
            "--production",
            "baseline",
            "baseline.hdf5",
            "--production",
            "candidate",
            "candidate.hdf5",
            "--output_path",
            "output",
        ],
    )

    args, _ = compare_productions.APPLICATION._parse()

    assert args["select"] == []


def test_main_rejects_unimplemented_comparison_level(mocker):
    app_context = mocker.MagicMock()
    app_context.args = {"comparison_level": "signals"}
    mock_application = mocker.patch("simtools.applications.compare_productions.APPLICATION")
    mock_application.start.return_value = app_context

    with pytest.raises(NotImplementedError, match="Comparison level 'signals' is not implemented"):
        compare_productions.main()
