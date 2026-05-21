from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import simtools.applications.compare_productions as app
import simtools.sim_events.production_comparison as production_comparison


def test_main_collects_metrics_and_plots(tmp_test_directory):
    """Test application orchestration from CLI args to plotting output."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={
            "production": [["baseline", "base_*.h5"], ["candidate", "cand_*.h5"]],
            "comparison_level": "events",
        },
        io_handler=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir

    parsed_productions = [MagicMock(), MagicMock()]
    collected_metrics = [MagicMock(), MagicMock()]

    with (
        patch(
            "simtools.applications.compare_productions.build_application",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.compare_productions.parse_production_arguments",
            return_value=parsed_productions,
        ) as mock_parse,
        patch(
            "simtools.applications.compare_productions.collect_production_metrics",
            return_value=collected_metrics,
        ) as mock_collect,
        patch(
            "simtools.applications.compare_productions.plot_event_level_production_comparison.plot"
        ) as mock_plot,
    ):
        app.main()

    mock_parse.assert_called_once_with(app_context.args["production"])
    mock_collect.assert_called_once_with(parsed_productions)
    mock_plot.assert_called_once_with(collected_metrics, output_path=output_dir)


def test_parse_production_arguments_accepts_single_production(mocker):
    """Test parser accepts a single production descriptor."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments([["baseline", "base.h5"]])

    assert len(descriptors) == 1
    assert descriptors[0].label == "baseline"
    assert descriptors[0].event_data_files == ["base.h5"]


def test_parse_production_arguments_resolves_flattened_pairs(mocker):
    """Test parser supports flattened label/file list from configuration files."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        ["baseline", "base_*.h5", "candidate", "cand_*.h5"]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]
    assert descriptors[0].event_data_files == ["base_*.h5"]
    assert descriptors[1].event_data_files == ["cand_*.h5"]


def test_parse_production_arguments_rejects_duplicate_labels(mocker):
    """Test parser rejects duplicated production labels."""
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
        ([["baseline", "  ,   "]], "has no event_data_file pattern"),
        ([["baseline", "a.h5"], ["candidate", 1]], "label/file pairs"),
    ],
)
def test_parse_production_arguments_error_paths(mocker, arguments, error_match):
    """Test parser validation failures for malformed production arguments."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match=error_match):
        production_comparison.parse_production_arguments(arguments)


def test_parse_production_arguments_rejects_unresolved_patterns(mocker):
    """Test parser rejects productions that resolve to no files."""
    mocker.patch("simtools.sim_events.production_comparison.resolve_file_patterns", return_value=[])

    with pytest.raises(ValueError, match="does not resolve to any files"):
        production_comparison.parse_production_arguments([["baseline", "missing_*.h5"]])


def test_parse_production_arguments_accepts_nested_flattened_strings(mocker):
    """Test parser accepts nested flattened string groups."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = production_comparison.parse_production_arguments(
        [["baseline", "a.h5", "candidate", "b.h5"]]
    )

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]
