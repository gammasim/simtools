from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import simtools.applications.compare_productions as app


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
