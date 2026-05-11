from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import simtools.applications.plot_simulated_event_distributions as app


def test_main_loads_histograms_and_plots(tmp_test_directory):
    """Test plotting application builds histograms from the configured event-data file."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={"event_data_file": "test_events.h5"},
        io_handler=MagicMock(),
        logger=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir

    with (
        patch(
            "simtools.applications.plot_simulated_event_distributions.build_application",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.plot_simulated_event_distributions.EventDataHistograms"
        ) as mock_histogram_class,
        patch(
            "simtools.applications.plot_simulated_event_distributions.plot_simtel_event_histograms.plot"
        ) as mock_plot,
    ):
        histogram_instance = MagicMock()
        mock_histogram_class.return_value = histogram_instance

        app.main()

    mock_histogram_class.assert_called_once_with("test_events.h5")
    histogram_instance.fill.assert_called_once()
    mock_plot.assert_called_once_with(
        histogram_instance.histograms,
        output_path=output_dir,
    )
