from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import simtools.applications.plot_simulated_event_distributions as app


def test_main_loads_precomputed_trigger_histograms_and_plots(tmp_test_directory):
    """Test plotting application can load a precomputed trigger-histogram file."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={"trigger_histogram_file": "trigger_histograms.hdf5"},
        io_handler=MagicMock(),
        logger=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir
    histogram_instance = MagicMock()

    with (
        patch(
            "simtools.applications.plot_simulated_event_distributions.build_application",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.plot_simulated_event_distributions.load_event_data_histograms",
            return_value=[(MagicMock(), histogram_instance)],
        ) as mock_load,
        patch(
            "simtools.applications.plot_simulated_event_distributions.plot_simtel_event_histograms.plot"
        ) as mock_plot,
    ):
        app.main()

    mock_load.assert_called_once_with("trigger_histograms.hdf5")
    mock_plot.assert_called_once_with(histogram_instance.histograms, output_path=output_dir)
