from pathlib import Path
from unittest.mock import MagicMock, patch

from simtools.telescope_trigger_rates import telescope_trigger_rates

FILE_SIMTEL = "test_file.simtel"


def test_telescope_trigger_rates_with_array_layout_name():
    args_dict = {
        "array_layout_name": "test_layout",
        "site": "test_site",
        "model_version": "1.0.0",
        "event_data_file": FILE_SIMTEL,
        "plot_histograms": False,
    }

    with (
        patch(
            "simtools.telescope_trigger_rates.get_array_elements_from_db_for_layouts"
        ) as mock_get_array_elements,
        patch("simtools.telescope_trigger_rates.SimtelIOEventHistograms") as mock_histograms,
    ):
        mock_get_array_elements.return_value = {"array1": [1, 2, 3]}
        mock_histograms_instance = MagicMock()
        mock_histograms.return_value = mock_histograms_instance

        telescope_trigger_rates(args_dict)

        mock_get_array_elements.assert_called_once_with("test_layout", "test_site", "1.0.0")
        mock_histograms.assert_called_once_with(
            FILE_SIMTEL, array_name="array1", telescope_list=[1, 2, 3]
        )
        mock_histograms_instance.fill.assert_called_once()
        mock_histograms_instance.plot.assert_not_called()


def test_telescope_trigger_rates_without_array_layout_name():
    args_dict = {
        "telescope_ids": Path("test_telescope_ids.txt"),
        "event_data_file": FILE_SIMTEL,
        "plot_histograms": True,
    }

    with (
        patch(
            "simtools.telescope_trigger_rates.ascii_handler.collect_data_from_file"
        ) as mock_collect_data,
        patch("simtools.telescope_trigger_rates.SimtelIOEventHistograms") as mock_histograms,
        patch("simtools.telescope_trigger_rates.plot_simtel_event_histograms.plot") as mock_plot,
        patch("simtools.telescope_trigger_rates.io_handler.IOHandler") as mock_io_handler,
    ):
        mock_collect_data.return_value = {"telescope_configs": {"array1": [1, 2, 3]}}
        mock_histograms_instance = MagicMock()
        mock_histograms.return_value = mock_histograms_instance
        mock_io_handler_instance = MagicMock()
        mock_io_handler.return_value = mock_io_handler_instance
        mock_io_handler_instance.get_output_directory.return_value = Path("output_dir")

        telescope_trigger_rates(args_dict)

        mock_collect_data.assert_called_once_with(Path("test_telescope_ids.txt"))
        mock_histograms.assert_called_once_with(
            FILE_SIMTEL, array_name="array1", telescope_list=[1, 2, 3]
        )
        mock_histograms_instance.fill.assert_called_once()
        mock_plot.assert_called_once_with(
            mock_histograms_instance.histograms, output_path=Path("output_dir"), array_name="array1"
        )
