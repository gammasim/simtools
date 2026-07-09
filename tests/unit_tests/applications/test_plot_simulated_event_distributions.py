from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import simtools.applications.plot_simulated_event_distributions as app


def test_main_loads_precomputed_trigger_histograms_and_plots(tmp_test_directory):
    """Test plotting application can load a precomputed trigger-histogram file."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={
            "trigger_histogram_file": "trigger_histograms.hdf5",
            "array_layout_name": "alpha",
        },
        io_handler=MagicMock(),
        logger=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir
    histogram_instance = MagicMock()
    histogram_instance.histograms = {
        "energy": {"histogram": MagicMock(ndim=1)},
        "energy_vs_core": {"histogram": MagicMock(ndim=2)},
        "energy_vs_core_vs_angle": {"histogram": MagicMock(ndim=3)},
        "raw_edges": MagicMock(),
    }

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

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=["alpha"])
    mock_plot.assert_called_once_with(
        {
            "energy": histogram_instance.histograms["energy"],
            "energy_vs_core": histogram_instance.histograms["energy_vs_core"],
        },
        output_path=output_dir,
    )


def test_main_raises_for_missing_array_layout(tmp_test_directory):
    """Test plotting application fails when the requested array layout is absent."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={
            "trigger_histogram_file": "trigger_histograms.hdf5",
            "array_layout_name": "missing_layout",
        },
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
            "simtools.applications.plot_simulated_event_distributions.load_event_data_histograms",
            return_value=[],
        ) as mock_load,
    ):
        with pytest.raises(
            ValueError,
            match=(
                r"Array layout 'missing_layout' not found in histogram file "
                r"'trigger_histograms\.hdf5'"
            ),
        ):
            app.main()

    mock_load.assert_called_once_with("trigger_histograms.hdf5", array_names=["missing_layout"])
