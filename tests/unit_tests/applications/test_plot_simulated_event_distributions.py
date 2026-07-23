from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import simtools.applications.plot_simulated_event_distributions as app


def test_main_loads_precomputed_trigger_histograms_and_plots(tmp_test_directory):
    """Test plotting application can load a precomputed trigger-histogram file."""
    output_dir = tmp_test_directory / "plots"
    app_context = SimpleNamespace(
        args={
            "trigger_histogram_file": "trigger_histograms.hdf5",
            "array_layout_name": "alpha",
        },
        io_handler=MagicMock(),
        logger=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir

    with (
        patch(
            "simtools.application.definition.ApplicationDefinition.start",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.plot_simulated_event_distributions."
            "plot_simtel_event_histograms.plot_trigger_histogram_file"
        ) as mock_plot,
    ):
        app.main()

    mock_plot.assert_called_once_with(
        "trigger_histograms.hdf5",
        output_dir,
        "alpha",
    )


def test_main_raises_for_missing_array_layout(tmp_test_directory):
    """Test plotting application fails when the requested array layout is absent."""
    output_dir = tmp_test_directory / "plots"
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
            "simtools.application.definition.ApplicationDefinition.start",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.plot_simulated_event_distributions."
            "plot_simtel_event_histograms.plot_trigger_histogram_file",
            side_effect=ValueError(
                "Array layout 'missing_layout' not found in histogram file "
                "'trigger_histograms.hdf5'."
            ),
        ) as mock_plot,
    ):
        with pytest.raises(
            ValueError,
            match=(
                r"Array layout 'missing_layout' not found in histogram file "
                r"'trigger_histograms\.hdf5'"
            ),
        ):
            app.main()

    mock_plot.assert_called_once_with(
        "trigger_histograms.hdf5",
        output_dir,
        "missing_layout",
    )
