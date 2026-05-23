from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from astropy.table import Column, Table

import simtools.applications.production_plot_corsika_limits as app


def _create_merged_table():
    return Table(
        [
            Column(data=[20.0, 40.0], name="zenith"),
            Column(data=[0.0, 180.0], name="azimuth"),
            Column(data=["dark", "moon"], name="nsb_level"),
            Column(data=["alpha", "alpha"], name="array_name"),
            Column(data=[0.1, 0.2], name="lower_energy_limit"),
            Column(data=[1200.0, 1500.0], name="upper_radius_limit"),
            Column(data=[8.0, 10.0], name="viewcone_radius"),
        ]
    )


def test_main_reads_table_and_plots(tmp_test_directory):
    """Test application orchestration from CLI input to plotting output."""
    output_dir = Path(tmp_test_directory) / "plots"
    app_context = SimpleNamespace(
        args={"input": "merged_limits.ecsv"},
        io_handler=MagicMock(),
    )
    app_context.io_handler.get_output_directory.return_value = output_dir

    merged_table = _create_merged_table()

    with (
        patch(
            "simtools.applications.production_plot_corsika_limits.build_application",
            return_value=app_context,
        ),
        patch(
            "simtools.applications.production_plot_corsika_limits.data_reader.read_table_from_file",
            return_value=merged_table,
        ) as mock_read_table,
        patch("simtools.applications.production_plot_corsika_limits.plot_limits") as mock_limits,
    ):
        app.main()

    mock_read_table.assert_called_once_with("merged_limits.ecsv")
    mock_limits.assert_called_once_with(merged_table, output_dir)
