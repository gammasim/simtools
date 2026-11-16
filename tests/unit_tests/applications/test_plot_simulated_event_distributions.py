from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from simtools.applications import plot_simulated_event_distributions as app


@pytest.mark.parametrize(
    ("parameter", "value", "expected"),
    [
        ("discriminator_threshold", 28, "threshold-28mV"),
        ("asum_threshold", 250.5, "threshold-250.5mV"),
        ("dsum_threshold", 55, "threshold-55count"),
    ],
)
def test_plot_threshold_scan_groups_by_threshold(tmp_test_directory, parameter, value, expected):
    root = Path(tmp_test_directory)
    product = root / "sample.trigger_histograms.hdf5"
    product.touch()
    product.with_suffix(".yml").write_text(
        yaml.safe_dump(
            {
                "configuration": {
                    "model_parameter_overrides": {"MSTN-01": {parameter: {"value": value}}}
                }
            }
        )
    )
    hist = SimpleNamespace(ndim=1)
    record = SimpleNamespace(array_name="MSTN-01", histograms={"energy": hist, "energy_mc": hist})
    with (
        patch.object(app, "load_event_data_histograms", return_value=[({}, record)]),
        patch.object(app.plot_simtel_event_histograms, "plot") as plot,
    ):
        app._plot_threshold_scan(str(root / "*.hdf5"), root / "plots", "MSTN-01")
    assert plot.call_args.args[0] == {"energy": hist}
    assert plot.call_args.kwargs["output_path"] == root / "plots" / expected


def test_plot_threshold_scan_rejects_empty_pattern(tmp_test_directory):
    with pytest.raises(FileNotFoundError, match="No trigger-histogram"):
        app._plot_threshold_scan(
            str(Path(tmp_test_directory) / "*.hdf5"), Path(tmp_test_directory), "MSTN-01"
        )
