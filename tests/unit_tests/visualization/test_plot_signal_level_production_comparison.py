import json
from pathlib import Path

import numpy as np

from simtools.sim_events.production_comparison import ProductionSignalMetrics
from simtools.visualization import plot_signal_level_production_comparison


def _metrics(label, scale=1.0):
    return ProductionSignalMetrics(
        label=label,
        pedestals=np.array([1.0, 2.0, 3.0]) * scale,
        signals=np.array([4.0, 5.0, 6.0]) * scale,
        peak_timing=np.array([2.0, 3.0, 4.0]),
        triggered_pixels=np.array([2.0, 3.0, 4.0]) * scale,
    )


def test_plot_writes_one_set_per_telescope(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    statistics_files = plot_signal_level_production_comparison.plot(
        {
            "LSTN-01": [_metrics("baseline"), _metrics("candidate", 1.2)],
            "MSTN-01": [_metrics("baseline"), _metrics("candidate", 1.1)],
        },
        output_path,
        bins=4,
    )

    assert statistics_files == [
        output_path / "LSTN-01" / "comparison_statistics.json",
        output_path / "MSTN-01" / "comparison_statistics.json",
    ]
    telescope_path = output_path / "LSTN-01"
    assert {path.name for path in telescope_path.iterdir()} == {
        "pedestals.png",
        "signals.png",
        "peak_timing.png",
        "triggered_pixels.png",
        "comparison_statistics.json",
    }

    with (telescope_path / "comparison_statistics.json").open(encoding="utf-8") as file_handle:
        statistics = json.load(file_handle)
    assert statistics["baseline"]["label"] == "baseline"
    assert statistics["comparison_sets"][0]["label"] == "candidate"
    assert set(statistics["plot_statistics"]) == {
        "pedestals",
        "signals",
        "peak_timing",
        "triggered_pixels",
    }
    assert statistics["plot_statistics"]["signals"]["comparisons"][0]["candidate_label"] == (
        "candidate"
    )
    assert statistics["plot_statistics"]["signals"]["comparisons"][0]["valid"]
    assert statistics["plot_statistics"]["signals"]["metric"] == "ks"
    assert statistics["plot_statistics"]["triggered_pixels"]["metric"] == "wasserstein"


def test_plot_skips_observable_without_values(tmp_test_directory):
    metrics = [_metrics("baseline"), _metrics("candidate")]
    for item in metrics:
        item.peak_timing = np.array([])

    output_path = Path(tmp_test_directory)
    plot_signal_level_production_comparison.plot(
        {"LSTN-01": metrics},
        output_path,
        bins=4,
    )

    telescope_path = output_path / "LSTN-01"
    assert not (telescope_path / "peak_timing.png").exists()
    with (telescope_path / "comparison_statistics.json").open(encoding="utf-8") as file_handle:
        statistics = json.load(file_handle)
    assert "peak_timing" not in statistics["plot_statistics"]
