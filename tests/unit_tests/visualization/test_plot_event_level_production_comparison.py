from collections import Counter
from pathlib import Path

import numpy as np

from simtools.sim_events.production_comparison import ProductionEventMetrics
from simtools.visualization import plot_event_level_production_comparison


def _build_metrics(label, simulated_scale, triggered_scale):
    """Build a compact metrics fixture for plotting tests."""
    return ProductionEventMetrics(
        label=label,
        simulated_energies=np.array([0.1, 0.3, 1.0, 3.0]) * simulated_scale,
        triggered_energies=np.array([0.3, 1.0]) * triggered_scale,
        simulated_core_distances=np.array([10.0, 20.0, 30.0, 40.0]),
        triggered_core_distances=np.array([20.0, 40.0]),
        trigger_multiplicity=np.array([2, 1]),
        trigger_combinations=Counter({"LSTN-01,MSTN-01": 1, "MSTN-01": 1}),
        telescope_participation=Counter({"LSTN-01": 1, "MSTN-01": 2}),
        simulated_event_count=4,
        triggered_event_count=2,
    )


def test_plot_writes_event_level_comparison_figures(tmp_test_directory):
    """Test full comparison plotting writes all expected output files."""
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics("candidate", simulated_scale=1.2, triggered_scale=1.1),
    ]

    plot_event_level_production_comparison.plot(metrics, output_path=output_path, bins=8)

    expected_files = [
        "trigger_fraction_comparison.png",
        "trigger_multiplicity_comparison.png",
        "trigger_combination_comparison.png",
        "triggered_fraction_vs_energy.png",
        "triggered_fraction_vs_core_distance.png",
        "telescope_participation_fraction.png",
    ]
    for file_name in expected_files:
        assert (output_path / file_name).exists()
