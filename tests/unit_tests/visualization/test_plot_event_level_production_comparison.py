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
        simulated_angular_distances=np.array([0.1, 0.2, 0.3, 0.4]),
        triggered_angular_distances=np.array([0.2, 0.3]),
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
        "trigger_multiplicity.png",
        "trigger_combination.png",
        "distribution_energy.png",
        "distribution_core_distance.png",
        "distribution_core_distance_cumulative.png",
        "distribution_angular_distance.png",
        "distribution_angular_distance_cumulative.png",
        "telescope_participation_fraction.png",
    ]
    for file_name in expected_files:
        assert (output_path / file_name).exists()


def test_plot_writes_per_type_comparison_figures(tmp_test_directory):
    """Test per-array-element-type comparison plots are written for each type."""
    output_path = Path(tmp_test_directory)
    per_type_lstn = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    per_type_mstn = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=0.8)
    metrics = [
        ProductionEventMetrics(
            label="baseline",
            simulated_energies=np.array([0.1, 0.3, 1.0, 3.0]),
            triggered_energies=np.array([0.3, 1.0]),
            simulated_core_distances=np.array([10.0, 20.0, 30.0, 40.0]),
            triggered_core_distances=np.array([20.0, 40.0]),
            trigger_multiplicity=np.array([2, 1]),
            trigger_combinations=Counter({"LSTN-01,MSTN-01": 1, "MSTN-01": 1}),
            telescope_participation=Counter({"LSTN-01": 1, "MSTN-01": 2}),
            simulated_event_count=4,
            triggered_event_count=2,
            per_type={"LSTN": per_type_lstn, "MSTN": per_type_mstn},
        )
    ]

    plot_event_level_production_comparison.plot(metrics, output_path=output_path, bins=8)

    per_type_files = [
        "trigger_multiplicity_LSTN.png",
        "trigger_multiplicity_MSTN.png",
        "distribution_energy_LSTN.png",
        "distribution_angular_distance_LSTN.png",
        "distribution_angular_distance_cumulative_LSTN.png",
    ]
    for file_name in per_type_files:
        assert (output_path / file_name).exists()


def test_mixed_trigger_filter_accepts_only_allowed_signatures():
    """Test mixed trigger selector keeps only 1+1 and 1+2/2+1 patterns."""
    assert plot_event_level_production_comparison._is_mixed_type_combination("LSTN-01,MSTN-01")
    assert plot_event_level_production_comparison._is_mixed_type_combination(
        "LSTN-01,LSTN-02,MSTN-01"
    )
    assert not plot_event_level_production_comparison._is_mixed_type_combination(
        "LSTN-01,MSTN-01,SSTS-01"
    )
    assert not plot_event_level_production_comparison._is_mixed_type_combination(
        "LSTN-01,LSTN-02,LSTN-03,MSTN-01"
    )
