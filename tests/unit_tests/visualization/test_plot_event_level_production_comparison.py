from collections import Counter
from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

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


def _assert_files_exist(output_path, file_names):
    """Assert all expected output files exist in output path."""
    for file_name in file_names:
        assert (output_path / file_name).exists()


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
    _assert_files_exist(output_path, expected_files)


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
    _assert_files_exist(output_path, per_type_files)


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


@pytest.mark.parametrize(
    ("quantity_name", "expected"),
    [("energy", (False,)), ("core_distance", (False, True)), ("angular_distance", (False, True))],
)
def test_distribution_cumulative_variants(quantity_name, expected):
    """Test cumulative variant selection by quantity."""
    assert (
        plot_event_level_production_comparison._distribution_cumulative_variants(quantity_name)
        == expected
    )


@pytest.mark.parametrize("cumulative", [False, True])
def test_normalized_histogram_values(cumulative):
    """Test normalized histogram values and Poisson errors helper."""
    values, errors = plot_event_level_production_comparison._normalized_histogram_values(
        np.array([1.0, 3.0]), cumulative=cumulative
    )
    assert values.shape == (2,)
    assert errors.shape == (2,)
    if cumulative:
        np.testing.assert_allclose(values, np.array([0.25, 1.0]))
        np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))
    else:
        np.testing.assert_allclose(values, np.array([0.25, 0.75]))
        np.testing.assert_allclose(errors, np.sqrt(np.array([1.0, 3.0])) / 4.0)


def test_normalized_histogram_values_zero_counts():
    """Test zero-count handling in histogram normalization helper."""
    values, errors = plot_event_level_production_comparison._normalized_histogram_values(
        np.array([0.0, 0.0]), cumulative=False
    )
    np.testing.assert_array_equal(values, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))


def test_plot_series_and_artist_color():
    """Test line and histogram rendering branches in _plot_series and artist color extraction."""
    fig, ax = plt.subplots(figsize=(4, 3))
    bin_edges = np.array([1.0, 2.0, 3.0])
    values = np.array([0.4, 0.6])
    line_artist = plot_event_level_production_comparison._plot_series(
        ax, bin_edges, values, "line", "core_distance"
    )
    hist_artist = plot_event_level_production_comparison._plot_series(
        ax, bin_edges, values, "hist", "core_distance", force_histogram=True
    )
    assert line_artist is not None
    assert hist_artist is not None
    assert plot_event_level_production_comparison._artist_color(None) == "black"
    assert plot_event_level_production_comparison._artist_color(line_artist) is not None
    assert plot_event_level_production_comparison._artist_color(hist_artist) is not None
    plt.close(fig)


def test_trigger_fraction_and_skip_paths(tmp_test_directory):
    """Test trigger fraction output and skip branches with empty inputs."""
    output_path = Path(tmp_test_directory)
    metrics = [_build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)]
    plot_event_level_production_comparison._plot_trigger_fraction(metrics, output_path)
    assert (output_path / "trigger_fraction.png").exists()

    empty_metric = _build_metrics("empty", simulated_scale=1.0, triggered_scale=1.0)
    empty_metric.trigger_multiplicity = np.array([], dtype=int)
    empty_metric.trigger_combinations = Counter()
    empty_metric.telescope_participation = Counter()
    plot_event_level_production_comparison._plot_trigger_multiplicity([empty_metric], output_path)
    plot_event_level_production_comparison._plot_trigger_combinations([empty_metric], output_path)
    plot_event_level_production_comparison._plot_telescope_participation(
        [empty_metric], output_path
    )
    assert not (output_path / "trigger_multiplicity_empty.png").exists()


def test_plot_invokes_triggered_fraction_branch_when_enabled(tmp_test_directory, mocker):
    """Test branch enabling triggered-fraction plotting in top-level plot function."""
    output_path = Path(tmp_test_directory)
    metrics = [_build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)]
    mock_trigger = mocker.patch(
        "simtools.visualization.plot_event_level_production_comparison._plot_triggered_vs_quantity"
    )
    mocker.patch(
        "simtools.visualization.plot_event_level_production_comparison._TRIGGERED_FRACTION_QUANTITIES",
        {"core_distance"},
    )

    plot_event_level_production_comparison.plot(metrics, output_path=output_path, bins=8)

    mock_trigger.assert_called_once()


def test_triggered_vs_quantity_outputs_and_empty_skip(tmp_test_directory):
    """Test triggered-vs-quantity output and empty-input skip path."""
    output_path = Path(tmp_test_directory)
    metrics = [_build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)]
    plot_event_level_production_comparison._plot_triggered_vs_quantity(
        metrics,
        output_path,
        quantity_name="core_distance",
        x_label="Core Distance (m)",
        x_scale="linear",
        bins=8,
    )
    assert (output_path / "triggered_fraction_vs_core_distance.png").exists()

    empty_metric = _build_metrics("empty", simulated_scale=1.0, triggered_scale=1.0)
    empty_metric.simulated_core_distances = np.array([], dtype=float)
    plot_event_level_production_comparison._plot_triggered_vs_quantity(
        [empty_metric],
        output_path,
        quantity_name="core_distance",
        x_label="Core Distance (m)",
        x_scale="linear",
        bins=8,
        suffix="_empty",
    )
    assert not (output_path / "triggered_fraction_vs_core_distance_empty.png").exists()


def test_single_and_mixed_trigger_skip_paths(tmp_test_directory):
    """Test skip branches when no single or mixed trigger combinations exist."""
    output_path = Path(tmp_test_directory)
    metric = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    metric.trigger_combinations = Counter({"LSTN-01,MSTN-01,SSTS-01": 2})
    plot_event_level_production_comparison._plot_single_telescope_trigger_frequencies(
        [metric], output_path
    )
    plot_event_level_production_comparison._plot_mixed_trigger_combinations([metric], output_path)
    assert not (output_path / "single_telescope_trigger_distribution.png").exists()
    assert not (output_path / "mixed_trigger_combinations.png").exists()
