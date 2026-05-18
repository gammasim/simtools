from types import SimpleNamespace

import numpy as np
import pytest

from simtools.sim_events.production_comparison import (
    collect_production_metrics,
)
from simtools.statistics import (
    chi2_test_histograms,
    compare_samples_with_statistics,
)


def test_collect_production_metrics_aggregates_event_data(mocker):
    """Test production metrics aggregation across all event-level quantities."""
    descriptor = SimpleNamespace(label="baseline", event_data_files=["baseline_file.h5"])

    shower_data = SimpleNamespace(
        simulated_energy=np.array([1.0, 2.0, 3.0]),
        core_distance_shower=np.array([10.0, 20.0, 30.0]),
        angular_distance=np.array([0.5, 1.0, 1.5]),
    )
    triggered_shower_data = SimpleNamespace(
        simulated_energy=np.array([1.0, 3.0, 2.0]),
        core_distance_shower=np.array([10.0, 30.0, 20.0]),
        angular_distance=np.array([0.5, 1.5, 1.2]),
    )
    triggered_data = SimpleNamespace(
        telescope_list=[
            np.array(["LSTN-01", "MSTN-01"]),
            np.array(["MSTN-01"]),
            np.array(["MSTN-01", "MSTN-02"]),
        ]
    )

    mock_reader = mocker.patch("simtools.sim_events.production_comparison.EventDataReader")
    mock_reader.return_value.data_sets = [{"SHOWERS": "SHOWERS", "TRIGGERS": "TRIGGERS"}]
    mock_reader.return_value.read_event_data.return_value = (
        SimpleNamespace(),
        shower_data,
        triggered_shower_data,
        triggered_data,
    )

    metrics = collect_production_metrics([descriptor])

    assert len(metrics) == 1
    assert metrics[0].label == "baseline"
    assert metrics[0].simulated_event_count == 3
    assert metrics[0].triggered_event_count == 3
    np.testing.assert_array_equal(metrics[0].simulated_energies, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(metrics[0].triggered_energies, np.array([1.0, 3.0, 2.0]))
    np.testing.assert_array_equal(metrics[0].trigger_multiplicity, np.array([2, 1, 2]))
    assert metrics[0].trigger_combinations["LSTN-01,MSTN-01"] == 1
    assert metrics[0].trigger_combinations["MSTN-01"] == 1
    assert metrics[0].trigger_combinations["MSTN-01,MSTN-02"] == 1
    assert metrics[0].telescope_participation["MSTN-01"] == 3
    assert metrics[0].telescope_participation["MSTN-02"] == 1
    assert metrics[0].telescope_participation["LSTN-01"] == 1
    assert metrics[0].trigger_fraction == pytest.approx(1.0)
    np.testing.assert_array_equal(metrics[0].simulated_angular_distances, np.array([0.5, 1.0, 1.5]))
    np.testing.assert_array_equal(metrics[0].triggered_angular_distances, np.array([0.5, 1.5, 1.2]))
    assert set(metrics[0].per_type.keys()) == {"MSTN", "single_telescope", "mixed_type"}
    assert metrics[0].per_type["MSTN"].triggered_event_count == 1
    np.testing.assert_array_equal(metrics[0].per_type["MSTN"].triggered_energies, [2.0])
    np.testing.assert_array_equal(metrics[0].per_type["MSTN"].trigger_multiplicity, [2])
    assert metrics[0].per_type["single_telescope"].triggered_event_count == 1
    assert metrics[0].per_type["mixed_type"].triggered_event_count == 1
    np.testing.assert_array_equal(metrics[0].per_type["single_telescope"].triggered_energies, [3.0])
    np.testing.assert_array_equal(metrics[0].per_type["mixed_type"].triggered_energies, [1.0])
    np.testing.assert_array_equal(metrics[0].per_type["MSTN"].triggered_angular_distances, [1.2])


def test_collect_production_metrics_handles_empty_and_partial_datasets(mocker):
    """Test collector skips missing shower and missing trigger data sets gracefully."""
    descriptor = SimpleNamespace(label="baseline", event_data_files=["baseline_file.h5"])
    shower_data = SimpleNamespace(
        simulated_energy=np.array([2.0]),
        core_distance_shower=np.array([20.0]),
        angular_distance=np.array([1.0]),
    )

    mock_reader = mocker.patch("simtools.sim_events.production_comparison.EventDataReader")
    mock_reader.return_value.data_sets = [{"A": "A"}, {"B": "B"}, {"C": "C"}]
    mock_reader.return_value.read_event_data.side_effect = [
        (SimpleNamespace(), None, None, None),
        (SimpleNamespace(), shower_data, None, None),
        (SimpleNamespace(), shower_data, shower_data, None),
    ]

    metrics = collect_production_metrics([descriptor])

    assert metrics[0].simulated_event_count == 2
    assert metrics[0].triggered_event_count == 0
    assert metrics[0].trigger_fraction == pytest.approx(0.0)
    assert metrics[0].trigger_combinations == {}
    np.testing.assert_array_equal(metrics[0].triggered_energies, np.array([]))


def test_collect_production_metrics_empty_input_keeps_arrays_empty(mocker):
    """Test empty shower input yields empty arrays and zero trigger fraction."""
    descriptor = SimpleNamespace(label="baseline", event_data_files=["baseline_file.h5"])

    mock_reader = mocker.patch("simtools.sim_events.production_comparison.EventDataReader")
    mock_reader.return_value.data_sets = [{"SHOWERS": "SHOWERS"}]
    mock_reader.return_value.read_event_data.return_value = (SimpleNamespace(), None, None, None)

    metrics = collect_production_metrics([descriptor])

    assert metrics[0].simulated_event_count == 0
    assert metrics[0].triggered_event_count == 0
    assert metrics[0].trigger_fraction == pytest.approx(0.0)
    np.testing.assert_array_equal(metrics[0].simulated_energies, np.array([]))
    np.testing.assert_array_equal(metrics[0].trigger_multiplicity, np.array([], dtype=int))


def test_compute_ks_chi2_for_samples_returns_statistics():
    """Test KS/Chi2 statistics for sample-based comparisons."""
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 2.5, 4.0])
    bin_edges = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    result = compare_samples_with_statistics(baseline, candidate, bin_edges)

    assert result["ks_statistic"] is not None
    assert result["ks_pvalue"] is not None
    assert result["chi2_statistic"] is not None
    assert result["chi2_pvalue"] is not None
    assert result["valid"]
    assert result["reason"] == "ok"
    assert result["bin_edges"] == [1.0, 2.0, 3.0, 4.0, 5.0]


def test_compute_ks_chi2_for_aligned_histograms_handles_empty_counts():
    """Test aligned histogram statistics return invalid for empty counts."""
    baseline = np.array([0.0, 0.0, 0.0])
    candidate = np.array([1.0, 0.0, 0.0])

    chi2_stat, chi2_pval, valid, reason = chi2_test_histograms(baseline, candidate)
    result = {
        "chi2_statistic": chi2_stat,
        "chi2_pvalue": chi2_pval,
        "valid": valid,
        "reason": reason,
    }

    assert result["ks_statistic"] is None
    assert result["chi2_statistic"] is None
    assert result["valid"] is False
    assert result["reason"] == "insufficient_data"
