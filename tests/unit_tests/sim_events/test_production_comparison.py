from types import SimpleNamespace

import numpy as np
import pytest

from simtools.sim_events.production_comparison import (
    collect_production_metrics,
    parse_production_arguments,
)


def test_parse_production_arguments_requires_two_productions():
    """Test parser rejects less than two productions."""
    with pytest.raises(ValueError, match="At least two productions"):
        parse_production_arguments([["baseline", "base.h5"]])


def test_parse_production_arguments_resolves_flattened_pairs(mocker):
    """Test parser supports flattened label/file list from configuration files."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    descriptors = parse_production_arguments(["baseline", "base_*.h5", "candidate", "cand_*.h5"])

    assert [descriptor.label for descriptor in descriptors] == ["baseline", "candidate"]
    assert descriptors[0].event_data_files == ["base_*.h5"]
    assert descriptors[1].event_data_files == ["cand_*.h5"]


def test_parse_production_arguments_rejects_duplicate_labels(mocker):
    """Test parser rejects duplicated production labels."""
    mocker.patch(
        "simtools.sim_events.production_comparison.resolve_file_patterns",
        side_effect=lambda patterns: patterns,
    )

    with pytest.raises(ValueError, match="labels must be unique"):
        parse_production_arguments([["same", "a.h5"], ["same", "b.h5"]])


def test_collect_production_metrics_aggregates_event_data(mocker):
    """Test production metrics aggregation across all event-level quantities."""
    descriptor = SimpleNamespace(label="baseline", event_data_files=["baseline_file.h5"])

    shower_data = SimpleNamespace(
        simulated_energy=np.array([1.0, 2.0, 3.0]),
        core_distance_shower=np.array([10.0, 20.0, 30.0]),
    )
    triggered_shower_data = SimpleNamespace(
        simulated_energy=np.array([1.0, 3.0]),
        core_distance_shower=np.array([10.0, 30.0]),
    )
    triggered_data = SimpleNamespace(
        telescope_list=[
            np.array(["LSTN-01", "MSTN-01"]),
            np.array(["MSTN-01"]),
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
    assert metrics[0].triggered_event_count == 2
    np.testing.assert_array_equal(metrics[0].simulated_energies, np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(metrics[0].triggered_energies, np.array([1.0, 3.0]))
    np.testing.assert_array_equal(metrics[0].trigger_multiplicity, np.array([2, 1]))
    assert metrics[0].trigger_combinations["LSTN-01,MSTN-01"] == 1
    assert metrics[0].trigger_combinations["MSTN-01"] == 1
    assert metrics[0].telescope_participation["MSTN-01"] == 2
    assert metrics[0].telescope_participation["LSTN-01"] == 1
    assert metrics[0].trigger_fraction == pytest.approx(2.0 / 3.0)
