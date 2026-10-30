"""Tests for production comparison statistics."""

import numpy as np
import pytest

from simtools.sim_events import production_comparison_statistics


def test_histogram_quantity_comparison_uses_binned_ks():
    baseline = {
        "simulated": np.array([3.0, 1.0]),
        "triggered": np.array([3.0, 1.0]),
        "simulated_samples": None,
        "triggered_samples": None,
    }
    candidate = {
        "simulated": np.array([1.0, 3.0]),
        "triggered": np.array([1.0, 3.0]),
        "simulated_samples": None,
        "triggered_samples": None,
    }
    baseline["bin_edges"] = np.array([0.0, 1.0, 2.0])
    candidate["bin_edges"] = np.array([0.0, 1.0, 2.0])

    result = production_comparison_statistics._compare_distribution_data(
        baseline, candidate, "simulated"
    )

    assert result["metric"] == "ks"
    assert result["ks_statistic"] == pytest.approx(0.5)
    assert result["value"] == pytest.approx(0.5)
