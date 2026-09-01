import numpy as np
import pytest

from simtools.statistics import compare_histogram_counts, compare_samples_with_statistics


def test_compare_histogram_counts_uses_binned_ks_statistic():
    result = compare_histogram_counts([3, 1], [1, 3])

    assert result["metric"] == "ks"
    assert result["ks_statistic"] == pytest.approx(0.5)
    assert result["value"] == pytest.approx(0.5)
    assert result["valid"]


def test_compare_histogram_counts_uses_jensen_shannon_distance():
    result = compare_histogram_counts([1, 0], [0, 1], metric="jensen_shannon")

    assert result["jensen_shannon_distance"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(1.0)
    assert result["valid"]


def test_compare_histogram_counts_uses_wasserstein_distance():
    result = compare_histogram_counts([1, 0], [0, 1], metric="wasserstein", bin_edges=[1, 2, 3])

    assert result["wasserstein_distance"] == pytest.approx(1.0)
    assert result["value"] == pytest.approx(1.0)
    assert result["valid"]


def test_compare_histogram_counts_is_invariant_to_count_scale():
    baseline = compare_histogram_counts([1, 3], [2, 2], metric="jensen_shannon")
    scaled = compare_histogram_counts([10, 30], [20, 20], metric="jensen_shannon")

    assert scaled["jensen_shannon_distance"] == pytest.approx(baseline["jensen_shannon_distance"])
    assert scaled["value"] == pytest.approx(baseline["value"])


def test_compare_histogram_counts_returns_invalid_for_empty_data():
    result = compare_histogram_counts([0, 0], [1, 0], metric="jensen_shannon")

    assert result == {
        "metric": "jensen_shannon",
        "value": None,
        "valid": False,
        "reason": "insufficient_data",
    }


@pytest.mark.parametrize(
    ("counts1", "counts2"),
    [([-1, 1], [1, 1]), ([np.nan, 1], [1, 1]), ([1, 2], [1])],
)
def test_compare_histogram_counts_rejects_invalid_counts(counts1, counts2):
    with pytest.raises(ValueError, match="must"):
        compare_histogram_counts(counts1, counts2)


def test_compare_histogram_counts_rejects_invalid_wasserstein_edges():
    with pytest.raises(ValueError, match="strictly increasing"):
        compare_histogram_counts([1, 1], [1, 1], metric="wasserstein", bin_edges=[0, 2, 1])


def test_compare_histogram_counts_rejects_unknown_metric():
    with pytest.raises(ValueError, match="Unsupported"):
        compare_histogram_counts([1], [1], metric="unknown")


def test_compare_samples_with_statistics_returns_statistics():
    baseline = np.array([1.0, 2.0, 3.0, 4.0])
    candidate = np.array([1.0, 2.0, 2.5, 4.0])

    result = compare_samples_with_statistics(baseline, candidate)

    assert result["value"] is not None
    assert result["pvalue"] is not None
    assert result["valid"]
    assert result["reason"] == "ok"


def test_compare_samples_with_statistics_empty_samples():
    result = compare_samples_with_statistics([], [], [0.0, 1.0])
    assert result["ks_statistic"] is None
    assert result["ks_pvalue"] is None


def test_compare_samples_with_statistics_uses_wasserstein_distance():
    result = compare_samples_with_statistics(
        baseline_samples=[1.1, 1.2],
        candidate_samples=[2.1, 2.2],
        bin_edges=[1.0, 2.0, 3.0],
        metric="wasserstein",
    )

    assert result["metric"] == "wasserstein"
    assert result["wasserstein_distance"] == pytest.approx(1.0)
    assert result["baseline_counts"] == [2, 0]
    assert result["candidate_counts"] == [0, 2]
    assert result["value"] == pytest.approx(1.0)
