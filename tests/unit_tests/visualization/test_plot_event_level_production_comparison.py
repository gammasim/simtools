import json
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from matplotlib import pyplot as plt

from simtools.constants import SCHEMA_PATH
from simtools.io import ascii_handler
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
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics("candidate", simulated_scale=1.2, triggered_scale=1.1),
    ]

    statistics_file = plot_event_level_production_comparison.plot(
        metrics, output_path=output_path, bins=8
    )

    expected_files = [
        "comparison_statistics.json",
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
    assert statistics_file == output_path / "comparison_statistics.json"

    with (output_path / "comparison_statistics.json").open(encoding="utf-8") as file_handle:
        stats_payload = json.load(file_handle)
    comparison_schema = ascii_handler.collect_data_from_file(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )
    jsonschema.Draft6Validator.check_schema(comparison_schema)
    jsonschema.validate(stats_payload, comparison_schema)
    assert stats_payload["baseline"]["label"] == "baseline"
    assert stats_payload["format_version"] == 1
    assert [item["label"] for item in stats_payload["comparison_sets"]] == ["candidate"]
    assert "distribution_energy" in stats_payload["plot_statistics"]
    assert "trigger_multiplicity" in stats_payload["plot_statistics"]
    assert (
        stats_payload["plot_statistics"]["trigger_multiplicity"]["comparisons"][0]["metric"]
        == "wasserstein"
    )
    assert (
        stats_payload["plot_statistics"]["trigger_combination"]["comparisons"][0]["metric"]
        == "jensen_shannon"
    )


def test_plot_writes_requested_pdf_figures_without_png(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics("candidate", simulated_scale=1.2, triggered_scale=1.1),
    ]

    plot_event_level_production_comparison.plot(
        metrics, output_path=output_path, bins=8, figure_format=["pdf"]
    )

    assert (output_path / "trigger_multiplicity.pdf").exists()
    assert (output_path / "distribution_energy.pdf").exists()
    assert not (output_path / "trigger_multiplicity.png").exists()


def test_plot_writes_matplotlib_supported_figure_format(tmp_test_directory):
    metrics = [
        _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics("candidate", simulated_scale=1.2, triggered_scale=1.1),
    ]

    plot_event_level_production_comparison.plot(
        metrics, output_path=Path(tmp_test_directory), figure_format=["svg"]
    )

    assert (Path(tmp_test_directory) / "trigger_multiplicity.svg").exists()


def test_comparison_statistics_schema_rejects_unknown_format_version(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metrics = [_build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)]
    plot_event_level_production_comparison.plot(metrics, output_path=output_path, bins=8)
    with (output_path / "comparison_statistics.json").open(encoding="utf-8") as file_handle:
        statistics = json.load(file_handle)
    comparison_schema = ascii_handler.collect_data_from_file(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )
    statistics["format_version"] = 2

    with pytest.raises(jsonschema.ValidationError, match="2 is not one of"):
        jsonschema.validate(statistics, comparison_schema)


def test_output_directory_for_array_layout_selection_joins_list_values(tmp_test_directory):
    output_dir = Path(tmp_test_directory) / "plots"
    array_layout_names = ["CTAO-North Alpha", "MSTN-01"]

    selected = plot_event_level_production_comparison._output_directory_for_array_layout_selection(
        output_dir,
        array_layout_names,
    )

    assert selected == output_dir.joinpath(*array_layout_names)
    assert selected.exists()


def test_plot_writes_per_type_comparison_figures(tmp_test_directory):
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
    assert _assert_files_exist(output_path, per_type_files) is None


@pytest.mark.parametrize("cumulative", [False, True])
def test_normalized_histogram_values(cumulative):
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
    values, errors = plot_event_level_production_comparison._normalized_histogram_values(
        np.array([0.0, 0.0]), cumulative=False
    )
    np.testing.assert_array_equal(values, np.array([0.0, 0.0]))
    np.testing.assert_array_equal(errors, np.array([0.0, 0.0]))


def test_plot_series_and_artist_color():
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
    output_path = Path(tmp_test_directory)

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
    output_path = Path(tmp_test_directory)
    metric = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    metric.trigger_combinations = Counter({"LSTN-01,MSTN-01,SSTS-01": 2})
    plot_event_level_production_comparison._plot_single_telescope_trigger_frequencies(
        [metric], output_path
    )
    plot_event_level_production_comparison._plot_mixed_trigger_combinations([metric], output_path)
    assert not (output_path / "single_telescope_trigger_distribution.png").exists()
    assert not (output_path / "mixed_trigger_combinations.png").exists()


def test_trigger_combination_metric_uses_categories_outside_top_n(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    baseline = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    candidate = _build_metrics("candidate", simulated_scale=1.0, triggered_scale=1.0)
    baseline.trigger_combinations = Counter({"LSTN-01": 100, "MSTN-01": 1})
    candidate.trigger_combinations = Counter({"LSTN-01": 100, "MSTN-01": 100})

    statistics = plot_event_level_production_comparison._plot_trigger_combinations(
        [baseline, candidate], output_path, top_n=1
    )

    comparison = statistics["comparisons"][0]
    assert comparison["metric"] == "jensen_shannon"
    assert comparison["jensen_shannon_distance"] > 0
    assert statistics["metadata"]["categories"] == ["LSTN-01", "MSTN-01"]
    assert statistics["metadata"]["display_categories"] == ["LSTN-01"]


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

    result = plot_event_level_production_comparison._compare_distribution_data(
        baseline, candidate, "simulated", np.array([0.0, 1.0, 2.0])
    )

    assert result["metric"] == "ks"
    assert result["ks_statistic"] == pytest.approx(0.5)


def test_global_quantity_bin_edges_union_histogram_supports():
    baseline = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    candidate = _build_metrics("candidate", simulated_scale=1.0, triggered_scale=1.0)
    baseline.quantity_histograms = {
        "energy": {
            "simulated": (np.array([1.0, 1.0]), np.array([0.0, 1.0, 3.0])),
            "triggered": (np.array([1.0, 1.0]), np.array([0.0, 1.0, 3.0])),
        }
    }
    candidate.quantity_histograms = {
        "energy": {
            "simulated": (np.array([1.0, 1.0]), np.array([0.0, 2.0, 3.0])),
            "triggered": (np.array([1.0, 1.0]), np.array([0.0, 2.0, 3.0])),
        }
    }

    edges = plot_event_level_production_comparison._get_global_quantity_bin_edges(
        [baseline, candidate], "energy", "log", bins=4
    )

    np.testing.assert_array_equal(edges, np.array([0.0, 1.0, 2.0, 3.0]))


def test_quantity_count_rebinning_preserves_distribution_when_edges_split():
    metric = _build_metrics("baseline", simulated_scale=1.0, triggered_scale=1.0)
    metric.quantity_histograms = {
        "energy": {
            "simulated": (np.array([100.0]), np.array([0.0, 2.0])),
        }
    }

    counts, samples = plot_event_level_production_comparison._get_quantity_counts(
        metric,
        "energy",
        np.array([0.0, 1.0, 2.0]),
        "simulated",
    )

    assert samples is None
    np.testing.assert_allclose(counts, np.array([50.0, 50.0]))
