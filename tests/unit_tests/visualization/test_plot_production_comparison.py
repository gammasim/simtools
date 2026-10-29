import json
from collections import Counter
from pathlib import Path

import jsonschema
import numpy as np
import pytest
from matplotlib import pyplot as plt

from simtools.constants import SCHEMA_PATH
from simtools.data_model import schema
from simtools.production_configuration.comparison import (
    ProductionEventMetrics,
    ProductionSignalMetrics,
)
from simtools.visualization import plot_production_comparison

_FIRST = "first"
_SECOND = "second"
_RENDERING_KEY = "rendering"

ENERGY = "energy"
ANGULAR_DISTANCE = "angular_distance"
CORE_DISTANCE = "core_distance"
COMPACT_REPORT = "compact"
CONFIGURATIONS_KEY = "configurations"
BASELINE = "baseline"
CANDIDATE = "candidate"
ENERGY_MIN_KEY = "energy_min"
ENERGY_MAX_KEY = "energy_max"
VIEW_CONE_MIN_KEY = "view_cone_min"
VIEW_CONE_MAX_KEY = "view_cone_max"
VALUE_KEY = "value"
PLOT_STATISTICS_KEY = "plot_statistics"
BASELINE_COUNTS_KEY = "baseline_counts"
TEXT_ENCODING = "utf-8"
CORE_DISTANCE_FIGURE = "distribution_core_distance.png"


@pytest.fixture
def fixed_pair_context():
    configuration = {
        ENERGY_MIN_KEY: {VALUE_KEY: 30.0, "unit": "GeV"},
        ENERGY_MAX_KEY: {VALUE_KEY: 0.03, "unit": "TeV"},
        VIEW_CONE_MIN_KEY: {VALUE_KEY: 0.0, "unit": "rad"},
        VIEW_CONE_MAX_KEY: {VALUE_KEY: 0.0, "unit": "deg"},
    }
    return {CONFIGURATIONS_KEY: {label: dict(configuration) for label in (BASELINE, CANDIDATE)}}


def test_metadata_suppression_converts_units_and_tolerates_roundoff(fixed_pair_context):
    maximum = 0.030000000000001
    assert np.isclose(maximum, 0.03, rtol=1e-6, atol=1e-12)
    fixed_pair_context[CONFIGURATIONS_KEY][CANDIDATE][ENERGY_MAX_KEY] = {
        VALUE_KEY: maximum,
        "unit": "TeV",
    }
    offset = 1e-13
    assert np.isclose(offset, 0.0, rtol=0.0, atol=1e-12)
    fixed_pair_context[CONFIGURATIONS_KEY][CANDIDATE][VIEW_CONE_MAX_KEY] = offset
    assert plot_production_comparison._metadata_suppressed_distributions(fixed_pair_context) == {
        ENERGY,
        ANGULAR_DISTANCE,
    }


@pytest.mark.parametrize(
    ("key", VALUE_KEY, "expected"),
    [
        (ENERGY_MAX_KEY, {VALUE_KEY: 31.0, "unit": "GeV"}, {ANGULAR_DISTANCE}),
        (ENERGY_MAX_KEY, {VALUE_KEY: 1.0, "unit": "m"}, {ANGULAR_DISTANCE}),
        (ENERGY_MAX_KEY, {VALUE_KEY: 30.0}, {ANGULAR_DISTANCE}),
        (ENERGY_MAX_KEY, float("nan"), {ANGULAR_DISTANCE}),
        (ENERGY_MAX_KEY, float("inf"), {ANGULAR_DISTANCE}),
        (VIEW_CONE_MAX_KEY, {VALUE_KEY: 1.0, "unit": "deg"}, {ENERGY}),
    ],
)
def test_metadata_suppression_keeps_uncertain_distributions(
    fixed_pair_context, key, value, expected
):
    fixed_pair_context[CONFIGURATIONS_KEY][CANDIDATE][key] = value
    assert (
        plot_production_comparison._metadata_suppressed_distributions(fixed_pair_context)
        == expected
    )


@pytest.mark.parametrize(
    "context",
    [
        None,
        {},
        {CONFIGURATIONS_KEY: {BASELINE: {}}},
        {CONFIGURATIONS_KEY: {BASELINE: {}, CANDIDATE: {}}},
    ],
)
def test_metadata_suppression_requires_both_productions(context):
    assert not plot_production_comparison._metadata_suppressed_distributions(context)


def test_metadata_suppression_keeps_different_fixed_energies(fixed_pair_context):
    configuration = fixed_pair_context[CONFIGURATIONS_KEY][CANDIDATE]
    configuration[ENERGY_MIN_KEY] = configuration[ENERGY_MAX_KEY] = 3000.0
    assert plot_production_comparison._metadata_suppressed_distributions(fixed_pair_context) == {
        ANGULAR_DISTANCE
    }


def test_metadata_suppression_keeps_nonzero_fixed_view_cone(fixed_pair_context):
    for configuration in fixed_pair_context[CONFIGURATIONS_KEY].values():
        configuration[VIEW_CONE_MIN_KEY] = configuration[VIEW_CONE_MAX_KEY] = 1.0
    assert plot_production_comparison._metadata_suppressed_distributions(fixed_pair_context) == {
        ENERGY
    }


def test_compact_figure_label_identifies_the_report_plot(tmp_test_directory):
    figure = plt.figure()
    figures = []
    token = plot_production_comparison._COMPACT_FIGURES.set(figures)
    try:
        plot_production_comparison._save_figure(
            figure, Path(tmp_test_directory), "distribution_core_distance.png"
        )
    finally:
        plot_production_comparison._COMPACT_FIGURES.reset(token)

    assert figures == [figure]
    assert figure.get_label() == "distribution_core_distance"
    plt.close(figure)


@pytest.mark.parametrize(
    ("report_mode", "has_metadata"),
    [(COMPACT_REPORT, True), (COMPACT_REPORT, False), ("diagnostic", True)],
)
def test_distribution_suppression_covers_overall_and_telescope_types(
    tmp_test_directory, mocker, fixed_pair_context, report_mode, has_metadata
):
    metrics = [_build_metrics(label, 1.0, 1.0) for label in (BASELINE, CANDIDATE)]
    for metric in metrics:
        metric.per_type = {"LSTN": _build_metrics(metric.label, 1.0, 1.0)}
    save = mocker.patch.object(
        plot_production_comparison,
        "_save_figure",
        side_effect=lambda figure, *_args, **_kwargs: plt.close(figure),
    )
    record = mocker.spy(plot_production_comparison, "record_plot_statistics")
    path = plot_production_comparison.plot_event_level_comparison(
        metrics,
        output_path=Path(tmp_test_directory),
        report_mode=report_mode,
        compact_figures=[],
        comparison_context=fixed_pair_context if has_metadata else None,
    )
    names = {call.args[2] for call in save.call_args_list}
    statistics = json.loads(path.read_text(encoding=TEXT_ENCODING))[PLOT_STATISTICS_KEY]
    if report_mode == COMPACT_REPORT:
        assert statistics
        assert BASELINE_COUNTS_KEY not in path.read_text(encoding=TEXT_ENCODING)
    recorded_names = {call.args[1] for call in record.call_args_list}
    for quantity in (ENERGY, ANGULAR_DISTANCE):
        variants = ("", "_cumulative") if quantity == ANGULAR_DISTANCE else ("",)
        for cumulative in variants:
            for suffix in ("", "_LSTN"):
                key = f"distribution_{quantity}{cumulative}{suffix}"
                assert key in recorded_names
                assert (f"{key}.png" in names) is (
                    report_mode != COMPACT_REPORT or not has_metadata
                )
    assert CORE_DISTANCE_FIGURE in names
    assert "distribution_core_distance_cumulative_LSTN.png" in names


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
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]

    statistics_file = plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=output_path, bins=8
    )

    expected_files = [
        "comparison_statistics.json",
        "trigger_multiplicity.png",
        "trigger_combination.png",
        "distribution_energy.png",
        CORE_DISTANCE_FIGURE,
        "distribution_core_distance_cumulative.png",
        "distribution_angular_distance.png",
        "distribution_angular_distance_cumulative.png",
        "telescope_participation_fraction.png",
    ]
    _assert_files_exist(output_path, expected_files)
    assert statistics_file == output_path / "comparison_statistics.json"

    with (output_path / "comparison_statistics.json").open(encoding=TEXT_ENCODING) as file_handle:
        stats_payload = json.load(file_handle)
    comparison_schema = schema.load_schema(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )
    jsonschema.Draft6Validator.check_schema(comparison_schema)
    jsonschema.validate(stats_payload, comparison_schema)
    assert stats_payload[BASELINE]["label"] == BASELINE
    assert stats_payload["format_version"] == 1
    assert [item["label"] for item in stats_payload["comparison_sets"]] == [CANDIDATE]
    assert "distribution_energy" in stats_payload[PLOT_STATISTICS_KEY]
    assert "trigger_multiplicity" in stats_payload[PLOT_STATISTICS_KEY]
    assert (
        stats_payload[PLOT_STATISTICS_KEY]["trigger_multiplicity"]["comparisons"][0]["metric"]
        == "wasserstein"
    )
    assert stats_payload[PLOT_STATISTICS_KEY]["trigger_multiplicity"]["comparisons"][0][
        "normalized_value"
    ] == pytest.approx(0.0)
    assert (
        stats_payload[PLOT_STATISTICS_KEY]["trigger_combination"]["comparisons"][0]["metric"]
        == "jensen_shannon"
    )
    participation = stats_payload[PLOT_STATISTICS_KEY]["telescope_participation_fraction_LSTN"]
    assert participation["comparisons"][0]["value"] == pytest.approx(0.0)


def test_plot_writes_requested_pdf_figures_without_png(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]

    plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=output_path, bins=8, figure_format=["pdf"]
    )

    assert (output_path / "trigger_multiplicity.pdf").exists()
    assert (output_path / "distribution_energy.pdf").exists()
    assert not (output_path / "trigger_multiplicity.png").exists()


def test_compact_report_writes_one_pdf_and_retains_statistics(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]

    statistics_file = plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=output_path, report_mode=COMPACT_REPORT
    )

    assert (output_path / "comparison-report.pdf").is_file()
    assert statistics_file.is_file()
    compact_text = statistics_file.read_text(encoding=TEXT_ENCODING)
    compact_payload = json.loads(compact_text)
    assert len(compact_text.splitlines()) < 1000
    assert compact_payload[PLOT_STATISTICS_KEY]
    assert BASELINE_COUNTS_KEY not in compact_text
    assert not list(output_path.glob("*.png"))


def test_no_plot_mode_retains_statistics(tmp_test_directory, monkeypatch):
    output_path = Path(tmp_test_directory)
    metrics = [
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]

    def reject_figure(*_args, **_kwargs):
        raise AssertionError("Statistics-only comparison must not allocate figures")

    monkeypatch.setattr(plt, "subplots", reject_figure)

    statistics_file = plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=output_path, report_mode="none"
    )

    assert statistics_file.is_file()
    assert not list(output_path.glob("*.png"))


def test_report_modes_keep_identical_scalar_statistics(tmp_test_directory):
    root = Path(tmp_test_directory)
    metrics = [
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]
    payloads = []
    for mode in (COMPACT_REPORT, "none"):
        path = plot_production_comparison.plot_event_level_comparison(
            metrics, output_path=root / mode, report_mode=mode
        )
        payloads.append(json.loads(path.read_text(encoding=TEXT_ENCODING)))
    for plot_statistics in payloads[1][PLOT_STATISTICS_KEY].values():
        assert _RENDERING_KEY not in plot_statistics
    for payload in payloads:
        for plot_statistics in payload[PLOT_STATISTICS_KEY].values():
            plot_statistics.pop(_RENDERING_KEY, None)
    assert payloads[0] == payloads[1]


def test_pdf_pages_are_closed_immediately_and_labels_are_retained(tmp_test_directory):
    output = Path(tmp_test_directory) / "streamed.pdf"
    with plot_production_comparison.pdf_report(output) as report:
        for label in (_FIRST, _SECOND):
            figure = plot_production_comparison.plt.figure()
            figure.set_label(label)
            report.add(figure)
            assert not plot_production_comparison.plt.fignum_exists(figure.number)
        assert report.page_labels == [_FIRST, _SECOND]
        assert len(report) == 2
    assert output.is_file()


def test_empty_pdf_report_does_not_create_an_output(tmp_test_directory):
    output = Path(tmp_test_directory) / "unused" / "report.pdf"
    with plot_production_comparison.pdf_report(output) as report:
        assert len(report) == 0
    assert not output.exists()


def test_plot_writes_matplotlib_supported_figure_format(tmp_test_directory):
    metrics = [
        _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0),
        _build_metrics(CANDIDATE, simulated_scale=1.2, triggered_scale=1.1),
    ]

    plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=Path(tmp_test_directory), figure_format=["svg"]
    )

    assert (Path(tmp_test_directory) / "trigger_multiplicity.svg").exists()


def test_comparison_statistics_schema_rejects_unknown_format_version(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metrics = [_build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)]
    plot_production_comparison.plot_event_level_comparison(metrics, output_path=output_path, bins=8)
    with (output_path / "comparison_statistics.json").open(encoding=TEXT_ENCODING) as file_handle:
        statistics = json.load(file_handle)
    comparison_schema = schema.load_schema(
        SCHEMA_PATH / "production_comparison_statistics.schema.yml"
    )
    statistics["format_version"] = 3

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(statistics, comparison_schema)


def test_array_layout_selection_writes_into_layout_directory(tmp_test_directory):
    output_dir = Path(tmp_test_directory) / "plots"
    metrics = [_build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)]

    plot_production_comparison.plot_event_level_comparison(
        metrics, output_path=output_dir, array_layout_name=["CTAO-North Alpha"], bins=8
    )

    assert (output_dir / "CTAO-North Alpha" / "comparison_statistics.json").exists()


def test_array_layout_selection_rejects_multiple_layouts(tmp_test_directory):
    metrics = [_build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)]
    with pytest.raises(ValueError, match="one array layout"):
        plot_production_comparison.plot_event_level_comparison(
            metrics,
            output_path=Path(tmp_test_directory),
            array_layout_name=["CTAO-North Alpha", "MSTN-01"],
        )


def test_plot_writes_per_type_comparison_figures(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    per_type_lstn = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    per_type_mstn = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=0.8)
    metrics = [
        ProductionEventMetrics(
            label=BASELINE,
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

    plot_production_comparison.plot_event_level_comparison(metrics, output_path=output_path, bins=8)

    per_type_files = [
        "trigger_multiplicity_LSTN.png",
        "trigger_multiplicity_MSTN.png",
        "distribution_energy_LSTN.png",
        "distribution_angular_distance_LSTN.png",
        "distribution_angular_distance_cumulative_LSTN.png",
    ]
    assert _assert_files_exist(output_path, per_type_files) is None


def test_artist_color():
    fig, ax = plt.subplots(figsize=(4, 3))
    stairs_artist = ax.stairs(np.array([0.4, 0.6]), np.array([1.0, 2.0, 3.0]))
    assert plot_production_comparison._artist_color(None) == "black"
    assert plot_production_comparison._artist_color(stairs_artist) is not None
    assert plot_production_comparison._artist_color(ax.plot([1, 2], [1, 2])[0]) is not None
    plt.close(fig)


def test_trigger_fraction_and_skip_paths(tmp_test_directory):
    output_path = Path(tmp_test_directory)

    empty_metric = _build_metrics("empty", simulated_scale=1.0, triggered_scale=1.0)
    empty_metric.trigger_multiplicity = np.array([], dtype=int)
    empty_metric.trigger_combinations = Counter()
    empty_metric.telescope_participation = Counter()
    plot_production_comparison._plot_trigger_multiplicity([empty_metric], output_path)
    plot_production_comparison._plot_trigger_combinations([empty_metric], output_path)
    plot_production_comparison._plot_telescope_participation([empty_metric], output_path)
    assert not (output_path / "trigger_multiplicity_empty.png").exists()


def test_single_and_mixed_trigger_skip_paths(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metric = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    metric.trigger_combinations = Counter({"LSTN-01,MSTN-01,SSTS-01": 2})
    plot_production_comparison._plot_single_telescope_trigger_frequencies([metric], output_path)
    plot_production_comparison._plot_mixed_trigger_combinations([metric], output_path)
    assert not (output_path / "single_telescope_trigger_distribution.png").exists()
    assert not (output_path / "mixed_trigger_combinations.png").exists()


def test_mixed_trigger_combination_plot_limits_categories(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    metric = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    metric.trigger_combinations = Counter(
        {
            "LSTN-01,MSTN-01": 10,
            "LSTN-01,MSTN-02": 5,
            "LSTN-02,MSTN-01": 1,
        }
    )

    statistics = plot_production_comparison._plot_mixed_trigger_combinations(
        [metric], output_path, top_n=2
    )

    assert statistics["metadata"]["category_count"] == 3
    assert len(statistics["metadata"]["display_categories"]) == 2
    assert (output_path / "mixed_trigger_combinations.png").exists()


def test_trigger_combination_metric_uses_categories_outside_top_n(tmp_test_directory):
    output_path = Path(tmp_test_directory)
    baseline = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    candidate = _build_metrics(CANDIDATE, simulated_scale=1.0, triggered_scale=1.0)
    baseline.trigger_combinations = Counter({"LSTN-01": 100, "MSTN-01": 1})
    candidate.trigger_combinations = Counter({"LSTN-01": 100, "MSTN-01": 100})

    statistics = plot_production_comparison._plot_trigger_combinations(
        [baseline, candidate], output_path, top_n=1
    )

    comparison = statistics["comparisons"][0]
    assert comparison["metric"] == "jensen_shannon"
    assert comparison["jensen_shannon_distance"] > 0
    assert statistics["metadata"]["categories"] == ["LSTN-01", "MSTN-01"]
    assert comparison[VALUE_KEY] > 0
    assert statistics["metadata"]["category_count"] == 2
    assert statistics["metadata"]["display_categories"] == ["LSTN-01"]


def test_global_quantity_bin_edges_union_histogram_supports():
    baseline = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    candidate = _build_metrics(CANDIDATE, simulated_scale=1.0, triggered_scale=1.0)
    baseline.quantity_histograms = {
        ENERGY: {
            "simulated": (np.array([1.0, 1.0]), np.array([0.0, 1.0, 3.0])),
            "triggered": (np.array([1.0, 1.0]), np.array([0.0, 1.0, 3.0])),
        }
    }
    candidate.quantity_histograms = {
        ENERGY: {
            "simulated": (np.array([1.0, 1.0]), np.array([0.0, 2.0, 3.0])),
            "triggered": (np.array([1.0, 1.0]), np.array([0.0, 2.0, 3.0])),
        }
    }

    edges = plot_production_comparison._get_global_quantity_bin_edges(
        [baseline, candidate], ENERGY, "log", bins=4
    )

    np.testing.assert_array_equal(edges, np.array([0.0, 1.0, 2.0, 3.0]))


def test_quantity_count_rebinning_preserves_distribution_when_edges_split():
    metric = _build_metrics(BASELINE, simulated_scale=1.0, triggered_scale=1.0)
    metric.quantity_histograms = {
        ENERGY: {
            "simulated": (np.array([100.0]), np.array([0.0, 2.0])),
        }
    }

    counts = plot_production_comparison._get_quantity_counts(
        metric,
        ENERGY,
        np.array([0.0, 1.0, 2.0]),
        "simulated",
    )

    np.testing.assert_allclose(counts, np.array([50.0, 50.0]))


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
    statistics_files = plot_production_comparison.plot_signal_level_comparison(
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
    assert statistics["plot_statistics"]["signals"]["metric_type"] == "sample_distribution"
    assert statistics["plot_statistics"]["signals"]["metric"] == "ks"
    assert statistics["plot_statistics"]["triggered_pixels"]["metric"] == "wasserstein"


def test_plot_skips_observable_without_values(tmp_test_directory):
    metrics = [_metrics("baseline"), _metrics("candidate")]
    for item in metrics:
        item.peak_timing = np.array([])

    output_path = Path(tmp_test_directory)
    plot_production_comparison.plot_signal_level_comparison(
        {"LSTN-01": metrics},
        output_path,
        bins=4,
    )

    telescope_path = output_path / "LSTN-01"
    assert not (telescope_path / "peak_timing.png").exists()
    with (telescope_path / "comparison_statistics.json").open(encoding="utf-8") as file_handle:
        statistics = json.load(file_handle)
    assert "peak_timing" not in statistics["plot_statistics"]


@pytest.mark.parametrize(
    "spec",
    [
        ("pedestals", "Pedestal", "pedestals.png", "ks", True),
        ("signals", "Integrated signal", "signals.png", "ks", True),
    ],
)
def test_plot_uses_log_y_axis_for_pedestals_and_signals(mocker, tmp_test_directory, spec):
    figure = mocker.MagicMock()
    axes = mocker.MagicMock()
    mocker.patch(
        "simtools.visualization.plot_production_comparison.plt.subplots",
        return_value=(figure, axes),
    )
    mocker.patch("simtools.visualization.plot_production_comparison.save_figure")

    plot_production_comparison._plot_observable(
        [_metrics("baseline")], Path(tmp_test_directory), spec, 4
    )

    axes.set_yscale.assert_called_once_with("log")


def test_plot_forwards_figure_format(mocker, tmp_test_directory):
    figure = mocker.MagicMock()
    axes = mocker.MagicMock()
    mocker.patch(
        "simtools.visualization.plot_production_comparison.plt.subplots",
        return_value=(figure, axes),
    )
    save_figure = mocker.patch("simtools.visualization.plot_production_comparison.save_figure")

    plot_production_comparison._plot_observable(
        [_metrics("baseline")],
        Path(tmp_test_directory),
        ("signals", "Integrated signal", "signals.png", "ks", True),
        4,
        figure_format=["pdf"],
    )

    save_figure.assert_called_once_with(
        figure,
        Path(tmp_test_directory) / "signals",
        figure_format=["pdf"],
        dpi=300,
        close=True,
        pil_kwargs={"compress_level": 1},
    )


def test_plot_annotates_comparison_statistics(mocker, tmp_test_directory):
    annotate = mocker.patch(
        "simtools.visualization.plot_production_comparison.annotate_comparison_statistics"
    )

    plot_production_comparison._plot_observable(
        [_metrics("baseline"), _metrics("candidate", 1.2)],
        Path(tmp_test_directory),
        ("signals", "Integrated signal", "signals.png", "ks", False),
        4,
    )

    annotate.assert_called_once()
    statistics = annotate.call_args.args[1]
    assert statistics["metric"] == "ks"
    assert statistics["comparisons"][0]["candidate_label"] == "candidate"
