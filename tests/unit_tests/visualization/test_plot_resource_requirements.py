"""Tests for resource requirement plots."""

import pytest

from simtools.visualization import plot_resource_requirements


def test_plot_writes_available_resource_figures(tmp_test_directory):
    rows = [
        {
            "role": "corsika",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 1.0,
            "cpu_time_seconds_per_event": 0.5,
            "peak_rss_bytes": 1_000_000.0,
            "sim_telarray_storage_bytes_per_event": None,
        },
        {
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 1000.0,
            "wall_time_seconds_per_event": 2.0,
            "cpu_time_seconds_per_event": 1.5,
            "peak_rss_bytes": 2_000_000.0,
            "sim_telarray_storage_bytes_per_event": 1_000_000.0,
        },
    ]

    output_files = plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert len(output_files) == 7
    assert {output_file.name for output_file in output_files} == {
        "resource_wall_time_corsika",
        "resource_wall_time_sim_telarray",
        "resource_cpu_time_corsika",
        "resource_cpu_time_sim_telarray",
        "resource_peak_rss_corsika",
        "resource_peak_rss_sim_telarray",
        "resource_storage_sim_telarray",
    }
    assert all(output_file.with_suffix(".png").is_file() for output_file in output_files)


def test_plot_uses_linear_y_axis_for_nonpositive_values(mocker, tmp_test_directory):
    rows = [
        {
            "role": "corsika",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 0.0,
            "cpu_time_seconds_per_event": 1.0,
            "peak_rss_bytes": 1_000_000.0,
        }
    ]
    axis = mocker.Mock()
    figure = mocker.Mock()
    mocker.patch.object(plot_resource_requirements.plt, "subplots", return_value=(figure, axis))
    mocker.patch.object(plot_resource_requirements, "save_figure")

    plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert axis.set_yscale.call_args_list[0].args == ("linear",)


def test_plot_writes_individual_output_size_figures(tmp_test_directory):
    common = {
        "zenith_angle_deg": 20.0,
        "energy_midpoint_gev": 100.0,
        "wall_time_seconds_per_event": None,
        "cpu_time_seconds_per_event": None,
        "peak_rss_bytes": None,
        "sim_telarray_storage_bytes_per_event": None,
    }
    rows = [
        {
            **common,
            "role": "corsika",
            "corsika_output_bytes_per_event": 1.0,
            "sim_telarray_output_bytes_per_event": None,
            "reduced_event_data_bytes_per_event": None,
            "sim_telarray_histogram_bytes_per_event": None,
        },
        {
            **common,
            "role": "sim_telarray",
            "corsika_output_bytes_per_event": None,
            "sim_telarray_output_bytes_per_event": 2.0,
            "reduced_event_data_bytes_per_event": 3.0,
            "sim_telarray_histogram_bytes_per_event": 4.0,
        },
    ]

    output_files = plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert {output_file.name for output_file in output_files} == {
        "resource_corsika_output_corsika",
        "resource_sim_telarray_output_sim_telarray",
        "resource_reduced_event_data_sim_telarray",
        "resource_sim_telarray_histogram_sim_telarray",
    }


def test_plot_separates_roles_and_only_draws_averages(mocker, tmp_test_directory):
    rows = [
        {
            "production_label": "baseline",
            "simtools_version": "0.37.1",
            "role": "corsika",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 1.0,
            "cpu_time_seconds_per_event": 0.5,
            "peak_rss_bytes": 1000.0,
            "sim_telarray_storage_bytes_per_event": None,
        },
        {
            "production_label": "baseline",
            "simtools_version": "0.37.1",
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 2.0,
            "cpu_time_seconds_per_event": 1.5,
            "peak_rss_bytes": 2000.0,
            "sim_telarray_storage_bytes_per_event": None,
        },
    ]
    axis = mocker.Mock()
    figure = mocker.Mock()
    subplots = mocker.patch.object(
        plot_resource_requirements.plt, "subplots", return_value=(figure, axis)
    )
    mocker.patch.object(plot_resource_requirements, "save_figure")

    plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert not axis.scatter.called
    errorbar_calls = axis.errorbar.call_args_list
    assert [call.kwargs["label"] for call in errorbar_calls] == ["za=20 deg"] * 6
    assert [call.kwargs["fmt"] for call in errorbar_calls] == ["o", "s"] * 3
    assert subplots.call_count == 6
    titles = [call.args[0] for call in axis.set_title.call_args_list]
    assert all("baseline" not in title for title in titles)
    assert all("model" not in title for title in titles)
    assert all("simtools" not in title for title in titles)


def test_plot_adds_energy_group_means_with_rms_error_bars(mocker, tmp_test_directory):
    rows = [
        {
            "production_label": "baseline",
            "simtools_version": "0.37.1",
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 1.0,
            "cpu_time_seconds_per_event": 1.0,
            "peak_rss_bytes": 1000.0,
            "sim_telarray_storage_bytes_per_event": 10.0,
        },
        {
            "production_label": "baseline",
            "simtools_version": "0.37.1",
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": 3.0,
            "cpu_time_seconds_per_event": 3.0,
            "peak_rss_bytes": 3000.0,
            "sim_telarray_storage_bytes_per_event": 30.0,
        },
    ]
    axis = mocker.Mock()
    figure = mocker.Mock()
    mocker.patch.object(plot_resource_requirements.plt, "subplots", return_value=(figure, axis))
    mocker.patch.object(plot_resource_requirements, "save_figure")

    plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    first_average = axis.errorbar.call_args_list[0]
    assert first_average.args[:2] == ([100.0], [2.0])
    assert first_average.kwargs["yerr"] == [1.0]


def test_plot_converts_peak_rss_to_megabytes(mocker, tmp_test_directory):
    rows = [
        {
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": None,
            "cpu_time_seconds_per_event": None,
            "peak_rss_bytes": 2_000_000.0,
            "sim_telarray_storage_bytes_per_event": None,
        }
    ]
    axis = mocker.Mock()
    figure = mocker.Mock()
    mocker.patch.object(plot_resource_requirements.plt, "subplots", return_value=(figure, axis))
    mocker.patch.object(plot_resource_requirements, "save_figure")

    plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    memory_call = axis.errorbar.call_args_list[0]
    assert memory_call.args[:2] == ([100.0], [2.0])
    assert axis.set_ylabel.call_args.args[0] == "Peak RSS (MB)"


def test_byte_plot_label_uses_gigabytes_for_large_values():
    label, scale = plot_resource_requirements._byte_plot_label(
        "sim_telarray_output_bytes_per_event",
        "sim_telarray output",
        [{"sim_telarray_output_bytes_per_event": 2_000_000_000.0}],
    )

    assert label == "sim_telarray output (GB/event)"
    assert scale == pytest.approx(1e-9)


def test_byte_plot_label_adds_triggered_event_unit():
    label, scale = plot_resource_requirements._byte_plot_label(
        "sim_telarray_storage_bytes_per_triggered_event",
        "sim_telarray storage",
        [{"sim_telarray_storage_bytes_per_triggered_event": 2_000_000.0}],
    )

    assert label == "sim_telarray storage (MB/triggered event)"
    assert scale == pytest.approx(1e-6)


def test_plot_writes_ratio_figures_for_baseline_and_candidate(mocker, tmp_test_directory):
    common = {
        "role": "sim_telarray",
        "zenith_angle_deg": 20.0,
        "energy_midpoint_gev": 100.0,
        "wall_time_seconds_per_event": 2.0,
        "cpu_time_seconds_per_event": 1.0,
        "peak_rss_bytes": 2_000_000.0,
        "sim_telarray_storage_bytes_per_event": 1_000_000.0,
    }
    rows = [
        {**common, "production_label": "reference", "comparison_role": "baseline"},
        {
            **common,
            "production_label": "optimized",
            "comparison_role": "candidate",
            "wall_time_seconds_per_event": 3.0,
            "cpu_time_seconds_per_event": 1.5,
            "peak_rss_bytes": 3_000_000.0,
            "sim_telarray_storage_bytes_per_event": 1_500_000.0,
        },
    ]
    axis = mocker.Mock()
    figure = mocker.Mock()
    mocker.patch.object(plot_resource_requirements.plt, "subplots", return_value=(figure, axis))
    mocker.patch.object(plot_resource_requirements, "save_figure")

    output_files = plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert {path.name for path in output_files} == {
        "resource_wall_time_sim_telarray",
        "resource_wall_time_ratio_sim_telarray",
        "resource_cpu_time_sim_telarray",
        "resource_cpu_time_ratio_sim_telarray",
        "resource_peak_rss_sim_telarray",
        "resource_peak_rss_ratio_sim_telarray",
        "resource_storage_sim_telarray",
        "resource_storage_ratio_sim_telarray",
    }
    assert axis.axhline.call_count == 4
    assert [call.kwargs["label"] for call in axis.errorbar.call_args_list] == [
        "_nolegend_",
        "za=20 deg",
        "za=20 deg",
    ] * 4
    assert all(
        call.args[0] == "optimized / reference" for call in axis.set_ylabel.call_args_list[1::2]
    )
    assert all("ratio:" in call.args[0] for call in axis.set_title.call_args_list[1::2])


def test_ratio_series_propagates_independent_rms_errors():
    rows = [
        {
            "production_label": "baseline",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": value,
        }
        for value in (2.0, 4.0)
    ] + [
        {
            "production_label": "candidate",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_event": value,
        }
        for value in (3.0, 5.0)
    ]

    series = plot_resource_requirements._ratio_series(rows, "wall_time_seconds_per_event")

    _, ratio, ratio_error = series[20.0][0]
    expected_ratio = 4.0 / 3.0
    expected_error = expected_ratio * ((1 / 4 / 2**0.5) ** 2 + (1 / 3 / 2**0.5) ** 2) ** 0.5
    assert ratio == pytest.approx(expected_ratio)
    assert ratio_error == pytest.approx(expected_error)


def test_plot_writes_trigger_normalized_simtel_figures(tmp_test_directory):
    rows = [
        {
            "role": "sim_telarray",
            "zenith_angle_deg": 20.0,
            "energy_midpoint_gev": 100.0,
            "wall_time_seconds_per_triggered_event": 2.0,
            "cpu_time_seconds_per_triggered_event": 1.0,
            "sim_telarray_storage_bytes_per_triggered_event": 1_000_000.0,
            "sim_telarray_output_bytes_per_triggered_event": 1_000_000.0,
            "reduced_event_data_bytes_per_triggered_event": 1_000_000.0,
            "sim_telarray_histogram_bytes_per_triggered_event": 1_000_000.0,
        }
    ]

    output_files = plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    assert {output_file.name for output_file in output_files} == {
        "resource_wall_time_triggered_sim_telarray",
        "resource_cpu_time_triggered_sim_telarray",
        "resource_storage_triggered_sim_telarray",
        "resource_sim_telarray_output_triggered_sim_telarray",
        "resource_reduced_event_data_triggered_sim_telarray",
        "resource_sim_telarray_histogram_triggered_sim_telarray",
    }


def test_plot_uses_zenith_colors_and_separate_averages(mocker, tmp_test_directory):
    rows = []
    for zenith, value in ((20.0, 1.0), (70.0, 3.0)):
        rows.append(
            {
                "production_label": "baseline",
                "simtools_version": "0.37.1",
                "role": "corsika",
                "zenith_angle_deg": zenith,
                "energy_midpoint_gev": 100.0,
                "wall_time_seconds_per_event": value,
                "cpu_time_seconds_per_event": value,
                "peak_rss_bytes": value,
                "sim_telarray_storage_bytes_per_event": None,
            }
        )
    axis = mocker.Mock()
    figure = mocker.Mock()
    mocker.patch.object(plot_resource_requirements.plt, "subplots", return_value=(figure, axis))
    mocker.patch.object(plot_resource_requirements, "save_figure")

    plot_resource_requirements.plot(rows, tmp_test_directory, figure_format=["png"])

    average_calls = axis.errorbar.call_args_list
    assert len(average_calls) == 6
    assert all(call.kwargs["yerr"] == [0.0] for call in average_calls)
    assert average_calls[0].kwargs["color"] != average_calls[1].kwargs["color"]
