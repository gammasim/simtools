"""Unit tests for bias_curve_generator."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from astropy import units as u
from astropy.table import Table

from simtools.simtel import bias_curve_generator


def _base_args(tmp_path):
    return {
        "data_dir": tmp_path,
        "output": tmp_path / "bias_curve.png",
        "site": "North",
        "model_version": "7.0.0",
        "telescope": "LSTN-01",
        "title": "Bias curve",
        "ymin": 1,
        "ymax": 1e6,
    }


def test_calculate_time_window_requires_telescope(tmp_path):
    args = _base_args(tmp_path)
    args["telescope"] = ""

    with pytest.raises(ValueError, match="telescope must be provided"):
        bias_curve_generator._calculate_time_window(args)


def test_calculate_time_window_reads_telescope_model(tmp_path):
    args = _base_args(tmp_path)

    telescope_model = MagicMock()
    telescope_model.get_parameter_value.side_effect = [68, 1024]

    with (
        patch("simtools.simtel.bias_curve_generator.TelescopeModel", return_value=telescope_model),
    ):
        time_window = bias_curve_generator._calculate_time_window(args)

    assert time_window == pytest.approx(68 / (1024 * 1e6))


def test_extract_nsb_rates_uses_direct_logs(tmp_path):
    args = _base_args(tmp_path)
    log_file = tmp_path / "gamma_run000001_asum220.simtel.log.gz"
    log_file.touch()

    with patch(
        "simtools.simtel.bias_curve_generator._run_nsb_trigger_derivation",
        return_value={220: {"rate_hz": 10.0}},
    ) as mock_run:
        result = bias_curve_generator._extract_nsb_rates(args, time_window=0.001)

    assert result == {220: {"rate_hz": 10.0}}
    mock_run.assert_called_once_with(tmp_path, args, 0.001)


def test_extract_nsb_rates_raises_when_no_logs(tmp_path):
    args = _base_args(tmp_path)

    with pytest.raises(FileNotFoundError, match=r"No \*\.simtel\.log\.gz files found"):
        bias_curve_generator._extract_nsb_rates(args, time_window=0.001)


def test_group_hdf5_files_by_threshold_and_run(tmp_path):
    valid = tmp_path / "proton_run000001_asum220.reduced_event_data.hdf5"
    valid.touch()
    (tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5").touch()
    (tmp_path / "proton_missing_threshold.reduced_event_data.hdf5").touch()

    grouped = bias_curve_generator._group_hdf5_files_by_threshold_and_run(tmp_path)

    assert grouped == {220: {1: valid}}


def test_extract_proton_rates_raises_when_no_files(tmp_path):
    args = _base_args(tmp_path)

    with patch(
        "simtools.simtel.bias_curve_generator._group_hdf5_files_by_threshold_and_run",
        return_value={},
    ):
        with pytest.raises(
            FileNotFoundError,
            match="No proton HDF5 files with threshold labels found",
        ):
            bias_curve_generator._extract_proton_rates(args)


def test_extract_proton_rates_calculates_statistics_per_threshold(tmp_path):
    args = _base_args(tmp_path)
    grouped_files = {
        240: {2: tmp_path / "run2.hdf5"},
        220: {1: tmp_path / "run1.hdf5", 3: tmp_path / "run3.hdf5"},
    }

    with (
        patch(
            "simtools.simtel.bias_curve_generator._group_hdf5_files_by_threshold_and_run",
            return_value=grouped_files,
        ),
        patch(
            "simtools.simtel.bias_curve_generator._calculate_proton_statistics_for_threshold",
            side_effect=[
                {"runs": {1: 10.0, 3: 12.0}, "rate_hz": 11.0, "error_hz": 1.0, "num_runs": 2},
                {"runs": {2: 5.0}, "rate_hz": 5.0, "error_hz": 0.0, "num_runs": 1},
            ],
        ) as mock_calc,
    ):
        stats = bias_curve_generator._extract_proton_rates(args)

    assert stats == {
        220: {"runs": {1: 10.0, 3: 12.0}, "rate_hz": 11.0, "error_hz": 1.0, "num_runs": 2},
        240: {"runs": {2: 5.0}, "rate_hz": 5.0, "error_hz": 0.0, "num_runs": 1},
    }
    assert mock_calc.call_count == 2
    assert mock_calc.call_args_list[0].args == (grouped_files[220], args)
    assert mock_calc.call_args_list[1].args == (grouped_files[240], args)


def test_calculate_proton_statistics_for_threshold_uses_non_none_rates(tmp_path):
    files = {1: tmp_path / "run1.hdf5", 2: tmp_path / "run2.hdf5", 3: tmp_path / "run3.hdf5"}

    with patch(
        "simtools.simtel.bias_curve_generator._calculate_proton_rate_for_file",
        side_effect=[10.0, None, 20.0],
    ):
        stats = bias_curve_generator._calculate_proton_statistics_for_threshold(files, {})

    assert stats["runs"] == {1: 10.0, 3: 20.0}
    assert stats["rate_hz"] == pytest.approx(15.0)
    assert stats["error_hz"] > 0
    assert stats["num_runs"] == 2


def test_calculate_proton_statistics_for_threshold_returns_nan_when_no_rates(tmp_path):
    with patch(
        "simtools.simtel.bias_curve_generator._calculate_proton_rate_for_file",
        return_value=None,
    ):
        stats = bias_curve_generator._calculate_proton_statistics_for_threshold(
            {1: tmp_path / "run1.hdf5"}, {}
        )

    assert stats["runs"] == {}
    assert np.isnan(stats["rate_hz"])
    assert np.isnan(stats["error_hz"])
    assert stats["num_runs"] == 0


def test_calculate_proton_rate_for_file_with_array_layout(tmp_path):
    args = _base_args(tmp_path)

    with patch(
        "simtools.simtel.bias_curve_generator.telescope_trigger_rates",
        return_value={"array": 5 * u.Hz},
    ):
        assert (
            bias_curve_generator._calculate_proton_rate_for_file(tmp_path / "events.hdf5", args)
            == 5
        )


def test_calculate_proton_rate_for_file_with_telescope_ids(tmp_path):
    args = {"telescope": "LSTN-01", "site": "North", "model_version": "7.0.0"}

    with patch(
        "simtools.simtel.bias_curve_generator.telescope_trigger_rates",
        return_value={"array": 2 * u.Hz},
    ):
        assert (
            bias_curve_generator._calculate_proton_rate_for_file(tmp_path / "events.hdf5", args)
            == 2
        )


def test_calculate_proton_rate_for_file_returns_zero_for_missing_trigger_histograms(tmp_path):
    args = _base_args(tmp_path)

    with patch(
        "simtools.simtel.bias_curve_generator.telescope_trigger_rates",
        side_effect=TypeError("'NoneType' object is not subscriptable"),
    ):
        assert (
            bias_curve_generator._calculate_proton_rate_for_file(tmp_path / "events.hdf5", args)
            == 0
        )


def test_calculate_proton_rate_for_file_returns_none_for_other_errors(tmp_path):
    args = _base_args(tmp_path)

    with patch(
        "simtools.simtel.bias_curve_generator.telescope_trigger_rates",
        side_effect=ValueError("bad file"),
    ):
        assert (
            bias_curve_generator._calculate_proton_rate_for_file(tmp_path / "events.hdf5", args)
            is None
        )


def test_calculate_proton_rate_for_file_returns_none_without_telescope_config(tmp_path):
    assert (
        bias_curve_generator._calculate_proton_rate_for_file(tmp_path / "events.hdf5", {}) is None
    )


def test_write_proton_ecsv_writes_table(tmp_path):
    output_file = tmp_path / "proton.ecsv"
    proton_stats = {
        220: {
            "runs": {1: 10.0, 2: 20.0},
            "rate_hz": 15.126,
            "error_hz": 5.889,
            "num_runs": 2,
        },
        240: {"runs": {1: 5.0}, "rate_hz": 5.0, "error_hz": 0.0, "num_runs": 1},
    }

    bias_curve_generator._write_proton_ecsv(proton_stats, output_file)

    table = Table.read(output_file, format="ascii.ecsv")
    assert list(table["threshold"]) == [220, 240]
    assert "run1" in table.colnames
    assert "run2" in table.colnames
    assert table["Rate (Hz)"][0] == pytest.approx(15.13)
    assert table["Error (Hz)"][0] == pytest.approx(5.89)

    output_text = output_file.read_text(encoding="utf-8")
    assert "220 10.0 20.0 15.13 5.89 2" in output_text


def test_write_proton_ecsv_raises_for_empty_stats(tmp_path):
    with pytest.raises(ValueError, match="No proton statistics to write"):
        bias_curve_generator._write_proton_ecsv({}, tmp_path / "empty.ecsv")


def test_write_bias_curve_ecsv_writes_combined_table(tmp_path):
    output_file = tmp_path / "bias.ecsv"

    bias_curve_generator._write_bias_curve_ecsv(
        nsb_stats={220: {"rate_hz": 100.125}},
        proton_stats={220: {"rate_hz": 5.987}, 240: {"rate_hz": 7.0}},
        output_file=output_file,
    )

    table = Table.read(output_file, format="ascii.ecsv")
    assert list(table["threshold"]) == [220, 240]
    assert table["NSB rate (Hz)"][0] == pytest.approx(100.12)
    assert np.isnan(table["NSB rate (Hz)"][1])
    assert table["Proton rate (Hz)"][0] == pytest.approx(5.99)
    assert table["Proton rate (Hz)"][1] == pytest.approx(7.0)

    output_text = output_file.read_text(encoding="utf-8")
    assert "220 100.12 5.99" in output_text


def test_generate_bias_curves_runs_full_pipeline(tmp_path):
    args = _base_args(tmp_path)
    args["proton_output"] = tmp_path / "proton.ecsv"
    args["nsb_output"] = tmp_path / "nsb.ecsv"

    with (
        patch("simtools.simtel.bias_curve_generator._calculate_time_window", return_value=0.001),
        patch(
            "simtools.simtel.bias_curve_generator._extract_nsb_rates",
            return_value={220: {"rate_hz": 100.0}},
        ),
        patch(
            "simtools.simtel.bias_curve_generator._extract_proton_rates",
            return_value={
                220: {"runs": {1: 10.0}, "rate_hz": 10.0, "error_hz": 0.0, "num_runs": 1}
            },
        ),
        patch("simtools.simtel.bias_curve_generator._write_proton_ecsv") as mock_write_proton,
        patch(
            "simtools.simtel.bias_curve_generator.plot_tables.resolve_plot_output_path",
            return_value=tmp_path / "bias.png",
        ),
        patch("simtools.simtel.bias_curve_generator.plot_tables.plot_bias_curves") as mock_plot,
        patch("simtools.simtel.bias_curve_generator._write_bias_curve_ecsv") as mock_write_bias,
    ):
        bias_curve_generator.generate_bias_curves(args)

    mock_write_proton.assert_called_once()
    mock_plot.assert_called_once()
    mock_write_bias.assert_called_once()


def test_generate_bias_curves_raises_when_no_nsb_inputs_exist(tmp_path):
    args = _base_args(tmp_path)
    args["nsb_output"] = tmp_path / "nsb.ecsv"

    with (
        patch("simtools.simtel.bias_curve_generator._calculate_time_window", return_value=0.001),
        patch("simtools.simtel.bias_curve_generator._extract_nsb_rates", return_value={}),
        patch("simtools.simtel.bias_curve_generator._extract_proton_rates") as mock_extract_proton,
        patch(
            "simtools.simtel.bias_curve_generator.plot_tables.resolve_plot_output_path",
            return_value=tmp_path / "bias.png",
        ),
        patch("simtools.simtel.bias_curve_generator.plot_tables.plot_bias_curves") as mock_plot,
        patch("simtools.simtel.bias_curve_generator._write_bias_curve_ecsv") as mock_write_bias,
    ):
        with pytest.raises(FileNotFoundError, match="No NSB input files found"):
            bias_curve_generator.generate_bias_curves(args)

    mock_extract_proton.assert_not_called()
    mock_plot.assert_not_called()
    mock_write_bias.assert_not_called()


def test_generate_bias_curves_raises_when_no_proton_inputs_exist(tmp_path):
    args = _base_args(tmp_path)

    with (
        patch("simtools.simtel.bias_curve_generator._calculate_time_window", return_value=0.001),
        patch(
            "simtools.simtel.bias_curve_generator._extract_nsb_rates",
            return_value={220: {"rate_hz": 100.0}},
        ),
        patch("simtools.simtel.bias_curve_generator._extract_proton_rates", return_value={}),
        patch("simtools.simtel.bias_curve_generator.plot_tables.plot_bias_curves") as mock_plot,
        patch("simtools.simtel.bias_curve_generator._write_bias_curve_ecsv") as mock_write_bias,
    ):
        with pytest.raises(FileNotFoundError, match="No proton input files found"):
            bias_curve_generator.generate_bias_curves(args)

    mock_plot.assert_not_called()
    mock_write_bias.assert_not_called()
