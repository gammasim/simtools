"""Unit tests for nsb_trigger_calculator."""

from unittest.mock import patch

import pytest
from astropy.table import Table

from simtools.simtel import nsb_trigger_calculator

LOG_TEXT = """
Tel. triggered: 10
Run(s) completed as expected after 100 events
Tel. triggered: 20
Run(s) completed as expected after 200 events
"""


def test_extract_threshold_from_file_name_asum_and_dsum():
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name(
            "gamma_run000001_asum220.simtel.log.gz"
        )
        == 220
    )
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name(
            "gamma_run000001_dsum450.simtel.log.gz"
        )
        == 450
    )


def test_extract_threshold_from_file_name_returns_none_when_missing():
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name("gamma_run000001.simtel.log.gz")
        is None
    )


def test_parse_nsb_log_file_returns_parsed_data(tmp_path):
    log_file = tmp_path / "gamma_run000001_asum220.simtel.log"
    log_file.write_text(LOG_TEXT, encoding="utf-8")

    result = nsb_trigger_calculator.parse_nsb_log_file(log_file)

    assert result == {
        "run": 1,
        "threshold": 220,
        "triggers": 20,
        "events": 200,
        "file_path": str(log_file),
    }


def test_parse_nsb_log_file_uses_zero_when_no_triggers_but_events_exist(tmp_path):
    log_file = tmp_path / "gamma_run000001_asum220.simtel.log"
    log_file.write_text("Run(s) completed as expected after 200 events\n", encoding="utf-8")

    result = nsb_trigger_calculator.parse_nsb_log_file(log_file)

    assert result["triggers"] == 0
    assert result["events"] == 200


def test_parse_nsb_log_file_returns_none_when_file_cannot_be_read(tmp_path):
    with patch(
        "simtools.simtel.nsb_trigger_calculator.read_log_file",
        side_effect=OSError("cannot read"),
    ):
        assert (
            nsb_trigger_calculator.parse_nsb_log_file(
                tmp_path / "gamma_run000001_asum220.simtel.log"
            )
            is None
        )


def test_parse_nsb_log_file_returns_none_when_critical_info_is_missing(tmp_path):
    log_file = tmp_path / "gamma_run000001.simtel.log"
    log_file.write_text(LOG_TEXT, encoding="utf-8")

    assert nsb_trigger_calculator.parse_nsb_log_file(log_file) is None


def test_parse_nsb_log_files_filters_failed_parses(tmp_path):
    files = [tmp_path / "good.log", tmp_path / "bad.log"]

    with patch(
        "simtools.simtel.nsb_trigger_calculator.parse_nsb_log_file",
        side_effect=[
            {"run": 1, "threshold": 220, "triggers": 10, "events": 100},
            None,
        ],
    ):
        assert nsb_trigger_calculator.parse_nsb_log_files(files) == [
            {"run": 1, "threshold": 220, "triggers": 10, "events": 100}
        ]


def test_parse_nsb_log_files_raises_when_all_parses_fail(tmp_path):
    with patch("simtools.simtel.nsb_trigger_calculator.parse_nsb_log_file", return_value=None):
        with pytest.raises(ValueError, match="No log files could be parsed successfully"):
            nsb_trigger_calculator.parse_nsb_log_files([tmp_path / "bad.log"])


def test_group_by_threshold_and_run():
    grouped = nsb_trigger_calculator.group_by_threshold_and_run(
        [
            {"threshold": 220, "run": 1, "triggers": 10, "events": 100},
            {"threshold": 220, "run": 2, "triggers": 20, "events": 100},
            {"threshold": 240, "run": 1, "triggers": 5, "events": 50},
        ]
    )

    assert grouped == {
        220: {
            1: {"triggers": 10, "events": 100},
            2: {"triggers": 20, "events": 100},
        },
        240: {
            1: {"triggers": 5, "events": 50},
        },
    }


def test_calculate_statistics_with_multiple_runs():
    stats = nsb_trigger_calculator.calculate_statistics(
        {
            220: {
                1: {"triggers": 10, "events": 100},
                2: {"triggers": 20, "events": 100},
            }
        },
        time_window=0.001,
    )

    threshold_stats = stats[220]
    assert threshold_stats["runs"] == {1: 10, 2: 20}
    assert threshold_stats["total_triggers"] == 30
    assert threshold_stats["total_events"] == 200
    assert threshold_stats["time_s"] == pytest.approx(0.2)
    assert threshold_stats["rate_hz"] == pytest.approx(150.0)
    assert threshold_stats["rate_khz"] == pytest.approx(0.15)
    assert threshold_stats["error_hz"] > 0
    assert threshold_stats["num_runs"] == 2


def test_calculate_statistics_with_no_events_returns_zero_rate():
    stats = nsb_trigger_calculator.calculate_statistics(
        {220: {1: {"triggers": 0, "events": None}}},
        time_window=0.001,
    )

    assert stats[220]["runs"] == {}
    assert stats[220]["total_triggers"] == 0
    assert stats[220]["total_events"] == 0
    assert stats[220]["time_s"] == 0
    assert stats[220]["rate_hz"] == 0
    assert stats[220]["error_hz"] == 0
    assert stats[220]["num_runs"] == 0


def test_calculate_statistics_skips_runs_with_missing_events():
    stats = nsb_trigger_calculator.calculate_statistics(
        {
            220: {
                1: {"triggers": 10, "events": 100},
                2: {"triggers": 20, "events": None},
            }
        },
        time_window=0.001,
    )

    assert stats[220]["runs"] == {1: 10}
    assert stats[220]["total_triggers"] == 10
    assert stats[220]["total_events"] == 100
    assert stats[220]["time_s"] == pytest.approx(0.1)
    assert stats[220]["rate_hz"] == pytest.approx(100.0)
    assert stats[220]["num_runs"] == 1


def test_generate_ecsv_output_writes_table(tmp_path):
    output_file = tmp_path / "nsb_rates.ecsv"
    statistics = {
        220: {
            "runs": {1: 10, 2: 20},
            "total_triggers": 30,
            "total_events": 200,
            "time_s": 0.2,
            "rate_hz": 150.0,
            "rate_khz": 0.15,
            "error_hz": 1.0,
            "num_runs": 2,
        },
        240: {
            "runs": {1: 5},
            "total_triggers": 5,
            "total_events": 100,
            "time_s": 0.1,
            "rate_hz": 50.0,
            "rate_khz": 0.05,
            "error_hz": 0.0,
            "num_runs": 1,
        },
    }

    nsb_trigger_calculator.generate_ecsv_output(statistics, output_file, time_window=0.001)

    table = Table.read(output_file, format="ascii.ecsv")
    assert list(table["threshold"]) == [220, 240]
    assert "run1" in table.colnames
    assert "run2" in table.colnames
    assert table.meta["comments"][0] == "Total events: 300"


def test_generate_ecsv_output_raises_for_empty_statistics(tmp_path):
    with pytest.raises(ValueError, match="No statistics to write"):
        nsb_trigger_calculator.generate_ecsv_output({}, tmp_path / "empty.ecsv", 0.001)


def test_derive_nsb_triggers_runs_full_pipeline_without_output(tmp_path):
    with (
        patch("simtools.simtel.nsb_trigger_calculator.crawl_log_files", return_value=["log1"]),
        patch(
            "simtools.simtel.nsb_trigger_calculator.parse_nsb_log_files",
            return_value=[{"threshold": 220, "run": 1, "triggers": 10, "events": 100}],
        ),
        patch(
            "simtools.simtel.nsb_trigger_calculator.group_by_threshold_and_run",
            return_value={220: {1: {"triggers": 10, "events": 100}}},
        ),
        patch(
            "simtools.simtel.nsb_trigger_calculator.calculate_statistics",
            return_value={
                220: {
                    "rate_hz": 100.0,
                    "rate_khz": 0.1,
                    "error_hz": 0.0,
                    "total_triggers": 10,
                    "num_runs": 1,
                }
            },
        ),
        patch("simtools.simtel.nsb_trigger_calculator.generate_ecsv_output") as mock_generate,
    ):
        stats = nsb_trigger_calculator.derive_nsb_triggers(
            {"root_dir": tmp_path, "time_window": 0.001}
        )

    assert stats[220]["rate_hz"] == 100
    mock_generate.assert_not_called()


def test_derive_nsb_triggers_writes_output_when_requested(tmp_path):
    output_file = tmp_path / "rates.ecsv"

    with (
        patch("simtools.simtel.nsb_trigger_calculator.crawl_log_files", return_value=["log1"]),
        patch(
            "simtools.simtel.nsb_trigger_calculator.parse_nsb_log_files",
            return_value=[{"threshold": 220, "run": 1, "triggers": 10, "events": 100}],
        ),
        patch(
            "simtools.simtel.nsb_trigger_calculator.group_by_threshold_and_run",
            return_value={220: {1: {"triggers": 10, "events": 100}}},
        ),
        patch(
            "simtools.simtel.nsb_trigger_calculator.calculate_statistics",
            return_value={
                220: {
                    "rate_hz": 100.0,
                    "rate_khz": 0.1,
                    "error_hz": 0.0,
                    "total_triggers": 10,
                    "num_runs": 1,
                }
            },
        ),
        patch("simtools.simtel.nsb_trigger_calculator.generate_ecsv_output") as mock_generate,
    ):
        nsb_trigger_calculator.derive_nsb_triggers(
            {"root_dir": tmp_path, "time_window": 0.001, "output": output_file}
        )

    mock_generate.assert_called_once()


def test_derive_nsb_triggers_raises_for_missing_time_window(tmp_path):
    with pytest.raises(ValueError, match="Missing required argument 'time_window'"):
        nsb_trigger_calculator.derive_nsb_triggers({"root_dir": tmp_path})


@pytest.mark.parametrize("time_window", [0, -1e-9])
def test_derive_nsb_triggers_raises_for_non_positive_time_window(tmp_path, time_window):
    with pytest.raises(ValueError, match="Argument 'time_window' must be > 0"):
        nsb_trigger_calculator.derive_nsb_triggers(
            {"root_dir": tmp_path, "time_window": time_window}
        )


def test_derive_nsb_triggers_raises_for_non_numeric_time_window(tmp_path):
    with pytest.raises(ValueError, match="must be a positive number"):
        nsb_trigger_calculator.derive_nsb_triggers(
            {"root_dir": tmp_path, "time_window": "not-a-number"}
        )
