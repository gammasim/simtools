"""Unit tests for nsb_trigger_calculator."""

from unittest.mock import patch

import numpy as np
import pytest
from astropy.table import Table

from simtools.simtel import nsb_trigger_calculator


def _write_nsb_hdf5(file_path, n_showers=0, n_triggers=None, threshold=None):
    """Create a minimal reduced event-data HDF5 file for NSB calculator tests."""
    showers_table = Table({"event_id": list(range(n_showers))})
    showers_table.write(file_path, path="SHOWERS", format="hdf5", overwrite=True)

    if n_triggers is not None:
        triggers_table = Table({"event_id": list(range(n_triggers))})
        triggers_table.write(file_path, path="TRIGGERS", format="hdf5", append=True)

    if threshold is not None:
        file_info_table = Table({"asum_threshold": [threshold]})
        file_info_table.write(file_path, path="FILE_INFO", format="hdf5", append=True)


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("", None),
        ("  ", None),
        ("245", 245),
        ("245.9", 245),
        ("asum250", 250),
        (b"260", 260),
        (270, 270),
        (270.9, 270),
        (np.nan, None),
        (np.inf, None),
        (object(), None),
    ],
)
def test_coerce_threshold_value_covers_string_bytes_and_numeric_paths(value, expected):
    assert nsb_trigger_calculator._coerce_threshold_value(value) == expected


def test_extract_run_number_from_parent_directory_parts(tmp_path):
    nested = tmp_path / "run000123" / "subdir" / "file.reduced_event_data.hdf5"
    assert nsb_trigger_calculator._extract_run_number(nested) == 123


def test_extract_run_number_returns_none_when_missing(tmp_path):
    missing = tmp_path / "subdir" / "file.reduced_event_data.hdf5"
    assert nsb_trigger_calculator._extract_run_number(missing) is None


def test_extract_threshold_from_file_name_asum_and_dsum():
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name(
            "gamma_run000001_asum220.reduced_event_data.hdf5"
        )
        == 220
    )
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name(
            "gamma_run000001_dsum450.reduced_event_data.hdf5"
        )
        == 450
    )


def test_extract_threshold_from_file_name_returns_none_when_missing():
    assert (
        nsb_trigger_calculator.extract_threshold_from_file_name(
            "gamma_run000001.reduced_event_data.hdf5"
        )
        is None
    )


def test_extract_threshold_from_hdf5_metadata_reads_threshold(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=10, n_triggers=2, threshold=250)

    assert nsb_trigger_calculator.extract_threshold_from_hdf5_metadata(hdf5_file) == 250


def test_extract_threshold_from_hdf5_metadata_returns_none_when_missing_file_info(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=10, n_triggers=2, threshold=None)

    assert nsb_trigger_calculator.extract_threshold_from_hdf5_metadata(hdf5_file) is None


def test_extract_threshold_from_hdf5_metadata_returns_none_for_empty_file_info(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=10, n_triggers=2, threshold=None)

    Table({"asum_threshold": []}).write(hdf5_file, path="FILE_INFO", format="hdf5", overwrite=True)

    assert nsb_trigger_calculator.extract_threshold_from_hdf5_metadata(hdf5_file) is None


def test_extract_threshold_from_hdf5_metadata_uses_first_known_threshold_column(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=10, n_triggers=2, threshold=None)

    file_info_table = Table(
        {
            "trigger_threshold": [" 300 "],
            "asum_threshold": [250],
            "dsum_threshold": [200],
        }
    )
    file_info_table.write(hdf5_file, path="FILE_INFO", format="hdf5", overwrite=True)

    assert nsb_trigger_calculator.extract_threshold_from_hdf5_metadata(hdf5_file) == 300


def test_parse_nsb_hdf5_file_returns_parsed_data(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=200, n_triggers=20)

    result = nsb_trigger_calculator.parse_nsb_hdf5_file(hdf5_file)

    assert result == {
        "run": 1,
        "threshold": 220,
        "triggers": 20,
        "events": 200,
        "file_path": str(hdf5_file),
    }


def test_parse_nsb_hdf5_file_uses_metadata_threshold_before_filename(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=100, n_triggers=10, threshold=245)

    result = nsb_trigger_calculator.parse_nsb_hdf5_file(hdf5_file)

    assert result["threshold"] == 245


def test_parse_nsb_hdf5_file_uses_zero_when_triggers_table_missing(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=200, n_triggers=None)

    result = nsb_trigger_calculator.parse_nsb_hdf5_file(hdf5_file)

    assert result["triggers"] == 0
    assert result["events"] == 200


def test_parse_nsb_hdf5_file_returns_none_when_events_table_missing(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5"
    Table({"event_id": [1]}).write(hdf5_file, path="TRIGGERS", format="hdf5", overwrite=True)

    assert nsb_trigger_calculator.parse_nsb_hdf5_file(hdf5_file) is None


def test_parse_nsb_hdf5_file_returns_none_when_critical_info_is_missing(tmp_path):
    hdf5_file = tmp_path / "gamma_run000001.reduced_event_data.hdf5"
    _write_nsb_hdf5(hdf5_file, n_showers=100, n_triggers=10)

    assert nsb_trigger_calculator.parse_nsb_hdf5_file(hdf5_file) is None


def test_parse_nsb_hdf5_files_filters_failed_parses(tmp_path):
    files = [tmp_path / "good.hdf5", tmp_path / "bad.hdf5"]

    with patch(
        "simtools.simtel.nsb_trigger_calculator.parse_nsb_hdf5_file",
        side_effect=[
            {"run": 1, "threshold": 220, "triggers": 10, "events": 100},
            None,
        ],
    ):
        assert nsb_trigger_calculator.parse_nsb_hdf5_files(files) == [
            {"run": 1, "threshold": 220, "triggers": 10, "events": 100}
        ]


def test_parse_nsb_hdf5_files_raises_when_all_parses_fail(tmp_path):
    with patch("simtools.simtel.nsb_trigger_calculator.parse_nsb_hdf5_file", return_value=None):
        with pytest.raises(ValueError, match="No HDF5 files could be parsed successfully"):
            nsb_trigger_calculator.parse_nsb_hdf5_files([tmp_path / "bad.hdf5"])


def test_find_hdf5_files_raises_for_missing_root(tmp_path):
    with pytest.raises(FileNotFoundError, match="Root directory not found"):
        nsb_trigger_calculator.find_hdf5_files(tmp_path / "missing")


def test_find_hdf5_files_raises_for_missing_matches(tmp_path):
    with pytest.raises(FileNotFoundError, match="No HDF5 files found"):
        nsb_trigger_calculator.find_hdf5_files(tmp_path)


def test_find_hdf5_files_returns_sorted_matches(tmp_path):
    first = tmp_path / "gamma_run000001_asum220.reduced_event_data.hdf5"
    second = tmp_path / "gamma_run000002_asum220.reduced_event_data.hdf5"
    second.touch()
    first.touch()

    matches = nsb_trigger_calculator.find_hdf5_files(tmp_path)
    assert matches == [first, second]


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
    assert threshold_stats["error_hz"] == pytest.approx(50.0)
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


@pytest.mark.parametrize("write_output", [False, True])
def test_derive_nsb_triggers_pipeline_output_toggle(tmp_path, write_output):
    output_file = tmp_path / "rates.ecsv"
    args = {"root_dir": tmp_path, "time_window": 0.001}
    if write_output:
        args["output"] = output_file

    with (
        patch("simtools.simtel.nsb_trigger_calculator.find_hdf5_files", return_value=["f1"]),
        patch(
            "simtools.simtel.nsb_trigger_calculator.parse_nsb_hdf5_files",
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
        stats = nsb_trigger_calculator.derive_nsb_triggers(args)

    assert stats[220]["rate_hz"] == 100
    if write_output:
        mock_generate.assert_called_once()
    else:
        mock_generate.assert_not_called()


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
    with pytest.raises(ValueError, match="must be a number"):
        nsb_trigger_calculator.derive_nsb_triggers(
            {"root_dir": tmp_path, "time_window": "not-a-number"}
        )
