#!/usr/bin/python3

import pytest

import simtools.simtel.simtel_table_writer as simtel_table_writer


def test_write_simtel_table_two_columns(tmp_test_directory):
    value = {
        "columns": ["time", "amplitude"],
        "rows": [[-1.0, 0.0], [0.0, 0.5], [1.0, 1.0]],
    }
    result = simtel_table_writer.write_simtel_table(
        "fadc_pulse_shape", value, tmp_test_directory, "LSTN-01"
    )

    assert result == "fadc_pulse_shape-LSTN-01.dat"
    out_file = tmp_test_directory / result
    assert out_file.exists()
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# time amplitude"
    assert lines[1] == "-1.0 0.0"
    assert lines[2] == "0.0 0.5"
    assert lines[3] == "1.0 1.0"


def test_write_simtel_table_three_columns(tmp_test_directory):
    value = {
        "columns": ["time", "amplitude", "amplitude (low gain)"],
        "rows": [[0.0, 1.0, 0.5], [1.0, 0.0, 0.0]],
    }
    result = simtel_table_writer.write_simtel_table(
        "fadc_pulse_shape", value, tmp_test_directory, "MSTN-05"
    )

    assert result == "fadc_pulse_shape-MSTN-05.dat"
    out_file = tmp_test_directory / result
    lines = out_file.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# time amplitude amplitude (low gain)"
    assert lines[1] == "0.0 1.0 0.5"


def test_write_simtel_table_raises_on_non_dict(tmp_test_directory):
    with pytest.raises(ValueError, match="'columns' and 'rows' keys"):
        simtel_table_writer.write_simtel_table(
            "fadc_pulse_shape", "some_file.dat", tmp_test_directory, "LSTN-01"
        )


def test_write_simtel_table_raises_on_missing_rows_key(tmp_test_directory):
    with pytest.raises(ValueError, match="'columns' and 'rows' keys"):
        simtel_table_writer.write_simtel_table(
            "fadc_pulse_shape",
            {"columns": ["time", "amplitude"]},
            tmp_test_directory,
            "LSTN-01",
        )
