#!/usr/bin/python3

import pytest
from astropy.table import QTable

import simtools.simtel.simtel_table_writer as simtel_table_writer


def test_write_mirror_segmentation(tmp_test_directory):
    result = simtel_table_writer.write_mirror_segmentation(
        [{"kind": "ring", "count": 2, "r_min_cm": 1, "r_max_cm": 2, "dphi_deg": 90}],
        tmp_test_directory / "segments.dat",
    )
    assert result == "segments.dat"
    assert "RING 2 1 2 90 0 0" in (tmp_test_directory / result).read_text(encoding="utf-8")


def test_write_ecsv_table_uses_original_filename(tmp_test_directory):
    table = QTable({"time": [0.0, 1.0], "amplitude": [0.0, 1.0]})
    table.meta["simtelarray_original_file_name"] = "pulse.dat"
    table.meta["simtelarray_table_format"] = "pulse"

    result = simtel_table_writer.write_simtel_table(table, tmp_test_directory)

    assert result == "pulse.dat"
    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "0.0 0.0",
        "1.0 1.0",
    ]


def test_write_ecsv_table_rejects_unsafe_filename(tmp_test_directory):
    table = QTable({"x": [1.0]})
    table.meta["simtelarray_original_file_name"] = "../pulse.dat"
    with pytest.raises(ValueError, match="Unsafe"):
        simtel_table_writer.write_simtel_table(table, tmp_test_directory)


def test_write_rpol_table_uses_reflectivity_column(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0, 300.0, 400.0, 400.0],
            "angle": [0.0, 10.0, 0.0, 10.0],
            "reflectivity": [0.8, 0.7, 0.9, 0.8],
            "reflectivity_rms": [0.1, 0.1, 0.1, 0.1],
        }
    )
    table.meta.update(
        {
            "simtelarray_original_file_name": "reflectivity.dat",
            "simtelarray_table_format": "rpol_matrix",
        }
    )

    result = simtel_table_writer.write_simtel_table(table, tmp_test_directory)

    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE= 0.0 10.0",
        "300.0 0.8 0.7",
        "400.0 0.9 0.8",
    ]


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
