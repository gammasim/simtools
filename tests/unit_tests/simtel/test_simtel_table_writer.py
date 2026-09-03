#!/usr/bin/python3

import pytest
from astropy.table import QTable

import simtools.simtel.simtel_table_writer as simtel_table_writer


def test_write_mirror_segmentation(tmp_test_directory):
    result = simtel_table_writer.write_mirror_segmentation(
        [{"kind": "ring", "count": 2, "r_min_cm": 1, "r_max_cm": 2, "dphi_deg": 90}],
        tmp_test_directory / "segments.dat",
        "primary_mirror_segmentation",
        "0.2.0",
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


def test_write_camera_file_preserves_pixel_fields_and_members(tmp_test_directory):
    configuration = {
        "rotate": 10.893,
        "pixel_types": [
            {
                "type_id": 1,
                "pmt_type": 0,
                "cathode_shape": 0,
                "cathode_diameter_cm": 2.5,
                "funnel_shape": 3,
                "funnel_diameter_cm": 4.9,
                "funnel_depth_cm": 5.1,
                "lightguide_angle_file": "angle.dat",
                "lightguide_wavelength_file": "wavelength.dat",
            }
        ],
        "pixels": [
            {
                "pixel_id": 0,
                "type_id": 1,
                "x_cm": 1.0,
                "y_cm": 2.0,
                "module": 3,
                "board": 4,
                "channel": 5,
                "module_id": "0x0a",
                "enabled": 1,
                "relative_qe": 0.9,
                "relative_gain": 1.1,
                "z_offset_cm": 0.2,
                "rotation_deg": 3.0,
                "normal_x": 0.01,
                "normal_y": -0.02,
            },
            {
                "pixel_id": 1,
                "type_id": 1,
                "x_cm": 3.0,
                "y_cm": 4.0,
                "module": 3,
                "board": 4,
                "channel": 6,
                "module_id": "0x0a",
                "enabled": 1,
                "relative_qe": 0.9,
                "relative_gain": 1.1,
                "z_offset_cm": 0.2,
                "rotation_deg": 3.0,
                "normal_x": 0.01,
                "normal_y": -0.02,
            },
            {
                "pixel_id": 2,
                "type_id": 1,
                "x_cm": 5.0,
                "y_cm": 6.0,
                "module": 3,
                "board": 4,
                "channel": 7,
                "module_id": "0x0a",
                "enabled": 1,
                "relative_qe": 0.9,
                "relative_gain": 1.1,
                "z_offset_cm": 0.2,
                "rotation_deg": 3.0,
                "normal_x": 0.01,
                "normal_y": -0.02,
            },
        ],
        "triggers": [
            {
                "group_id": 0,
                "kind": "majority",
                "use_default_multiplicity": 1,
                "multiplicity": 0,
            }
        ],
        "trigger_members": [
            {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0, "required": 1},
            {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 1, "required": 0},
            {"group_id": 0, "member_order": 1, "pixel_order": 0, "pixel_id": 2, "required": 0},
        ],
    }

    result = simtel_table_writer.write_camera_file(configuration, tmp_test_directory / "camera.dat")

    assert result == "camera.dat"
    lines = (tmp_test_directory / result).read_text(encoding="utf-8").splitlines()
    assert lines[1] == "Rotate 10.893"
    assert lines[2].endswith('"wavelength.dat"')
    assert lines[3].split() == [
        "Pixel",
        "0",
        "1",
        "1.0",
        "2.0",
        "3",
        "4",
        "5",
        "0xa",
        "1",
        "0.9",
        "1.1",
        "0.2",
        "3.0",
        "0.01",
        "-0.02",
    ]
    assert lines[6] == "MajorityTrigger * of +0[1] 2"


def test_write_camera_file_rejects_invalid_trigger_and_module_id(tmp_test_directory):
    configuration = {
        "pixel_types": [
            {
                "type_id": 1,
                "pmt_type": 0,
                "cathode_shape": 0,
                "cathode_diameter_cm": 1,
                "funnel_shape": 2,
                "funnel_diameter_cm": 1,
                "funnel_depth_cm": 0,
                "lightguide_angle_file": "angle.dat",
            }
        ],
        "pixels": [
            {
                "pixel_id": 0,
                "type_id": 1,
                "x_cm": 0,
                "y_cm": 0,
                "module": 0,
                "board": 0,
                "channel": 0,
                "module_id": "bad",
                "enabled": 1,
                "relative_qe": 1,
                "relative_gain": 1,
                "z_offset_cm": 0,
                "rotation_deg": 0,
                "normal_x": 0,
                "normal_y": 0,
            }
        ],
        "triggers": [],
        "trigger_members": [],
    }
    with pytest.raises(ValueError, match="Invalid camera module ID"):
        simtel_table_writer.write_camera_file(configuration, tmp_test_directory / "camera.dat")
