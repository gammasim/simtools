#!/usr/bin/python3

from copy import deepcopy
from pathlib import Path

import astropy.units as u
import pytest
from astropy.table import QTable

import simtools.simtel.simtel_file_writer as simtel_file_writer
import simtools.simtel.table_serializers as table_serializers


def _camera_configuration():
    """Return a minimal complete camera configuration."""
    return {
        "pixel_types": [
            {
                "type_id": 1,
                "pmt_type": 0,
                "cathode_shape": 0,
                "cathode_diameter_cm": 1,
                "funnel_shape": 2,
                "funnel_diameter_cm": 1,
                "funnel_depth_cm": 0,
                "funnel_transparency": 0.9,
                "funnel_wall_reflectivity": 0.8,
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
                "module_id": 10,
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


def _contract(table_format, columns, **kwargs):
    """Build a complete test serialization contract."""
    return {
        "table_format": table_format,
        "columns": columns,
        "row_sort_keys": kwargs.pop("row_sort_keys", []),
        "float_format": kwargs.pop("float_format", ".12g"),
        "write_comments": False,
        "units": kwargs.pop("units", dict.fromkeys(columns, "dimensionless")),
        **kwargs,
    }


def test_write_ecsv_table_uses_explicit_filename(tmp_test_directory):
    table = QTable({"time": [0.0, 1.0], "amplitude": [0.0, 1.0]})

    result = table_serializers.write_simtel_table(
        table,
        tmp_test_directory,
        output_name="pulse.dat",
        contract=_contract("pulse", ["time", "amplitude"], float_format=".1f"),
    )

    assert result == "pulse.dat"
    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "0.0 0.0",
        "1.0 1.0",
    ]


def test_write_ecsv_table_rejects_unsafe_filename(tmp_test_directory):
    table = QTable({"time": [0.0], "amplitude": [1.0]})
    with pytest.raises(ValueError, match="Unsafe"):
        table_serializers.write_simtel_table(
            table,
            tmp_test_directory,
            output_name="../pulse.dat",
            contract=_contract("pulse", ["time", "amplitude"]),
        )


def test_write_rpol_table_uses_reflectivity_column(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0, 300.0, 400.0, 400.0],
            "angle": [0.0, 10.0, 0.0, 10.0],
            "reflectivity": [0.8, 0.7, 0.9, 0.8],
            "reflectivity_rms": [0.1, 0.1, 0.1, 0.1],
        }
    )
    result = table_serializers.write_simtel_table(
        table,
        tmp_test_directory,
        contract=_contract(
            "rpol_matrix",
            ["wavelength", "reflectivity"],
            allowed_columns=["wavelength", "angle", "reflectivity", "reflectivity_rms"],
            row_sort_keys=["wavelength", "angle"],
            matrix_axes=["wavelength", "angle"],
            value_column="reflectivity",
            float_format=".1f",
        ),
    )

    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "#@RPOL@[ANGLE=] 2",
        "ANGLE= 0.0 10.0",
        "300.0 0.8 0.7",
        "400.0 0.9 0.8",
    ]


def test_write_rpol_table_preserves_one_dimensional_table(tmp_test_directory):
    table = QTable({"wavelength": [300.0, 400.0], "reflectivity": [0.8, 0.9]})
    result = table_serializers.write_simtel_table(
        table,
        tmp_test_directory,
        contract=_contract(
            "rpol_matrix",
            ["wavelength", "reflectivity"],
            allowed_columns=["wavelength", "angle", "reflectivity"],
            row_sort_keys=["wavelength", "angle"],
            matrix_axes=["wavelength", "angle"],
            value_column="reflectivity",
            float_format=".1f",
        ),
    )

    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "300.0 0.8",
        "400.0 0.9",
    ]


def test_write_atmospheric_transmission_groups_rows_by_wavelength(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0, 300.0, 400.0, 400.0],
            "altitude": [2.0, 1.0, 2.0, 1.0],
            "extinction": [0.2, 0.1, 0.4, 0.3],
        }
    )
    table.meta["observatory_level"] = 1.5 * u.km

    result = table_serializers.write_simtel_table(
        table,
        tmp_test_directory,
        contract=_contract(
            "atmospheric_transmission",
            ["wavelength", "altitude", "extinction"],
            row_sort_keys=["wavelength", "altitude"],
            matrix_axes=["wavelength", "altitude"],
            value_column="extinction",
            float_format=".1f",
        ),
    )

    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "# H2= 1.5, H1= 1.0 2.0",
        "300.0 0.1 0.2",
        "400.0 0.3 0.4",
    ]


def test_write_atmospheric_transmission_fills_sparse_cells(tmp_test_directory):
    table = QTable(
        {
            "wavelength": [300.0, 400.0, 400.0],
            "altitude": [1.0, 1.0, 2.0],
            "extinction": [0.1, 0.3, 0.4],
        }
    )
    result = table_serializers.write_simtel_table(
        table,
        tmp_test_directory,
        contract=_contract(
            "atmospheric_transmission",
            ["wavelength", "altitude", "extinction"],
            row_sort_keys=["wavelength", "altitude"],
            matrix_axes=["wavelength", "altitude"],
            value_column="extinction",
            missing_value=99999,
            float_format=".1f",
        ),
    )

    assert (tmp_test_directory / result).read_text(encoding="utf-8").splitlines() == [
        "# H1= 1.0 2.0",
        "300.0 0.1 99999",
        "400.0 0.3 0.4",
    ]


def test_write_simtel_table_rejects_structured_table_payload(tmp_test_directory):
    with pytest.raises(TypeError, match="Astropy table"):
        table_serializers.write_simtel_table(
            {"columns": ["time", "amplitude"], "rows": [[0.0, 1.0]]},
            tmp_test_directory,
            contract=_contract("pulse", ["time", "amplitude"]),
        )


def test_write_simtel_table_is_invariant_to_input_row_order(tmp_test_directory):
    contract = _contract("plain", ["time", "amplitude"], row_sort_keys=["time"], float_format=".3f")
    first = QTable({"time": [2.0, 0.0, 1.0], "amplitude": [0.2, 0.0, 0.1]})
    second = QTable({"time": [1.0, 2.0, 0.0], "amplitude": [0.1, 0.2, 0.0]})

    table_serializers.write_simtel_table(
        first, tmp_test_directory, contract=contract, output_name="first.dat"
    )
    table_serializers.write_simtel_table(
        second, tmp_test_directory, contract=contract, output_name="second.dat"
    )

    assert (tmp_test_directory / "first.dat").read_text(encoding="utf-8") == (
        tmp_test_directory / "second.dat"
    ).read_text(encoding="utf-8")


def test_validate_simtel_serialization_rejects_incomplete_matrix():
    table = QTable(
        {
            "wavelength": [300.0, 400.0, 400.0],
            "incidence_angle": [0.0, 0.0, 10.0],
            "reflectivity": [0.8, 0.9, 0.8],
        }
    )
    contract = _contract(
        "rpol_matrix",
        ["wavelength", "reflectivity"],
        allowed_columns=["wavelength", "incidence_angle", "reflectivity"],
        matrix_axes=["wavelength", "incidence_angle"],
        value_column="reflectivity",
    )

    with pytest.raises(ValueError, match="complete Cartesian grid"):
        table_serializers.validate_simtel_serialization(table, contract)


def test_validate_simtel_serialization_rejects_undeclared_column():
    table = QTable({"time": [0.0], "amplitude": [1.0], "unexpected": [2.0]})
    with pytest.raises(ValueError, match="undeclared columns"):
        table_serializers.validate_simtel_serialization(
            table,
            _contract("plain", ["time", "amplitude"], row_sort_keys=["time"]),
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

    result = simtel_file_writer.write_camera_file(configuration, tmp_test_directory / "camera.dat")

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
        simtel_file_writer.write_camera_file(configuration, tmp_test_directory / "camera.dat")


def test_write_camera_file_supports_transparency_and_rejects_unsafe_path(tmp_test_directory):
    configuration = _camera_configuration()
    result = simtel_file_writer.write_camera_file(
        configuration, tmp_test_directory / "nested" / "camera.dat"
    )
    assert "0.9 0.8" in (tmp_test_directory / "nested" / result).read_text(encoding="utf-8")

    with pytest.raises(ValueError, match="Unsafe camera configuration path"):
        simtel_file_writer.write_camera_file(
            configuration, Path(tmp_test_directory).joinpath("..", "camera.dat")
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.clear(), "requires pixel types"),
        (
            lambda config: config["pixel_types"].append(deepcopy(config["pixel_types"][0])),
            "type IDs",
        ),
        (lambda config: config["pixels"][0].update(pixel_id=1), "contiguous"),
        (lambda config: config["pixels"][0].update(enabled=0), "enabled pixel"),
        (lambda config: config["pixels"][0].update(type_id=99), "unknown pixel type"),
    ],
)
def test_camera_validation_rejects_invalid_components(mutation, message):
    configuration = _camera_configuration()
    mutation(configuration)

    with pytest.raises(ValueError, match=message):
        simtel_file_writer._validate_camera_components(configuration)


def test_camera_validation_requires_resolved_pixel_type_data():
    configuration = _camera_configuration()
    configuration["pixel_types"][0].pop("funnel_transparency")
    configuration["pixel_types"][0].pop("funnel_wall_reflectivity")

    with pytest.raises(ValueError, match="no resolved lightguide"):
        simtel_file_writer._validate_camera_components(configuration)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda config: config.update(triggers=[{"group_id": 1, "kind": "majority"}]), "group IDs"),
        (
            lambda config: config.update(
                triggers=[{"group_id": 0, "kind": "unknown", "use_default_multiplicity": True}],
                trigger_members=[
                    {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0}
                ],
            ),
            "Unsupported",
        ),
        (
            lambda config: config.update(
                triggers=[
                    {
                        "group_id": 0,
                        "kind": "majority",
                        "use_default_multiplicity": True,
                        "multiplicity": 1,
                    }
                ],
                trigger_members=[
                    {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0}
                ],
            ),
            "must not be positive",
        ),
        (
            lambda config: config.update(
                triggers=[
                    {
                        "group_id": 0,
                        "kind": "majority",
                        "use_default_multiplicity": False,
                        "multiplicity": 0,
                    }
                ],
                trigger_members=[
                    {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0}
                ],
            ),
            "must be positive",
        ),
    ],
)
def test_camera_validation_rejects_invalid_triggers(mutation, message):
    configuration = _camera_configuration()
    mutation(configuration)

    with pytest.raises(ValueError, match=message):
        simtel_file_writer._validate_camera_components(configuration)


def test_camera_validation_rejects_trigger_member_references():
    configuration = _camera_configuration()
    configuration["triggers"] = [
        {"group_id": 0, "kind": "majority", "use_default_multiplicity": True}
    ]
    configuration["trigger_members"] = [
        {"group_id": 1, "member_order": 0, "pixel_order": 0, "pixel_id": 0}
    ]
    with pytest.raises(ValueError, match="unknown group"):
        simtel_file_writer._validate_camera_components(configuration)

    configuration["trigger_members"][0]["group_id"] = 0
    configuration["trigger_members"][0]["member_order"] = 1
    with pytest.raises(ValueError, match="member orders"):
        simtel_file_writer._validate_camera_components(configuration)


def test_camera_validation_rejects_trigger_pixel_order_and_requirements():
    configuration = _camera_configuration()
    configuration["triggers"] = [
        {"group_id": 0, "kind": "majority", "use_default_multiplicity": True}
    ]
    configuration["trigger_members"] = [
        {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 0}
    ]
    with pytest.raises(ValueError, match="pixel orders"):
        simtel_file_writer._validate_camera_components(configuration)

    configuration["trigger_members"] = [
        {"group_id": 0, "member_order": 0, "pixel_order": 0, "pixel_id": 0, "required": True},
        {"group_id": 0, "member_order": 0, "pixel_order": 1, "pixel_id": 0, "required": True},
    ]
    with pytest.raises(ValueError, match="first pixel"):
        simtel_file_writer._validate_camera_components(configuration)

    configuration["trigger_members"][1]["required"] = False
    configuration["trigger_members"][1]["pixel_id"] = 99
    with pytest.raises(ValueError, match="unknown pixel ID"):
        simtel_file_writer._validate_camera_components(configuration)


def test_camera_validation_rejects_empty_trigger_group():
    configuration = _camera_configuration()
    configuration["triggers"] = [
        {"group_id": 0, "kind": "majority", "use_default_multiplicity": True}
    ]

    with pytest.raises(ValueError, match="has no members"):
        simtel_file_writer._validate_camera_components(configuration)


def test_camera_helpers_validate_names_and_module_ids():
    with pytest.raises(ValueError, match="Unsafe lightguide"):
        simtel_file_writer._safe_basename("nested/angle.dat", "lightguide")
    with pytest.raises(ValueError, match="Invalid camera module ID"):
        simtel_file_writer._module_id(-1)
    assert simtel_file_writer._module_id("0x0a") == "0xa"


def test_write_light_pulse_table_gauss_exp_conv(tmp_test_directory):
    output = simtel_file_writer.write_light_pulse_table_gauss_exp_conv(
        tmp_test_directory / "pulse.dat",
        width_ns=1.0,
        exp_decay_ns=2.0,
        fadc_sum_bins=1,
        dt_ns=1.0,
    )

    assert output == tmp_test_directory / "pulse.dat"
    assert (
        (tmp_test_directory / "pulse.dat")
        .read_text(encoding="utf-8")
        .startswith("# time[ns] amplitude\n")
    )


@pytest.mark.parametrize("missing", ["width", "decay"])
def test_write_light_pulse_table_requires_shape_parameters(tmp_test_directory, missing):
    width = None if missing == "width" else 1.0
    decay = None if missing == "decay" else 2.0
    with pytest.raises(ValueError, match="required"):
        simtel_file_writer.write_light_pulse_table_gauss_exp_conv(
            tmp_test_directory / "pulse.dat", width, decay, fadc_sum_bins=1
        )


def test_write_angular_distribution_table_lambertian(tmp_test_directory):
    output = simtel_file_writer.write_angular_distribution_table_lambertian(
        tmp_test_directory / "angles.dat", max_angle_deg=120, n_samples=4
    )

    assert output == tmp_test_directory / "angles.dat"
    lines = (tmp_test_directory / "angles.dat").read_text(encoding="utf-8").splitlines()
    assert lines[0] == "# angle[deg] relative_intensity"
    assert lines[-1].endswith("0.00000000")


def test_write_angular_distribution_handles_zero_intensity_maximum(tmp_test_directory, monkeypatch):
    simtools_table = tmp_test_directory / "angles.dat"
    monkeypatch.setattr(simtel_file_writer.np, "linspace", lambda *_args, **_kwargs: [180.0])
    simtel_file_writer.write_angular_distribution_table_lambertian(
        simtools_table, max_angle_deg=180, n_samples=1
    )

    assert simtools_table.read_text(encoding="utf-8").splitlines()[-1].endswith("0.00000000")


def test_write_ascii_table_helpers(tmp_test_directory):
    pulse_path = simtel_file_writer.write_ascii_pulse_table(
        tmp_test_directory / "pulse.dat", [0.0, 1.0], [0.5, 1.0]
    )
    angle_path = simtel_file_writer.write_ascii_angle_distribution_table(
        tmp_test_directory / "angles.dat", [0.0, 10.0], [1.0, 0.5]
    )

    assert pulse_path.name == "pulse.dat"
    assert angle_path.name == "angles.dat"
    assert "0.000000 0.50000000" in pulse_path.read_text(encoding="utf-8")
    assert "10.000000 0.50000000" in angle_path.read_text(encoding="utf-8")
