from pathlib import Path

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.io import ascii_handler
from simtools.production_configuration import job_grid_io


def _job_rows():
    return [
        {
            "run_number": 10,
            "primary": "gamma",
            "azimuth_angle": 45 * u.deg,
            "zenith_angle": 20 * u.deg,
            "ha": 123 * u.deg,
            "dec": -45 * u.deg,
            "energy_min": 30 * u.GeV,
            "energy_max": 10 * u.TeV,
            "cores_per_shower": 10,
            "core_scatter_max": 200 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 5 * u.deg,
            "showers_per_run": 1000,
            "nsb_rate": 0.24,
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
        }
    ]


def _metadata():
    return {
        "site": "North",
        "simulation_software": "corsika_sim_telarray",
        "coordinate_system": "ha_dec",
    }


def test_serialize_and_read_job_grid_ecsv(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"

    job_grid_io.serialize_job_grid(_job_rows(), output_file, metadata=_metadata())
    rows, metadata = job_grid_io.read_job_grid(output_file)
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert metadata["site"] == "North"
    assert metadata["job_grid_format_version"] == job_grid_io.JOB_GRID_SCHEMA.version
    assert metadata["cta"]["product"]["data"]["model"]["url"] == job_grid_io._JOB_GRID_SCHEMA_URL
    assert output_table.colnames[0] == "run_number"
    assert output_table["energy_min"].unit == u.GeV
    assert output_table["energy_max"].unit == u.GeV
    assert output_table["core_scatter_max"].unit == u.m
    assert output_table["azimuth_angle"].unit == u.deg
    assert output_table["nsb_rate"][0] == pytest.approx(0.24)
    assert rows[0]["energy_min"] == 30 * u.GeV
    assert rows[0]["cores_per_shower"] == 10
    assert rows[0]["array_layout_name"] == "CTAO-North-Alpha"
    assert rows[0]["ha"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg
    assert rows[0]["nsb_rate"].to_value(output_table["nsb_rate"].unit) == pytest.approx(0.24)
    assert metadata["job_grid_summary"]["simulation_rows"] == 1
    assert metadata["job_grid_summary"]["total_showers"] == 1000
    assert metadata["job_grid_summary"]["energy_min_used"] == "30 GeV"


def test_serialize_job_grid_and_read_multiple_rows(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows_to_write = _job_rows()
    second_row = dict(rows_to_write[0], run_number=11, array_layout_name="layout with spaces")
    rows_to_write.append(second_row)
    job_grid_io.serialize_job_grid(rows_to_write, output_file, metadata=_metadata())
    rows, _ = job_grid_io.read_job_grid(output_file)

    assert [row["run_number"] for row in rows] == [10, 11]
    assert rows[1]["array_layout_name"] == "layout with spaces"


def test_serialize_job_grid_omits_hadec_columns_when_not_provided(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows_to_write = _job_rows()
    rows_to_write[0].pop("ha")
    rows_to_write[0].pop("dec")

    job_grid_io.serialize_job_grid(rows_to_write, output_file, metadata=_metadata())
    rows, _ = job_grid_io.read_job_grid(output_file)
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert "ha" not in output_table.colnames
    assert "dec" not in output_table.colnames
    assert "ha" not in rows[0]
    assert "dec" not in rows[0]


def test_serialize_job_grid_writes_empty_grid_header(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "empty_job_grid.ecsv"

    job_grid_io.serialize_job_grid([], output_file, metadata=_metadata())
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert output_table.colnames in (
        list(job_grid_io.JOB_GRID_SCHEMA.columns),
        job_grid_io.JOB_GRID_COLUMNS,
        job_grid_io.JOB_GRID_COLUMNS + list(job_grid_io._OPTIONAL_COORDINATE_FIELDS),
        job_grid_io.JOB_GRID_COLUMNS + list(job_grid_io._OPTIONAL_STRING_FIELDS),
        job_grid_io.JOB_GRID_COLUMNS
        + list(job_grid_io._OPTIONAL_COORDINATE_FIELDS)
        + list(job_grid_io._OPTIONAL_STRING_FIELDS),
    )
    assert output_table.meta["job_grid_summary"]["simulation_rows"] == 0


def test_serialize_and_read_job_grid_with_optional_string_fields(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows.append(
        {
            **rows[0],
            "run_number": 11,
            "overwrite_model_parameters": "overwrite file.yaml",
            "scan_label": "asum220",
            "telescope": "LSTN-01",
        }
    )

    job_grid_io.serialize_job_grid(rows, output_file, metadata=_metadata())
    read_rows, _ = job_grid_io.read_job_grid(output_file)

    assert "overwrite_model_parameters" not in read_rows[0]
    assert read_rows[1]["overwrite_model_parameters"] == "overwrite file.yaml"
    assert read_rows[1]["scan_label"] == "asum220"
    assert read_rows[1]["telescope"] == "LSTN-01"


def test_job_grid_density_schema_defines_supported_columns():
    schema = ascii_handler.collect_data_from_file(
        SCHEMA_PATH / "job_grid_density.schema.yml",
    )
    table_columns = schema["data"][0]["table_columns"]
    required_columns = [column["name"] for column in table_columns if column.get("required")]
    optional_columns = [column["name"] for column in table_columns if not column.get("required")]

    assert required_columns == job_grid_io.JOB_GRID_COLUMNS
    assert optional_columns == list(
        job_grid_io._OPTIONAL_COORDINATE_FIELDS + job_grid_io._OPTIONAL_STRING_FIELDS
    )
    schema_units = {
        column["name"]: u.Unit(column["unit"]) for column in table_columns if "unit" in column
    }
    assert schema_units == job_grid_io.JOB_GRID_SCHEMA.column_units


def test_serialize_job_grid_rejects_non_ecsv_output(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.txt"

    with pytest.raises(ValueError, match="\\.ecsv"):
        job_grid_io.serialize_job_grid(_job_rows(), output_file, metadata=_metadata())


def test_serialize_job_grid_requires_nsb_rate(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows[0].pop("nsb_rate")

    with pytest.raises(KeyError, match="nsb_rate"):
        job_grid_io.serialize_job_grid(rows, output_file, metadata=_metadata())


def test_read_job_grid_rejects_non_ecsv_input(tmp_test_directory):
    input_file = Path(tmp_test_directory) / "job_grid.txt"
    input_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="\\.ecsv"):
        job_grid_io.read_job_grid(input_file)


def test_read_job_grid_validates_schema_units(tmp_test_directory):
    input_file = Path(tmp_test_directory) / "job_grid.ecsv"
    job_grid_io.serialize_job_grid(_job_rows(), input_file, metadata=_metadata())

    table = Table.read(input_file, format="ascii.ecsv")
    table["energy_min"].unit = None
    table.write(input_file, format="ascii.ecsv", overwrite=True)

    with pytest.raises(u.UnitConversionError, match="not convertible"):
        job_grid_io.read_job_grid(input_file)


def test_read_job_grid_rejects_non_integral_integer_columns(tmp_test_directory):
    input_file = Path(tmp_test_directory) / "job_grid.ecsv"
    job_grid_io.serialize_job_grid(_job_rows(), input_file, metadata=_metadata())

    table = Table.read(input_file, format="ascii.ecsv")
    table["run_number"] = [10.5]
    table.write(input_file, format="ascii.ecsv", overwrite=True)

    with pytest.raises(TypeError):
        job_grid_io.read_job_grid(input_file)


def test_read_job_grid_row_returns_correct_row(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    second_row = dict(rows[0], run_number=11, azimuth_angle=90 * u.deg)
    job_grid_io.serialize_job_grid([rows[0], second_row], output_file, metadata=_metadata())

    row, metadata = job_grid_io.read_job_grid_row(output_file, 2)

    assert row["run_number"] == 11
    assert row["azimuth_angle"] == 90 * u.deg
    assert metadata["site"] == "North"


def test_read_job_grid_row_raises_on_out_of_range(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    job_grid_io.serialize_job_grid(_job_rows(), output_file, metadata=_metadata())

    with pytest.raises(IndexError, match="out of range"):
        job_grid_io.read_job_grid_row(output_file, 5)

    with pytest.raises(IndexError, match="out of range"):
        job_grid_io.read_job_grid_row(output_file, 0)


def test_job_grid_row_to_simulate_prod_args_maps_fields():
    row = _job_rows()[0]

    args = job_grid_io.job_grid_row_to_simulate_prod_args(row)

    assert args["primary"] == "gamma"
    assert args["azimuth_angle"] == 45 * u.deg
    assert args["zenith_angle"] == 20 * u.deg
    assert args["energy_range"] == (30 * u.GeV, 10 * u.TeV)
    assert args["core_scatter"] == (10, 200 * u.m)
    assert args["view_cone"] == (0 * u.deg, 5 * u.deg)
    assert args["showers_per_run"] == 1000
    assert args["model_version"] == "7.0.0"
    assert args["array_layout_name"] == "CTAO-North-Alpha"
    assert args["corsika_le_interaction"] == "urqmd"
    assert args["corsika_he_interaction"] == "epos"
    assert args["run_number"] == 10
    assert "site" not in args


def test_job_grid_row_to_simulate_prod_args_includes_metadata_site_and_software():
    row = _job_rows()[0]
    metadata = _metadata()

    args = job_grid_io.job_grid_row_to_simulate_prod_args(row, metadata)

    assert args["site"] == "North"
    assert args["simulation_software"] == "corsika_sim_telarray"


def test_job_grid_row_to_simulate_prod_args_skips_empty_metadata():
    row = _job_rows()[0]

    args_no_meta = job_grid_io.job_grid_row_to_simulate_prod_args(row, metadata=None)
    args_empty_meta = job_grid_io.job_grid_row_to_simulate_prod_args(row, metadata={})

    for args in (args_no_meta, args_empty_meta):
        assert "site" not in args
        assert "simulation_software" not in args
