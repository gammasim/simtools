from pathlib import Path

import astropy.units as u
import numpy as np
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
            "ra": 123 * u.deg,
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
        "coordinate_system": "ra_dec",
    }


def test_serialize_and_read_job_grid_ecsv(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"

    job_grid_io.serialize_job_grid(_job_rows(), output_file, metadata=_metadata())
    rows, metadata = job_grid_io.read_job_grid(output_file)
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert metadata["site"] == "North"
    assert output_table.colnames[0] == "run_number"
    assert rows[0]["energy_min"] == 30 * u.GeV
    assert rows[0]["cores_per_shower"] == 10
    assert rows[0]["array_layout_name"] == "CTAO-North-Alpha"
    assert rows[0]["ra"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg
    assert metadata["job_grid_summary"]["simulation_rows"] == 1
    assert metadata["job_grid_summary"]["total_showers"] == 1000
    assert metadata["job_grid_summary"]["energy_min_used"] == "30 GeV"


def test_serialize_job_grid_stream_and_read_job_grid_ecsv(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"

    row_count = job_grid_io.serialize_job_grid_stream(
        iter(_job_rows()), output_file, metadata=_metadata()
    )
    rows, metadata = job_grid_io.read_job_grid(output_file)

    assert row_count == 1
    assert metadata["site"] == "North"
    assert rows[0]["energy_min"] == 30 * u.GeV
    assert rows[0]["ra"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg


def test_serialize_job_grid_stream_appends_astropy_formatted_chunks(
    tmp_test_directory, monkeypatch
):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows_to_write = _job_rows()
    second_row = dict(rows_to_write[0], run_number=11, array_layout_name="layout with spaces")
    rows_to_write.append(second_row)
    monkeypatch.setattr(job_grid_io, "_STREAM_CHUNK_SIZE", 1)

    row_count = job_grid_io.serialize_job_grid_stream(
        iter(rows_to_write), output_file, metadata=_metadata()
    )
    rows, _ = job_grid_io.read_job_grid(output_file)

    assert row_count == 2
    assert [row["run_number"] for row in rows] == [10, 11]
    assert rows[1]["array_layout_name"] == "layout with spaces"


def test_serialize_job_grid_stream_skips_empty_optional_radec_columns(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows_to_write = _job_rows()
    rows_to_write[0]["ra"] = None
    rows_to_write[0]["dec"] = None

    job_grid_io.serialize_job_grid_stream(iter(rows_to_write), output_file, metadata=_metadata())
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert "ra" not in output_table.colnames
    assert "dec" not in output_table.colnames


def test_serialize_job_grid_stream_writes_empty_grid_header(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "empty_job_grid.ecsv"

    row_count = job_grid_io.serialize_job_grid_stream(iter([]), output_file, metadata=_metadata())
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert row_count == 0
    assert output_table.colnames == job_grid_io.JOB_GRID_COLUMNS
    assert output_table["run_number"].dtype == np.dtype("uint32")
    assert output_table["azimuth_angle_value"].dtype == np.dtype("float64")
    assert output_table["primary"].dtype.kind == "U"


def test_serialize_and_read_job_grid_with_optional_string_fields(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows[0]["overwrite_model_parameters"] = "overwrite_nsb_asum220.yaml"
    rows[0]["scan_label"] = "asum220"
    rows[0]["telescope"] = "LSTN-01"

    job_grid_io.serialize_job_grid(rows, output_file, metadata=_metadata())
    read_rows, _ = job_grid_io.read_job_grid(output_file)

    assert read_rows[0]["overwrite_model_parameters"] == "overwrite_nsb_asum220.yaml"
    assert read_rows[0]["scan_label"] == "asum220"
    assert read_rows[0]["telescope"] == "LSTN-01"


def test_serialize_job_grid_stream_with_optional_string_fields(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows[0]["overwrite_model_parameters"] = "overwrite file.yaml"
    rows[0]["scan_label"] = "asum220"
    rows[0]["telescope"] = "LSTN-01"

    job_grid_io.serialize_job_grid_stream(iter(rows), output_file, metadata=_metadata())
    read_rows, _ = job_grid_io.read_job_grid(output_file)

    assert read_rows[0]["overwrite_model_parameters"] == "overwrite file.yaml"
    assert read_rows[0]["scan_label"] == "asum220"
    assert read_rows[0]["telescope"] == "LSTN-01"


def test_job_grid_density_schema_defines_supported_columns():
    schema = ascii_handler.collect_data_from_file(
        SCHEMA_PATH / "job_grid_density.schema.yml",
    )
    table_columns = schema["data"][0]["table_columns"]
    required_columns = [column["name"] for column in table_columns if column.get("required")]
    optional_columns = [column["name"] for column in table_columns if not column.get("required")]

    assert required_columns == job_grid_io.JOB_GRID_COLUMNS
    assert optional_columns == [
        "ra",
        "dec",
        "overwrite_model_parameters",
        "scan_label",
        "telescope",
    ]
    assert all("unit" not in column for column in table_columns)


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
