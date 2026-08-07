from pathlib import Path

import astropy.units as u
import pytest
from astropy.table import Table

from simtools.constants import SCHEMA_PATH
from simtools.io.ascii_handler import collect_data_from_file
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


def test_job_grid_schema_keeps_previous_format_document():
    schemas = collect_data_from_file(SCHEMA_PATH / "job_grid_density.schema.yml")

    assert [schema["schema_version"] for schema in schemas] == ["0.4.0", "0.3.0"]
    assert "telescope" not in job_grid_io.JOB_GRID_SCHEMA.columns
    assert "telescope" in {column["name"] for column in schemas[1]["data"][0]["table_columns"]}


def test_serialize_job_grid_writes_empty_grid_header(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "empty_job_grid.ecsv"

    job_grid_io.serialize_job_grid([], output_file, metadata=_metadata())
    output_table = Table.read(output_file, format="ascii.ecsv")

    assert output_table.colnames == [
        *job_grid_io.JOB_GRID_COLUMNS,
        *job_grid_io.JOB_GRID_SCHEMA.optional_columns,
    ]
    assert output_table.meta["job_grid_summary"]["simulation_rows"] == 0
    assert output_table.meta["job_grid_format_version"] == "0.4.0"
    assert "telescope" not in output_table.colnames


def test_serialize_and_read_job_grid_with_optional_string_fields(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows.append(
        {
            **rows[0],
            "run_number": 11,
            "overwrite_model_parameters": "overwrite file.yaml",
            "scan_label": "asum220",
        }
    )

    job_grid_io.serialize_job_grid(rows, output_file, metadata=_metadata())
    read_rows, _ = job_grid_io.read_job_grid(output_file)

    assert "overwrite_model_parameters" not in read_rows[0]
    assert read_rows[1]["overwrite_model_parameters"] == "overwrite file.yaml"
    assert read_rows[1]["scan_label"] == "asum220"


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


def test_read_job_grid_rejects_non_integral_integer_columns(tmp_test_directory):
    input_file = Path(tmp_test_directory) / "job_grid.ecsv"
    job_grid_io.serialize_job_grid(_job_rows(), input_file, metadata=_metadata())

    table = Table.read(input_file, format="ascii.ecsv")
    table["run_number"] = [10.5]
    table.write(input_file, format="ascii.ecsv", overwrite=True)

    with pytest.raises(TypeError):
        job_grid_io.read_job_grid(input_file)


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
    assert "corsika_hadronic_transition_energy" not in args
    assert args["ha"] == 123 * u.deg
    assert args["dec"] == -45 * u.deg
    assert args["run_number"] == 10
    assert "site" not in args


def test_job_grid_row_to_simulate_prod_args_omits_missing_hadec_coordinates():
    row = _job_rows()[0]
    row.pop("ha")
    row.pop("dec")

    args = job_grid_io.job_grid_row_to_simulate_prod_args(row)

    assert "ha" not in args
    assert "dec" not in args


def test_job_grid_row_to_simulate_prod_args_includes_explicit_transition_energy():
    row = _job_rows()[0]
    row["corsika_hadronic_transition_energy"] = 120 * u.GeV

    args = job_grid_io.job_grid_row_to_simulate_prod_args(row)

    assert args["corsika_hadronic_transition_energy"] == 120 * u.GeV
