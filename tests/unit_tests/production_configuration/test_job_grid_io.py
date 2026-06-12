from pathlib import Path

import astropy.units as u
import pytest

from simtools.production_configuration.job_grid_io import (
    read_job_grid,
    serialize_job_grid,
    serialize_job_grid_stream,
)


def _job_rows():
    return [
        {
            "primary": "gamma",
            "azimuth_angle": 45 * u.deg,
            "zenith_angle": 20 * u.deg,
            "ra": 123 * u.deg,
            "dec": -45 * u.deg,
            "energy_min": 30 * u.GeV,
            "energy_max": 10 * u.TeV,
            "core_scatter_number": 10,
            "core_scatter_max": 200 * u.m,
            "view_cone_min": 0 * u.deg,
            "view_cone_max": 5 * u.deg,
            "showers_per_run": 1000,
            "nsb_rate": 0.24,
            "model_version": "7.0.0",
            "array_layout_name": "CTAO-North-Alpha",
            "corsika_le_interaction": "urqmd",
            "corsika_he_interaction": "epos",
            "run_number": 10,
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

    serialize_job_grid(_job_rows(), output_file, metadata=_metadata())
    rows, metadata = read_job_grid(output_file)

    assert metadata["site"] == "North"
    assert rows[0]["energy_min"] == 30 * u.GeV
    assert rows[0]["core_scatter_number"] == 10
    assert rows[0]["array_layout_name"] == "CTAO-North-Alpha"
    assert rows[0]["nsb_rate"] == pytest.approx(0.24)
    assert rows[0]["ra"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg


def test_serialize_job_grid_stream_and_read_job_grid_ecsv(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"

    row_count = serialize_job_grid_stream(iter(_job_rows()), output_file, metadata=_metadata())
    rows, metadata = read_job_grid(output_file)

    assert row_count == 1
    assert metadata["site"] == "North"
    assert rows[0]["energy_min"] == 30 * u.GeV
    assert rows[0]["ra"] == 123 * u.deg
    assert rows[0]["dec"] == -45 * u.deg


def test_serialize_job_grid_rejects_non_ecsv_output(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.txt"

    with pytest.raises(ValueError, match="\\.ecsv"):
        serialize_job_grid(_job_rows(), output_file, metadata=_metadata())


def test_serialize_job_grid_requires_nsb_rate(tmp_test_directory):
    output_file = Path(tmp_test_directory) / "job_grid.ecsv"
    rows = _job_rows()
    rows[0].pop("nsb_rate")

    with pytest.raises(KeyError, match="nsb_rate"):
        serialize_job_grid(rows, output_file, metadata=_metadata())


def test_read_job_grid_rejects_non_ecsv_input(tmp_test_directory):
    input_file = Path(tmp_test_directory) / "job_grid.txt"
    input_file.write_text("dummy", encoding="utf-8")

    with pytest.raises(ValueError, match="\\.ecsv"):
        read_job_grid(input_file)
