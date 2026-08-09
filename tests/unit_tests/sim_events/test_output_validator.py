import numpy as np
import pytest
from astropy.table import Table

from simtools.sim_events.output_validator import (
    validate_event_numbers,
    validate_reduced_event_data_file,
    validate_sim_events,
)


def _reduced_event_tables(trigger_shower_id=1):
    """Return minimal linked reduced-event tables for structural validation tests."""
    return {
        "SHOWERS": Table(
            {"file_id": np.array([1], dtype=np.uint32), "event_id": [2], "shower_id": [1]}
        ),
        "TRIGGERS": Table(
            {
                "file_id": np.array([1], dtype=np.uint32),
                "event_id": [2],
                "shower_id": [trigger_shower_id],
            }
        ),
        "FILE_INFO": Table({"file_id": np.array([1], dtype=np.uint32)}),
    }


def _mock_reduced_event_dependencies(mocker, tables):
    """Mock file access and table-schema details for reduced-event validator tests."""
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="HDF5",
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_list",
        return_value={
            name: [name]
            for name in ("SHOWERS", "TRIGGERS", "FILE_INFO", "METADATA", "SIMULATION_METADATA")
        },
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_tables", return_value=tables
    )
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_metadata_document", return_value={}
    )
    mocker.patch("simtools.sim_events.output_validator.schema.validate_dict_using_schema")
    mocker.patch("simtools.sim_events.output_validator.validate_simulation_metadata")
    return mocker.patch("simtools.sim_events.output_validator.validate_data.DataValidator")


def test_validate_reduced_event_data_file_validates_all_tables(mocker):
    mock_validator = _mock_reduced_event_dependencies(mocker, _reduced_event_tables())

    assert validate_reduced_event_data_file("events.hdf5") is True
    assert mock_validator.call_count == 3


def test_validate_reduced_event_data_file_rejects_unmatched_trigger(mocker):
    _mock_reduced_event_dependencies(mocker, _reduced_event_tables(trigger_shower_id=3))

    with pytest.raises(ValueError, match="triggers without matching showers"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_reduced_event_data_file_rejects_missing_entries(mocker):
    _mock_reduced_event_dependencies(mocker, _reduced_event_tables())
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_list",
        return_value={"SHOWERS": ["SHOWERS"], "TRIGGERS": [], "FILE_INFO": ["FILE_INFO"]},
    )

    with pytest.raises(ValueError, match="missing required entries: TRIGGERS"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_reduced_event_data_file_rejects_non_hdf5_input(mocker):
    mocker.patch(
        "simtools.sim_events.output_validator.table_handler.read_table_file_type",
        return_value="FITS",
    )

    with pytest.raises(ValueError, match="must be an HDF5 file"):
        validate_reduced_event_data_file("events.fits")


def test_validate_reduced_event_data_file_rejects_unknown_file_id(mocker):
    tables = _reduced_event_tables()
    tables["TRIGGERS"]["file_id"][0] = 3
    _mock_reduced_event_dependencies(mocker, tables)

    with pytest.raises(ValueError, match=r"references unknown file_id values: \[3\]"):
        validate_reduced_event_data_file("events.hdf5")


def test_validate_sim_events_calls_validate_event_numbers(tmp_path, monkeypatch):
    test_files = [tmp_path / f"test_{i}.hdf5" for i in range(2)]
    for f in test_files:
        f.touch()

    mock_tables = {"SHOWERS": [1, 2, 3, 4]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    assert validate_sim_events(test_files, 4) is None


def test_validate_sim_events_rejects_fits_files(tmp_path):
    test_file = tmp_path / "test_events.fits"
    test_file.touch()

    with pytest.raises(ValueError, match="Only HDF5 files"):
        validate_sim_events(str(test_file), 3)


def test_validate_event_numbers_single_file_mismatch(tmp_path, monkeypatch):
    test_file = tmp_path / "test_events.hdf5"
    test_file.touch()

    mock_tables = {"SHOWERS": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="Inconsistent event counts found in reduced event lists"):
        validate_event_numbers(str(test_file), 5)


def test_validate_event_numbers_missing_showers_table(tmp_path, monkeypatch):
    test_file = tmp_path / "test_events.hdf5"
    test_file.touch()

    mock_tables = {"OTHER_TABLE": [1, 2, 3]}
    monkeypatch.setattr(
        "simtools.sim_events.output_validator.table_handler.read_tables",
        lambda *args, **kwargs: mock_tables,
    )

    with pytest.raises(ValueError, match="SHOWERS table not found in reduced event list"):
        validate_event_numbers(str(test_file), 3)
