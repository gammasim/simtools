import pytest

from simtools.sim_events.output_validator import validate_event_numbers, validate_sim_events


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
